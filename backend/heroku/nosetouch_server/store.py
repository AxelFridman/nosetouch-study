"""Idempotent storage of uploads, deletion on request, and the data of the admin page.

The phone retries every upload until it reads an answer, and re-sends the session record after every
trial, so storing is an upsert by primary key with these rules (all answered 200 ok, so the phone
stops sending):
  sessions      a different version replaces the stored one only if its updated_at is later (equal
                updated_at: the later envelope sent_at wins); an older retry is ignored ('kept_newer').
  trials        a trial_id is one attempt. An identical re-send changes nothing ('unchanged'). A
                different body is a newer version of the same attempt. The app saves the copies of an
                attempt in this order (copy_rank): a placeholder before the participant answered
                (outcome.note 'review_interrupted': tap review; 'hand_not_reported': the v2 choice trial
                before its hand question), then the decided copy, and possibly later a revoked one
                ('redo_requested': "Redo last movement" / "Last one was the wrong hand", final). A copy
                later in that order always replaces an earlier one and never the other way round
                (docs/02 section 10, motionlab.io); between copies of the same rank the later envelope
                sent_at wins. receiver.py and Apps Script simply overwrite; this keeps their
                last-write-wins result for the app's own sequence while an out-of-order or stale copy can
                no longer undo a decision or a revocation.
  participants  no envelope of their own: the profile travels in each session (`participant`, already
                redacted by the app). The newest profile by its updated_at is kept.
Every outcome is logged in ingest_log (identifiers and sizes only).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import case, delete, func, or_, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError

from .db import Database, ingest_log, participants, sessions, trials, utc
from .validation import (
    PLACEHOLDER_NOTES,
    REDO_REQUESTED_NOTE,
    as_int,
    envelope_identifiers,
    record_path,
    time_key,
)

TEXT_MAX = 200
_RETRYABLE_SQLSTATES = frozenset({'40001', '40P01', '23505'})  # serialization, deadlock, unique race
BIGINT_MAX = 2**63 - 1


@dataclass(frozen=True)
class Stored:
    outcome: str  # stored | replaced | unchanged | kept_newer
    kind: str
    record_id: str
    stored_as: str
    reason: str | None = None  # why a different re-send replaced or did not replace the stored copy


# ───────────────────────────── derived columns ─────────────────────────────


def _text(v: Any, limit: int = TEXT_MAX) -> str | None:
    """A string field for a derived column (anything else -> None), cut to `limit` characters."""
    if not isinstance(v, str):
        return None
    v = v.encode('utf-8', 'replace').decode('utf-8').replace('\x00', '')
    return v[:limit]


def _bool(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _int(v: Any) -> int | None:
    n = as_int(v)
    return n if n is not None and abs(n) <= BIGINT_MAX else None


def _get(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def session_columns(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        'session_id': rec['session_id'],
        'participant_id': rec['participant_id'],
        'study_code': rec['study_code'],
        'status': _text(rec.get('status'), 32),
        'started_at': _text(rec.get('started_at'), 40),
        'updated_at': _text(rec.get('updated_at'), 40),
        'ended_at': _text(rec.get('ended_at'), 40),
        'simulated': _bool(rec.get('simulated')),
        'app_version': _text(rec.get('app_version'), 64),
        'schema_version': _text(rec.get('schema_version'), 32),
        'platform': _text(_get(rec, 'device', 'platform'), 32),
        'browser': _text(_get(rec, 'device', 'browser'), 80),
        'preset': _text(_get(rec, 'protocol', 'preset'), 32),
        'n_planned': _int(_get(rec, 'protocol', 'n_planned_trials')),
        'n_attempts': _int(_get(rec, 'progress', 'n_attempts')),
        'n_accepted': _int(_get(rec, 'progress', 'n_accepted')),
    }


#: copy_rank values: where a copy stands in the app's save sequence of one attempt (see the module doc)
PLACEHOLDER, DECIDED, REVOKED = 0, 1, 2


def copy_rank(rec: Any) -> int:
    """PLACEHOLDER (saved before the participant answered), DECIDED, or REVOKED (redo_requested)."""
    note = _get(rec, 'outcome', 'note') if isinstance(rec, dict) else None
    if note in PLACEHOLDER_NOTES:
        return PLACEHOLDER
    if note == REDO_REQUESTED_NOTE:
        return REVOKED
    return DECIDED


def is_review_interrupted(rec: dict[str, Any]) -> bool:
    """The `review_interrupted` column: a placeholder copy (review_interrupted or hand_not_reported)."""
    return copy_rank(rec) == PLACEHOLDER


def trial_columns(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        'trial_id': rec['trial_id'],
        'session_id': rec['session_id'],
        'participant_id': rec['participant_id'],
        'study_code': rec['study_code'],
        'trial_index': as_int(rec['trial_index']),
        'attempt': as_int(rec['attempt']),
        'hand': _text(_get(rec, 'condition', 'hand'), 16),
        'orientation': _text(_get(rec, 'condition', 'orientation'), 32),
        'speed': _text(_get(rec, 'condition', 'speed'), 16),
        'variant': _text(_get(rec, 'condition', 'variant'), 32),
        'status': _text(_get(rec, 'outcome', 'status'), 32),
        'accepted': _bool(_get(rec, 'outcome', 'accepted')),
        'review_interrupted': is_review_interrupted(rec),
        'simulated': _bool(rec.get('simulated')),
        'created_at': _text(rec.get('created_at'), 40),
        'app_version': _text(rec.get('app_version'), 64),
        'schema_version': _text(rec.get('schema_version'), 32),
    }


def _profile_digest(profile: dict[str, Any]) -> str:
    data = json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8', 'replace')).hexdigest()


# ───────────────────────────── decisions ─────────────────────────────


def session_replaces(inc_updated: Any, inc_sent: Any, old_updated: Any, old_sent: Any) -> tuple[bool, str]:
    """Does an incoming (different) session version replace the stored one? (decision, reason)"""
    a, b = time_key(inc_updated), time_key(old_updated)
    if a != b:
        return (True, 'newer_updated_at') if a > b else (False, 'older_updated_at')
    if time_key(inc_sent) >= time_key(old_sent):
        return True, 'newer_sent_at'
    return False, 'older_sent_at'


def trial_replaces(inc_rank: int, inc_sent: Any, old_rank: int, old_sent: Any) -> tuple[bool, str]:
    """Does an incoming (different) copy of a trial replace the stored one? (decision, reason)

    Ranks are copy_rank(): a copy later in the app's save sequence wins whatever the envelope times
    (an out-of-order retry can neither undo a decision nor a revocation); within a rank the later
    envelope sent_at wins."""
    if inc_rank > old_rank:
        return True, ('revoked_copy' if inc_rank == REVOKED else 'decided_copy')
    if inc_rank < old_rank:
        return False, ('kept_revoked' if old_rank == REVOKED else 'review_interrupted_copy')
    if time_key(inc_sent) >= time_key(old_sent):
        return True, 'newer_sent_at'
    return False, 'older_sent_at'


# ───────────────────────────── ingest ─────────────────────────────


def store_envelope(db: Database, env: dict[str, Any], *, digest: str, size: int, body_bytes: int,
                   body_sha256: str | None, received_at: _dt.datetime) -> Stored:
    """Stores one validated envelope (see the module doc). `digest`/`size`: canonical record hash and
    size (validation.canonical_record); `body_bytes`/`body_sha256`: the request body, for the log."""
    kind = env['kind']
    rec = env['record']
    folder, name = record_path(env)
    sent_at = _text(env.get('sent_at'), 40)
    ids = envelope_identifiers(env)
    for attempt in range(3):
        try:
            with db.write() as conn:
                if kind == 'session':
                    outcome, reason = _upsert_session(db, conn, rec, sent_at, digest, size, received_at)
                    profile = rec.get('participant')
                else:
                    outcome, reason = _upsert_trial(db, conn, rec, sent_at, digest, size, received_at)
                    profile = None
                _touch_participant(db, conn, rec['participant_id'], rec['study_code'], profile, received_at)
                db.log_entry(conn, at=received_at, outcome=outcome, code=reason, ids=ids, size=body_bytes,
                             sha256=body_sha256)
            break
        except DBAPIError as e:
            if attempt < 2 and getattr(e.orig, 'sqlstate', None) in _RETRYABLE_SQLSTATES:
                continue
            raise
    record_id = rec['session_id'] if kind == 'session' else rec['trial_id']
    return Stored(outcome, kind, record_id, f'{folder}/{name}', reason)


def _upsert_session(db: Database, conn: Connection, rec: dict[str, Any], sent_at: str | None, digest: str,
                    size: int, now: _dt.datetime) -> tuple[str, str | None]:
    cols = session_columns(rec)
    values = {**cols, 'record': rec, 'sent_at': sent_at, 'body_sha256': digest, 'size_bytes': size,
              'first_received_at': now, 'last_received_at': now}
    if db.insert_ignore(conn, sessions, values):
        return 'stored', None
    pk = sessions.c.session_id == rec['session_id']
    old = conn.execute(
        select(sessions.c.updated_at, sessions.c.sent_at, sessions.c.body_sha256).where(pk).with_for_update()
    ).one()
    if old.body_sha256 == digest:
        return 'unchanged', None
    replace, reason = session_replaces(cols['updated_at'], sent_at, old.updated_at, old.sent_at)
    if not replace:
        return 'kept_newer', reason
    values.pop('first_received_at')
    conn.execute(update(sessions).where(pk).values(**values))
    return 'replaced', reason


def _upsert_trial(db: Database, conn: Connection, rec: dict[str, Any], sent_at: str | None, digest: str,
                  size: int, now: _dt.datetime) -> tuple[str, str | None]:
    cols = trial_columns(rec)
    values = {**cols, 'record': rec, 'sent_at': sent_at, 'body_sha256': digest, 'size_bytes': size,
              'first_received_at': now, 'received_at': now}
    if db.insert_ignore(conn, trials, values):
        return 'stored', None
    pk = trials.c.trial_id == rec['trial_id']
    old = conn.execute(
        select(trials.c.review_interrupted, trials.c.sent_at, trials.c.body_sha256).where(pk).with_for_update()
    ).one()
    if old.body_sha256 == digest:
        return 'unchanged', None
    # a stored placeholder is known from its column; otherwise its note tells decided from revoked (this
    # reads the stored body, only for the rare different re-send of an attempt)
    old_rank = PLACEHOLDER if old.review_interrupted else copy_rank(conn.execute(select(trials.c.record).where(pk)).scalar_one())
    replace, reason = trial_replaces(copy_rank(rec), sent_at, old_rank, old.sent_at)
    if not replace:
        return 'kept_newer', reason
    values.pop('first_received_at')
    conn.execute(update(trials).where(pk).values(**values))
    return 'replaced', reason


def _touch_participant(db: Database, conn: Connection, pid: str, study_code: str, profile: Any,
                       now: _dt.datetime) -> None:
    prof = profile if isinstance(profile, dict) and profile.get('participant_id') == pid else None
    prof_updated = _text(prof.get('updated_at'), 40) if prof is not None else None
    prof_digest = _profile_digest(prof) if prof is not None else None
    values = {'participant_id': pid, 'study_code': study_code, 'record': prof, 'profile_updated_at': prof_updated,
              'profile_sha256': prof_digest, 'first_seen': now, 'last_seen': now}
    if db.insert_ignore(conn, participants, values):
        return
    pk = participants.c.participant_id == pid
    old = conn.execute(
        select(participants.c.profile_updated_at, participants.c.profile_sha256).where(pk).with_for_update()
    ).one()
    changes: dict[str, Any] = {'last_seen': now}
    if prof is not None and old.profile_sha256 != prof_digest and (
        old.profile_sha256 is None or time_key(prof_updated) >= time_key(old.profile_updated_at)
    ):
        changes.update(record=prof, profile_updated_at=prof_updated, profile_sha256=prof_digest, study_code=study_code)
    conn.execute(update(participants).where(pk).values(**changes))


# ───────────────────────────── deletion ─────────────────────────────


def delete_participant(db: Database, pid: str) -> dict[str, int]:
    """Deletes a participant with every session and trial (also trials filed under one of their
    sessions) and their ingest_log rows. Returns the number of rows deleted per table."""
    with db.write() as conn:
        sids = list(conn.execute(select(sessions.c.session_id).where(sessions.c.participant_id == pid)).scalars())
        n_trials = conn.execute(
            delete(trials).where(or_(trials.c.participant_id == pid, trials.c.session_id.in_(sids)))
        ).rowcount
        n_sessions = conn.execute(delete(sessions).where(sessions.c.participant_id == pid)).rowcount
        n_participants = conn.execute(delete(participants).where(participants.c.participant_id == pid)).rowcount
        n_log = conn.execute(delete(ingest_log).where(ingest_log.c.participant_id == pid)).rowcount
    return {'participants': n_participants or 0, 'sessions': n_sessions or 0, 'trials': n_trials or 0,
            'ingest_log': n_log or 0}


# ───────────────────────────── admin page data ─────────────────────────────


def overview(db: Database, now: _dt.datetime, *, n_sessions: int = 25, n_trials: int = 40,
             n_log: int = 40) -> dict[str, Any]:
    """Everything the admin page shows, from small columns (never the trial streams)."""
    day_ago = now - _dt.timedelta(days=1)
    with db.read() as conn:
        counts = db.counts(conn)
        trial_totals = conn.execute(select(
            func.count(),
            func.coalesce(func.sum(case((trials.c.accepted.is_(True), 1), else_=0)), 0),
            func.coalesce(func.sum(case((trials.c.simulated.is_(True), 1), else_=0)), 0),
            func.coalesce(func.sum(trials.c.size_bytes), 0),
            func.max(trials.c.received_at),
        )).one()
        session_status = {
            (status or '?'): n
            for status, n in conn.execute(select(sessions.c.status, func.count()).group_by(sessions.c.status)).all()
        }
        last_session = conn.execute(select(func.max(sessions.c.last_received_at))).scalar_one()
        people = conn.execute(select(
            participants.c.participant_id, participants.c.study_code, participants.c.record,
            participants.c.first_seen, participants.c.last_seen,
        ).order_by(participants.c.last_seen.desc())).all()
        session_rows = conn.execute(select(
            sessions.c.participant_id, sessions.c.platform, sessions.c.browser, sessions.c.simulated,
            sessions.c.last_received_at,
        )).all()
        trial_groups = conn.execute(select(
            trials.c.participant_id, trials.c.hand, trials.c.accepted, trials.c.simulated, func.count(),
        ).group_by(trials.c.participant_id, trials.c.hand, trials.c.accepted, trials.c.simulated)).all()
        recent_sessions = conn.execute(select(
            sessions.c.session_id, sessions.c.participant_id, sessions.c.status, sessions.c.preset,
            sessions.c.n_planned, sessions.c.n_attempts, sessions.c.n_accepted, sessions.c.platform,
            sessions.c.browser, sessions.c.simulated, sessions.c.app_version, sessions.c.started_at,
            sessions.c.last_received_at,
        ).order_by(sessions.c.last_received_at.desc()).limit(n_sessions)).all()
        recent_trials = conn.execute(select(
            trials.c.trial_id, trials.c.session_id, trials.c.participant_id, trials.c.trial_index,
            trials.c.attempt, trials.c.hand, trials.c.orientation, trials.c.speed, trials.c.variant,
            trials.c.status, trials.c.accepted, trials.c.simulated, trials.c.size_bytes, trials.c.received_at,
        ).order_by(trials.c.received_at.desc()).limit(n_trials)).all()
        log_rows = conn.execute(select(
            ingest_log.c.at, ingest_log.c.outcome, ingest_log.c.code, ingest_log.c.kind,
            ingest_log.c.participant_id, ingest_log.c.record_id, ingest_log.c.app_version, ingest_log.c.bytes,
        ).where(ingest_log.c.outcome.not_in(('stored', 'replaced', 'unchanged')))
            .order_by(ingest_log.c.id.desc()).limit(n_log)).all()
        last_day = dict(conn.execute(
            select(ingest_log.c.outcome, func.count()).where(ingest_log.c.at >= day_ago).group_by(ingest_log.c.outcome)
        ).all())
        db_size = db.size_bytes(conn)

    per: dict[str, dict[str, Any]] = {}
    for p in people:
        rec = p.record if isinstance(p.record, dict) else {}
        per[p.participant_id] = {
            'participant_id': p.participant_id, 'study_code': p.study_code, 'profile': rec,
            'first_seen': utc(p.first_seen), 'last_seen': utc(p.last_seen),
            'sessions': 0, 'sessions_simulated': 0, 'trials': 0, 'accepted': 0, 'trials_simulated': 0,
            'left': 0, 'right': 0, 'left_accepted': 0, 'right_accepted': 0,
            'platform': None, 'browser': None, '_latest': None,
        }
    for s in session_rows:
        row = per.get(s.participant_id)
        if row is None:
            continue
        row['sessions'] += 1
        row['sessions_simulated'] += 1 if s.simulated else 0
        at = utc(s.last_received_at)
        if row['_latest'] is None or (at is not None and at > row['_latest']):
            row['_latest'], row['platform'], row['browser'] = at, s.platform, s.browser
    for pid, hand, accepted, simulated, n in trial_groups:
        row = per.get(pid)
        if row is None:
            continue
        row['trials'] += n
        row['trials_simulated'] += n if simulated else 0
        if accepted:
            row['accepted'] += n
        if hand in ('left', 'right'):
            row[hand] += n
            if accepted:
                row[f'{hand}_accepted'] += n
    for row in per.values():
        row.pop('_latest')

    return {
        'now': now,
        'counts': counts,
        'trials_accepted': int(trial_totals[1] or 0),
        'trials_simulated': int(trial_totals[2] or 0),
        'trial_bytes': int(trial_totals[3] or 0),
        'last_trial_at': utc(trial_totals[4]),
        'last_session_at': utc(last_session),
        'session_status': {str(k): int(v) for k, v in session_status.items()},
        'participants': list(per.values()),
        'recent_sessions': [dict(r._mapping) for r in recent_sessions],
        'recent_trials': [dict(r._mapping) for r in recent_trials],
        'log': [dict(r._mapping) for r in log_rows],
        'last_day': {str(k): int(v) for k, v in last_day.items()},
        'db_size': db_size,
        'dialect': db.dialect,
    }
