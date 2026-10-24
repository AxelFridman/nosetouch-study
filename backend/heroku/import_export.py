#!/usr/bin/env python3
"""Backfill: post an export file from the app to the upload service, record by record, as the app does.

    python import_export.py nosetouch_P-XXXXXX_20261024-0159.json --url https://<app>.herokuapp.com/api/ingest
    python import_export.py export.json --url ... --study-key K       the service's STUDY_KEY
    python import_export.py export.json --dry-run                     check locally, send nothing

The file is the app's export (format 'nose-touch-export': participants, sessions, trials; also
.json.txt and .json.gz). Every session and trial is sent as the UploadEnvelope the app's uploader
sends ({kind, study_key, sent_at, record}, Content-Type: application/json), sessions before their
trials, one request at a time, with retries and backoff on network errors, 5xx, 408 and 429. sent_at
is the file's exported_at: the records are as current as the export, so the service never lets them
overwrite a copy the phone uploaded later. There is no envelope for participants: their profiles travel
inside the session records (as with the app), and the summary says which file profiles are covered.
Re-running is safe: records already stored answer 'unchanged'.

Defaults come from the environment (NOSETOUCH_INGEST_URL, NOSETOUCH_STUDY_KEY) or from
%USERPROFILE%\\.nosetouch\\heroku.env (lines KEY=VALUE: APP_URL, STUDY_KEY). Exit code 0 when every
record was accepted, 1 when some were refused or could not be sent, 2 on a usage or file error.
Standard library only (the local checks of --dry-run use nosetouch_server/validation.py next to it).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nosetouch_server.validation import (  # noqa: E402 (stdlib only)
    Rejection,
    canonical_record,
    parse_envelope,
    validate_envelope,
)

USER_AGENT = 'nosetouch-import/1.0'
DEFAULT_MAX_BYTES = 4_000_000
ENV_FILE = Path.home() / '.nosetouch' / 'heroku.env'
OUTCOMES = ('stored', 'replaced', 'unchanged', 'kept_newer', 'checked', 'rejected', 'failed')

# (status, lower-cased headers, body) of one POST
Poster = Callable[[bytes], tuple[int, dict[str, str], bytes]]


@dataclass
class Envelope:
    kind: str
    record_id: str
    session_id: str
    body: bytes


@dataclass
class Result:
    kind: str
    record_id: str
    outcome: str
    detail: str | None = None
    attempts: int = 0


@dataclass
class Summary:
    results: list[Result] = field(default_factory=list)
    bytes_sent: int = 0
    seconds: float = 0.0
    participants_in_file: list[str] = field(default_factory=list)
    participants_covered: list[str] = field(default_factory=list)

    def count(self, kind: str, outcome: str) -> int:
        return sum(1 for r in self.results if r.kind == kind and r.outcome == outcome)

    @property
    def ok(self) -> bool:
        return not any(r.outcome in ('rejected', 'failed') for r in self.results)


# ───────────────────────────── input ─────────────────────────────


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """KEY=VALUE lines (comments with #, optional quotes). Missing file -> {}."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding='utf-8-sig')
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip().removeprefix('export ').strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        out[key] = value
    return out


def load_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = p.read_bytes()
    if p.name.lower().endswith('.gz'):
        data = gzip.decompress(data)
    doc = json.loads(data.decode('utf-8-sig'))
    if not isinstance(doc, dict) or doc.get('format') != 'nose-touch-export':
        raise ValueError(f"{p.name} is not an app export (expected format 'nose-touch-export')")
    for key in ('sessions', 'trials'):
        if not isinstance(doc.get(key, []), list):
            raise ValueError(f'{p.name}: {key} must be a list')
    return doc


def _sort_key(t: dict[str, Any]) -> tuple[Any, ...]:
    def num(v: Any) -> float:
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else float('inf')
    return (num(t.get('trial_index')), num(t.get('attempt')), str(t.get('created_at')), str(t.get('trial_id')))


def build_envelopes(bundle: dict[str, Any], study_key: str | None, sent_at: str | None = None) -> list[Envelope]:
    """The app's envelopes for every session and trial: each session first, then its trials (plan
    order), then trials whose session is not in the file."""
    when = sent_at or bundle.get('exported_at')
    if not isinstance(when, str) or not when:
        when = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    sessions = [s for s in bundle.get('sessions') or [] if isinstance(s, dict)]
    trials = [t for t in bundle.get('trials') or [] if isinstance(t, dict)]
    by_session: dict[Any, list[dict[str, Any]]] = {}
    for t in trials:
        by_session.setdefault(t.get('session_id'), []).append(t)

    def envelope(kind: str, rec: dict[str, Any]) -> Envelope:
        body = json.dumps({'kind': kind, 'study_key': study_key or None, 'sent_at': when, 'record': rec},
                          ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        rid = rec.get('session_id') if kind == 'session' else rec.get('trial_id')
        return Envelope(kind, str(rid), str(rec.get('session_id')), body)

    out: list[Envelope] = []
    seen_sessions = set()
    for s in sessions:
        out.append(envelope('session', s))
        seen_sessions.add(s.get('session_id'))
        out.extend(envelope('trial', t) for t in sorted(by_session.get(s.get('session_id'), []), key=_sort_key))
    for sid, group in by_session.items():
        if sid not in seen_sessions:
            out.extend(envelope('trial', t) for t in sorted(group, key=_sort_key))
    return out


def participant_coverage(bundle: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(participant ids in the file's participants list, those whose profile a session carries)."""
    listed = [p['participant_id'] for p in bundle.get('participants') or []
              if isinstance(p, dict) and isinstance(p.get('participant_id'), str)]
    carried = {s['participant']['participant_id'] for s in bundle.get('sessions') or []
               if isinstance(s, dict) and isinstance(s.get('participant'), dict)
               and isinstance(s['participant'].get('participant_id'), str)}
    return listed, [p for p in listed if p in carried]


# ───────────────────────────── sending ─────────────────────────────


def urllib_poster(url: str, timeout: float = 60.0) -> Poster:
    def post(body: bytes) -> tuple[int, dict[str, str], bytes]:
        req = urllib.request.Request(url, data=body, method='POST',
                                     headers={'Content-Type': 'application/json', 'User-Agent': USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (https URL from the user)
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read()
    return post


def _interpret(status: int, headers: dict[str, str], raw: bytes) -> tuple[str, str | None, bool]:
    """(outcome, detail, transient) like the app's uploader (uploader.ts interpretResponse)."""
    try:
        body = json.loads(raw.decode('utf-8')) if raw.strip() else None
    except (UnicodeDecodeError, ValueError):
        body = 'unreadable'
    obj = body if isinstance(body, dict) else {}
    error = obj.get('error') if isinstance(obj.get('error'), str) else None
    code = obj.get('code') if isinstance(obj.get('code'), str) else None
    detail = f'{code}: {error}' if code and error else error or code
    if 200 <= status < 300:
        if body == 'unreadable':
            return 'failed', 'unexpected answer (not JSON)', True
        if obj.get('ok') is False:
            return ('failed', detail, True) if obj.get('retry') is True else ('rejected', detail, False)
        outcome = headers.get('x-nosetouch-outcome', 'stored')  # receiver.py and Apps Script: no header
        return (outcome if outcome in ('stored', 'replaced', 'unchanged', 'kept_newer') else 'stored'), None, False
    transient = status >= 500 or status in (408, 429) or obj.get('retry') is True
    return ('failed' if transient else 'rejected'), f'HTTP {status}' + (f' {detail}' if detail else ''), transient


def send(env: Envelope, post: Poster, *, retries: int = 4, backoff_s: float = 2.0,
         sleep: Callable[[float], None] = time.sleep) -> Result:
    attempt = 0
    while True:
        attempt += 1
        try:
            outcome, detail, transient = _interpret(*post(env.body))
        except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as e:
            outcome, detail, transient = 'failed', f'network error: {type(e).__name__}: {e}', True
        if not transient or attempt > retries:
            return Result(env.kind, env.record_id, outcome, detail, attempt)
        sleep(min(backoff_s * 2 ** (attempt - 1), 60.0))


def check_locally(env: Envelope, max_bytes: int) -> Result:
    """--dry-run: the service's own validation (minus the study key, which only the server knows)."""
    try:
        if len(env.body) > max_bytes:
            raise Rejection('too_large', f'payload too large ({len(env.body)} bytes > {max_bytes})', 413)
        parsed = parse_envelope(env.body)
        validate_envelope(parsed, None)
        canonical_record(parsed['record'])
    except Rejection as rej:
        return Result(env.kind, env.record_id, 'rejected', f'{rej.code}: {rej.message}')
    return Result(env.kind, env.record_id, 'checked')


def run_import(bundle: dict[str, Any], post: Poster | None, *, study_key: str | None, dry_run: bool = False,
               max_bytes: int = DEFAULT_MAX_BYTES, retries: int = 4, backoff_s: float = 2.0,
               sleep: Callable[[float], None] = time.sleep,
               progress: Callable[[int, int, Result], None] | None = None) -> Summary:
    envelopes = build_envelopes(bundle, study_key)
    summary = Summary()
    summary.participants_in_file, summary.participants_covered = participant_coverage(bundle)
    started = time.perf_counter()
    for i, env in enumerate(envelopes, 1):
        if dry_run or post is None:
            result = check_locally(env, max_bytes)
        else:
            result = send(env, post, retries=retries, backoff_s=backoff_s, sleep=sleep)
            summary.bytes_sent += len(env.body) * result.attempts
        summary.results.append(result)
        if progress:
            progress(i, len(envelopes), result)
    summary.seconds = time.perf_counter() - started
    return summary


# ───────────────────────────── report ─────────────────────────────


def format_summary(summary: Summary, dry_run: bool) -> str:
    cols = [o for o in OUTCOMES if any(r.outcome == o for r in summary.results)] or ['stored']
    lines = [f"{'':10}" + ''.join(f'{c:>12}' for c in cols)]
    for kind in ('session', 'trial'):
        lines.append(f'{kind + "s":10}' + ''.join(f'{summary.count(kind, c):>12}' for c in cols))
    listed, covered = summary.participants_in_file, summary.participants_covered
    missing = [p for p in listed if p not in covered]
    lines.append(f'participants: {len(listed)} in the file, {len(covered)} carried by their session records'
                 + (f'; NOT carried (no session in the file): {", ".join(missing)}' if missing else ''))
    if not dry_run:
        lines.append(f'sent {summary.bytes_sent / 1e6:.1f} MB in {summary.seconds:.1f} s')
    problems = [r for r in summary.results if r.outcome in ('rejected', 'failed')]
    if problems:
        lines.append('problems:')
        lines.extend(f'  {r.kind} {r.record_id}: {r.outcome} ({r.detail})' for r in problems[:10])
        if len(problems) > 10:
            lines.append(f'  ... and {len(problems) - 10} more')
    codes = Counter(r.detail.split(':')[0] for r in problems if r.detail)
    if codes:
        lines.append('by cause: ' + ', '.join(f'{k} x{v}' for k, v in codes.most_common()))
    return '\n'.join(lines)


def _resolve_url(arg: str | None, env_file: dict[str, str]) -> str | None:
    url = arg or os.environ.get('NOSETOUCH_INGEST_URL')
    if not url and env_file.get('APP_URL'):
        url = env_file['APP_URL'].rstrip('/') + '/api/ingest'
    return url


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='import_export.py',
                                description="Post an app export file to the NoseTouch upload service, as the app's uploader would.")
    p.add_argument('export', help="the app's export file (.json, .json.txt or .json.gz)")
    p.add_argument('--url', help='the ingest endpoint, e.g. https://<app>.herokuapp.com/api/ingest '
                                 '(default: NOSETOUCH_INGEST_URL, or APP_URL in ~/.nosetouch/heroku.env)')
    p.add_argument('--study-key', help='the service\'s STUDY_KEY (default: NOSETOUCH_STUDY_KEY or heroku.env)')
    p.add_argument('--dry-run', action='store_true', help='check every record locally and send nothing')
    p.add_argument('--retries', type=int, default=4, help='retries per record on network errors and 5xx (default 4)')
    p.add_argument('--timeout', type=float, default=60.0, help='seconds per request (default 60)')
    p.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES, help='size limit for --dry-run checks')
    p.add_argument('--quiet', action='store_true', help='no line per record')
    return p


def main(argv: list[str] | None = None, post: Poster | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_file = read_env_file()
    try:
        bundle = load_bundle(args.export)
    except (OSError, ValueError) as e:
        print(f'cannot read {args.export}: {e}', file=sys.stderr)
        return 2
    url = _resolve_url(args.url, env_file)
    if not args.dry_run and post is None:
        if not url:
            print('no --url given (nor NOSETOUCH_INGEST_URL, nor APP_URL in ~/.nosetouch/heroku.env)', file=sys.stderr)
            return 2
        if not url.startswith(('https://', 'http://127.0.0.1', 'http://localhost')):
            print(f'refusing to send participant data over plain http: {url}', file=sys.stderr)
            return 2
        post = urllib_poster(url, args.timeout)
    study_key = args.study_key or os.environ.get('NOSETOUCH_STUDY_KEY') or env_file.get('STUDY_KEY') or None

    def progress(i: int, n: int, r: Result) -> None:
        if not args.quiet:
            extra = f' ({r.detail})' if r.detail else ''
            print(f'  [{i}/{n}] {r.kind:<7} {r.record_id}: {r.outcome}{extra}', flush=True)

    target = 'local checks only (--dry-run)' if args.dry_run else (url or 'custom poster')
    print(f"{Path(args.export).name}: {len(bundle.get('sessions') or [])} sessions, "
          f"{len(bundle.get('trials') or [])} trials, exported {bundle.get('exported_at')} -> {target}")
    summary = run_import(bundle, None if args.dry_run else post, study_key=study_key, dry_run=args.dry_run,
                         max_bytes=args.max_bytes, retries=max(0, args.retries), progress=progress)
    print(format_summary(summary, args.dry_run))
    return 0 if summary.ok else 1


if __name__ == '__main__':
    sys.exit(main())
