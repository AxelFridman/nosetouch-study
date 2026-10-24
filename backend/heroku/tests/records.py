"""Minimal records and envelopes (the shapes of server/test_receiver.py) and small request helpers."""
from __future__ import annotations

import copy
import json
from typing import Any

STUDY = 'nose-touch-v1'
PID = 'P-ABCDEF'
SID = 'sess-0001'
T0 = '2026-10-23T10:00:00.000Z'
T1 = '2026-10-23T10:05:00.000Z'
T2 = '2026-10-23T10:10:00.000Z'


def profile(**over: Any) -> dict[str, Any]:
    p = {
        'participant_id': PID, 'nickname': None, 'age_years': 31, 'sex': 'female', 'handedness_self': 'right',
        'ehi_sf': None, 'ehi_lq': 87.5, 'ehi_class': 'right', 'phone_hand': 'right', 'height_cm': None,
        'arm_issue': None, 'phone_model': None, 'phone_case': None, 'created_at': T0, 'updated_at': T0,
    }
    p.update(over)
    return p


def make_session(**over: Any) -> dict[str, Any]:
    rec = {
        'record_type': 'session', 'schema_version': '1.1.0', 'app_version': '0.1.0',
        'session_id': SID, 'participant_id': PID, 'study_code': STUDY,
        'started_at': T0, 'updated_at': T0, 'ended_at': None, 'status': 'in_progress',
        'participant': profile(),
        'device': {'platform': 'android', 'browser': 'Chrome 155.0.0.0'},
        'protocol': {'preset': 'quick', 'n_planned_trials': 24},
        'progress': {'next_trial_index': 0, 'n_attempts': 0, 'n_accepted': 0},
        'timezone_offset_min': 180, 'simulated': False, 'notes': None,
    }
    rec.update(over)
    return rec


def make_trial(**over: Any) -> dict[str, Any]:
    rec = {
        'record_type': 'trial', 'schema_version': '1.1.0', 'app_version': '0.1.0',
        'trial_id': 'trial-0001', 'session_id': SID, 'participant_id': PID, 'study_code': STUDY,
        'created_at': T0, 'block_index': 0, 'trial_index': 0, 'trial_in_block': 0, 'attempt': 1,
        'simulated': False,
        'condition': {'hand': 'right', 'orientation': 'upright_toward', 'speed': 'natural', 'variant': 'standard'},
        'outcome': {'status': 'complete', 'accepted': True, 'participant_flag': None, 'note': None},
        'streams': {'motion': {'t': [0, 16.7, 33.4], 'acc_x': [0.01, -0.5, None], 'rr_alpha': [0.5, 120.25, 10]}},
    }
    rec.update(over)
    return rec


def envelope(kind: str, record: Any, key: str | None = None, sent_at: str = T0, **over: Any) -> dict[str, Any]:
    env = {'kind': kind, 'study_key': key, 'sent_at': sent_at, 'record': record}
    env.update(over)
    return env


def post(client: Any, payload: Any, path: str = '/api/ingest', content_type: str = 'application/json',
         headers: dict[str, str] | None = None) -> Any:
    body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    return client.post(path, content=body.encode('utf-8') if isinstance(body, str) else body,
                       headers={'Content-Type': content_type, **(headers or {})})


def deep(rec: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(rec)
