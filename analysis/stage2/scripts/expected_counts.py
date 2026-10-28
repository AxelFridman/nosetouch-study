"""The snapshot counts the tests pin (stage2/tests/expected_counts.json), and the way to update them.

    cd analysis
    python -m stage2.scripts.expected_counts                  # compare the current cache with the fixture
    python -m stage2.scripts.expected_counts --without P21    # the fixture must be reproduced WITHOUT P21
    python -m stage2.scripts.expected_counts --write          # then rewrite the fixture from the current cache

When a participant is added (raw folder in the snapshot + curation row, then
``python -m stage2.pipeline all --force``), every count of the fixture changes. Before rewriting it, check
that the change is explained by the new person alone: ``--without <id>`` recomputes every count on the
current data with that person's sessions and trials removed (the exclusion cascade is re-run) and must
reproduce the old fixture exactly. Then ``--write`` and commit the new fixture with the data.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # run as a file
    import _bootstrap  # type: ignore # noqa: F401

FIXTURE = Path(__file__).resolve().parents[1] / 'tests' / 'expected_counts.json'
NOTE = ('Snapshot counts pinned by the stage2 tests. Update after adding a participant: rebuild '
        '(python -m stage2.pipeline all --force), check that the old values are reproduced without the new person '
        '(python -m stage2.scripts.expected_counts --without P21), then rewrite this file '
        '(python -m stage2.scripts.expected_counts --write).')
FLOW_COLUMNS = ('removed', 'remaining', 'sessions', 'persons', 'natural_use_added', 'natural_use_removed', 'natural_use')


def snapshot_counts(without: Sequence[str] = ()) -> dict[str, Any]:
    """Every pinned count, from the cache; ``without`` drops those persons (sessions, trials, arrays) first."""
    import numpy as np
    import pandas as pd

    from stage2 import config as C
    from stage2 import screen
    from stage2.data import FLAG_COLUMNS, GATE_COLUMNS, apply_exclusions, load_cache, load_curation, load_tables

    drop = set(without)
    cur = load_curation()
    cur = cur[~cur['person_id'].isin(drop)]
    T = load_tables()
    t, S = T['trials'], T['sessions']
    t = t[~t['person_id'].isin(drop)].copy()
    S = S[~S['person_id'].isin(drop)]
    flow, e1 = T['exclusion_flow'], T['exclusion_e1']
    if drop:  # the cascade is a function of the trial table: re-run it without the dropped persons
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, flow, e1 = apply_exclusions(t.drop(columns=['analysis_set', 'excl_rule', 'excl_detail'],
                                                  errors='ignore'))
    inc = cur[cur['include']]
    n_sess = inc.groupby('person_id').size()
    prim = t[t['analysis_set'] == 'primary']
    nat = t[t['analysis_set'] == 'natural_use']
    both = t[t['analysis_set'].isin(['natural_use', 'natural_use_repeat'])]
    rep = t[t['analysis_set'] == 'natural_use_repeat']
    d = load_cache()
    d = d.subset(~d.trials['person_id'].isin(drop).to_numpy())
    _, blocks = screen.block_consistency(d.trials, d.approach)
    hit = blocks[blocks['fhy_swap_signature']]
    f = flow.set_index('step')
    return {
        '_note': NOTE,
        'snapshot': C.SNAPSHOT_NAME,
        'curation': {
            'sessions': int(len(cur)), 'included': int(cur['include'].sum()), 'excluded': int((~cur['include']).sum()),
            'persons': int(inc['person_id'].nunique()),
            'multi_session_persons': {k: int(v) for k, v in n_sess[n_sess > 1].sort_index().items()},
            'unassigned_excluded_sessions': int((~cur['include'] & cur['person_id'].isna()).sum()),
            'superseded_persons': sorted(cur.loc[cur['exclude_code'] == 'superseded', 'person_id'].dropna().unique().tolist()),
        },
        'primary': {'trials': int(len(prim)), 'persons': int(prim['person_id'].nunique()),
                    'left': int((prim['hand'] == 'left').sum()), 'right': int((prim['hand'] == 'right').sum())},
        'natural_use': {'headline': int(len(nat)), 'with_repeats': int(len(both)),
                        'persons': int(both['person_id'].nunique()),
                        'repeat_session_labels': sorted(rep['trial_key'].str.rsplit('-', n=1).str[0].tolist())},
        'flow': {step: {c: int(f.loc[step, c]) for c in FLOW_COLUMNS if c in f.columns} for step in f.index},
        'e1_attempts': int(e1['attempts'].sum()),
        'flags_primary': {c: int(prim[c].sum()) for c in FLAG_COLUMNS},
        'grip_mismatch_persons': sorted(prim.loc[prim['grip_mismatch'], 'person_id'].unique().tolist()),
        'gates_failed_primary': {c: int((~prim[c].astype(bool)).sum()) for c in (*GATE_COLUMNS, 'gate_pass')},
        'report_sessions': int(len(S)),
        'swap_signature_blocks': sorted([str(a), int(b)] for a, b in zip(hit['session_label'], hit['block_index'])),
        'arrays_rows': {'primary': int(len(d.trials)), 'channels': int(np.asarray(d.approach).shape[1])},
    }


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def diff(a: Any, b: Any, path: str = '') -> list[str]:
    """Paths where two nested structures differ (the note is ignored)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k == '_note':
                continue
            out += diff(a.get(k), b.get(k), f'{path}.{k}' if path else k)
        return out
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--without', nargs='+', default=(), metavar='PERSON', help='recompute without these persons')
    ap.add_argument('--write', action='store_true', help='rewrite the fixture from the current cache')
    a = ap.parse_args(argv)
    if a.write:
        if a.without:
            ap.error('--write takes the full current data (no --without)')
        FIXTURE.write_text(json.dumps(snapshot_counts(), indent=1) + '\n', encoding='utf-8', newline='\n')
        print(f'wrote {FIXTURE}')
        return 0
    now = snapshot_counts(a.without)
    d = diff(now, load_fixture())
    what = f"current data without {', '.join(a.without)}" if a.without else 'current cache'
    if d:
        print(f'{what} differs from the fixture in {len(d)} place(s):')
        print('\n'.join('  ' + x for x in d))
        return 1
    print(f'{what} reproduces the fixture exactly')
    return 0


if __name__ == '__main__':
    sys.exit(main())
