"""Lane x1: the exploratory persons' own cache, built by the SAME code path as the main cache.

Only lane x1 reads ``analysis/data/exploratory/`` and ``analysis/data/curation/exploratory.json``
(``config.EXPLORATORY_*``). For each exploratory person (PD01, ...) this module

1. writes a derived, main-format curation file (``{"sessions": [{session_id, person_id, include}]}``) into the
   person's folder of the exploratory cache (``analysis/data/exploratory_cache/<person>/curation.json``, git-ignored);
2. runs ``stage2.data.build_cache`` on the person's raw folder with that curation: the same loader
   (``motionlab.io.load_dataset``: schema validation, deduplication, norm_v2 per session), the same
   standardisation (``standardize.stream_from_record`` -> ``canonical`` -> windows: 60 Hz grid, 15 Hz zero-phase
   low-pass, gravity frame, time-normalised approach, fixed and return windows), the same flags, gates and exclusion
   cascade (E0..E4, N1) and the same cache format as the main cache;
3. records the person's data identity (``data.data_identity`` of the raw folder + derived curation + parameters +
   builder code) so the cache is rebuilt when the raw data, the curation or the builder code change.

Nothing here ever writes into the main snapshot, curation or cache, and nothing here fits a model: exploratory
persons are never trained on, validated on, tuned with or scored by any model. :func:`assert_apart` re-checks at
every build that none of their ids reached the main curation or cache (``tests/test_exploratory_guard.py`` checks
the same independently).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from stage2 import config as C

GROUP_LABELS = {'parkinsons': "Parkinson's disease"}


def exploratory_doc() -> dict[str, Any]:
    """The researcher's exploratory curation file (only its ``persons`` list is used)."""
    p = C.assert_not_private(C.EXPLORATORY_CURATION)
    return json.loads(Path(p).read_text(encoding='utf-8'))


def exploratory_persons() -> list[dict[str, Any]]:
    """Every exploratory person of the curation file with an id, a raw folder and sessions, sorted by id."""
    persons = [p for p in exploratory_doc().get('persons') or []
               if p.get('person_id') and p.get('raw_dir') and p.get('sessions')]
    return sorted(persons, key=lambda p: p['person_id'])


def person_spec(person_id: str) -> dict[str, Any]:
    for p in exploratory_persons():
        if p['person_id'] == person_id:
            return p
    raise KeyError(f'{person_id} is not an exploratory person of {C.EXPLORATORY_CURATION.name}')


def raw_dir(person_id: str) -> Path:
    """The person's raw folder (backend layout nose-touch-v1/<code>/<session>/...); must lie under the exploratory
    data folder, never under the main snapshot."""
    p = (C.REPO_DIR / person_spec(person_id)['raw_dir']).resolve()
    base = C.EXPLORATORY_DIR.resolve()
    if base not in [p, *p.parents]:
        raise ValueError(f'{person_id}: raw_dir must lie under {C.EXPLORATORY_DIR}')
    snap = C.SNAPSHOT_DIR.resolve()
    if snap in [p, *p.parents] or p in [snap, *snap.parents]:
        raise ValueError(f'{person_id}: raw_dir overlaps the main snapshot')
    return p


def cache_dir(person_id: str) -> Path:
    d = (C.EXPLORATORY_CACHE_DIR / person_id).resolve()
    if d == C.CACHE_DIR.resolve() or C.CACHE_DIR.resolve() in d.parents:
        raise ValueError('the exploratory cache must lie outside the main cache')
    return d


def curation_path(person_id: str) -> Path:
    return cache_dir(person_id) / 'curation.json'


def derived_curation(person_id: str) -> dict[str, Any]:
    """The main-format curation of one exploratory person: every listed session included, person_id set."""
    spec = person_spec(person_id)
    return {'sessions': [{'session_id': s['session_id'], 'person_id': person_id, 'include': True}
                         for s in spec['sessions']]}


def write_curation(person_id: str) -> Path:
    """Write the derived curation (only when its content changes, so its sha256 and the data identity stay stable)."""
    p = curation_path(person_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(derived_curation(person_id), indent=1) + '\n'
    if not p.exists() or p.read_text(encoding='utf-8') != text:
        p.write_text(text, encoding='utf-8', newline='\n')
    return p


def identity(person_id: str) -> dict[str, Any]:
    """``data.data_identity`` of the person's raw folder and derived curation (what a cache built now would be)."""
    from stage2 import data as D
    write_curation(person_id)
    return D.data_identity(raw_dir(person_id), curation_path(person_id))


def cache_status(person_id: str) -> tuple[str, str]:
    """('missing' | 'stale' | 'current', detail) of the person's exploratory cache."""
    d = cache_dir(person_id)
    if not ((d / 'arrays.npz').exists() and (d / 'trials.pkl').exists() and (d / 'meta.json').exists()):
        return 'missing', f'no exploratory cache in {d}'
    have = (json.loads((d / 'meta.json').read_text(encoding='utf-8')).get('data_identity') or {})
    now = identity(person_id)
    if have.get('data_id') == now['data_id']:
        return 'current', f"data_id {now['data_id'][:12]}"
    diff = [k for k in now if k != 'data_id' and have.get(k) != now[k]]
    return 'stale', 'changed: ' + ', '.join(diff)


def assert_apart(person_id: str) -> None:
    """None of the person's ids (person, sessions, app codes) is in the main curation or the main cache."""
    import pandas as pd
    spec = person_spec(person_id)
    sessions = {s['session_id'] for s in spec['sessions']}
    codes = {s.get('participant_code') for s in spec['sessions'] if s.get('participant_code')}
    main = json.loads(C.assert_not_private(C.CURATION_FILE).read_text(encoding='utf-8'))
    if {s['session_id'] for s in main['sessions']} & sessions or person_id in {s.get('person_id') for s in main['sessions']}:
        raise RuntimeError(f'{person_id} appears in the main curation: exploratory persons never enter it')
    if (C.CACHE_DIR / 'trials.pkl').exists():
        t = pd.read_pickle(C.CACHE_DIR / 'trials.pkl')
        if t['session_id'].isin(sessions).any() or t['person_id'].astype(str).eq(person_id).any() \
                or t['participant_code'].isin(codes).any():
            raise RuntimeError(f'{person_id} appears in the main cache: exploratory persons never enter it')


def ensure_cache(person_id: str, force: bool = False, progress: bool = False) -> Path:
    """Build (or rebuild when stale or forced) the person's exploratory cache with ``data.build_cache``."""
    from stage2 import data as D
    assert_apart(person_id)
    write_curation(person_id)               # the derived curation must exist before build_cache reads it
    status, why = cache_status(person_id)
    d = cache_dir(person_id)
    if force or status != 'current':
        print(f'x1: building the exploratory cache of {person_id} ({status}: {why})', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            D.build_cache(snapshot=raw_dir(person_id), curation=curation_path(person_id), cache_dir=d, progress=progress)
    assert_apart(person_id)
    return d


def load(person_id: str, sets: Any = None) -> Any:
    """The person's Stage2Data (default: every cached attempt, i.e. the complete ones) from the exploratory cache."""
    from stage2 import data as D
    return D.load_cache(sets=sets, cache_dir=cache_dir(person_id))


def load_tables(person_id: str) -> dict[str, Any]:
    from stage2 import data as D
    return D.load_tables(cache_dir(person_id))


def session_specs(person_id: str) -> list[dict[str, Any]]:
    """The curation's description of each session (status, app version, platform, attempt counts), without ids."""
    keep = ('status', 'app_version', 'preset', 'platform', 'model_hint', 'n_attempts', 'n_accepted')
    return [{k: s.get(k) for k in keep} for s in person_spec(person_id)['sessions']]
