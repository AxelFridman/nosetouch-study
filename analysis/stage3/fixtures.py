"""Golden fixtures: raw motion exactly as the app stores it, plus ``reference_predict``'s outputs.

* **Synthetic, committed** (``app/src/predict/__fixtures__/*.json.gz``, :func:`synthetic_cases`): multi-touch streams
  concatenated from ``motionlab.synth.simulate_trial`` records (protocol v2: each record holds the hold, the approach,
  the hold at the nose and the return), with the edge cases of PORT_SPEC 4.6 - both engine signs, a 314 ms iOS gap, an
  isolated repeated sample, the screen-away grip, a mirrored stream, a stream that starts moving, an unended last
  bout and a missing linear acceleration. No participant data.
* **Real, local only** (``analysis/data/stage3_fixtures/``, git-ignored; :func:`real_cases`): about 40 recordings of
  P01..P20 from the snapshot - one per person, extra grips and speeds, natural-use trials, excluded hard cases (sampling
  gaps, rejected attempts, no turn), screen-away twins (``flip_device_vectors`` on the raw vectors, the raw-level
  equivalent of ``standardize.screen_flip_channels``) and multi-touch concatenations of one person and hand. Keyed by
  trial_key; no name, app code or timestamp.

Every case is normalised as the page does it (the engine prior; checked equal to the session's own normalisation for
real recordings) and rebased so that its first sample is t = 0, with the app's rounding (``core/recorder.ts``
``roundValue``: t 0.01 ms, acc / accg 1e-4, rr 1e-3, halves away from zero): a LiveBuffer fed these samples stores
exactly these values. The reference is computed from the EXPORTED JSON models (``runtime.json_set``); the tests check
that the in-memory fits give the same numbers.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from stage2 import config as C

from . import reference as REF
from . import runtime as RT

SYN_DIR = C.REPO_DIR / 'app' / 'src' / 'predict' / '__fixtures__'
REAL_DIR = C.DATA_DIR / 'stage3_fixtures'
FORMAT = 'nosetouch-fixture-1'
COLS = REF.MOTION_COLUMNS
VEC = {'acc': ('acc_x', 'acc_y', 'acc_z'), 'accg': ('accg_x', 'accg_y', 'accg_z'), 'rr': ('rr_alpha', 'rr_beta', 'rr_gamma')}
DECIMALS = {'t': 2, 'acc': 4, 'accg': 4, 'rr': 3}
SCREEN_FLIP = (-1.0, 1.0, -1.0)                          # F = diag(-1, 1, -1): polar and axial vectors alike
MIRROR = {'acc': (-1.0, 1.0, 1.0), 'accg': (-1.0, 1.0, 1.0), 'rr': (1.0, -1.0, -1.0)}   # a -> S a, w -> -S w


# ───────────────────────────── the app's rounding and raw-level transforms ─────────────────────────────


def round_js(v: Any, decimals: int) -> float | None:
    """core/recorder.ts roundValue: halves away from zero, null for missing / non-finite, never -0."""
    if v is None:
        return None
    x = float(v)
    if not math.isfinite(x):
        return None
    f = 10.0 ** decimals
    scaled = abs(x) * f
    if not scaled < 2.0 ** 52:
        return x
    r = math.floor(scaled)
    if scaled - r >= 0.5:
        r += 1
    out = math.copysign(r, x) / f
    return 0.0 if out == 0 else out


def normalise_motion(m: dict[str, list]) -> dict[str, list]:
    """Rebased to the first sample (t = 0) and rounded like the app; a vector with a missing component is all null."""
    t = [float(v) for v in m['t']]
    t0 = t[0]
    out: dict[str, list] = {'t': [round_js(v - t0, DECIMALS['t']) for v in t]}
    for k, names in VEC.items():
        cols = [[round_js(v, DECIMALS[k]) for v in m[nm]] for nm in names]
        for j, nm in enumerate(names):
            out[nm] = [cols[j][i] if all(c[i] is not None for c in cols) else None for i in range(len(t))]
    return out


def changed_values(a: dict[str, list], b: dict[str, list]) -> int:
    """Value cells (not t) that differ between two motion dicts of the same length."""
    n = 0
    for c in COLS[1:]:
        for x, y in zip(a[c], b[c]):
            if not (x == y or (x is None and y is None)):
                n += 1
    return n


def concat(parts: list[dict[str, list]]) -> dict[str, list]:
    """Raw-level concatenation: each part's clock continues one 60 Hz sample after the previous part's last sample."""
    out: dict[str, list] = {c: [] for c in COLS}
    for m in parts:
        t = [float(v) for v in m['t']]
        shift = 0.0 if not out['t'] else out['t'][-1] + 1000.0 / 60.0 - t[0]
        out['t'] += [v + shift for v in t]
        for c in COLS[1:]:
            out[c] += list(m[c])
    return out


def scale_vectors(m: dict[str, list], signs: dict[str, tuple[float, float, float]]) -> dict[str, list]:
    out = {c: list(v) for c, v in m.items()}
    for k, s in signs.items():
        for j, nm in enumerate(VEC[k]):
            out[nm] = [None if v is None else (s[j] * v if s[j] != 1.0 else v) for v in m[nm]]
            out[nm] = [None if v is None else (0.0 if v == 0 else v) for v in out[nm]]
    return out


def screen_flip(m: dict[str, list]) -> dict[str, list]:
    """The same movement with the phone turned 180 deg about its y axis (motionlab flip_device_vectors on the raw acc,
    accg and the W3C rotation rate)."""
    return scale_vectors(m, {'acc': SCREEN_FLIP, 'accg': SCREEN_FLIP, 'rr': SCREEN_FLIP})


def mirror(m: dict[str, list]) -> dict[str, list]:
    """The left/right mirror image (motionlab mirror_device_vectors on the raw vectors): the other hand."""
    return scale_vectors(m, MIRROR)


def cut(m: dict[str, list], t_from: float = -math.inf, t_to: float = math.inf) -> dict[str, list]:
    keep = [i for i, t in enumerate(m['t']) if t_from <= float(t) <= t_to]
    return {c: [m[c][i] for i in keep] for c in COLS}


def drop_gap(m: dict[str, list], t_from: float, t_to: float) -> dict[str, list]:
    """No sample in (t_from, t_to): a sampling stall."""
    keep = [i for i, t in enumerate(m['t']) if not (t_from < float(t) < t_to)]
    return {c: [m[c][i] for i in keep] for c in COLS}


def repeat_sample(m: dict[str, list], i: int) -> dict[str, list]:
    """Sample i + 1 carries sample i's nine values (WebKit's duplicated snapshot), its own time."""
    out = {c: list(v) for c, v in m.items()}
    for c in COLS[1:]:
        out[c][i + 1] = out[c][i]
    return out


def no_acc(m: dict[str, list]) -> dict[str, list]:
    out = {c: list(v) for c, v in m.items()}
    for nm in VEC['acc']:
        out[nm] = [None] * len(out['t'])
    return out


# ───────────────────────────── one case ─────────────────────────────


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else None
    return v


def build_case(case_id: str, kind: str, motion: dict[str, list], engine: str, models: Any, *,
               description: str, tags: list[str], truth: dict[str, Any] | None = None,
               source: dict[str, Any] | None = None) -> dict[str, Any]:
    m = normalise_motion(motion)
    ref = REF.reference_predict(m, models, engine=engine, mode='offline')
    live = REF.reference_predict(m, models, engine=engine, mode='live', include_stream=False, check=False)
    doc = {'format': FORMAT, 'id': case_id, 'kind': kind, 'description': description, 'tags': tags,
           'engine': engine, 'normalization': ref['normalization'], 'truth': truth, 'source': source,
           'models': {k: hashlib.sha256((Path(models_dir()) / f).read_bytes()).hexdigest()
                      for k, f in RT.FILES.items()},
           'motion': m, 'reference': ref,
           'live': {'settle_ms': 1000.0, 'max_stream_ms': None, 'attempts': live['attempts'],
                    'decisions': live['decisions'],
                    'rule': 'attempts frozen by the end of the recording: ended, look-ahead complete, decided_ms + '
                            'settle_ms <= the last grid time; without a final decision the status is watching'}}
    return clean(doc)


def models_dir() -> Path:
    from .export import APP_MODELS_DIR
    return APP_MODELS_DIR


def write_case(doc: dict[str, Any], folder: Path) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(raw)
    data = buf.getvalue()
    path = folder / f"{doc['id']}.json.gz"
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)
    ref = doc['reference']
    return {'id': doc['id'], 'file': path.name, 'kind': doc['kind'], 'tags': doc['tags'], 'engine': doc['engine'],
            'bytes': len(data), 'bytes_json': len(raw), 'sha256': hashlib.sha256(data).hexdigest(),
            'n_raw': ref['n_raw'], 'n_grid': ref['n_grid'], 'n_segments': len(ref['segments']),
            'n_attempts': len(ref['attempts']), 'n_attempts_live': len(doc['live']['attempts']),
            'decisions': {k: [v['stopped'], v['percent_right'], v['n_used']] for k, v in ref['decisions'].items()},
            'truth_hand': (doc.get('truth') or {}).get('hand')}


def read_case(path: str | Path) -> dict[str, Any]:
    with gzip.open(path, 'rb') as fh:
        return json.loads(fh.read().decode('utf-8'))


# ───────────────────────────── synthetic cases (committed) ─────────────────────────────


def _cond(hand: str, orientation: str = 'upright_toward', speed: str = 'natural') -> dict[str, Any]:
    return {'hand': hand, 'orientation': orientation, 'speed': speed, 'start_position': 'natural', 'variant': 'standard',
            'posture': 'standing', 'rep': 0}


def _synth(platform: str, hand: str, n: int, seed: int, *, orientations: tuple[str, ...] = ('upright_toward',),
           speeds: tuple[str, ...] = ('natural',)) -> tuple[dict[str, list], dict[str, Any], str]:
    """n simulated touches of one participant and hand, concatenated (raw values in the platform's own sign)."""
    from motionlab import synth
    rng = np.random.default_rng(seed)
    part = synth.random_participant(rng)
    plat = synth.default_platform(platform, rng)
    parts, touches, off = [], [], 0.0
    for k in range(n):
        rec, tr = synth.simulate_trial(part, plat, _cond(hand, orientations[k % len(orientations)],
                                                         speeds[k % len(speeds)]), seed=seed * 100 + k, v2=True)
        m = rec['streams']['motion']
        parts.append({c: list(m[c]) for c in COLS})
        t = [float(v) for v in m['t']]
        shift = 0.0 if k == 0 else off + 1000.0 / 60.0 - t[0]
        touches.append({'onset_ms': tr.onset_ms + shift, 'end_ms': tr.end_ms + shift,
                        'orientation': orientations[k % len(orientations)], 'speed': speeds[k % len(speeds)]})
        off = t[-1] + shift
    return concat(parts), {'hand': hand, 'n_touches': n, 'touches': touches, 'platform': platform}, plat.engine


def synthetic_cases(models: Any) -> list[dict[str, Any]]:
    """The committed cases. Seeds are chosen so that the simulated rest pose faces the user (rest_up_z > 0.1): in
    motionlab.synth more than half of the seeds hold the phone with the screen tilted slightly downwards, which the
    grip check refuses."""
    out = []
    # 1. Android, right hand, three touches: the plain case
    m, tr, eng = _synth('android', 'right', 3, 63)
    out.append(build_case('syn01_android_right_3touch', 'synthetic', m, eng, models, truth=tr,
                          tags=['android', 'blink', 'right', 'multi_touch', 'returns'],
                          description='three simulated touches with the right hand on Android (W3C sign)'))
    base1 = (m, tr, eng)
    # 2. iOS, left hand, four touches, grips varied; a 314 ms stall in the second hold and an isolated repeat
    m, tr, eng = _synth('ios', 'left', 4, 79, orientations=('upright_toward', 'top_left_toward', 'top_right_toward'))
    t1 = tr['touches'][1]['onset_ms']
    m = drop_gap(m, t1 - 700.0, t1 - 700.0 + 314.0)
    m = repeat_sample(m, 40)
    out.append(build_case('syn02_ios_left_gap_repeat', 'synthetic', m, eng, models, truth=tr,
                          tags=['ios', 'webkit', 'left', 'multi_touch', 'gap_314ms', 'repeated_sample', 'grips'],
                          description='four simulated left-hand touches on iOS (accelerometer negated), three grips, a '
                                      '314 ms sampling stall before the second touch and one duplicated snapshot'))
    # 3. the screen-away grip: the grip gate refuses to answer
    m, tr, eng = _synth('android', 'right', 2, 60)
    out.append(build_case('syn03_android_screen_away', 'synthetic', screen_flip(m), eng, models,
                          truth={**tr, 'screen': 'away', 'expect_status': 'grip'},
                          tags=['android', 'blink', 'right', 'screen_away', 'grip_gate'],
                          description='two right-hand touches held with the screen away from the face '
                                      '(flip_device_vectors on the raw vectors): the grip check must refuse'))
    # 4. the mirror image of case 1: the other hand, the same movements
    m1, tr1, eng1 = base1
    out.append(build_case('syn04_android_mirror_of_syn01', 'synthetic', mirror(m1), eng1, models,
                          truth={**tr1, 'hand': 'left', 'mirror_of': 'syn01_android_right_3touch'},
                          tags=['android', 'blink', 'left', 'mirror', 'multi_touch'],
                          description='the left/right mirror image of syn01 (mirror_device_vectors on the raw '
                                      'vectors): mirror-even features, so the same segments; chiral evidence reversed'))
    # 5. Firefox-like engine, five touches, starts moving, no linear acceleration, the last touch cut short
    m, tr, eng = _synth('android', 'left', 6, 37, speeds=('natural', 'fast', 'slow'))
    t_start = tr['touches'][0]['onset_ms'] + 300.0
    t_stop = tr['touches'][5]['onset_ms'] + 800.0
    m = no_acc(cut(m, t_start, t_stop))
    tr = {**tr, 'note': 'the stream starts 300 ms into the first touch and stops 800 ms into the sixth'}
    out.append(build_case('syn05_gecko_left_edges', 'synthetic', m, 'gecko', models, truth=tr,
                          tags=['gecko', 'left', 'multi_touch', 'starts_moving', 'unended_last_bout', 'no_acc', 'speeds'],
                          description='six simulated left-hand touches (natural, fast, slow) with a W3C-sign engine: the '
                                      'stream starts mid-movement, has no linear acceleration and stops during the '
                                      'last touch'))
    return out


# ───────────────────────────── real cases (local only) ─────────────────────────────


def _dataset() -> Any:
    import warnings
    from motionlab.io import load_dataset
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return load_dataset(C.SNAPSHOT_DIR, simulated='include')


def _motion_of(ds: Any, tid: str) -> dict[str, list]:
    m = ds.record(tid)['streams']['motion']
    return {c: list(m[c]) for c in COLS}


def _norm_ok(ds: Any, tid: str, engine: str) -> bool:
    n = ds.trial_normalization(tid)
    return (n.sign == REF.ENGINE_SIGN.get(engine, 1) and n.rates.verdict == 'w3c' and n.rates.units == 'deg_per_s')


def _pick(T: Any, rng: np.random.Generator, **eq: Any) -> Any:
    sel = T
    for k, v in eq.items():
        sel = sel[sel[k] == v]
    return None if not len(sel) else sel.iloc[int(rng.integers(len(sel)))]


def real_cases(models: Any, ds: Any | None = None, *, verbose: bool = True) -> list[dict[str, Any]]:
    """About 40 real recordings (module docstring). Deterministic (seeded)."""
    from stage2.data import load_cache
    ds = ds if ds is not None else _dataset()
    T = load_cache(sets=None).trials.reset_index(drop=True)
    T = T[T['person_id'].astype(str).str.match(r'^P\d\d$')]
    rng = np.random.default_rng(C.SEED + 3)
    prim = T[T['analysis_set'] == 'primary']
    out: list[dict[str, Any]] = []
    stats = {'values_rounded': 0, 'norm_skipped': 0}
    grips = ('upright_toward', 'top_left_toward', 'top_right_toward')
    speeds = ('natural', 'fast', 'slow')

    def add(row: Any, tags: list[str], motion: dict[str, list] | None = None, suffix: str = '',
            description: str = '') -> None:
        tid, eng = str(row['trial_id']), str(row['engine'])
        if not _norm_ok(ds, tid, eng):
            stats['norm_skipped'] += 1
            return
        raw = _motion_of(ds, tid)
        stats['values_rounded'] += changed_values(raw, normalise_motion(raw) | {'t': raw['t']})
        m = raw if motion is None else motion
        src = {'trial_keys': [str(row['trial_key'])], 'person_id': str(row['person_id']), 'platform': str(row['platform']),
               'hand': str(row['hand']), 'orientation': str(row['orientation']), 'speed': str(row['speed']),
               'set': str(row['analysis_set']), 'excl_rule': None if not isinstance(row['excl_rule'], str)
               else str(row['excl_rule'])}
        case_id = f"real_{len(out) + 1:02d}_{row['trial_key']}{suffix}"
        out.append(build_case(case_id, 'real', m, eng, models, tags=['real', str(row['platform']), str(row['hand'])] + tags,
                              description=description or 'one recorded trial', source=src,
                              truth={'hand': str(row['hand']) if not suffix.startswith('_mirror') else None}))
        if verbose:
            print(f'  stage3 fixtures: real case {len(out)}', flush=True)

    # one primary trial per person, the hand alternating, grips and speeds cycling
    persons = sorted(prim['person_id'].astype(str).unique())
    for k, p in enumerate(persons):
        hand = 'right' if k % 2 == 0 else 'left'
        row = _pick(prim, rng, person_id=p, hand=hand, orientation=grips[k % 3], speed=speeds[k % 3])
        if row is None:
            row = _pick(prim, rng, person_id=p, hand=hand, orientation=grips[k % 3])
        if row is None:
            row = _pick(prim, rng, person_id=p, hand=hand)
        if row is not None:
            add(row, ['single', 'primary', str(row['orientation']), str(row['speed'])])
    # five more: slow reaches and tilted grips on both platforms
    for plat, hand, o, s in (('ios', 'right', 'top_right_toward', 'slow'), ('android', 'left', 'top_left_toward', 'slow'),
                             ('ios', 'left', 'upright_toward', 'fast'), ('android', 'right', 'top_right_toward', 'fast'),
                             ('ios', 'right', 'top_left_toward', 'natural')):
        row = _pick(prim, rng, platform=plat, hand=hand, orientation=o, speed=s)
        if row is None:
            row = _pick(prim, rng, platform=plat, hand=hand, speed=s)
        if row is not None:
            add(row, ['single', 'primary', str(row['orientation']), str(row['speed'])])
    # natural use (free choice, never trained on)
    nat = T[T['analysis_set'] == 'natural_use']
    for plat in ('android', 'android', 'ios'):
        row = _pick(nat, rng, platform=plat)
        if row is not None:
            add(row, ['single', 'natural_use'], description='a natural-use (free-choice) trial')
    # excluded hard cases: sampling gaps near the movement, rejected attempts, no turn
    exc = T[T['analysis_set'] == 'excluded']
    for rule, k in (('E4d', 2), ('E1', 2), ('E4c', 1)):
        sub = exc[exc['excl_rule'] == rule]
        for i in rng.permutation(len(sub))[:k]:
            add(sub.iloc[int(i)], ['single', 'excluded', rule], description=f'an excluded trial (rule {rule})')
    # screen-away twins
    for plat, hand in (('android', 'right'), ('ios', 'left'), ('android', 'left')):
        row = _pick(prim, rng, platform=plat, hand=hand, orientation='upright_toward')
        if row is not None:
            add(row, ['single', 'screen_away', 'grip_gate'], motion=screen_flip(_motion_of(ds, str(row['trial_id']))),
                suffix='_flip', description='a recorded trial turned screen-away (flip_device_vectors on the raw vectors)')
    # multi-touch sessions: four consecutive recordings of one person and hand
    for plat, hand in (('android', 'right'), ('ios', 'left'), ('android', 'left'), ('ios', 'right')):
        sub = prim[(prim['platform'] == plat) & (prim['hand'] == hand)]
        cells = sorted({str(p) for p in sub['person_id']})
        if not cells:
            continue
        p = cells[int(rng.integers(len(cells)))]
        rows = sub[sub['person_id'].astype(str) == p].sort_values(['session_order', 'trial_index', 'attempt'])
        rows = rows.iloc[:4]
        if len(rows) < 2 or not all(_norm_ok(ds, str(r['trial_id']), str(r['engine'])) for _, r in rows.iterrows()):
            continue
        m = concat([_motion_of(ds, str(r['trial_id'])) for _, r in rows.iterrows()])
        r0 = rows.iloc[0]
        src = {'trial_keys': [str(k) for k in rows['trial_key']], 'person_id': p, 'platform': plat, 'hand': hand,
               'set': 'primary', 'concatenated': int(len(rows))}
        case_id = f"real_{len(out) + 1:02d}_{p}_{plat}_{hand}_x{len(rows)}"
        out.append(build_case(case_id, 'real', m, str(r0['engine']), models, source=src, truth={'hand': hand},
                              tags=['real', plat, hand, 'multi_touch', 'concatenated'],
                              description=f'{len(rows)} consecutive recordings of one person and hand, concatenated'))
        if verbose:
            print(f'  stage3 fixtures: real case {len(out)}', flush=True)
    out_stats = {'n': len(out), **stats}
    for doc in out:
        doc['_stats'] = out_stats
    return out


# ───────────────────────────── writing ─────────────────────────────


def _clear(folder: Path) -> None:
    """Remove the fixture files of an earlier run (case ids may shift when the selection changes)."""
    if folder.exists():
        for f in folder.glob('*.json.gz'):
            f.unlink()


def write_fixtures(*, synthetic: bool = True, real: bool = True, verbose: bool = True) -> dict[str, Any]:
    t0 = time.time()
    models = RT.json_set(models_dir())
    summary: dict[str, Any] = {}
    if synthetic:
        t1 = time.time()
        docs = synthetic_cases(models)
        _clear(SYN_DIR)
        rows = [write_case(doc, SYN_DIR) for doc in docs]
        index = {'format': 'nosetouch-fixtures-index-1', 'kind': 'synthetic', 'cases': rows,
                 'models': {k: hashlib.sha256((models_dir() / f).read_bytes()).hexdigest() for k, f in RT.FILES.items()},
                 'generated_by': 'analysis/stage3 fixtures (python -m stage3 fixtures)',
                 'note': 'synthetic streams only (motionlab.synth): no participant data'}
        (SYN_DIR / 'index.json').write_text(json.dumps(index, sort_keys=True, indent=1) + '\n', encoding='utf-8')
        summary['synthetic'] = {'n': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'seconds': round(time.time() - t1, 1)}
        if verbose:
            print(f"stage3: {len(rows)} synthetic fixtures ({summary['synthetic']['bytes'] / 1024:.0f} kB gz)", flush=True)
    if real:
        t1 = time.time()
        docs = real_cases(models, verbose=verbose)
        stats = docs[0].pop('_stats') if docs else {}
        for d in docs[1:]:
            d.pop('_stats', None)
        _clear(REAL_DIR)
        rows = [write_case(doc, REAL_DIR) for doc in docs]
        index = {'format': 'nosetouch-fixtures-index-1', 'kind': 'real', 'cases': rows, 'stats': stats,
                 'models': {k: hashlib.sha256((models_dir() / f).read_bytes()).hexdigest() for k, f in RT.FILES.items()},
                 'note': 'LOCAL ONLY (git-ignored): real recordings of P01..P20'}
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        (REAL_DIR / 'index.json').write_text(json.dumps(clean(index), sort_keys=True, indent=1) + '\n', encoding='utf-8')
        summary['real'] = {'n': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'seconds': round(time.time() - t1, 1),
                           **stats}
        if verbose:
            print(f"stage3: {len(rows)} real fixtures ({summary['real']['bytes'] / 1024:.0f} kB gz, local only)", flush=True)
    summary['seconds'] = round(time.time() - t0, 1)
    return summary
