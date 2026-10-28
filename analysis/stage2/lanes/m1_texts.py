"""Lane m1 texts: the number book and the caption / decision of every figure, generated from saved files.

Every number that appears in the summary or in a figure text is rendered through a :class:`Book`, which registers
it as an entry of results/m1_numbers.json (key, value, format, source, description). The numbers step renders the
summary and every figure text with ONE book and writes it, so the numbers file holds every number the texts cite;
the figures step renders the same texts with a throw-away book. Decision sentences are built from the saved
results (comparisons, counts, directions), never typed.

Text conventions (:func:`render`): a '%' format prints ``83.3 %``; percentage-point differences are stored in
points (format '+.1f'); p values use the '.2g' format.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import report_export as RX
from stage2.lanes import m1_experiments as E
from stage2.lanes import m1_features as MF
from stage2.pipeline import Lane

L = Lane('m1')
SRC_M = 'tables/m1/m1_models.json'
SRC_CAT = 'tables/m1/m1_catalogue.json'
SRC_CAL = 'tables/m1/m1_calibration.json'
SRC_AGG = 'tables/m1/m1_k_aggregation.json'
SRC_FAIL = 'tables/m1/m1_failures.json'
SRC_STRATA = 'tables/m1/m1_strata.json'
SRC_NAT = 'tables/m1/m1_natural.json'
SRC_DRIFT = 'tables/m1/m1_drift.json'
SRC_GRID = 'tables/m1/m1_grid.json'
SRC_SENS = 'tables/m1/m1_sensitivity.json'
SRC_FIN = 'results/finalists_m1.json'
ALPHA = 0.05
PCT, PTS, F3, F2, PV, INT = '.1%', '+.1f', '.3f', '.2f', '.2g', 'd'


# ───────────────────────────── rendering ─────────────────────────────


def render(v: Any, fmt: str, bare: bool = False) -> str:
    """The text of a number: '%' formats print the percentage with a space before ' %' (none when ``bare``)."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '--'
    if isinstance(v, str):
        return v
    if fmt.endswith('%'):
        return f'{100 * float(v):{fmt[:-1]}f}' + ('' if bare else ' %')
    if fmt == 'd':
        return str(int(round(float(v))))
    return format(float(v), fmt)


def _key(k: str) -> str:
    k = re.sub(r'[^a-z0-9_]+', '_', str(k).lower()).strip('_')
    k = re.sub(r'_+', '_', k)
    return k if k and k[0].isalpha() else f'n_{k}'


def mkey(name: str) -> str:
    """Short number-key stem of an experiment name."""
    return 'ref' if name == E.REFERENCE else _key(name[3:] if name.startswith('m1_') else name)


class Book:
    """Collects the numbers as text is rendered (see the module docstring)."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def reg(self, key: str, value: Any, fmt: str, source: str, desc: str) -> str:
        key = _key(key)
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if fmt == 'd' and value is not None and not (isinstance(value, float) and not math.isfinite(value)):
            value = int(round(float(value)))
        e = RX.numbers_entry(value, fmt, source, desc)
        old = self.entries.get(key)
        if old is not None:
            same = old['value'] == e['value'] or (
                isinstance(old['value'], float) and isinstance(e['value'], float)
                and math.isclose(old['value'], e['value'], rel_tol=1e-12, abs_tol=1e-15))
            if not same or old['format'] != e['format']:
                raise ValueError(f'number {key!r} registered twice: {old["value"]!r} ({old["format"]}) and '
                                 f'{e["value"]!r} ({e["format"]})')
            return key
        self.entries[key] = e
        return key

    def t(self, key: str, value: Any, fmt: str, source: str, desc: str, bare: bool = False) -> str:
        """Register and render one number."""
        self.reg(key, value, fmt, source, desc)
        return render(value, fmt, bare)

    def ci(self, key: str, lo: Any, hi: Any, fmt: str, source: str, desc: str) -> str:
        """Register ``key_lo`` / ``key_hi`` and render '[lo, hi]' (bare numbers); '' when undefined."""
        if lo is None or hi is None or any(isinstance(v, float) and not math.isfinite(v) for v in (lo, hi)):
            return ''
        return (f'[{self.t(f"{key}_lo", lo, fmt, source, f"{desc}: 95 % CI low", True)}, '
                f'{self.t(f"{key}_hi", hi, fmt, source, f"{desc}: 95 % CI high", True)}]')


def _isnum(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and not math.isfinite(v)) and not pd.isna(v)


# ───────────────────────────── saved files ─────────────────────────────


def models() -> pd.DataFrame:
    return pd.DataFrame(L.read_json('m1_models')['rows']).set_index('name', drop=False)


def rows(name: str, key: str = 'rows') -> pd.DataFrame:
    return pd.DataFrame(L.read_json(name)[key])


def finalists() -> list[dict[str, Any]]:
    p = L.finalists_path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else []


def slot_names() -> dict[str, str]:
    out = dict(E.ANCHORS)
    for f in finalists():
        if f.get('slot') in out:
            out[f['slot']] = f['name']
    return out


def focus_name() -> str:
    """The model of the failure analysis (the small-model finalist), as recorded by the derive step."""
    try:
        return str(L.read_json('m1_failures').get('model') or E.BASELINE)
    except FileNotFoundError:
        return E.BASELINE


def label(mt: pd.DataFrame, name: str) -> str:
    """The display label of a model; a data-treatment option of PHYS6 is named with its base outside the
    treatment table ('PHYS6, asinh-compressed tails')."""
    if name not in mt.index:
        return name
    lab = str(mt.loc[name, 'label'])
    if mt.loc[name, 'group'] == 'treatment' and 'PHYS6' not in lab:
        lab = f'PHYS6, {lab}'
    return lab


#: short design names for wide tables (E.DESIGN_LABELS has the long ones)
DESIGN_SHORT = {'pre-specified': 'pre-specified', 'post-pooled': 'post-pooled', 'nested': 'nested',
                'family': 'exploratory', 'sensitivity': 'sensitivity', 'analysis': 'analysis'}
#: a signed calibration lean below this (in share units) is reported as 'no clear lean'
LEAN_TOL = 0.01


def lean_word(v: float) -> str:
    return 'no clear lean' if abs(v) < LEAN_TOL else ('under-confident' if v > 0 else 'over-confident')


# ───────────────────────────── model-level renderers ─────────────────────────────


def acc(b: Book, r: pd.Series) -> str:
    k = mkey(r['name'])
    return (f'{b.t(f"{k}_acc", r["accuracy"], PCT, SRC_M, f"{r.label}: LOSO accuracy")} '
            f'{b.ci(f"{k}_acc", r.get("accuracy_lo"), r.get("accuracy_hi"), PCT, SRC_M, f"{r.label}: accuracy")}').strip()


def auc(b: Book, r: pd.Series, ci: bool = True) -> str:
    k = mkey(r['name'])
    s = b.t(f'{k}_auc', r['roc_auc'], F3, SRC_M, f'{r.label}: ROC AUC')
    return f'{s} {b.ci(f"{k}_auc", r.get("roc_auc_lo"), r.get("roc_auc_hi"), F3, SRC_M, f"{r.label}: AUC")}'.strip() if ci else s


def ll(b: Book, r: pd.Series, ci: bool = True) -> str:
    if r.get('hard'):
        return 'n/a (P is 0 or 1)'
    k = mkey(r['name'])
    s = b.t(f'{k}_ll', r['log_loss'], F3, SRC_M, f'{r.label}: log loss')
    return f'{s} {b.ci(f"{k}_ll", r.get("log_loss_lo"), r.get("log_loss_hi"), F3, SRC_M, f"{r.label}: log loss")}'.strip() if ci else s


def brier(b: Book, r: pd.Series) -> str:
    return b.t(f'{mkey(r["name"])}_brier', r['brier'], F3, SRC_M, f'{r.label}: Brier score')


def ece(b: Book, r: pd.Series, src: str = SRC_M, key: str | None = None) -> str:
    """'0.047 (null 95 % quantile 0.052, p 0.14)': the ECE against its perfect-calibration null."""
    if r.get('hard') or not _isnum(r.get('ece')):
        return 'n/a'
    k = key or mkey(r['name'])
    lab = r.get('label', r.get('model', k))
    return (f'{b.t(f"{k}_ece", r["ece"], F3, src, f"{lab}: ECE ({C.ECE_BINS} equal-mass bins)")} (null 95 % quantile '
            f'{b.t(f"{k}_ece_null_q95", r["ece_null_q95"], F3, src, f"{lab}: ECE of a perfectly calibrated model, 95th percentile")}, '
            f'p {b.t(f"{k}_ece_p", r["ece_p"], PV, src, f"{lab}: p of the ECE under perfect calibration")})')


def cox(b: Book, r: pd.Series, src: str = SRC_M, key: str | None = None) -> str:
    if r.get('hard') or not _isnum(r.get('cox_slope')):
        return 'n/a'
    k = key or mkey(r['name'])
    lab = r.get('label', r.get('model', k))
    return (f'{b.t(f"{k}_cox_slope", r["cox_slope"], F2, src, f"{lab}: calibration slope")} '
            f'{b.ci(f"{k}_cox_slope", r["cox_slope_lo"], r["cox_slope_hi"], F2, src, f"{lab}: calibration slope")}')


def nat(b: Book, r: pd.Series, short: bool = False) -> str:
    """'15 of 20 (75 %, Wilson [53, 89])', with the ties when there are any."""
    if not _isnum(r.get('nat_n')) or not r.get('nat_n'):
        return 'n/a'
    from stage2.lanes.m1_tables import wilson
    k = mkey(r['name'])
    n = int(r['nat_n'])
    lo, hi = wilson(r['nat_acc_half'], n)
    s = (f'{b.t(f"{k}_nat_correct", r["nat_correct"], INT, SRC_M, f"{r.label}: natural-use trials answered correctly")} of '
         f'{b.t("nat_n", n, INT, SRC_M, "natural-use trials (one per person, never trained on)")}')
    if short:
        return s
    s += f' ({b.t(f"{k}_nat", r["nat_acc_half"], ".0%", SRC_M, f"{r.label}: natural-use accuracy (a tie counts one half)")}'
    s += f', Wilson {b.ci(f"{k}_nat", lo, hi, ".0%", SRC_M, f"{r.label}: natural-use accuracy, Wilson interval")})'
    if int(r.get('nat_ties') or 0):
        s += (f'; {b.t(f"{k}_nat_ties", r["nat_ties"], INT, SRC_M, f"{r.label}: natural-use trials with P exactly 0.5")} '
              f'tie(s) counted one half (coverage {b.t(f"{k}_nat_coverage", r["nat_coverage"], ".0%", SRC_M, f"{r.label}: share of natural-use trials answered")})')
    return s


def k3(b: Book, r: pd.Series) -> str:
    if not _isnum(r.get('acc_k3')):
        return 'n/a'
    k = mkey(r['name'])
    return (f'{b.t(f"{k}_k3", r["acc_k3"], PCT, SRC_M, f"{r.label}: per-person balanced accuracy, 3 attempts averaged")} '
            f'{b.ci(f"{k}_k3", r["acc_k3_lo"], r["acc_k3_hi"], PCT, SRC_M, f"{r.label}: k = 3 accuracy")}').strip()


def configs(b: Book, r: pd.Series) -> str:
    return b.t(f'{mkey(r["name"])}_configs', r['n_configs_tried'], INT, SRC_M,
               f'{r.label}: configurations of its family compared on the outer folds')


def paired(b: Book, r: pd.Series, pre: str, what: str = 'both', holm: bool = True, src: str = SRC_M,
           key: str | None = None) -> str:
    """A paired comparison stored with prefix ``pre`` (vs_ref, vs_base, vs_rule, vs_anchor): accuracy in points and /
    or the per-person log-loss difference, paired 95 % CI, sign-flip p and the Holm-adjusted p of its family;
    'identical' when the two models agree within solver precision."""
    k = key or f'{mkey(r["name"])}_{pre}'
    if not _isnum(r.get(f'{pre}_acc_diff')):
        return ''
    if bool(r.get(f'{pre}_identical')):
        return 'identical predictions'
    lab = r.get('label', k)
    out = []
    if what in ('both', 'acc'):
        s = (f'{b.t(f"{k}_acc", 100 * r[f"{pre}_acc_diff"], PTS, src, f"{lab}: accuracy difference ({pre}), points")} points '
             f'{b.ci(f"{k}_acc", 100 * r[f"{pre}_acc_diff_lo"], 100 * r[f"{pre}_acc_diff_hi"], PTS, src, f"{lab}: accuracy difference ({pre})")}'
             f', p {b.t(f"{k}_p_acc", r[f"{pre}_p_acc"], PV, src, f"{lab}: sign-flip p of the accuracy difference ({pre})")}')
        if holm and _isnum(r.get(f'{pre}_p_acc_holm')):
            s += f', Holm {b.t(f"{k}_p_acc_holm", r[f"{pre}_p_acc_holm"], PV, src, f"{lab}: Holm-adjusted p, accuracy ({pre})")}'
        out.append(s)
    if what in ('both', 'll'):
        s = (f'log loss {b.t(f"{k}_ll", r[f"{pre}_ll_diff"], "+.3f", src, f"{lab}: per-person log-loss difference ({pre})")} '
             f'{b.ci(f"{k}_ll", r[f"{pre}_ll_diff_lo"], r[f"{pre}_ll_diff_hi"], "+.3f", src, f"{lab}: log-loss difference ({pre})")}'
             f', p {b.t(f"{k}_p_ll", r[f"{pre}_p_ll"], PV, src, f"{lab}: sign-flip p of the log-loss difference ({pre})")}')
        if holm and _isnum(r.get(f'{pre}_p_ll_holm')):
            s += f', Holm {b.t(f"{k}_p_ll_holm", r[f"{pre}_p_ll_holm"], PV, src, f"{lab}: Holm-adjusted p, log loss ({pre})")}'
        out.append(s)
    return '; '.join(out)


def family_size(b: Book, mt: pd.DataFrame, pre: str, fam: str) -> str:
    fams = L.read_json('m1_models').get('families', {})
    n = fams.get(fam, {}).get('size', int((mt.get(f'{pre}_family') == fam).sum()))
    return b.t(f'family_{fam}_size', n, INT, SRC_M, f'comparisons in the Holm family {fam}')


def better(r: pd.Series, pre: str, metric: str = 'll') -> bool | None:
    """True / False when the comparison is beyond its paired CI AND survives Holm (lower log loss or higher accuracy
    is better), None otherwise."""
    lo, hi = r.get(f'{pre}_{metric}_diff_lo'), r.get(f'{pre}_{metric}_diff_hi')
    ph = r.get(f'{pre}_p_{metric}_holm')
    if not (_isnum(lo) and _isnum(hi)) or bool(r.get(f'{pre}_identical')):
        return None
    sig = _isnum(ph) and ph < ALPHA
    good = (hi < 0) if metric == 'll' else (lo > 0)
    bad = (lo > 0) if metric == 'll' else (hi < 0)
    if sig and good:
        return True
    if sig and bad:
        return False
    return None


BAND_MISS = 0.05      # nominal miss rate of a 95 % reliability band


def bins_outside_p(q: pd.Series) -> float:
    """P(at least this many of the reliability bins miss their 95 % band) when each band misses with its nominal
    rate BAND_MISS (binomial tail; bins treated as independent): ten bands already give two misses about one time in
    eleven, so a single or a second miss is not evidence by itself."""
    from scipy.stats import binom
    k, n = int(q.get('bins_outside') or 0), int(q.get('n_bins') or 0)
    return float(binom.sf(k - 1, n, BAND_MISS)) if n else math.nan


def miscalibrated(q: pd.Series) -> bool:
    """Miscalibration detected: the ECE beyond its perfect-calibration null, or more reliability bins outside their
    person-bootstrap band than the bands' nominal miss rate explains (both at ALPHA)."""
    return bool((_isnum(q.get('ece_p')) and q['ece_p'] < ALPHA) or bins_outside_p(q) < ALPHA)


def bins_text(b: Book, q: pd.Series, key: str, lab: str, src: str = SRC_CAL) -> str:
    """'2 of 10 bins outside their band (2 under-, 0 over-confident; binomial p 0.086)'."""
    return (f'{b.t(f"{key}_bins_outside", q["bins_outside"], INT, src, f"{lab}: reliability bins outside their band")} of '
            f'{b.t("cal_n_bins", q["n_bins"], INT, src, "reliability bins")} bins outside their band '
            f'({b.t(f"{key}_bins_under", q["bins_under"], INT, src, f"{lab}: under-confident bins")} under-, '
            f'{b.t(f"{key}_bins_over", q["bins_over"], INT, src, f"{lab}: over-confident bins")} over-confident; binomial p '
            f'{b.t(f"{key}_bins_p", bins_outside_p(q), PV, src, f"{lab}: binomial p of the bins outside against the bands nominal 5 % miss rate")})')


def miscal_shape(q: pd.Series) -> str:
    """The kind of miscalibration from the saved reliability bins and calibration slope: 'none'; 'over' / 'under'
    (the slope CI below / above 1); else, with a slope compatible with 1, 'S' (bins missing on both sides: no single
    map of the logit fixes it), 'under_bins' / 'over_bins' (every miss on one side) or 'ece_only'."""
    if not miscalibrated(q):
        return 'none'
    lo, hi = q.get('cox_slope_lo'), q.get('cox_slope_hi')
    if _isnum(hi) and hi < 1:
        return 'over'
    if _isnum(lo) and lo > 1:
        return 'under'
    u, o = int(q.get('bins_under') or 0), int(q.get('bins_over') or 0)
    if u and o:
        return 'S'
    return 'under_bins' if u else ('over_bins' if o else 'ece_only')


SHAPE_TEXT = {'none': 'no miscalibration beyond chance',
              'S': 'an S-shaped error: bins miss the diagonal on both sides while the calibration slope is compatible '
                   'with 1, so no single global map of the logit can fix it',
              'over': 'over-confidence overall (calibration slope below 1)',
              'under': 'under-confidence overall (calibration slope above 1)',
              'under_bins': 'under-confidence: every bin outside its band misses on the under-confident side (the '
                            'calibration slope CI includes 1)',
              'over_bins': 'over-confidence: every bin outside its band misses on the over-confident side (the '
                           'calibration slope CI includes 1)',
              'ece_only': 'an ECE beyond its perfect-calibration null, with no bin outside its band'}


def ci_excludes(r: pd.Series, pre: str, metric: str) -> int:
    """+1 / -1 when the paired CI lies above / below zero (unadjusted), else 0."""
    lo, hi = r.get(f'{pre}_{metric}_diff_lo'), r.get(f'{pre}_{metric}_diff_hi')
    if not (_isnum(lo) and _isnum(hi)) or bool(r.get(f'{pre}_identical')):
        return 0
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def level_text(b: Book, stratum: str, level: str, idx: Any = None, src: str = SRC_STRATA) -> str:
    """The text of a stratum level; a duration quartile's edges are registered numbers ('Q2 (1.25-1.65 s)')."""
    if stratum == 'duration_bin':
        m = re.match(r'\s*([0-9.]+)-([0-9.]+)\s*s', str(level))
        if m and _isnum(idx):
            q = int(idx)
            lo = b.t(f'duration_q{q}_lo', float(m.group(1)), F2, src, f'lower edge of duration quartile {q}, s')
            hi = b.t(f'duration_q{q}_hi', float(m.group(2)), F2, src, f'upper edge of duration quartile {q}, s')
            return f'Q{q} ({lo}-{hi} s)'
    return str(level)


def join(items: list[str], last: str = 'and') -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1]) + f' {last} ' + items[-1]


# ───────────────────────────── figure texts ─────────────────────────────

FAMILY_ORDER = ('yaw', 'roll', 'lateral', 'dedrift', 'device', 'fixed', 'control', 'even')


def t_M1_01(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    cat = L.read_json('m1_catalogue')
    df = pd.DataFrame(cat['rows'])
    odd, even = df[df['parity'] == 'odd'], df[df['parity'] == 'even']
    best = df.loc[df['auc'].idxmax()]
    flips = int((~df['direction_consistent'].astype(bool)).sum())
    cap = (f'Univariate leave-one-person-out AUC of every catalogue feature '
           f'({b.t("cat_n_features", len(df), INT, SRC_CAT, "catalogue features")} features, '
           f'{b.t("n_trials", cat["n_trials"], INT, SRC_CAT, "primary trials")} trials of '
           f'{b.t("n_persons", cat["n_persons"], INT, SRC_CAT, "persons")} persons) with the direction-only rule '
           'P(right) = sigmoid(s x / RMS(x)): its direction s and scale are fitted on the training persons, with no '
           'intercept (an intercept would track the training class balance and bias the pooled AUC below 0.5), so for '
           'an odd feature the decision is its sign; bars give the share of persons whose own trials the feature '
           f'orders correctly more often than not. Odd (chiral) features reach AUC up to '
           f'{b.t("cat_best_auc", best["auc"], F2, SRC_CAT, "largest univariate AUC")} ({best["name"]}); even features stay between '
           f'{b.t("cat_even_auc_min", even["auc"].min(), F2, SRC_CAT, "smallest AUC of an even feature")} and '
           f'{b.t("cat_even_auc_max", even["auc"].max(), F2, SRC_CAT, "largest AUC of an even feature")}. An AUC below 0.5 means '
           'that the direction learned on the other persons fails on the held-out person: for an uninformative feature '
           f'the learned direction flips from fold to fold ({b.t("cat_n_direction_flips", flips, INT, SRC_CAT, "features whose learned direction changes across folds")} '
           'features change direction across folds).')
    dec = ('Only mirror-odd features carry the hand: models are built from chiral features, and even descriptors enter '
           f'at most as modulators ({b.t("cat_n_odd", len(odd), INT, SRC_CAT, "odd catalogue features")} odd, '
           f'{b.t("cat_n_even", len(even), INT, SRC_CAT, "even catalogue features")} even features).')
    return cap, dec


def _peak_order(b: Book) -> list[tuple[str, float]]:
    """(component, time of the largest right-minus-left difference of the mean rate, % of the approach)."""
    pc = rows('m1_physics_curves')
    out = []
    for ser, nm in (('roll_rate', 'roll'), ('yaw_rate', 'yaw'), ('acc_lat', 'lateral')):
        g = pc[pc['series'] == ser]
        rr = g[g['hand'] == 'right'].sort_values('tau')
        lf = g[g['hand'] == 'left'].sort_values('tau')
        dif = np.abs(rr['mean'].to_numpy() - lf['mean'].to_numpy())
        out.append((nm, float(100 * rr['tau'].to_numpy()[int(np.argmax(dif))])))
    return sorted(out, key=lambda t: t[1])


def t_M1_02(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    pc = rows('m1_physics_curves')
    n = int(pc['n_persons'].max())
    order = _peak_order(b)
    cap = ('Mean time course of the three chiral components of the movement, per hand (mean over persons of each '
           f'person\'s mean; {b.t("physics_n_persons", n, INT, "tables/m1/m1_physics_curves.json", "persons in the class-mean time courses")} persons; '
           'bands: person-bootstrap 95 % CI). Top: rates; bottom: their integrals over the approach. Dashed: the '
           'left-hand mean mirrored (sign flipped). Yaw is rotation about gravity, roll the rotation about the '
           'horizontal axis through the screen, lateral the acceleration towards the person\'s right.')
    peaks = join([f'{c} at {b.t(f"physics_peak_{c}", v, ".0f", "tables/m1/m1_physics_curves.json", f"time of the largest right-minus-left difference of the mean {c} rate, % of the approach")} %'
                  for c, v in order])
    dec = ('The hand shows in three mirror-odd components with different timing: the right-minus-left difference of '
           f'the mean rate peaks for {peaks}. The PHYS6 feature set takes each component at the half and at the end '
           'of the approach.')
    return cap, dec


def t_M1_03(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    kf = L.read_json('m1_key_features')
    cat = pd.DataFrame(L.read_json('m1_catalogue')['rows']).set_index('name')
    ev = [f for f in kf['features'] if cat.loc[f, 'parity'] == 'even']
    od = [f for f in kf['features'] if cat.loc[f, 'parity'] == 'odd']
    aucs = {f: b.t(f'cat_{_key(f)}_auc_2', cat.loc[f, 'auc'], F2, SRC_CAT, f'univariate AUC of {f}') for f in kf['features']}
    cap = (f'Per-hand distributions of key features (clipped at the {o["clip_quantile"] * 100:g}th and '
           f'{100 - o["clip_quantile"] * 100:g}th percentiles for display). Odd features shift in opposite directions for '
           'the two hands; even ones overlap. Titles: leave-one-person-out AUC of the feature alone ('
           + join([f'{f} {aucs[f]}' for f in od + ev]) + ').')
    dec = ('Even descriptors (' + join([f'{f} AUC {aucs[f]}' for f in ev]) + ') cannot tell the hand alone; they are '
           'candidates only for modulating chiral slopes.')
    return cap, dec


CHANNEL_ROWS = ['m1_lr_yaw2', 'm1_lr_roll2', 'm1_lr_lat2', 'm1_lr_yaw_roll', 'm1_lr_yaw_lat', 'm1_lr_phys6',
                'm1_lr_sets_nested', 'm1_lr_chiral_all', 'm1_lr_fwd', 'm1_lr_fix6', 'm1_lr_lead', 'm1_lr_control']


def _options_chosen(name: str) -> pd.Series:
    g = rows('m1_grid')
    q = g[(g['model'] == name) & (g['hyperparameter'] == 'option')]
    if not len(q):
        return pd.Series(dtype=int)
    return pd.Series({k: int(v) for k, v in (x.split(':') for x in q['chosen'].iloc[0].split(';'))})


def t_M1_04(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    mt = models()
    base = mt.loc[E.BASELINE]
    cap = ('Antisymmetric logistic models (no intercept, odd features, L2 strength chosen on inner person splits) on '
           'one component (yaw, roll, lateral; each at the half and the end of the approach), on pairs, on all three '
           '(PHYS6), with the set chosen inside every fold, on every chiral approach feature, with features chosen '
           'in-fold, and on three windows: absolute time after the onset, the lead-in before the onset and the still '
           f'hold before it (negative control, grey). PHYS6: {acc(b, base)}.')
    parts = []
    if 'm1_lr_yaw_lat' in mt.index:
        r = mt.loc['m1_lr_yaw_lat']
        if ci_excludes(r, 'vs_base', 'acc') == 0 and ci_excludes(r, 'vs_base', 'll') == 0:
            parts.append(f'adding roll to yaw and lateral changes nothing measurable (yaw + lateral against PHYS6: '
                         f'{paired(b, r, "vs_base")})')
        else:
            parts.append(f'roll adds to yaw and lateral (yaw + lateral against PHYS6: {paired(b, r, "vs_base")})')
    if 'm1_lr_sets_nested' in mt.index:
        ch = _options_chosen('m1_lr_sets_nested')
        if len(ch):
            top = ch.idxmax()
            parts.append(f'chosen inside the folds, the set is {top} in '
                         f'{b.t("sets_nested_top_folds", ch.max(), INT, SRC_GRID, "folds in which the nested channel-set choice picks its modal set")} '
                         f'of {b.t("n_folds", ch.sum(), INT, SRC_GRID, "leave-one-person-out folds")} folds '
                         f'({paired(b, mt.loc["m1_lr_sets_nested"], "vs_base")} against PHYS6)')
    ctl = mt.loc['m1_lr_control'] if 'm1_lr_control' in mt.index else None
    if ctl is not None:
        if ctl['roc_auc_lo'] > 0.5:
            parts.append(f'the still hold before the onset is not at chance (AUC {auc(b, ctl)}): the held phone already '
                         'carries a weak chiral trace of the hand')
        else:
            parts.append(f'the negative control stays at chance (AUC {auc(b, ctl)})')
    dec = 'Which components to keep: ' + '; '.join(parts) + '.'
    return cap, dec


def ladder_names(mt: pd.DataFrame) -> list[str]:
    """The model ladder of M1_05: the rules, the finalists and one representative of every model form."""
    want = ['m1_rule_fhy_sign', 'm1_rule_fhy_sign_cal', 'm1_rule_fhy_threshold', 'm1_rule_nested', 'm1_lr_fhy',
            'm1_lr_yawq', 'm1_lr_phys6', 'm1_lr_phys6_win_asinh', 'm1_lr_phys6_nested', 'm1_lr_phys6_free',
            'm1_lr_phys6_ix', 'm1_lr_fwd', 'm1_tf_yaw', 'm1_tf_multi', 'm1_tree', 'm1_gam_phys6',
            *[f['name'] for f in finalists()], E.REFERENCE]
    return [n for n in dict.fromkeys(want) if n in mt.index]


def t_M1_05(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    mt = models()
    t = mt.loc[ladder_names(mt)]
    cap = ('Accuracy and log loss of the interpretable models (grey: rules; blue: fitted models; black: the reference '
           'floor, dotted line). Every hyperparameter is chosen on inner person splits; the zero-parameter rule outputs '
           '0 or 1 and has no finite log loss. Dashed: the log loss of a constant 0.5. Finalists are marked with a '
           'diamond.')
    fit = t[(t['name'] != E.REFERENCE)]
    fam = 'vs_reference'
    n_better = int(sum(better(r, 'vs_ref', 'acc') is True for _, r in fit.iterrows()))
    n_raw = int(sum(ci_excludes(r, 'vs_ref', 'acc') > 0 for _, r in fit.iterrows()))
    shown = t[~t['hard'].astype(bool)]
    fins = {f['name'] for f in finalists()}
    low = shown[shown['log_loss'] <= shown['log_loss'].min() + 1e-12]           # ties (identical models): a finalist
    best = low.loc[[n for n in low.index if n in fins] or list(low.index)].iloc[0]
    best_lab = o.get('labels', {}).get(best['name'], best['label'])
    dec = (f'Against the reference floor, {b.t("ladder_n_better_raw", n_raw, INT, SRC_M, "ladder models above the reference beyond the unadjusted paired CI (accuracy)")} '
           f'of the {b.t("ladder_n_models", len(fit), INT, SRC_M, "models of the ladder figure")} models shown are more '
           f'accurate beyond the unadjusted paired 95 % CI and {b.t("ladder_n_better_holm", n_better, INT, SRC_M, "ladder models above the reference after the Holm correction")} '
           f'after the Holm correction over all {family_size(b, mt, "vs_ref", fam)} m1 models paired with it. Among the '
           f'models shown the lowest log loss is {best_lab} ({ll(b, best, ci=False)})'
           + (f', tied with {join([o.get("labels", {}).get(n, label(mt, n)) for n in low.index if n != best["name"]])}'
              if len(low) > 1 else '') + '.')
    return cap, dec


def t_M1_06(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    js = L.read_json('m1_tf_yaw_profile')
    g = pd.DataFrame(js['rows']).sort_values('tau')
    w, tau = g['coef_std_median'].to_numpy(), g['tau'].to_numpy()
    early, late = w[tau < 0.5], w[tau >= 0.5]
    first_neg = tau[(tau >= 0.5) & (w < 0)]
    src = 'tables/m1/m1_tf_yaw_profile.json'
    cap = ('Coefficient profiles of antisymmetric logistic "temporal filters" on the binned chiral rates '
           f'({b.t("tc_bins", js.get("n_bins", 16), INT, src, "time bins of the temporal filters")} bins of the '
           'time-normalised approach; one scale per channel, so weights compare across time): median over the '
           f'{b.t("tf_n_folds", js["n_folds"], INT, src, "folds of the temporal-filter profile")} leave-one-person-out folds, band '
           + ('10th to 90th percentile across folds.' if o.get('band') == 'q10_q90' else 'range across folds.')
           + ' A positive weight means a positive (counter-clockwise, rightward) rate in that bin argues for the right hand.')
    dec = (f'Yaw-rate evidence: mean weight {b.t("tf_yaw_w_early", early.mean(), "+.2f", src, "mean temporal-filter weight, first half of the approach")} '
           f'in the first half of the approach and {b.t("tf_yaw_w_late", late.mean(), "+.2f", src, "mean temporal-filter weight, second half")} in the second'
           + (f', where it turns negative from {b.t("tf_yaw_first_negative", 100 * first_neg.min(), ".0f", src, "first negative bin of the second half, % of the approach")} % (the turn back)'
              if first_neg.size else '') + ': features must be time-resolved, not whole-movement integrals.')
    return cap, dec


def t_M1_07(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    ts = L.read_json('m1_tree_summary')
    roots = pd.DataFrame(ts['roots_per_fold'])
    mode = roots['feature'].mode().iloc[0]
    nmode = int((roots['feature'] == mode).sum())
    src = 'tables/m1/m1_tree_summary.json'
    cap = (f'Left: the shallow tree with the hyperparameters chosen most often in the folds '
           f'({b.t("tree_config_folds", ts["n_folds_with_this_config"], INT, src, "folds choosing the drawn tree configuration")} of '
           f'{b.t("tree_n_folds", ts["n_folds"], INT, src, "folds of the tree")}), refitted on every trial and its mirror image '
           '(for illustration; the reported accuracy comes from the per-fold trees, whose predictions are symmetrised); '
           'a leaf shows P(right) and its number of (augmented) training rows. Right: how often each feature is split on '
           f'across the fold trees.')
    even_used = [f for f in ts['features_used_per_fold'] if f in MF.BY_NAME and MF.BY_NAME[f].parity == 'even']
    dec = (('The tree asks about chiral features only' if not even_used else
            f'The tree splits mostly on chiral features (even features used: {join(even_used)})')
           + f'; its first question ({mode}) is the same in '
           f'{b.t("tree_root_mode_folds", nmode, INT, src, "folds whose tree root is the modal feature")} of {b.t("tree_n_folds", ts["n_folds"], INT, src, "folds of the tree")} folds, '
           'so the rule is stable enough to be drawn and explained.')
    return cap, dec


def t_M1_08(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    gs = L.read_json('m1_gam_shapes')
    mt = models()
    src = 'tables/m1/m1_gam_shapes.json'
    desc = pd.DataFrame(gs.get('shape_descriptors', []))
    kn = pd.Series(gs['knots_per_fold']).value_counts()
    cap = ('Shape functions of the GAM on PHYS6 (cubic splines, mirror-augmented and symmetrised; number of knots and L2 '
           f'strength chosen on inner splits): median over the {b.t("gam_n_folds", gs["n_folds"], INT, src, "folds of the GAM")} '
           'folds and the range across folds (training 2-98 % range of each fold), in the units of each feature. '
           'Contributions are 0 at a zero feature value (odd shapes).')
    sat = desc[desc['saturating'].astype(bool)] if len(desc) else desc
    turn = (desc[np.sign(desc['slope_outer']) != np.sign(desc['slope_centre'])] if len(desc) else desc)
    g = mt.loc['m1_gam_phys6'] if 'm1_gam_phys6' in mt.index else None
    cmp_ = ''
    if g is not None:
        verdict = better(g, 'vs_base', 'll')
        cmp_ = (f' Against the linear PHYS6 model: {paired(b, g, "vs_base", "ll")}'
                + {True: ', lower after the Holm correction over its family',
                   False: ', higher after the Holm correction over its family'}.get(
                    verdict, ', no difference after the Holm correction over its family') + '.')
    knots = []
    for kk, v in kn.sort_index().items():
        nk = int(kk)
        knots.append(f'{nk} in ' + b.t(f'gam_knots_{nk}_folds', v, INT, src, f'folds choosing {nk} knots') + ' folds')
    dec = (f'The median shapes are steepest around zero: the slope over the outer fifths of the range is smaller than '
           f'the central slope for {b.t("gam_n_saturating", len(sat), INT, src, "GAM shapes whose outer-fifth slope is below the central slope")} '
           f'of {b.t("gam_n_shapes", len(desc), INT, src, "GAM shape functions")} features and reverses its sign for '
           f'{b.t("gam_n_turning", len(turn), INT, src, "GAM shapes whose outer-fifth slope has the opposite sign")}'
           + (f' ({join(list(turn["feature"]))})' if len(turn) else '')
           + (': beyond the bulk of the trials a larger value adds no further evidence' if len(sat) == len(desc) and len(desc) else '')
           + f' (knots chosen: {join(knots)}).' + cmp_)
    return cap, dec


def t_M1_09(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    fl = L.read_json('m1_failures')
    mt = models()
    pp = pd.DataFrame(fl['per_person']).sort_values('accuracy')
    lab = label(mt, fl['model'])
    low = pp.iloc[0]
    cap = (f'Leave-one-person-out accuracy of every person ({b.t("fail_n_persons", len(pp), INT, SRC_FAIL, "persons of the failure analysis")}) with '
           f'the small-model finalist ({lab}; bars, coloured by platform) and per hand (ticks). * marks the left-handed '
           'persons.')
    pid = str(low['person_id'])
    low_txt = b.t(f'person_{_key(pid)}_acc', low['accuracy'], PCT, SRC_FAIL, f'{pid}: accuracy of the failure-analysis model')
    dec = (f'Persons differ: from {low_txt} '
           f'({pid}) to {b.t("person_max_acc", pp["accuracy"].max(), PCT, SRC_FAIL, "highest per-person accuracy")}; '
           f'{b.t("persons_below_70", int((pp["accuracy"] < 0.7).sum()), INT, SRC_FAIL, "persons below 70 % accuracy")} '
           f'persons are below {b.t("threshold_70", 0.7, ".0%", SRC_FAIL, "per-person accuracy threshold of the failure count")}.')
    return cap, dec


STRATUM_LABELS = {'platform': 'platform', 'instructed_rotation': 'grip', 'speed': 'speed', 'variant': 'variant',
                  'start_position': 'start', 'posture': 'posture', 'hand': 'hand', 'handedness': 'handedness',
                  'duration_bin': 'duration', 'grip_mismatch': 'grip mismatch flag', 'gate_pass': 'passes gates G1, G2'}


def t_M1_10(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    st = rows('m1_strata')
    focus = focus_name()
    mt = models()
    s = st[(st['model'] == focus) & st['stratum'].isin(STRATUM_LABELS) & (st['n_trials'] >= o['min_trials'])]
    js = L.read_json('m1_strata')
    cap = (f'Accuracy of the small-model finalist ({label(mt, focus)}) by condition (strata after prediction; levels '
           f'with at least {b.t("strata_min_trials", o["min_trials"], INT, SRC_STRATA, "minimum trials of a level shown in M1_10")} trials; '
           'person-bootstrap 95 % CIs only for levels with at least '
           f'{b.t("strata_min_persons_ci", js.get("min_persons_ci", 5), INT, SRC_STRATA, "minimum persons of a level with a CI")} persons). '
           'Conditions are never model inputs.')
    cand = s[s['n_persons'] >= js.get('min_persons_ci', 5)]
    worst = cand.loc[cand['accuracy'].idxmin()] if len(cand) else None
    dec = 'Where the model is weak: ' + (
        f'the lowest level with enough persons is {STRATUM_LABELS[worst["stratum"]]} '
        f'{level_text(b, worst["stratum"], worst["level"], worst.get("level_index"))} '
        f'({b.t(f"strata_{mkey(focus)}_worst_acc", worst["accuracy"], PCT, SRC_STRATA, "accuracy of the weakest condition level")}, '
        f'{b.t(f"strata_{mkey(focus)}_worst_n", worst["n_trials"], INT, SRC_STRATA, "trials of the weakest condition level")} trials).'
        if worst is not None else 'no level has enough persons.')
    return cap, dec


def t_M1_11(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    cal = L.read_json('m1_calibration')
    met = pd.DataFrame(cal['metrics'])
    pc = pd.DataFrame(cal['paired'])
    a = met[(met['model'] == E.BASELINE) & (met['scope'] == 'all')].iloc[0]
    tr = pc[(pc['kind'] == 'treatment')]
    rem = pc[(pc['kind'] == 'remedy') & (pc['baseline'] == E.BASELINE)]
    cap = ('Calibration of the PHYS6 antisymmetric logistic model and its remedies. Left, middle: reliability diagrams '
           f'({b.t("ece_bins", cal.get("ece_bins", C.ECE_BINS), INT, SRC_CAL, "equal-mass bins of the ECE")} equal-mass bins; '
           'band: person-bootstrap 95 % CI of the observed share; dashed: perfect calibration; circled: bins whose band '
           'misses the diagonal) of PHYS6 and of the small-model finalist (its data treatment chosen inside every fold). '
           'Right: the change of the log loss of each remedy against PHYS6 (paired person '
           'bootstrap; every post-hoc calibrator is fitted on the inner cross-fitted predictions of the training persons).')
    parts = [('PHYS6 is miscalibrated' if miscalibrated(a) else 'PHYS6 shows no miscalibration beyond chance')
             + f': ECE {ece(b, a, SRC_CAL, "cal_phys6")}, {bins_text(b, a, "cal_phys6", "PHYS6")}']
    good_post = rem[(rem['ll_diff_hi'] < 0)]
    rem_txt = []
    for x, y in zip(rem['model'], rem['ll_diff']):
        rem_txt.append(_remedy(x) + ' ' + b.t(f'cal_{mkey(x)}_dll', y, '+.3f', SRC_CAL, f'{x}: log-loss change against its base'))
    parts.append(f'no post-hoc map lowers its log loss beyond the paired CI ({join(rem_txt)})'
                 if not len(good_post) else f'{join(list(good_post["model"]))} lower its log loss beyond the paired CI')
    for _, t in tr.iterrows():
        m2 = met[(met['model'] == t['model']) & (met['scope'] == 'all')]
        if not len(m2):
            continue
        m2 = m2.iloc[0]
        kk = f'cal_{mkey(t["model"])}'
        nm = str(t['model'])
        parts.append(f'{label(models(), nm)}: log loss '
                     + b.t(f'{kk}_dll', t['ll_diff'], '+.3f', SRC_CAL, f'{nm}: log-loss change against PHYS6') + ' '
                     + b.ci(f'{kk}_dll', t['ll_diff_lo'], t['ll_diff_hi'], '+.3f', SRC_CAL, f'{nm}: log-loss change against PHYS6')
                     + f', ECE {ece(b, m2, SRC_CAL, kk)}, {bins_text(b, m2, kk, nm)}')
    dec = '; '.join(parts) + '.'
    return cap, dec


def _remedy(name: str) -> str:
    for cal, short, lab in E.REMEDIES:
        if name.endswith(f'_cal_{short}'):
            return lab
    return name


def t_M1_12(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    js = L.read_json('m1_k_aggregation')
    ka = pd.DataFrame(js['rows'])
    model = agg_model(ka)
    g = ka[ka['model'] == model]
    alpha = g.loc[g['method'] == 'fitted_alpha', 'alpha_median']
    n_p = int(g['n_persons'].max())
    mt = models()
    cap = (f'Combining k attempts of one person and hand with {label(mt, model)} '
           f'({b.t("agg_n_persons", n_p, INT, SRC_AGG, "persons of the aggregation analysis")} persons with at least '
           f'{b.t("agg_k_max", g["k"].max(), INT, SRC_AGG, "largest k of the aggregation analysis")} trials per hand; '
           f'{b.t("agg_n_draws", js.get("n_draws", 200), INT, SRC_AGG, "random k-subsets per person and hand")} random draws per cell; '
           'bands: person-bootstrap 95 % CI; the ECE is shown without a band). The exponent alpha of k^(-alpha) x sum of '
           'logits is fitted on the inner cross-fitted predictions of the training persons'
           + (f' (median over folds {b.t("agg_alpha_median", float(alpha.median()), F2, SRC_AGG, "median in-fold aggregation exponent")}).'
              if len(alpha) else '.'))
    return cap, agg_decision(b, ka, model)


def agg_model(ka: pd.DataFrame) -> str:
    """The model the aggregation figure and text use: the small-model finalist when it carries an exponent."""
    s = slot_names()['small']
    return s if s in set(ka['model']) else E.BASELINE


def agg_decision(b: Book, ka: pd.DataFrame, model: str) -> str:
    js = L.read_json('m1_k_aggregation')
    pr = pd.DataFrame(js.get('paired', []))
    g = ka[(ka['model'] == model) & (ka['k'] == 3)].set_index('method')
    lab = {'mean_p': 'mean of P', 'mean_logit': 'mean of logits', 'sum_logit': 'sum of logits',
           'fitted_alpha': 'k^(-alpha) x sum'}
    k = mkey(model)
    lean_txt = join([f'{lab[m]} ECE {b.t(f"agg_{k}_k3_ece_{m}", g.loc[m, "ece"], F3, SRC_AGG, f"{model}: ECE of 3-attempt aggregates, {m}")} '
                     f'({lean_word(g.loc[m, "lean"])})' for m in g.index])
    p3 = pr[(pr['model'] == model) & (pr['k'] == 3)] if len(pr) else pr
    if len(p3):
        q = p3.iloc[0]
        pair = (f'fitted alpha minus the plain sum: log loss {b.t(f"agg_{k}_k3_dll", q["ll_diff"], "+.3f", SRC_AGG, f"{model}: k = 3 log loss, fitted alpha minus sum")} '
                f'{b.ci(f"agg_{k}_k3_dll", q["ll_diff_lo"], q["ll_diff_hi"], "+.3f", SRC_AGG, f"{model}: k = 3 log-loss difference")}, ECE '
                f'{b.t(f"agg_{k}_k3_dece", q["ece_diff"], "+.3f", SRC_AGG, f"{model}: k = 3 ECE, fitted alpha minus sum")} '
                f'{b.ci(f"agg_{k}_k3_dece", q["ece_diff_lo"], q["ece_diff_hi"], "+.3f", SRC_AGG, f"{model}: k = 3 ECE difference")}')
        if q['ll_diff_hi'] < 0:
            verdict = 'the fitted exponent is better beyond its paired CI: aggregate with k^(-alpha) x sum.'
        elif q['ll_diff_lo'] > 0:
            verdict = 'the plain sum is better beyond its paired CI: aggregate with the parameter-free sum of logits.'
        else:
            verdict = 'the two logit sums cannot be told apart, so the parameter-free sum of logits is preferred.'
    else:
        pair, verdict = '', 'aggregate the logits.'
    leans = [float(g.loc[m, 'lean']) for m in ('mean_p', 'mean_logit') if m in g.index]
    avg = ('Averaging P or logits of correlated attempts is under-confident here; ' if leans and min(leans) >= LEAN_TOL else
           'Averaging P or logits of correlated attempts is over-confident here; ' if leans and max(leans) <= -LEAN_TOL else '')
    return f'At k = 3: {lean_txt}. ' + avg + (f'{pair}: ' if pair else '') + verdict


TREATMENT_ROWS = ['m1_lr_phys6_scale_none', 'm1_lr_phys6_scale_robust', 'm1_lr_phys6_win_clip', 'm1_lr_phys6_win_asinh',
                  'm1_lr_phys6_ampnorm', 'm1_lr_phys6_dedrift', 'm1_lr_phys6_nested', 'm1_lr_phys6_aug',
                  'm1_lr_phys6_free', 'm1_lr_phys6_ix', 'm1_lr_chiral_all', 'm1_lr_fwd', 'm1_lr_sets_nested',
                  'm1_tf_multi', 'm1_gam_phys6']


def t_M1_13(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    mt = models()
    fam = f'{E.BASELINE}:options'
    cap = ('Each option against the PHYS6 antisymmetric logistic model on the same trials: scaling (none, robust), '
           'winsorising (clip at the training 99th percentile of |x|, asinh), per-trial amplitude normalisation, '
           'de-drifted lateral velocity, the treatment chosen inside every fold, mirror augmentation with an intercept, '
           'an unconstrained model with even descriptors, odd x even interactions, more features, in-fold selections, '
           'splines and the multi-channel temporal filter. Paired person bootstrap; text: Holm-adjusted p of the '
           f'per-person log-loss difference over the {family_size(b, mt, "vs_base", fam)} options paired with PHYS6.')
    return cap, treatment_decision(b, mt)


def treatment_decision(b: Book, mt: pd.DataFrame) -> str:
    fam = f'{E.BASELINE}:options'
    t = mt[mt.get('vs_base_family') == fam]
    win = [n for n, r in t.iterrows() if better(r, 'vs_base', 'll') is True]
    raw = [n for n, r in t.iterrows() if ci_excludes(r, 'vs_base', 'll') < 0]
    worse = [n for n, r in t.iterrows() if better(r, 'vs_base', 'll') is False]
    s = []
    if win:
        s.append('Lower log loss than PHYS6 after the Holm correction: ' + join([label(mt, n) for n in win]) + '.')
    else:
        s.append(f'No option lowers the log loss of PHYS6 after the Holm correction over the '
                 f'{family_size(b, mt, "vs_base", fam)} options; '
                 + (('beyond the unadjusted paired CI: ' + join([f'{label(mt, n)} ({paired(b, mt.loc[n], "vs_base", "ll")})' for n in raw]) + '.')
                    if raw else 'none does even before the correction.'))
    if worse:
        s.append('Worse after the correction: ' + join([label(mt, n) for n in worse]) + '.')
    nest = 'm1_lr_phys6_nested'
    if nest in mt.index:
        ch = _options_chosen(nest)
        if len(ch):
            s.append(f'Chosen inside every fold, the treatment is {ch.idxmax()} in '
                     f'{b.t("nested_top_folds", ch.max(), INT, SRC_GRID, "folds choosing the modal treatment of the nested model")} of '
                     f'{b.t("n_folds", ch.sum(), INT, SRC_GRID, "leave-one-person-out folds")} folds (the nested model: '
                     f'{paired(b, mt.loc[nest], "vs_base")}).')
    return ' '.join(s)


def t_M1_14(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    fl = L.read_json('m1_failures')
    mt = models()
    ov = pd.DataFrame(fl['overrepresentation'])
    ov = ov[ov['stratum'].isin(['instructed_rotation', 'speed', 'platform', 'hand', 'handedness'])]
    top = ov.loc[ov['ratio'].idxmax()]
    bc = fl['block_clustering']
    cap = (f'Left: the mean yaw rate of correctly classified trials and of confident errors (P(true hand) < '
           f'{b.t("confident_threshold", fl["confident_threshold"], ".1f", SRC_FAIL, "P(true hand) below which an error is confident")}; '
           f'{b.t("fail_n_confident", fl["n_confident_errors"], INT, SRC_FAIL, "confident errors of the small-model finalist")} of '
           f'{b.t("fail_n_trials", fl["n_trials"], INT, SRC_FAIL, "trials of the failure analysis")} trials, {label(mt, fl["model"])}), '
           'by the instructed hand. Descriptive only: the errors are selected on this evidence, so their curves must '
           'look like the other hand. Right: share of confident errors over share of trials per condition, with the '
           'count of confident errors (1 = as often as among all trials).')
    lo, hi = top['share_lo'], top['share_hi']
    dec = (f'Most over-represented: {STRATUM_LABELS.get(top["stratum"], top["stratum"])} {top["level"]} '
           f'({b.t("fail_top_n", top["n_confident_errors"], INT, SRC_FAIL, "confident errors in the most over-represented level")} of '
           f'{b.t("fail_n_confident", fl["n_confident_errors"], INT, SRC_FAIL, "confident errors of the small-model finalist")} confident errors against '
           f'{b.t("fail_top_share_trials", top["share_trials"], ".0%", SRC_FAIL, "share of trials of that level")} of the trials; '
           f'Wilson {b.ci("fail_top_share", lo, hi, ".0%", SRC_FAIL, "share of confident errors in that level")} %). Confident errors '
           f'fall in {b.t("fail_blocks_with_confident", bc["blocks_with_confident"], INT, SRC_FAIL, "blocks holding a confident error")} '
           f'blocks against {b.t("fail_blocks_null_mean", bc["null_mean"], ".1f", SRC_FAIL, "blocks expected under a within-person permutation")} '
           f'expected within persons (p {b.t("fail_blocks_p", bc["p_clustered"], PV, SRC_FAIL, "one-sided p of block clustering")}): '
           + ('they cluster in blocks, as a hand swap for a whole block would make them.' if bc['p_clustered'] < ALPHA
              else 'no clustering in blocks beyond each person\'s error count.'))
    return cap, dec


def t_M1_15(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    cal = L.read_json('m1_calibration')
    met = pd.DataFrame(cal['metrics'])
    mt = models()
    fins = [f['name'] for f in finalists()]
    cap = ('Reliability of the lane m1 finalists on the primary set (leave-one-person-out probabilities; equal-mass '
           'bins of P(right); band: person-bootstrap 95 % CI of the observed share of right hands; dashed: perfect '
           'calibration). Text: ECE with the 95th percentile of its perfect-calibration null, the calibration slope with '
           'its person-bootstrap CI, Brier score and log loss over all trials, and the ECE per platform. No calibration '
           'step is applied. The rule outputs its training hit rate q or 1 - q, with q fitted per fold, so equal-mass '
           'bins would only sort the folds by q; its panel shows the two decision groups instead (no band).')
    parts = []
    for n in fins:
        q = met[(met['model'] == n) & (met['scope'] == 'all')]
        if not len(q) or (n in mt.index and mt.loc[n, 'group'] == 'rules'):
            continue
        q = q.iloc[0]
        k = f'fin_{mkey(n)}'
        verdict = 'miscalibrated' if miscalibrated(q) else 'consistent with calibration'
        parts.append(f'{label(mt, n)}: ECE {ece(b, q, SRC_CAL, k)}, slope {cox(b, q, SRC_CAL, k)}, '
                     f'{bins_text(b, q, k, n)}: {verdict}')
    dec = 'Fitted finalists: ' + '; '.join(parts) + '.' if parts else 'No fitted finalist.'
    return cap, dec


def t_M1_16(b: Book, o: dict[str, Any]) -> tuple[str, str]:
    nj = L.read_json('m1_natural')
    nt = pd.DataFrame(nj['transfer'])
    n = int(nt['n'].max())
    cap = ('Natural use: does each component transfer to the free-choice trials? Left: for each chiral feature, the '
           'number of free-choice trials whose sign points to the reported hand (dot) against the number expected from '
           'the same persons\' instructed trials of that hand (bar: 95 % range of the Poisson-binomial count; the '
           'direction of each person is learned on the other persons). Right: the mean logit contribution of each '
           'component of the small-model finalist towards the reported hand, on the free-choice trials and on the same '
           f'persons\' instructed trials of that hand ({b.t("nat_n_persons", n, INT, SRC_NAT, "persons in the natural-use transfer test")} persons, '
           'one free-choice trial each).')
    return cap, natural_decision(b, nt, nj)


def natural_decision(b: Book, nt: pd.DataFrame, nj: dict[str, Any]) -> str:
    drop = nt[(nt['p_two_sided_holm'] < ALPHA) & (nt['observed'] < nt['expected'])]
    keep = nt[nt['p_two_sided_holm'] >= ALPHA]
    s = []
    for _, r in nt.iterrows():
        k = f'nat_{_key(r["feature"])}'
        b.reg(f'{k}_observed', r['observed'], INT, SRC_NAT, f'{r.feature}: free-choice trials pointing to the reported hand')
        b.reg(f'{k}_expected', r['expected'], '.1f', SRC_NAT, f'{r.feature}: expected count from the persons\' instructed trials')
        b.reg(f'{k}_p_holm', r['p_two_sided_holm'], PV, SRC_NAT, f'{r.feature}: Holm-adjusted Poisson-binomial p')
    if len(drop):
        s.append('Transfers worse than on the same persons\' instructed trials (Holm over the '
                 f'{b.t("nat_n_features", len(nt), INT, SRC_NAT, "features of the natural-use transfer test")} features): '
                 + join([f'{r["feature"]} {render(r["observed"], INT)} of '
                         f'{b.t("nat_transfer_n", r["n"], INT, SRC_NAT, "free-choice trials in the transfer test")} against '
                         f'{render(r["expected"], ".1f")} expected (p {render(r["p_two_sided_holm"], PV)})' for _, r in drop.iterrows()]) + '.')
    else:
        s.append(f'No feature transfers worse after the Holm correction over {b.t("nat_n_features", len(nt), INT, SRC_NAT, "features of the natural-use transfer test")} features.')
    ok = keep[keep['component'] == 'yaw']
    if len(ok):
        s.append('Yaw shows no such drop: ' + join([f'{r["feature"]} {render(r["observed"], INT)} of '
                                          f'{b.t("nat_transfer_n", r["n"], INT, SRC_NAT, "free-choice trials in the transfer test")} against '
                                          f'{render(r["expected"], ".1f")} expected' for _, r in ok.iterrows()]) + '.')
    return ' '.join(s)


FIG_TEXTS: dict[str, Callable[[Book, dict[str, Any]], tuple[str, str]]] = {
    'M1_01_feature_catalogue': t_M1_01, 'M1_02_chiral_components': t_M1_02, 'M1_03_key_distributions': t_M1_03,
    'M1_04_channel_sets': t_M1_04, 'M1_05_model_ladder': t_M1_05, 'M1_06_temporal_filter': t_M1_06,
    'M1_07_tree': t_M1_07, 'M1_08_gam_shapes': t_M1_08, 'M1_09_per_person': t_M1_09, 'M1_10_conditions': t_M1_10,
    'M1_11_calibration': t_M1_11, 'M1_12_k_attempts': t_M1_12, 'M1_13_data_treatment': t_M1_13,
    'M1_14_failures': t_M1_14, 'M1_15_finalist_calibration': t_M1_15, 'M1_16_natural_transfer': t_M1_16,
}


def fig_text(name: str, book: Book | None = None, settings: dict[str, Any] | None = None) -> tuple[str, str]:
    """(caption, decision) of figure ``name`` from the saved files (settings: the figure's FIGURES entry)."""
    from stage2.lanes.m1_figures import FIGURES
    return FIG_TEXTS[name](book if book is not None else Book(), {**FIGURES.get(name, {}), **(settings or {})})
