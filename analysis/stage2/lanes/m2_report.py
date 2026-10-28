"""Lane m2 tables (CSV + LaTeX booktabs) and numbers (results/m2_numbers.json), from the saved derived files only.

    python -m stage2.pipeline tables --lanes m2      # tables/m2/m2_*.csv + .tex
    python -m stage2.pipeline numbers --lanes m2     # results/m2_numbers.json (then results/numbers.tex), m2_SUMMARY.md

``number_values()`` builds every number the lane's text cites (the summary, the figure captions and decisions) as
``report_export.numbers_entry`` records; ``fmt(n, key)`` prints one exactly as its LaTeX macro does. The summary and
the figure texts use only ``fmt``, so every number they show is a number of the numbers file.
"""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from stage2 import config as C
from stage2 import report_export as RX
from stage2.lanes import m2_pipeline as P
from stage2.lanes import m2_stats as S
from stage2.pipeline import Lane, selected

L = Lane('m2')
SRC_EXP = 'tables/m2/m2_experiments.json'
SRC_CAL = 'tables/m2/m2_calibration.json'
SRC_K = 'tables/m2/m2_k_attempts.json'
SRC_SUM = 'tables/m2/m2_summary.json'
SRC_IMP = 'tables/m2/m2_importance.json'
SRC_LC = 'tables/m2/m2_lc_runs.csv'
SRC_BD = 'tables/m2/m2_strata.csv'
SRC_PP = 'tables/m2/m2_per_person.csv'
SRC_SEL = 'tables/m2/m2_select_choices.csv'
SRC_REG = 'tables/m2/m2_feature_registry.csv'
SRC_CODE = 'stage2/lanes/m2_pipeline.py'
SRC_CFG = 'stage2/config.py'
FAMILY_TEXT = {
    'bins': 'mean in k equal time bins of the window',
    'stats': 'mean, SD, min, max, arg-min/max, skewness, kurtosis, energy, zero crossings',
    'fft': 'log band powers and real/imaginary parts of the lowest Fourier coefficients',
    'wavelet': 'log energy of every db2 sub-band',
    'xcorr': 'correlation of every channel pair within a frame group',
    'cumint': 'cumulative integral at quarters of the window (rates, accelerations)',
    'global': 'log approach duration',
}
AGG_SHORT = {'mean_p': 'meanp', 'mean_logit': 'meanlogit', 'sum_logit': 'sum', 'sum_logit_tempered': 'tempered'}
AGG_TEXT = {'mean_p': 'mean P(right)', 'mean_logit': 'mean logit', 'sum_logit': 'sum of logits',
            'sum_logit_tempered': 'sum of logits / nested tau_k'}
CAL_TEXT = {'none': 'uncalibrated', 'platt': 'Platt', 'temperature': 'temperature', 'isotonic': 'isotonic',
            'beta': 'beta', 'platt_platform': 'Platt per platform'}


# ───────────────────────────── loading and formatting ─────────────────────────────


def _exp() -> pd.DataFrame:
    return pd.DataFrame(L.read_json('m2_experiments')['rows'])


def short(name: str) -> str:
    """Number-key prefix of an experiment (its name without 'm2_')."""
    return name[3:] if name.startswith('m2_') else name


def slug(text: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')


def fmt(n: dict[str, Any], key: str) -> str:
    """Number ``key`` printed exactly as its LaTeX macro prints it (plain text). A missing key raises."""
    e = n[key]
    return RX.format_value(e['value'], e['format'])


def p_is(n: dict[str, Any], key: str) -> str:
    """A p-value number as a relation: '= 0.042' or '< 0.001'."""
    s = fmt(n, key)
    return f'< {s[1:]}' if s.startswith('<') else f'= {s}'


def _nan(v: Any) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v)) or (not isinstance(v, str) and math.isnan(float(v)))
    except (TypeError, ValueError):
        return False


class Numbers(dict):
    """The numbers being built: ``add`` skips missing values (NaN / None)."""

    def add(self, key: str, value: Any, f: str, src: str, desc: str) -> None:
        if _nan(value):
            return
        if f == 'd':
            value = int(round(float(value)))
        self[key] = RX.numbers_entry(value, f, src, desc)

    def p(self, key: str, value: Any, src: str, desc: str) -> None:
        """A p-value: '<0.001' below 0.001 (text), else 3 decimals."""
        if _nan(value):
            return
        v = float(value)
        if v < 1e-3:
            self[key] = RX.numbers_entry('<0.001', '', src, desc)
        else:
            self[key] = RX.numbers_entry(v, '.3f', src, desc)

    def pts(self, key: str, value: Any, src: str, desc: str) -> None:
        """An accuracy difference in percentage points (value x 100, signed)."""
        if not _nan(value):
            self.add(key, 100.0 * float(value), '+.1f', src, desc + ' (percentage points)')


# ───────────────────────────── numbers ─────────────────────────────


def _exp_block(n: Numbers, r: pd.Series, full: bool) -> None:
    s, lab = short(r['name']), str(r['label'])
    n.add(f'{s}_accuracy', r['accuracy'], '.1%', SRC_EXP, f'{lab}: LOSO accuracy')
    n.add(f'{s}_accuracy_lo', r['accuracy_lo'], '.1%', SRC_EXP, f'{lab}: accuracy 95% CI low')
    n.add(f'{s}_accuracy_hi', r['accuracy_hi'], '.1%', SRC_EXP, f'{lab}: accuracy 95% CI high')
    n.add(f'{s}_auc', r['roc_auc'], '.3f', SRC_EXP, f'{lab}: ROC AUC')
    n.add(f'{s}_log_loss', r['log_loss'], '.3f', SRC_EXP, f'{lab}: log loss')
    n.add(f'{s}_ece', r['ece'], '.3f', SRC_EXP, f'{lab}: ECE (10 equal-mass bins)')
    n.add(f'{s}_configs', r['n_configs_tried'], 'd', SRC_EXP, f'{lab}: configurations compared on the outer folds')
    n.add(f'{s}_n_features', r['n_features'], 'd', SRC_EXP, f'{lab}: features')
    if full:
        for k, f in (('roc_auc', '.3f'), ('log_loss', '.3f'), ('brier', '.3f')):
            kk = 'auc' if k == 'roc_auc' else k
            n.add(f'{s}_{kk}_lo', r[f'{k}_lo'], f, SRC_EXP, f'{lab}: {k} 95% CI low')
            n.add(f'{s}_{kk}_hi', r[f'{k}_hi'], f, SRC_EXP, f'{lab}: {k} 95% CI high')
        n.add(f'{s}_brier', r['brier'], '.3f', SRC_EXP, f'{lab}: Brier score')
        n.add(f'{s}_nat_acc', r.get('natural_use_accuracy'), '.0%', SRC_EXP,
              f'{lab}: right-hand recall on the free-choice trials (an abstention scores 1/2)')
        n.add(f'{s}_nat_lo', r.get('natural_use_lo'), '.0%', SRC_EXP, f'{lab}: free-choice recall 95% CI low')
        n.add(f'{s}_nat_hi', r.get('natural_use_hi'), '.0%', SRC_EXP, f'{lab}: free-choice recall 95% CI high')
        n.add(f'{s}_nat_correct', r.get('natural_use_correct_answered'), 'd', SRC_EXP,
              f'{lab}: free-choice trials answered correctly')
        n.add(f'{s}_nat_n', r.get('natural_use_n'), 'd', SRC_EXP, f'{lab}: free-choice trials scored')
        n.add(f'{s}_nat_coverage', r.get('natural_use_coverage'), '.0%', SRC_EXP,
              f'{lab}: share of free-choice trials answered (P(right) not 0.5)')
        n.add(f'{s}_right_recall', r.get('primary_right_recall'), '.1%', SRC_EXP,
              f'{lab}: right-hand recall on the primary trials (comparator of the free-choice recall)')
        for k in C.K_ATTEMPTS:
            n.add(f'{s}_acc_k{k}', r.get(f'acc_k{k}'), '.1%', SRC_EXP, f'{lab}: accuracy, {k} attempts averaged')
            n.add(f'{s}_acc_k{k}_lo', r.get(f'acc_k{k}_lo'), '.1%', SRC_EXP, f'{lab}: k={k} accuracy CI low')
            n.add(f'{s}_acc_k{k}_hi', r.get(f'acc_k{k}_hi'), '.1%', SRC_EXP, f'{lab}: k={k} accuracy CI high')
        n.add(f'{s}_k_persons', r.get('k_n_persons'), 'd', SRC_EXP, f'{lab}: persons of the k-attempt curve')
        n.add(f'{s}_person_median', r.get('person_acc_median'), '.1%', SRC_EXP, f'{lab}: median per-person accuracy')
        n.add(f'{s}_below80', r.get('n_persons_below_80'), 'd', SRC_EXP, f'{lab}: persons below 80% accuracy')
        n.add(f'{s}_gated_acc', r.get('gated_accuracy'), '.1%', SRC_EXP,
              f'{lab}: accuracy on the trials passing the predictor gates')
        n.add(f'{s}_gated_n', r.get('gated_n'), 'd', SRC_EXP, f'{lab}: trials passing the predictor gates')
        if not _nan(r.get('vs_ref_diff_acc')):
            n.pts(f'{s}_ref_d_acc', r['vs_ref_diff_acc'], SRC_EXP, f'{lab}: accuracy minus the reference floor')
            n.pts(f'{s}_ref_d_acc_lo', r['vs_ref_diff_acc_lo'], SRC_EXP, f'{lab}: minus reference, CI low')
            n.pts(f'{s}_ref_d_acc_hi', r['vs_ref_diff_acc_hi'], SRC_EXP, f'{lab}: minus reference, CI high')
            n.p(f'{s}_ref_p', r.get('vs_ref_p_acc'), SRC_EXP, f'{lab}: sign-flip p vs reference (unadjusted)')
            n.p(f'{s}_ref_p_holm', r.get('vs_ref_p_acc_holm'), SRC_EXP,
                f'{lab}: sign-flip p vs reference, Holm-adjusted over the comparisons with the reference')
    if not _nan(r.get('vs_base_diff_acc')):
        n.pts(f'{s}_d_acc', r['vs_base_diff_acc'], SRC_EXP, f'{lab}: accuracy minus its baseline')
        n.pts(f'{s}_d_acc_lo', r['vs_base_diff_acc_lo'], SRC_EXP, f'{lab}: minus baseline, CI low')
        n.pts(f'{s}_d_acc_hi', r['vs_base_diff_acc_hi'], SRC_EXP, f'{lab}: minus baseline, CI high')
        n.add(f'{s}_d_ll', r['vs_base_diff_ll'], '+.3f', SRC_EXP, f'{lab}: log loss minus its baseline')
        n.add(f'{s}_d_ll_lo', r['vs_base_diff_ll_lo'], '+.3f', SRC_EXP, f'{lab}: log-loss difference CI low')
        n.add(f'{s}_d_ll_hi', r['vs_base_diff_ll_hi'], '+.3f', SRC_EXP, f'{lab}: log-loss difference CI high')
        n.p(f'{s}_p_acc', r.get('vs_base_p_acc'), SRC_EXP, f'{lab}: sign-flip p of the accuracy difference (unadjusted)')
        n.p(f'{s}_p_acc_holm', r.get('vs_base_p_acc_holm'), SRC_EXP,
            f'{lab}: sign-flip p of the accuracy difference, Holm-adjusted within its comparison family')
        n.p(f'{s}_p_ll', r.get('vs_base_p_ll'), SRC_EXP, f'{lab}: sign-flip p of the log-loss difference (unadjusted)')
        n.p(f'{s}_p_ll_holm', r.get('vs_base_p_ll_holm'), SRC_EXP,
            f'{lab}: sign-flip p of the log-loss difference, Holm-adjusted within its comparison family')
        n.add(f'{s}_max_dp', r.get('vs_base_max_abs_dp'), '.1e', SRC_EXP,
              f'{lab}: largest |P(right) difference| against its baseline')
        n.add(f'{s}_class_changes', r.get('vs_base_n_class_changes'), 'd', SRC_EXP,
              f'{lab}: trials whose predicted hand differs from its baseline')
        verdict = ('identical' if r.get('vs_base_identical') else
                   ('better' if r.get('vs_base_sig_ll') and r['vs_base_diff_ll'] < 0 else
                    'worse' if r.get('vs_base_sig_ll') else 'not different'))
        n.add(f'{s}_ll_verdict', verdict, '', SRC_EXP,
              f'{lab}: log loss against its baseline (Holm-adjusted p < alpha; identical = solver-level difference)')
        verdict = ('identical' if r.get('vs_base_identical') else
                   ('better' if r.get('vs_base_sig_acc') and r['vs_base_diff_acc'] > 0 else
                    'worse' if r.get('vs_base_sig_acc') else 'not different'))
        n.add(f'{s}_acc_verdict', verdict, '', SRC_EXP, f'{lab}: accuracy against its baseline (Holm-adjusted)')


def _cal_numbers(n: Numbers) -> None:
    cal = L.read_json('m2_calibration')
    for src in cal.get('order', []):
        m = cal['models'][src]
        s = f'cal_{short(src)}'
        dec = m['decision']
        n.add(f'{s}_detected', 'yes' if dec.get('miscalibration_detected') else 'no', '', SRC_CAL,
              f'{src}: miscalibration detected (ECE above its null band, slope CI excluding 1 or intercept CI excluding 0)')
        n.add(f'{s}_adopted', CAL_TEXT.get(dec.get('adopted'), dec.get('adopted')), '', SRC_CAL,
              f'{src}: calibrator adopted (paired log-loss gain, Holm-adjusted p < alpha)')
        if dec.get('best_calibrator'):
            n.add(f'{s}_best', CAL_TEXT.get(dec['best_calibrator'], dec['best_calibrator']), '', SRC_CAL,
                  f'{src}: calibrator with the lowest log loss')
        n.add(f'{s}_n_calibrators', dec.get('n_calibrators'), 'd', SRC_CAL, f'{src}: calibrators compared')
        for c, e in m['variants'].items():
            k = f'{s}_{c}'
            what = f'{src} + {CAL_TEXT.get(c, c)}'
            plats = e['platform'].items() if c in ('none', 'platt_platform') else []
            for sub, st in [('', e['overall'])] + [(f'_{slug(pl)}', v) for pl, v in plats]:
                w = what + (f' ({sub[1:]})' if sub else '')
                n.add(f'{k}{sub}_ece', st.get('ece'), '.3f', SRC_CAL, f'{w}: ECE')
                n.add(f'{k}{sub}_ece_null_mean', st.get('ece_null_mean'), '.3f', SRC_CAL,
                      f'{w}: mean ECE of perfectly calibrated outcomes (y ~ Bernoulli(p))')
                n.add(f'{k}{sub}_ece_null_hi', st.get('ece_null_hi'), '.3f', SRC_CAL,
                      f'{w}: 97.5th percentile of the ECE under perfect calibration')
                n.p(f'{k}{sub}_ece_null_p', st.get('ece_null_p'), SRC_CAL, f'{w}: share of null ECEs >= the observed')
                n.add(f'{k}{sub}_slope', st.get('cal_slope'), '.2f', SRC_CAL, f'{w}: calibration slope')
                n.add(f'{k}{sub}_slope_lo', st.get('cal_slope_lo'), '.2f', SRC_CAL, f'{w}: slope 95% CI low')
                n.add(f'{k}{sub}_slope_hi', st.get('cal_slope_hi'), '.2f', SRC_CAL, f'{w}: slope 95% CI high')
                n.add(f'{k}{sub}_citl', st.get('cal_in_the_large'), '+.2f', SRC_CAL, f'{w}: calibration-in-the-large')
                n.add(f'{k}{sub}_citl_lo', st.get('cal_in_the_large_lo'), '+.2f', SRC_CAL, f'{w}: CITL 95% CI low')
                n.add(f'{k}{sub}_citl_hi', st.get('cal_in_the_large_hi'), '+.2f', SRC_CAL, f'{w}: CITL 95% CI high')
                n.add(f'{k}{sub}_log_loss', st.get('log_loss'), '.3f', SRC_CAL, f'{w}: log loss')
                n.add(f'{k}{sub}_brier', st.get('brier'), '.3f', SRC_CAL, f'{w}: Brier score')
                n.add(f'{k}{sub}_n_persons', st.get('n_persons'), 'd', SRC_CAL, f'{w}: persons')
                n.add(f'{k}{sub}_n', st.get('n'), 'd', SRC_CAL, f'{w}: trials')
            for vs, tag in (('vs_uncalibrated', ''), ('vs_platt', '_vs_platt')):
                v = e.get(vs)
                if not v:
                    continue
                fam = (f'the calibrators of {src}' if vs == 'vs_uncalibrated' else
                       'the analysed models (one platform-specific against global Platt comparison each)')
                for met, f in (('ll', '+.3f'), ('brier', '+.4f')):
                    n.add(f'{k}{tag}_d_{met}', v[f'diff_{met}'], f, SRC_CAL, f'{what}: {met} minus {vs[3:]}')
                    n.add(f'{k}{tag}_d_{met}_lo', v[f'diff_{met}_lo'], f, SRC_CAL, f'{what}: {met} difference CI low')
                    n.add(f'{k}{tag}_d_{met}_hi', v[f'diff_{met}_hi'], f, SRC_CAL, f'{what}: {met} difference CI high')
                    n.p(f'{k}{tag}_p_{met}', v.get(f'p_{met}'), SRC_CAL, f'{what}: sign-flip p ({met}, unadjusted)')
                    n.p(f'{k}{tag}_p_{met}_holm', v.get(f'p_{met}_holm'), SRC_CAL,
                        f'{what}: sign-flip p ({met}), Holm-adjusted over {fam}')
                if v.get('verdict_ll'):
                    n.add(f'{k}{tag}_verdict', v['verdict_ll'], '', SRC_CAL,
                          f'{what}: log loss against {vs[3:]} (Holm-adjusted p < alpha over {fam})')
                if v.get('family_size'):
                    n.add(f'{k}{tag}_family_size', v['family_size'], 'd', SRC_CAL, f'comparisons in the family: {fam}')


def _k_numbers(n: Numbers) -> None:
    ka = L.read_json('m2_k_attempts')
    for name, d in ka.items():
        if not isinstance(d, dict) or 'rows' not in d:
            continue
        s = f'ka_{short(name)}'
        for r in d.get('k_curve', []):
            k = int(r['k'])
            n.add(f'{s}_k{k}_acc', r['accuracy'], '.1%', SRC_K, f'{name}: per-person balanced accuracy, {k} attempts')
            n.add(f'{s}_k{k}_acc_lo', r['lo'], '.1%', SRC_K, f'{name}: k={k} accuracy CI low')
            n.add(f'{s}_k{k}_acc_hi', r['hi'], '.1%', SRC_K, f'{name}: k={k} accuracy CI high')
            n.add(f'{s}_n_persons', r['n_persons'], 'd', SRC_K, f'{name}: persons of the k-attempt analysis')
        for r in d['rows']:
            k, a = int(r['k']), AGG_SHORT[r['aggregator']]
            w = f'{name}: {AGG_TEXT[r["aggregator"]]} of {k} attempts'
            n.add(f'{s}_k{k}_{a}_ll', r['log_loss'], '.3f', SRC_K, f'{w}: log loss')
            n.add(f'{s}_k{k}_{a}_ll_lo', r['log_loss_lo'], '.3f', SRC_K, f'{w}: log loss CI low')
            n.add(f'{s}_k{k}_{a}_ll_hi', r['log_loss_hi'], '.3f', SRC_K, f'{w}: log loss CI high')
            n.add(f'{s}_k{k}_{a}_ece', r['ece'], '.3f', SRC_K, f'{w}: ECE')
            n.add(f'{s}_k{k}_{a}_conf', r['mean_confidence'], '.3f', SRC_K, f'{w}: mean confidence max(P, 1-P)')
            n.add(f'{s}_k{k}_{a}_acc', r['accuracy'], '.3f', SRC_K, f'{w}: share of aggregates correct')
            n.add(f'{s}_k{k}_{a}_gap', r['conf_gap'], '+.3f', SRC_K,
                  f'{w}: mean confidence minus share correct (above 0 = over-confident)')
            n.add(f'{s}_k{k}_{a}_gap_lo', r['conf_gap_lo'], '+.3f', SRC_K, f'{w}: confidence gap CI low')
            n.add(f'{s}_k{k}_{a}_gap_hi', r['conf_gap_hi'], '+.3f', SRC_K, f'{w}: confidence gap CI high')
            n.add(f'{s}_k{k}_{a}_bal_acc', r['balanced_accuracy'], '.3f', SRC_K, f'{w}: per-person balanced accuracy')
        for r in d.get('pairs', []):
            k = int(r['k'])
            key = f'{s}_k{k}_{AGG_SHORT[r["a"]]}_vs_{AGG_SHORT[r["b"]]}'
            w = f'{name}, {k} attempts: log loss of {AGG_TEXT[r["a"]]} minus {AGG_TEXT[r["b"]]}'
            n.add(f'{key}_d_ll', r['diff'], '+.3f', SRC_K, w)
            n.add(f'{key}_d_ll_lo', r['diff_lo'], '+.3f', SRC_K, w + ', CI low')
            n.add(f'{key}_d_ll_hi', r['diff_hi'], '+.3f', SRC_K, w + ', CI high')
            n.p(f'{key}_p', r['p'], SRC_K, w + ', sign-flip p (unadjusted)')
            if 'p_holm' in r:
                n.p(f'{key}_p_holm', r['p_holm'], SRC_K, w + ', sign-flip p Holm-adjusted over the pairs at this k')
        for k, v in (d.get('tau_nested_median') or {}).items():
            n.add(f'{s}_tau_k{k}', v, '.2f', SRC_K, f'{name}: median over outer folds of the nested tau_{k}')
        for k, v in (d.get('effective_independent_attempts') or {}).items():
            n.add(f'{s}_eff_k{k}', v, '.1f', SRC_K, f'{name}: k / tau_k, independent attempts carried by {k}')
        if d.get('recommended'):
            n.add(f'{s}_recommended', AGG_TEXT[d['recommended']], '', SRC_K,
                  f'{name}: aggregate with the lowest log loss at the largest k (the displayed probability)')
        n.add(f'{s}_k_max', d.get('k_max'), 'd', SRC_K, f'{name}: largest k of the analysis')
        n.add(f'{s}_left_out', ', '.join(d.get('persons_left_out') or []) or 'none', '', SRC_K,
              f'{name}: persons with too few trials of a hand for the largest k')


def _strata_numbers(n: Numbers, s_json: dict[str, Any]) -> None:
    p = L.tab_dir / 'm2_strata.csv'
    if not p.exists():
        return
    bd = pd.read_csv(p)
    head = (s_json.get('headline') or [None])[0]
    for name, g in bd[bd['model'] == head].groupby('model', sort=False):
        s = f'bd_{short(name)}'
        for _, r in g.iterrows():
            k = f'{s}_{slug(r["factor"])}_{slug(r["level"])}'
            w = f'{name}: {r["factor_label"]} = {r["level"]}'
            n.add(f'{k}_acc', r['accuracy'], '.1%', SRC_BD, f'{w}: accuracy')
            n.add(f'{k}_acc_lo', r['acc_lo'], '.1%', SRC_BD, f'{w}: accuracy CI low (>= {S.MIN_PERSONS_CI} persons)')
            n.add(f'{k}_acc_hi', r['acc_hi'], '.1%', SRC_BD, f'{w}: accuracy CI high')
            n.add(f'{k}_n', r['n_trials'], 'd', SRC_BD, f'{w}: trials')
            n.add(f'{k}_persons', r['n_persons'], 'd', SRC_BD, f'{w}: persons')
            n.add(f'{k}_log_loss', r['log_loss'], '.3f', SRC_BD, f'{w}: log loss')
    p = L.tab_dir / 'm2_strata_persons.csv'
    if p.exists():
        sp = pd.read_csv(p)
        sp = sp[sp['model'] == head] if len(sp) else sp
        for _, r in sp.iterrows():
            n.add(f'bdp_{short(r["model"])}_{slug(r["factor"])}_{slug(r["level"])}_{slug(r["person_id"])}',
                  r['accuracy'], '.1%', 'tables/m2/m2_strata_persons.csv',
                  f'{r["model"]}: {r["factor"]} = {r["level"]}, person {r["person_id"]}: accuracy')
    p = L.tab_dir / 'm2_per_person.csv'
    if p.exists():
        pp = pd.read_csv(p)
        for name, g in pp.groupby('model', sort=False):
            s = f'pp_{short(name)}'
            g = g.sort_values(['accuracy', 'person_id'])
            n.add(f'{s}_median', g['accuracy'].median(), '.1%', SRC_PP, f'{name}: median per-person accuracy')
            n.add(f'{s}_min', g['accuracy'].iloc[0], '.1%', SRC_PP, f'{name}: lowest per-person accuracy')
            n.add(f'{s}_min_person', str(g['person_id'].iloc[0]), '', SRC_PP, f'{name}: the person with the lowest')
            n.add(f'{s}_max', g['accuracy'].iloc[-1], '.1%', SRC_PP, f'{name}: highest per-person accuracy')
            n.add(f'{s}_below80', int((g['accuracy'] < 0.8).sum()), 'd', SRC_PP, f'{name}: persons below 80%')
            n.add(f'{s}_n', len(g), 'd', SRC_PP, f'{name}: persons')
    mh = s_json.get('mirror_handedness') or {}
    for r in mh.get('groups', []):
        k = f'mh_{short(r["name"])}_{slug(r["handedness"])}'
        n.add(f'{k}_acc', r['accuracy'], '.1%', SRC_SUM, f'{r["name"]}: accuracy of {r["handedness"]}-handed persons')
        n.add(f'{k}_acc_lo', r.get('acc_lo'), '.1%', SRC_SUM, f'{r["name"]}: CI low ({r["handedness"]})')
        n.add(f'{k}_acc_hi', r.get('acc_hi'), '.1%', SRC_SUM, f'{r["name"]}: CI high ({r["handedness"]})')
        n.add(f'{k}_persons', r['n_persons'], 'd', SRC_SUM, f'{r["handedness"]}-handed persons')
    for r in mh.get('persons', []):
        n.add(f'mh_{short(r["name"])}_{slug(r["person_id"])}', r['accuracy'], '.1%', SRC_SUM,
              f'{r["name"]}: accuracy of {r["handedness"]}-handed person {r["person_id"]}')


def _lc_numbers(n: Numbers) -> None:
    p = L.tab_dir / 'm2_lc_runs.csv'
    if not p.exists():
        return
    lc = pd.read_csv(p)
    sub = lc[lc['n_train_persons_requested'].astype(str) != 'all']
    for k, g in sub.groupby('n_train_persons_requested', sort=True):
        s = f'lc_n{int(float(k)):02d}'
        for col, f, w in (('accuracy', '.1%', 'accuracy'), ('log_loss', '.3f', 'log loss')):
            c = 'acc' if col == 'accuracy' else 'll'
            n.add(f'{s}_{c}_mean', g[col].mean(), f, SRC_LC, f'{int(float(k))} training persons: mean {w} over repetitions')
            n.add(f'{s}_{c}_min', g[col].min(), f, SRC_LC, f'{int(float(k))} training persons: lowest {w}')
            n.add(f'{s}_{c}_max', g[col].max(), f, SRC_LC, f'{int(float(k))} training persons: highest {w}')
        n.add(f'{s}_reps', len(g), 'd', SRC_LC, f'{int(float(k))} training persons: repetitions')
        n.add(f'{s}_persons', int(float(k)), 'd', SRC_LC, 'training persons kept per outer fold')
    full = lc[lc['n_train_persons_requested'].astype(str) == 'all']
    if len(full):
        r = full.iloc[0]
        n.add('lc_all_persons', r['n_train_persons'], 'd', SRC_LC, 'training persons per outer fold of the full model')
        n.add('lc_all_acc', r['accuracy'], '.1%', SRC_LC, 'learning curve, every training person: accuracy')
        n.add('lc_all_ll', r['log_loss'], '.3f', SRC_LC, 'learning curve, every training person: log loss')
    if len(sub):
        n.add('lc_sizes', len(sub['n_train_persons_requested'].unique()), 'd', SRC_LC, 'subsample sizes')
    if len(sub) and len(full):
        last = sub[sub['n_train_persons_requested'].astype(float) == sub['n_train_persons_requested'].astype(float).max()]
        r = full.iloc[0]
        n.pts('lc_last_step_acc', r['accuracy'] - last['accuracy'].mean(), SRC_LC,
              'accuracy with every training person minus the mean of the largest subsample')
        n.add('lc_last_step_ll', r['log_loss'] - last['log_loss'].mean(), '+.3f', SRC_LC,
              'log loss with every training person minus the mean of the largest subsample')
        for col, key in (('accuracy', 'acc'), ('log_loss', 'll')):
            within = bool(last[col].min() <= r[col] <= last[col].max())
            n.add(f'lc_last_step_{key}_within', 'yes' if within else 'no', '', SRC_LC,
                  f'the full-model {col} lies within the range of the largest subsample repetitions')


def number_values() -> Numbers:
    """Every number the lane's text cites, from the saved derived files (no model is fitted)."""
    from stage2.standardize import CHANNELS
    t = _exp()
    s_json = L.read_json('m2_summary')
    n = Numbers()
    idx = t.set_index('name')
    for _, r in t.iterrows():
        _exp_block(n, r, full=r['group'] in ('default', 'model', 'select'))
        if r['group'] == 'bins':
            n.add(f"{short(r['name'])}_k", r['n_bins'], 'd', SRC_EXP, f"{r['label']}: time bins")
        if r['group'] == 'learning_curve':
            n.add(f"{short(r['name'])}_persons", r['n_train_persons'], 'd', SRC_EXP, f"{r['label']}: training persons")
            n.add(f"{short(r['name'])}_rep", r['rep'], 'd', SRC_EXP, f"{r['label']}: repetition")
    # global facts (counts from the data, settings from the code)
    if 'm2_lr_full' in idx.index:
        r = idx.loc['m2_lr_full']
        n.add('n_features', r['n_features'], 'd', SRC_EXP, 'features of the default set')
        n.add('n_persons', r['n_persons'], 'd', SRC_EXP, 'persons (outer folds)')
        n.add('n_trials', r['n_trials'], 'd', SRC_EXP, 'primary trials scored')
        n.add('nat_n', r.get('natural_use_n'), 'd', SRC_EXP, 'free-choice (natural-use) trials')
        n.add('nat_n_right', r.get('natural_use_n_right'), 'd', SRC_EXP, 'free-choice trials with the right hand')
        n.add('nat_n_left', r.get('natural_use_n_left'), 'd', SRC_EXP, 'free-choice trials with the left hand')
        n.add('nat_n_persons', r.get('natural_use_n_persons'), 'd', SRC_EXP, 'persons of the free-choice set')
    n.add('n_experiments', len(t), 'd', SRC_EXP, 'lane m2 experiments computed')
    n.add('n_experiments_planned', len(P.experiments()), 'd', SRC_CODE, 'lane m2 experiments declared')
    models = t[t['group'].isin(['default', 'model'])]
    n.add('n_models_compared', len(models), 'd', SRC_EXP, 'model pipelines compared on the outer folds')
    n.add('inner_folds', P.INNER, 'd', SRC_CODE, 'inner person folds of the nested tuning')
    for m, k in sorted(P.TRIALS.items()):
        n.add(f'trials_{m}', k, 'd', SRC_CODE, f'Optuna trials per outer fold, {m}')
    n.add('lc_reps_planned', P.LC_REPS, 'd', SRC_CODE, 'learning-curve repetitions per size')
    n.add('winsor_sd', P.WINSOR_SD, '.0f', SRC_CODE, 'winsorising threshold (SD)')
    n.add('max_iter_unscaled', P.MAX_ITER_UNSCALED, 'd', SRC_CODE, 'lbfgs iteration cap of the unscaled model')
    from stage2.lanes.m2_models import FOREST_DEFAULT
    n.add('forest_n_estimators', FOREST_DEFAULT['n_estimators'], 'd', 'stage2/lanes/m2_models.py',
          'trees of the fixed random-forest / extra-trees configuration')
    n.add('forest_min_leaf', FOREST_DEFAULT['min_samples_leaf'], 'd', 'stage2/lanes/m2_models.py',
          'smallest leaf of the fixed random-forest / extra-trees configuration (rows)')
    from stage2.lanes.m2_features import DEFAULT_SPEC, FAMILIES
    n.add('n_channels', len(CHANNELS), 'd', 'stage2/standardize.py', 'channels of the cache')
    n.add('n_families', len(FAMILIES), 'd', 'stage2/lanes/m2_features.py', 'feature families')
    n.add('n_bins', DEFAULT_SPEC.n_bins, 'd', 'stage2/lanes/m2_features.py', 'time bins of the default set')
    n.add('alpha', S.ALPHA, '.2f', 'stage2/lanes/m2_stats.py', 'significance level of the Holm-adjusted tests')
    n.add('min_persons_ci', S.MIN_PERSONS_CI, 'd', 'stage2/lanes/m2_stats.py', 'fewest persons for a bootstrap CI')
    n.add('equiv_dp', S.EQUIV_MAX_DP, '.0e', 'stage2/lanes/m2_stats.py', 'identical pipelines: largest |dP| below this')
    n.add('ece_bins', C.ECE_BINS, 'd', SRC_CFG, 'equal-mass bins of the ECE')
    n.add('n_boot', C.N_BOOT, 'd', SRC_CFG, 'person-bootstrap resamples of the experiments')
    from stage2.lanes import m2_derive as D
    n.add('n_boot_derived', D.N_BOOT_DERIVED, 'd', 'stage2/lanes/m2_derive.py', 'resamples of the derived CIs')
    n.add('n_boot_attempts', D.N_BOOT_ATTEMPTS, 'd', 'stage2/lanes/m2_derive.py', 'resamples of the aggregate CIs')
    n.add('lowpass_hz', C.LOWPASS_HZ, '.0f', SRC_CFG, 'cache low-pass cut-off (Hz)')
    n.add('fs_hz', C.FS_HZ, '.0f', SRC_CFG, 'cache sampling rate (Hz)')
    n.add('max_gap_ms', C.MAX_GAP_MS, '.0f', SRC_CFG, 'longest interpolated gap (ms)')
    fs = s_json.get('feature_scales') or {}
    n.add('feature_sd_orders', fs.get('log10_sd_ratio'), '.1f', SRC_SUM,
          'orders of magnitude between the largest and smallest feature SD (default set)')
    un = s_json.get('unscaled') or {}
    n.add('unscaled_inner_fits', un.get('n_inner_fits'), 'd', SRC_SUM, 'inner fits of the unscaled model')
    n.add('unscaled_inner_nonconverged', un.get('n_inner_nonconverged'), 'd', SRC_SUM,
          'of which stopped at the iteration cap')
    n.add('unscaled_final_nonconverged', un.get('n_final_nonconverged'), 'd', SRC_SUM,
          'outer-fold final fits stopped at the cap')
    fw = s_json.get('fixed_window') or {}
    n.add('fixed_n_frozen', fw.get('n_frozen_tail'), 'd', SRC_SUM, 'trials whose fixed window has a frozen tail')
    n.add('fixed_n_trials', fw.get('n_trials'), 'd', SRC_SUM, 'trials of the fixed-window record')
    n.add('fixed_frozen_share', fw.get('median_frozen_share_of_frozen'), '.0%', SRC_SUM,
          'median share of the post-onset window that is frozen (trials with a frozen tail)')
    n.add('fixed_rho', fw.get('spearman_freeze_vs_duration'), '.2f', SRC_SUM,
          'Spearman correlation of the freeze point with the oracle approach duration')
    n.add('fixed_window_s', fw.get('window_s'), '.0f', SRC_SUM, 'fixed window length (s)')
    for fam, d in (s_json.get('families') or {}).items():
        n.add(f'fam_{slug(fam)}_size', d.get('size'), 'd', SRC_SUM, f'comparisons in the family: {d.get("text")}')
    best = s_json.get('best_by_log_loss')
    if best:
        n.add('best_model', str(idx.loc[best, 'label']), '', SRC_EXP, 'pipeline with the lowest outer log loss')
        n.add('best_model_name', best, '', SRC_EXP, 'its experiment name')
    ref = s_json.get('reference_metrics') or {}
    n.add('ref_accuracy', ref.get('accuracy'), '.1%', SRC_SUM, 'reference floor: accuracy')
    n.add('ref_accuracy_lo', ref.get('accuracy_lo'), '.1%', SRC_SUM, 'reference floor: CI low')
    n.add('ref_accuracy_hi', ref.get('accuracy_hi'), '.1%', SRC_SUM, 'reference floor: CI high')
    n.add('ref_log_loss', ref.get('log_loss'), '.3f', SRC_SUM, 'reference floor: log loss')
    # feature registry
    rp = L.tab_dir / 'm2_feature_registry.csv'
    if rp.exists():
        reg = pd.read_csv(rp)
        for fam, g in reg.groupby('family'):
            n.add(f'reg_{fam}_n', len(g), 'd', SRC_REG, f'features of family {fam}')
            n.add(f'reg_{fam}_odd', int((g['parity'] != 'even').sum()), 'd', SRC_REG,
                  f'odd or swapped (mirror-sensitive) features of family {fam}')
        n.add('reg_odd', int((reg['parity'] != 'even').sum()), 'd', SRC_REG, 'odd or swapped features (default set)')
        n.add('reg_even', int((reg['parity'] == 'even').sum()), 'd', SRC_REG, 'even features (default set)')
    _cal_numbers(n)
    _k_numbers(n)
    _strata_numbers(n, s_json)
    _lc_numbers(n)
    imp = L.read_json('m2_importance')
    for m in ('m2_lr_full', 'm2_lgbm'):
        if m not in imp:
            continue
        s = f'imp_{short(m)}'
        for fam, v in imp[m]['family_share'].items():
            n.add(f'{s}_fam_{fam}', v, '.0%', SRC_IMP, f'{m}: share of family {fam}')
        ch = pd.Series(imp[m]['channel_share']).sort_values(ascending=False)
        n.add(f'{s}_top_channel', str(ch.index[0]), '', SRC_IMP, f'{m}: channel with the largest share')
        for c, v in ch.items():
            n.add(f'{s}_ch_{slug(c)}', v, '.0%', SRC_IMP, f'{m}: share of channel {c} (pairs split equally)')
        for fr, v in (imp[m].get('frame_share') or {}).items():
            n.add(f'{s}_frame_{fr}', v, '.0%', SRC_IMP, f'{m}: share of the {fr} channels')
        for par, v in (imp[m].get('parity_share') or {}).items():
            n.add(f'{s}_parity_{par}', v, '.0%', SRC_IMP, f'{m}: share of the {par} features')
    p = L.tab_dir / 'm2_select_choices.csv'
    if p.exists():
        sc = pd.read_csv(p)
        for c in P.SELECT_CANDIDATES:
            n.add(f'sel_choice_{c}', int((sc['chosen'] == c).sum()), 'd', SRC_SEL, f'outer folds in which {c} was chosen')
        n.add('sel_n_folds', len(sc), 'd', SRC_SEL, 'outer folds of the nested selection')
        n.add('sel_margin_median', sc['margin'].median(), '.3f', SRC_SEL,
              'median inner log-loss margin between the chosen and the runner-up candidate')
        for c in P.SELECT_CANDIDATES:
            if f'inner_ll_{c}' in sc:
                n.add(f'sel_inner_ll_{c}', sc[f'inner_ll_{c}'].mean(), '.3f', SRC_SEL,
                      f'{c}: inner cross-validated log loss of the tuned configuration, mean over outer folds')
    return n


def numbers() -> None:
    L.write_numbers(dict(number_values()))


# ───────────────────────────── tables ─────────────────────────────


def _ci(lo: Any, hi: Any, scale: float = 100.0, f: str = '.1f') -> str:
    if _nan(lo) or _nan(hi):
        return '--'
    return f'[{format(scale * float(lo), f)}, {format(scale * float(hi), f)}]'


def _pm(d: Any, lo: Any, hi: Any, scale: float = 100.0, f: str = '+.1f') -> str:
    if _nan(d):
        return '--'
    return f'{format(scale * float(d), f)} [{format(scale * float(lo), f)}, {format(scale * float(hi), f)}]'


def _pfmt(p: Any) -> str:
    return S.format_p(p) if not _nan(p) else '--'


def table_models(t: pd.DataFrame) -> None:
    m = t[t['group'].isin(['default', 'model', 'select'])].sort_values(['log_loss', 'name']).reset_index(drop=True)
    out = pd.DataFrame({
        'Model': m['label'], 'Accuracy': m['accuracy'],
        '95% CI': [_ci(a, b) for a, b in zip(m['accuracy_lo'], m['accuracy_hi'])],
        'AUC': m['roc_auc'], 'Log loss': m['log_loss'], 'ECE': m['ece'], 'Brier': m['brier'],
        'Free choice: right recall': [f'{int(c)}/{int(k)}' for c, k in zip(m['natural_use_correct_answered'].fillna(0),
                                                                            m['natural_use_n'].fillna(0))],
        'Primary: right recall': m['primary_right_recall'], 'k = 3': m.get('acc_k3'),
        'p vs default (Holm)': [_pfmt(p) for p in m.get('vs_base_p_ll_holm', pd.Series([None] * len(m)))],
        'Configs compared': m['n_configs_tried'].astype(int)})
    L.export_table(out, 'm2_models', caption=(
        'Lane m2: classical models on the default per-trial features (leave-one-person-out, every model tuned inside '
        'the outer fold on inner person folds, mirror-augmented; 95\\,\\% person-cluster bootstrap CI). Free choice: '
        'right-hand recall on the free-choice trials (correct / scored; all scored trials are right-handed, so this '
        'is not an accuracy over both hands); primary: right-hand recall on the instructed trials as its comparator. '
        'k = 3: per-person balanced accuracy with 3 attempts averaged. p vs default: sign-flip test of the per-person '
        'log-loss difference against the pre-specified default, Holm-adjusted over the pipelines compared. '
        '"Configs compared": the size of the outer-fold comparison (1 = pre-specified or nested).'),
        column_formats={'Accuracy': '.1%', 'AUC': '.3f', 'Log loss': '.3f', 'ECE': '.3f', 'Brier': '.3f',
                        'Primary: right recall': '.1%', 'k = 3': '.1%'},
        align='lrlrrrrrrrrr')


DT_GROUPS = ['window', 'frame', 'bins', 'scaling', 'mirror', 'family_drop', 'family_only', 'family_drop_nomirror',
             'family_only_nomirror']


def table_data_treatment(t: pd.DataFrame) -> None:
    d = t[t['group'].isin(DT_GROUPS) & t['vs_base_diff_acc'].notna()].copy()
    d['g'] = d['group'].map({g: i for i, g in enumerate(DT_GROUPS)})
    d = d.sort_values(['g', 'name']).reset_index(drop=True)

    def verdict(r: pd.Series) -> str:
        if r.get('vs_base_identical'):
            return 'identical'
        return ('yes' if r.get('vs_base_sig_ll') else 'no')
    out = pd.DataFrame({
        'Choice': d['group'].str.replace('_', ' '), 'Variant': d['label'], 'Features': d['n_features'].astype(int),
        'Accuracy': d['accuracy'], 'Diff. vs baseline (pts)': [_pm(a, b, c) for a, b, c in zip(
            d['vs_base_diff_acc'], d['vs_base_diff_acc_lo'], d['vs_base_diff_acc_hi'])],
        'p acc. (Holm)': [_pfmt(p) for p in d['vs_base_p_acc_holm']], 'Log loss': d['log_loss'],
        'Diff. log loss': [_pm(a, b, c, 1.0, '+.3f') for a, b, c in zip(d['vs_base_diff_ll'], d['vs_base_diff_ll_lo'],
                                                                          d['vs_base_diff_ll_hi'])],
        'p log loss (Holm)': [_pfmt(p) for p in d['vs_base_p_ll_holm']],
        'Log loss differs': [verdict(r) for _, r in d.iterrows()]})
    L.export_table(out, 'm2_data_treatment', caption=(
        'Lane m2: data-treatment and representation choices, each paired against its baseline on the same trials '
        '(the default L2 logistic regression; the ablations without mirroring against the default without mirroring): '
        'difference with paired person-bootstrap 95\\,\\% CI; p: exact sign-flip test over persons, Holm-adjusted '
        'within the comparison family (data treatment; families with augmentation; families without mirroring). '
        '"identical": predictions within solver precision of the baseline (no trial changes hand), not a test.'),
        column_formats={'Accuracy': '.1%', 'Log loss': '.3f'}, align='llrrlrrlrl')


def table_calibration() -> None:
    cal = L.read_json('m2_calibration')
    rows = []
    for src in cal.get('order', []):
        for c, e in cal['models'][src]['variants'].items():
            ov = e['overall']
            v = e.get('vs_uncalibrated') or {}
            rows.append({'Model': src.replace('m2_', ''), 'Calibrator': CAL_TEXT.get(c, c), 'ECE': ov['ece'],
                         'ECE null 97.5%': ov.get('ece_null_hi'),
                         'Slope [95% CI]': f"{ov['cal_slope']:.2f} {_ci(ov.get('cal_slope_lo'), ov.get('cal_slope_hi'), 1.0, '.2f')}",
                         'CITL [95% CI]': f"{ov['cal_in_the_large']:+.2f} "
                                          f"{_ci(ov.get('cal_in_the_large_lo'), ov.get('cal_in_the_large_hi'), 1.0, '+.2f')}",
                         'Brier': ov['brier'], 'Log loss': ov['log_loss'],
                         'Diff. log loss vs none': _pm(v['diff_ll'], v['diff_ll_lo'], v['diff_ll_hi'], 1.0, '+.3f') if v else '--',
                         'p (Holm)': _pfmt(v.get('p_ll_holm')) if v else '--',
                         'Diff. Brier vs none': _pm(v['diff_brier'], v['diff_brier_lo'], v['diff_brier_hi'], 1.0, '+.4f') if v else '--'})
    out = pd.DataFrame(rows)
    L.export_table(out, 'm2_calibration', caption=(
        'Lane m2: calibration of the analysed models on held-out persons, uncalibrated and with each post-hoc '
        'calibrator fitted inside every outer fold on cross-fitted predictions of the training persons (inner person '
        'folds). ECE (10 equal-mass bins) is shown with the 97.5th percentile of the ECE that perfectly calibrated '
        'outcomes would give (y drawn from Bernoulli(p)); slope and calibration-in-the-large (CITL) of the logistic '
        'recalibration with person-bootstrap 95\\,\\% CIs (perfect: 1 and 0); log-loss and Brier differences paired '
        'against the uncalibrated model, p Holm-adjusted over the calibrators of a model.'),
        column_formats={'ECE': '.3f', 'ECE null 97.5%': '.3f', 'Brier': '.3f', 'Log loss': '.3f'},
        align='llrrllrrlll')


def table_calibration_platform() -> None:
    cal = L.read_json('m2_calibration')
    rows = []
    for src in cal.get('order', []):
        e = cal['models'][src]['variants'].get('none')
        if not e:
            continue
        for plat, st in sorted(e['platform'].items()):
            rows.append({'Model': src.replace('m2_', ''), 'Platform': plat, 'Persons': st['n_persons'],
                         'Trials': st['n'], 'ECE': st['ece'], 'ECE null 97.5%': st.get('ece_null_hi'),
                         'Slope [95% CI]': f"{st['cal_slope']:.2f} {_ci(st.get('cal_slope_lo'), st.get('cal_slope_hi'), 1.0, '.2f')}",
                         'CITL [95% CI]': f"{st['cal_in_the_large']:+.2f} "
                                          f"{_ci(st.get('cal_in_the_large_lo'), st.get('cal_in_the_large_hi'), 1.0, '+.2f')}",
                         'Log loss': st['log_loss'], 'Brier': st['brier']})
        pp = cal['models'][src]['variants'].get('platt_platform')
        if pp and pp.get('vs_platt'):
            v = pp['vs_platt']
            rows.append({'Model': src.replace('m2_', ''), 'Platform': 'per-platform vs global Platt',
                         'Persons': None, 'Trials': v['paired_n'],
                         'ECE': None, 'ECE null 97.5%': None, 'Slope [95% CI]': '--', 'CITL [95% CI]': '--',
                         'Log loss': None, 'Brier': None,
                         'Diff. log loss': _pm(v['diff_ll'], v['diff_ll_lo'], v['diff_ll_hi'], 1.0, '+.3f')})
    out = pd.DataFrame(rows)
    if 'Diff. log loss' not in out:
        out['Diff. log loss'] = '--'
    out['Diff. log loss'] = out['Diff. log loss'].fillna('--')
    L.export_table(out, 'm2_calibration_platform', caption=(
        'Lane m2: calibration of the uncalibrated models per platform (statistics as in the calibration table, within '
        'the platform), and the paired log-loss difference of a platform-specific Platt map against one global Platt '
        'map (both fitted inside the outer folds; platform is known to production from the browser).'),
        column_formats={'ECE': '.3f', 'ECE null 97.5%': '.3f', 'Log loss': '.3f', 'Brier': '.3f', 'Persons': 'd',
                        'Trials': 'd'}, align='llrrrrllrrl')


def table_breakdown() -> None:
    p = L.tab_dir / 'm2_strata.csv'
    if not p.exists():
        return
    bd = pd.read_csv(p)
    s = L.read_json('m2_summary')
    m = s['headline'][0]
    d = bd[bd['model'] == m]
    out = pd.DataFrame({'Factor': d['factor_label'], 'Level': d['level'].astype(str).str.replace('_', ' '),
                        'Trials': d['n_trials'], 'Persons': d['n_persons'], 'Accuracy': d['accuracy'],
                        '95% CI': [_ci(a, b) for a, b in zip(d['acc_lo'], d['acc_hi'])], 'Log loss': d['log_loss']})
    L.export_table(out, 'm2_breakdown', caption=(
        f'Lane m2: accuracy of the headline model ({m.replace("_", chr(92) + "_")}) by stratum: conditions, handedness, '
        'flags and predictor gates stratify predictions only. 95\\,\\% person-bootstrap CI within the stratum, shown '
        f'only with at least {S.MIN_PERSONS_CI} persons.'),
        column_formats={'Accuracy': '.1%', 'Log loss': '.3f'})


def table_learning_curve() -> None:
    p = L.tab_dir / 'm2_lc_runs.csv'
    if not p.exists():
        return
    lc = pd.read_csv(p)
    if 'n_train_persons' not in lc.columns or lc['n_train_persons'].nunique() < 2:
        return
    g = lc.groupby('n_train_persons').agg(reps=('accuracy', 'size'), accuracy=('accuracy', 'mean'),
                                          acc_min=('accuracy', 'min'), acc_max=('accuracy', 'max'),
                                          log_loss=('log_loss', 'mean'), ll_min=('log_loss', 'min'),
                                          ll_max=('log_loss', 'max'), auc=('roc_auc', 'mean')).reset_index()
    out = pd.DataFrame({'Training persons': g['n_train_persons'].astype(int), 'Repetitions': g['reps'],
                        'Accuracy (mean)': g['accuracy'], 'Min': g['acc_min'], 'Max': g['acc_max'],
                        'Log loss (mean)': g['log_loss'], 'LL min': g['ll_min'], 'LL max': g['ll_max'],
                        'AUC': g['auc']})
    L.export_table(out, 'm2_learning_curve', caption=(
        'Lane m2: learning curve of the default L2 logistic regression (training persons subsampled inside every '
        'outer fold, seeded repetitions; min and max over the repetitions; the last row uses every training person, '
        'one run).'),
        column_formats={'Accuracy (mean)': '.1%', 'Min': '.1%', 'Max': '.1%', 'Log loss (mean)': '.3f',
                        'LL min': '.3f', 'LL max': '.3f', 'AUC': '.3f'})


def table_k_attempts() -> None:
    ka = L.read_json('m2_k_attempts')
    s = L.read_json('m2_summary')
    m = s['headline'][0]
    if m not in ka:
        return
    ag = pd.DataFrame(ka[m]['rows'])
    kc = pd.DataFrame(ka[m]['k_curve']).set_index('k')
    rows = []
    for k, g in ag.groupby('k'):
        row = {'k': int(k), 'Accuracy (mean P)': kc.loc[k, 'accuracy'], '95% CI': _ci(kc.loc[k, 'lo'], kc.loc[k, 'hi'])}
        for _, r in g.iterrows():
            a = AGG_TEXT[r['aggregator']]
            row[f'LL {a}'] = r['log_loss']
            row[f'Gap {a}'] = r['conf_gap']
        tau = (ka[m].get('tau_nested_median') or {}).get(str(int(k)))
        row['tau_k (nested, median)'] = tau
        rows.append(row)
    out = pd.DataFrame(rows)
    fm = {'Accuracy (mean P)': '.1%', 'tau_k (nested, median)': '.2f'}
    fm |= {c: '.3f' for c in out.columns if c.startswith('LL ')} | {c: '+.3f' for c in out.columns if c.startswith('Gap ')}
    L.export_table(out, 'm2_k_attempts', caption=(
        f'Lane m2: k attempts of one person and hand ({m.replace("_", chr(92) + "_")}): per-person balanced accuracy of '
        'the mean P(right) (95\\,\\% person-bootstrap CI), and the log loss and confidence gap (mean confidence minus '
        'share correct: above 0 over-confident, below 0 under-confident) of four aggregates; the tempered sum divides '
        'the sum of logits by tau_k fitted inside each outer fold on the training persons (median over folds shown).'),
        column_formats=fm)


def table_features() -> None:
    reg = pd.read_csv(L.tab_dir / 'm2_feature_registry.csv')
    from stage2.lanes.m2_features import DEFAULT_SPEC, FAMILIES
    from stage2.standardize import CHANNELS
    rows = []
    for fam in FAMILIES:
        g = reg[reg['family'] == fam]
        rows.append({'Family': fam, 'Content': FAMILY_TEXT.get(fam, ''), 'Features': len(g),
                     'Even': int((g['parity'] == 'even').sum()), 'Odd': int((g['parity'] == 'odd').sum()),
                     'Swap pairs': int((g['parity'] == 'swap').sum())})
    out = pd.DataFrame(rows)
    L.export_table(out, 'm2_feature_families', caption=(
        f'Lane m2: feature families of the default set (approach window, {len(CHANNELS)} channels, k = '
        f'{DEFAULT_SPEC.n_bins} bins). Parity under the left/right mirror: even (unchanged), odd (sign flip) or swapped '
        'pairs (min/max and arg-min/arg-max of chiral channels); the mirror map is exact (unit test). With mirror '
        'augmentation a linear model gives the even features zero weight.'), align='llrrrr')


def table_finalists() -> None:
    s = L.read_json('m2_summary')
    t = _exp().set_index('name')
    rows = []
    for f in s.get('finalists') or []:
        r = t.loc[f['name']]
        rows.append({'Finalist': f['name'], 'Pipeline': r['label'], 'Accuracy': r['accuracy'],
                     '95% CI': _ci(r['accuracy_lo'], r['accuracy_hi']), 'Log loss': r['log_loss'], 'ECE': r['ece'],
                     'Pipelines compared for the finalist choice': int(f['n_configs_tried'])})
    if rows:
        L.export_table(pd.DataFrame(rows), 'm2_finalists', caption=(
            'Lane m2 finalists handed to the cross-lane nested selection (results/finalists\\_m2.json): the best '
            'linear, tree and ensemble pipelines by outer log loss, an exploratory choice among all the model '
            'pipelines compared on the outer folds.'),
            column_formats={'Accuracy': '.1%', 'Log loss': '.3f', 'ECE': '.3f'})


TABLES = {'m2_models': lambda: table_models(_exp()), 'm2_data_treatment': lambda: table_data_treatment(_exp()),
          'm2_calibration': table_calibration, 'm2_calibration_platform': table_calibration_platform,
          'm2_breakdown': table_breakdown, 'm2_learning_curve': table_learning_curve,
          'm2_k_attempts': table_k_attempts, 'm2_feature_families': table_features, 'm2_finalists': table_finalists}


def tables(only: list[str] | None = None) -> None:
    for name, fn in TABLES.items():
        if selected(name, only):
            fn()
