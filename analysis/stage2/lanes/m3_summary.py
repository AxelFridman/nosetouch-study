"""Lane m3: reports/stage2/m3_SUMMARY.md, generated from the saved files only (numbers step).

Every number in the PROSE is read from results/m3_numbers.json (``N``: value and format of the numbers file, so the
summary and the LaTeX macros agree); the tables are rendered from tables/m3/*.json (the same files as the .tex tables);
every verdict and decision sentence is assembled from saved verdicts (m3_paired.json, m3_calibration.json, ...), so the
summary stays true when a participant is added and the pipeline is re-run (python -m stage2.pipeline numbers --lanes m3)."""
from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from stage2 import config as C
from stage2 import report_export as RX
from stage2.lanes import m3_common as M
from stage2.lanes import m3_experiments as E
from stage2.lanes import m3_stats as S
from stage2.lanes.m3_derive import (AGG_LABELS, ENGINEERED, EQUAL_INPUT, FAMILIES, ROLE_LABELS, _k, comparison_key,
                                    shipped_name)
from stage2.lanes.m3_pipeline import L


def _r(path: Any) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


class Numbers:
    """Formatted access to results/m3_numbers.json: ``N('deep_accuracy')`` -> '89.3%' (keys are normalised exactly as
    ``m3_derive.numbers`` writes them)."""

    def __init__(self, path: Any):
        self.e = _r(path)

    def has(self, key: str) -> bool:
        key = _k(key)
        return key in self.e and self.e[key]['value'] is not None

    def v(self, key: str) -> Any:
        key = _k(key)
        return self.e[key]['value'] if key in self.e else None

    def __call__(self, key: str) -> str:
        key = _k(key)
        if key not in self.e:
            raise KeyError(f'm3 summary cites {key!r}, which is not in results/m3_numbers.json')
        return RX.format_value(self.e[key]['value'], self.e[key]['format'])

    def ci(self, key: str, lo: str = '_lo', hi: str = '_hi') -> str:
        return f'[{self(key + lo)}, {self(key + hi)}]' if self.has(key + lo) and self.has(key + hi) else ''

    def p(self, key: str) -> str:
        v = self.v(key)
        return 'p < 0.001' if v is not None and v < 0.001 else f'p = {self(key)}'

    def cmp(self, key: str, holm: bool = True) -> str:
        """'+7.2 points [+3.6, +11.1] (Holm p = 0.004; better)' (mean per-person accuracy difference)."""
        s = f"{self(key + '_diff')} points {self.ci(key)}"
        if holm and self.has(key + '_p_holm'):
            s += f" (Holm {self.p(key + '_p_holm')}; {self(key + '_verdict')})"
        return s


def vphrase(verdict: str, other: str) -> str:
    """'better than X', 'worse than X', 'identical to X' or 'not detectably different from X'."""
    return {'better': f'better than {other}', 'worse': f'worse than {other}', 'identical': f'identical to {other}'}.get(
        verdict, f'not detectably different from {other}')


def _p(v: Any, d: int = 1) -> str:
    return '--' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:.{d}f}%'


def _ci(lo: Any, hi: Any, d: int = 1) -> str:
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)) or (isinstance(hi, float) and math.isnan(hi)):
        return ''
    return f'[{100 * lo:.{d}f}, {100 * hi:.{d}f}]'


def _f(v: Any, d: int = 3) -> str:
    return '--' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{d}f}'


def _pn(v: Any) -> str:
    return '--' if v is None or (isinstance(v, float) and math.isnan(v)) else ('< 0.001' if v < 0.001 else f'{v:.3f}')


def _methods_text() -> str:
    from stage2.lanes import m3_deep as D
    from stage2.lanes import m3_minirocket as MR
    from stage2.lanes import m3_tf as TF
    dp, mp = D.DEFAULTS, MR.DEFAULTS
    return f"""\
## How every m3 model is trained and evaluated

- **Evaluation.** Only through `stage2.cv.run_experiment`: leave-one-person-out (one outer fold per person), the
  docs/09 D2 cap of {C.CAP_PER_PERSON} training trials per person, natural-use trials scored apart (`extra=`),
  person-cluster bootstrap 95% CIs (B = {C.N_BOOT}), and the leak check on **all** folds of every experiment (the leak
  re-run reuses the cached fit and repeats the prediction stage; `fit()` is never given test rows).
- **One contract for every family** (`m3_common.make_fit_predict`): `fit(X_train, y_train, groups_train,
  fold.inner_splits(k), fold.seed)` returns a state that holds a **cross-fitted logit for every training row**, made by
  a model that did not see that row's person. Calibrators and the attempt-aggregation scale c_k are fitted on these
  logits, never on the test person.
- **Fit cache** (`analysis/data/m3_fit_cache/`, git-ignored): every fitted fold, keyed by the family's code and
  configuration, the runtime (numpy, scipy, scikit-learn and torch versions, the training thread count) and every input
  of `fit`. Calibration variants, nested choices and the saliency step reuse the trained networks.
- **Reproducibility.** Every deep fit, prediction and attribution runs with {D.TRAIN_THREADS} intra-op thread and
  deterministic algorithms (`m3_deep.torch_runtime`), whatever the process's thread setting: a refit gives the same
  predictions bit for bit. The lane's thread budget is used by worker processes (`M3_WORKERS`).
- **Deep networks** (`m3_nets.py`, `m3_deep.py`; PyTorch on CPU): every outer fold trains a **{dp['n_inner']}-member
  cross-validated ensemble**. Member k trains on the other parts of `fold.inner_splits({dp['n_inner']})` and is
  **early-stopped on part k** (log loss on real, unaugmented trials; patience {dp['patience']}, at least
  {dp['min_epochs']} and at most {dp['max_epochs']} epochs), with its own seed. Each member fits its **channel normaliser
  on its training rows and their mirror images** (the centre of every chiral channel is then exactly 0, so normalising
  commutes with the mirror). Augmentation: the **mirror image with the flipped label** (the training set is doubled), a
  smooth monotone **time warp** (up to {dp['warp']} of the window, approach) or a **shift of up to {dp['shift']} samples
  and a speed change of up to {100 * dp['speed']:.0f}% about the onset** (fixed window), **magnitude scaling** (x
  {1 - dp['scale']:.2f}-{1 + dp['scale']:.2f}) and **jitter** ({dp['jitter']} SD). The ensemble logit is the mean of the
  member logits. Heads keep coarse time position ({dp['n_bins']} time bins), because the early yaw reverses later in
  the approach.
- **Antisymmetric variant**: z(x) = g(x) - g(mirror x), so p(mirror x) = 1 - p(x) exactly; mirror augmentation is then
  redundant and not used.
- **MiniRocket** (`m3_minirocket.py`, from scratch in numpy): the 84 fixed kernels, dilations on a log grid, every
  other combination unpadded, random channel subsets, PPV features, biases = quantiles of the convolution output of one
  random **training** example per combination (training rows and their mirrors); {mp['num_features']} features. L2
  logistic regression with C in {list(mp['Cs'])} chosen on `fold.inner_splits()` (log loss). A test compares the
  vectorised transform with a plain loop.
- **Time-frequency inputs** (`m3_tf.py`, per-trial transforms of the time-normalised approach): STFT (Hann,
  {TF.STFT_NPERSEG} samples, hop {TF.STFT_HOP}) and the complex-Morlet CWT (`{TF.CWT_WAVELET}`, {len(TF.CWT_SCALES)}
  scales from {TF.CWT_SCALES[0]:.0f} to {TF.CWT_SCALES[-1]:.0f} samples), both keeping the **real and imaginary parts**.
  Non-deep baseline: the planes pooled into frequency bands x time blocks, then an L2 logistic regression (C nested on
  `fold.inner_splits()`); deep: a small 2D CNN.
- **Body frame** (`m3_body.py`, per-trial): the heading-following body frame (lane m1's definition, re-implemented),
  h = the horizontal direction out of the screen and s = u x h, gives roll rate (w.h) and lateral acceleration (a.s),
  odd under the mirror, and pitch rate (w.s) and forward acceleration (a.h), even; a rotation of the phone about its
  screen normal leaves all four unchanged (`test_m3.py` checks invariance and parity).
- **Calibration remedies**: Platt, temperature, isotonic and beta, each fitted inside every outer fold on the training
  persons' cross-fitted logits, and the calibrator **chosen inside the fold** (every candidate cross-fitted over the
  training persons' logits, the lowest log loss kept: no outer-fold choice).
- **Statistics** (`m3_stats.py`): one estimand per comparison, the mean per-person difference, with its paired
  person-bootstrap 95% CI and the exact sign-flip p over persons, Holm-adjusted within its family; a verdict needs the
  adjusted p below {S.ALPHA} and the CI away from 0. ECE is judged against its distribution under perfect calibration
  (consistency resampling), never with a percentile bootstrap CI; calibration slope and intercept carry person-bootstrap
  CIs; strata with fewer than {S.MIN_PERSONS_CI} persons get no CI.
"""


COMMANDS = """\
## Regenerate everything (from `analysis/`)

```
.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m3            # missing / stale experiments (worker streams), phase 2, derived files
.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m3 --force    # all of them (see the training cost below)
.venv/Scripts/python.exe -m stage2.scripts.run_m3 --threads 1 --names m3_cnn m3_gru   # chosen base experiments, one process
.venv/Scripts/python.exe -m stage2.pipeline figures --lanes m3            # redraw from saved files (no fitting, seconds)
.venv/Scripts/python.exe -m stage2.pipeline tables --lanes m3             # CSV + booktabs .tex, registry_m3.csv
.venv/Scripts/python.exe -m stage2.pipeline numbers --lanes m3            # results/m3_numbers.json and this summary
.venv/Scripts/python.exe -m pytest stage2/tests/test_m3.py -q -p no:cacheprovider
```

`M3_WORKERS` sets the worker processes of the compute step (default: the thread budget, 3), `M3_EXTENDED=1` adds the
optional experiments (`m3_experiments.OPTIONAL`). Presentation settings live in `m3_figures.FIGURES`. Deep fits are
identical whatever the thread setting (one intra-op thread per fit), so any of these commands reproduces the saved
predictions bit for bit with the same library versions.
"""


def write_summary() -> str:
    T = L.tab_dir
    N = Numbers(L.numbers_path)
    Sm = _r(T / 'm3_summary.json')
    ex = pd.DataFrame(Sm['experiments']).set_index('name')
    sel = _r(T / 'm3_selection.json')['roles']
    pj = _r(T / 'm3_paired.json')
    pp = pd.DataFrame(pj['rows'])
    calj = _r(T / 'm3_calibration.json')
    cal = calj['roles']
    agj = _r(T / 'm3_aggregate.json')
    bdj = _r(T / 'm3_breakdown.json')
    bd = pd.DataFrame(bdj['rows'])
    sal = _r(T / 'm3_saliency.json') if (T / 'm3_saliency.json').exists() else None
    gtj = _r(T / 'm3_grip_transfer.json') if (T / 'm3_grip_transfer.json').exists() else None
    lnj = _r(T / 'm3_label_noise.json') if (T / 'm3_label_noise.json').exists() else None
    ship = {r: shipped_name(r, sel, cal) for r in sel}

    def key(family: str, a: str, b: str, subset: str = 'all') -> str:
        return comparison_key({'family': family, 'name': a, 'baseline': b, 'subset': subset})

    def row_of(family: str, a: str, b: str, subset: str = 'all') -> Any:
        h = pp[(pp['family'] == family) & (pp['name'] == a) & (pp['baseline'] == b) & (pp['subset'] == subset)]
        return h.iloc[0] if len(h) else None

    out: list[str] = []
    w = out.append
    w('# Lane m3: time-frequency and deep sequence models\n')
    w('*Generated by `python -m stage2.pipeline numbers --lanes m3` from the saved results. Every number in the prose is '
      'read from `results/m3_numbers.json` (LaTeX macros `\\MThree...` in `results/numbers.tex`); the tables come from '
      '`tables/m3/*.json`, the same files as the `.tex` tables.*\n')
    w('## Goal\n')
    w('Do representation-learning models (1D CNN, ResNet1D, InceptionTime-lite, TCN, GRU, 2D CNNs on spectrograms and '
      'scalograms, MiniRocket) beat engineered features on the street data, and do they find the same evidence (the early '
      f"yaw about gravity)? {N('n_trials')} primary trials of {N('n_persons')} persons, leave-one-person-out; the "
      f"{N('n_natural_use')} natural-use trials are scored apart.\n")

    # ───────── headline ─────────
    w('## Headline (leave-one-person-out, 95% person-cluster bootstrap CIs)\n')
    w('| Model | Accuracy [95% CI] | AUC [95% CI] | Log loss [95% CI] | ECE (null 95th pct) | Natural use [exact 95% CI] '
      '| k = 3, mean P [95% CI] | CPU-min | Configs | Selection |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    rows = []
    for role in ('deep', 'mr', 'tf'):
        if role not in sel:
            continue
        rows.append((f"{ROLE_LABELS[role]}, shipped: {ex.loc[ship[role], 'label']} (`{ship[role]}`)", ship[role]))
        rows.append((f"  its base, uncalibrated (`{sel[role]['best']}`)", sel[role]['best']))
        top = (sel[role].get('grip_rule') or {}).get('most_accurate')
        if top and top != sel[role]['best'] and top in ex.index:
            rows.append((f"  most accurate {role} configuration, grip-dependent (`{top}`)", top))
    for nn in E.NESTED_SPECS:
        if nn in ex.index:
            rows.append((f"{ex.loc[nn, 'label']} (`{nn}`)", nn))
    for lab, n in rows:
        r = ex.loc[n]
        nat = (f"{_p(r['natural_use_answered_accuracy'], 0)} {_ci(r['natural_use_answered_lo'], r['natural_use_answered_hi'], 0)}"
               if pd.notna(r['natural_use_answered_accuracy']) else 'abstains')
        w(f"| {lab} | {_p(r['accuracy'])} {_ci(r['accuracy_lo'], r['accuracy_hi'])} | {_f(r['roc_auc'])} "
          f"[{_f(r['roc_auc_lo'])}, {_f(r['roc_auc_hi'])}] | {_f(r['log_loss'])} [{_f(r['log_loss_lo'])}, "
          f"{_f(r['log_loss_hi'])}] | {_f(r['ece'])} ({_f(r['ece_null_q95'])}) | {nat} | {_p(r.get('acc_k3'))} "
          f"{_ci(r.get('acc_k3_lo'), r.get('acc_k3_hi'))} | {r['cpu_min']:.1f} | {int(r['n_configs_tried'])} | "
          f"{r['selection']}{', post hoc' if r['post_hoc'] else ''} |")
    ref = Sm.get('reference') or {}
    if ref:
        w(f"| reference floor (`example_yaw_logreg`, engineered) | {_p(ref['accuracy'])} {_ci(ref['accuracy_lo'], ref['accuracy_hi'])} "
          f"| {_f(ref['roc_auc'])} | {_f(ref['log_loss'])} | {_f(ref['ece'])} | | | | 1 | pre-specified |")
    w('')
    w('CPU-min: training cost of one LOSO run (for a nested choice, of every candidate). Configs: configurations compared '
      'on the outer folds before keeping this one; "best of K" is an exploratory choice whose accuracy is optimistic; '
      f'"post hoc": designed after outer-fold results had been seen. ECE: {C.ECE_BINS} equal-mass bins, with the 95th percentile of '
      f"its distribution under perfect calibration in brackets. Natural use: the one first free-choice trial per person; "
      'every one was made with the right hand, so this is the recall of the right hand over the answered trials (a '
      'P(right) of exactly 0.5 abstains), with an exact binomial CI. k = 3: per-person balanced accuracy when the P(right) '
      f"of 3 attempts of one person and hand are averaged ({N('k_n_persons')} persons with at least "
      f"{max(C.K_ATTEMPTS)} trials of each hand); the recommended aggregator is in the aggregation section.\n")
    gr = (sel.get('deep') or {}).get('grip_rule') or {}
    if gr:
        w(f"**Deep finalist rule** (declared before the body-frame results existed): {gr['rule']}. ")
        if 'acc_diff' in gr:
            w(f"The most accurate deep configuration, `{gr['most_accurate']}`, uses device-frame channels; against the most "
              f"accurate grip-invariant one, `{gr['grip_invariant_alternative']}`, it differs by {N('deep_grip_rule_diff')} "
              f"points {N.ci('deep_grip_rule')}, so the deep finalist is `{sel['deep']['best']}`"
              f"{' (replaced)' if gr.get('replaced') else ' (kept, grip-dependent)'}.\n")
        else:
            w(f"The most accurate deep configuration, `{gr['most_accurate']}`, is grip-invariant, so the rule did not "
              'have to act.\n')
    # selection honesty
    inv, alln = 'm3_deep_nested_inv', 'm3_deep_nested_all'
    if inv in ex.index and alln in ex.index:
        best = sel['deep']['best']
        k_inv = f"x_deep_nested_inv_chose_{best.replace('m3_', '')}"
        w(f"**How much of the deep number is selection?** The deep finalist's base was kept among {N('deep_n_configs')} "
          f"deep configurations on the outer folds{' and its input was designed post hoc' if sel['deep'].get('post_hoc') else ''}. "
          f"Choosing inside every fold among the {N('x_deep_nested_inv_n_candidates')} grip-invariant configurations (the "
          f"shipping policy) gives {N('x_deep_nested_inv_accuracy')} {N.ci('x_deep_nested_inv_accuracy')}"
          + (f", choosing `{best}` in {N(k_inv)} of {N('n_persons')} folds" if N.has(k_inv) else '')
          + f". Letting the grip-dependent configurations compete (`{alln}`, {N('x_deep_nested_all_n_candidates')} "
          f"candidates) gives {N('x_deep_nested_all_accuracy')} {N.ci('x_deep_nested_all_accuracy')}, with a "
          f"grip-dependent configuration chosen in {N('x_deep_nested_all_grip_dependent_folds')} folds: that difference "
          'comes from admitting grip-dependent inputs, not from the outer-fold choice. Neither nested estimate removes the '
          'optimism of the post-hoc design of the body-frame input (`post hoc` above).\n')

    # equal input and engineered
    eq = pp[pp['family'] == 'equal_input']
    if len(eq):
        w(f"**Input or model class?** Deep against linear models on matched information (family `equal_input`, "
          f"{N('family_equal_input_size')} comparisons, Holm-adjusted):\n")
        for _, r in eq.iterrows():
            k = comparison_key(r.to_dict())
            w(f"- {r['what']}: `{r['name']}` vs `{r['baseline']}`, {N.cmp(k)}.")
        k_body = key('treatment', 'm3_cnn_body', 'm3_cnn')
        if N.has(k_body + '_diff'):
            w(f"- Within the CNN, adding the body frame to the gravity frame: {N.cmp(k_body)} (family `treatment`).")
        w('')
    eng = pp[pp['family'] == 'engineered']
    if len(eng):
        w(f"**Against engineered features** (paired on the same trials; family `engineered`, "
          f"{N('family_engineered_size')} comparisons, Holm-adjusted; mean per-person accuracy difference):\n")
        for _, r in eng.iterrows():
            k = comparison_key(r.to_dict())
            w(f"- {r['option']}: {N.cmp(k)}; per-person log-loss difference {N(k + '_ll')} {N.ci(k + '_ll')}.")
        w('')

    # ───────── every experiment ─────────
    w('## Every experiment\n')
    w('One line per experiment; the results JSON is `analysis/reports/stage2/results/<name>.json` (registry: '
      '`results/registry_m3.csv`, where `n_configs_tried` records the outer-fold choice).\n')
    w('| Experiment | Model | Accuracy [95% CI] | AUC | Log loss | ECE | Natural use | k = 3 | CPU-min | Configs | Selection |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for n, r in ex.iterrows():
        cal_tag = f" (calibrated: {M.CALIB_LABELS.get(r['calib'], r['calib'])})" if r['group'] == 'calib' else ''
        nat = _p(r['natural_use_answered_accuracy'], 0) if pd.notna(r['natural_use_answered_accuracy']) else 'abstains'
        w(f"| `{n}` | {r['label']}{cal_tag} | {_p(r['accuracy'])} {_ci(r['accuracy_lo'], r['accuracy_hi'])} | "
          f"{_f(r['roc_auc'])} | {_f(r['log_loss'])} | {_f(r['ece'])} | {nat} | {_p(r.get('acc_k3'))} | {r['cpu_min']:.1f} | "
          f"{int(r['n_configs_tried'])} | {r['selection']}{', post hoc' if r['post_hoc'] else ''} |")
    opt = [n for n in E.OPTIONAL if n not in ex.index]
    if opt:
        w(f"\nSpecified but not run by default (CPU budget; `M3_EXTENDED=1` runs them): {', '.join(f'`{n}`' for n in opt)}.")
    w('')
    w(_methods_text())

    # ───────── data treatment ─────────
    w('## Data treatment and model choices: options, evidence, decision\n')
    w('Each option is paired against its family default on the same trials (mean per-person accuracy difference, paired '
      'person-bootstrap 95% CI, exact sign-flip p, Holm-adjusted within the family; per-person log-loss difference). '
      'Families: ' + '; '.join(f"`{k}` = {v} ({N('family_' + k + '_size')} comparisons)" for k, v in FAMILIES.items()
                               if N.has('family_' + k + '_size')) + '. Figure `M3_03_treatments`, table `m3_treatments`.\n')
    w('| Family | Choice | Option | Against | Accuracy | Difference [95% CI] | p | p (Holm) | Log-loss diff. [95% CI] | Verdict |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    tr = pp[pp['family'].isin(['treatment', 'treatment_body', 'nested', 'physics_demo'])]
    for _, r in tr.iterrows():
        w(f"| {r['family']} | {r['treatment']} | {r['option']} | `{r['baseline']}` | {_p(r['accuracy'])} | "
          f"{100 * r['diff']:+.1f} {_ci(r['diff_lo'], r['diff_hi'])} | {_pn(r['p'])} | {_pn(r['p_holm'])} | "
          f"{r['logloss_diff']:+.3f} [{r['logloss_lo']:+.3f}, {r['logloss_hi']:+.3f}] | {r['verdict']}"
          f"{'; ' + r['verdict_logloss'] if r['verdict_logloss'] not in ('no detectable difference', 'identical') else ''} |")
    w('')
    w('**Decision rule.** An option replaces its family default only when it is better (Holm-adjusted p below '
      f"{N('alpha')} and the paired CI above 0); it is rejected when worse by the same rule; otherwise the default is kept "
      '(a log loss that is lower after adjustment is reported as a candidate improvement, not adopted). The finalist of '
      'each role is a separate, exploratory choice: the highest LOSO accuracy among all the configurations of the role '
      '(`n_configs_tried` recorded on the result), plus the deep grip rule; so a finalist can use an option whose '
      'comparison with the default is not detectable (the MiniRocket finalist below). Upstream preprocessing '
      f'({C.FS_HZ:.0f} Hz grid, {C.LOWPASS_HZ:.0f} Hz low-pass, gap bridging and exclusions) belongs to the core cache and '
      'lane p1 and is not varied here.\n')
    grp = {'arch': '1D CNN', 'treat': '1D CNN', 'mr': 'MiniRocket', 'tf': 'time-frequency', 'treat_body': '1D CNN, body frame'}
    dec_rows = tr[tr['family'].isin(['treatment', 'treatment_body'])].copy()
    dec_rows['fam'] = [grp.get(E.BY_NAME[n].group, n) if n in E.BY_NAME else n for n in dec_rows['name']]
    for (fam, choice), g in dec_rows.groupby(['fam', 'treatment'], sort=False):
        opts = '; '.join(f"{r['option']} {N(comparison_key(r.to_dict()) + '_diff')} points "
                         f"{N.ci(comparison_key(r.to_dict()))} ({r['verdict']})" for _, r in g.iterrows())
        better = g[g['verdict'] == 'better']
        worse = g[g['verdict'] == 'worse']
        lower = g[g['verdict_logloss'] == 'lower log loss']
        recheck = bool((g['family'] == 'treatment_body').all())
        if recheck:                      # sensitivity checks of the finalist's input: never adopted on these folds
            parts = []
            for _, r in g.iterrows():
                g0 = tr[(tr['family'] == 'treatment') & (tr['baseline'] == E.D) & (tr['option'] == r['option'])]
                v0 = g0['verdict'].iloc[0] if len(g0) else None
                ll = (f" (log loss: {r['verdict_logloss']})" if r['verdict_logloss'] not in
                      ('no detectable difference', 'identical') else '')
                parts.append(f"{r['option']}: accuracy {r['verdict']} here{ll}" + (
                    f", {v0} on the gravity frame ({'the same accuracy verdict' if v0 == r['verdict'] else 'the accuracy verdict differs'})"
                    if v0 else ''))
            dec = '; '.join(parts)
            if len(better) or len(lower):
                dec += ('. A re-check is not a candidate of the deep role (adopting it would be one more choice on these '
                        'outer folds), so an option that is better or lowers the log loss here is reported as a candidate '
                        'improvement for a future nested selection, not adopted')
        elif len(better):
            dec = 'evidence for ' + ', '.join(f"**{o}**" for o in better['option'])
        elif len(worse):
            dec = 'keep the default; worse: ' + ', '.join(f"**{o}**" for o in worse['option'])
        else:
            dec = 'no detectable difference; keep the default' + (
                f" (candidate improvement in log loss: {', '.join(lower['option'])})" if len(lower) else '')
        w(f"- **{fam}, {choice}:** {opts}. Decision: {dec}.")
    for role in ('mr', 'tf'):
        if role in sel and sel[role]['best'] in E.BY_NAME:
            s = E.BY_NAME[sel[role]['best']]
            if s.baseline:
                w(f"- **{ROLE_LABELS[role]}:** the role's finalist is `{s.name}` ({s.option or s.label}), the most "
                  f"accurate of its {N(role + '_n_configs')} configurations (exploratory); its comparison with the default "
                  f"above is `{comparison_key({'family': 'treatment', 'name': s.name, 'baseline': s.baseline, 'subset': 'all'})}`.")
    w('')

    # ───────── TF physics ─────────
    w('## Time-frequency inputs: the sign is the evidence\n')
    w(f"- **Detected (physics, then numbers).** Under the left/right mirror a chiral channel only changes sign, so "
      f"|STFT(x)| = |STFT(-x)| and |CWT(x)| = |CWT(-x)|: over every chiral channel ("
      f"{', '.join(_r(T / 'm3_tf_example.json')['chiral_channels'])}) of all {N('n_trials')} trials the largest magnitude "
      f"difference is {N('tf_max_mag_diff_stft')} (STFT) and {N('tf_max_mag_diff_cwt')} (CWT), while the real parts "
      'change sign (`M3_02_tf_sign`).')
    for n in ('m3_tflog_stft', 'm3_tflog_stft_mag', 'm3_tflog_stft_mag_noaug', 'm3_tflog_cwt', 'm3_tflog_cwt_mag',
              'm3_tflog_cwt_mag_noaug', 'm3_cnn2d_stft', 'm3_cnn2d_cwt', 'm3_cnn2d_stft_mag'):
        if n in ex.index:
            x = n.replace('m3_', 'x_')
            w(f"  - `{n}` ({ex.loc[n, 'label']}): AUC {N(x + '_roc_auc')} {N.ci(x + '_roc_auc')}, accuracy "
              f"{N(x + '_accuracy')}.")
    w('- **Why it matters.** A magnitude-only input sees a trial and its mirror image as the same input. With mirror '
      'augmentation it gets every input twice with opposite labels and is at chance by construction (it answers 0.5, an '
      'abstention, on every trial). Without it, it can only use what is not mirror-symmetric between the hands, and that '
      'does not transfer.')
    w('- **Remedy.** Keep the sign: real and imaginary planes (or the signed low-frequency band).')
    w('- **Evidence.** The signed models above against their magnitude-only twins (family `physics_demo` in the table).\n')

    # ───────── calibration ─────────
    w('## Calibration (detected, why it matters, remedy, evidence)\n')
    w('The product shows P(right) as a percentage, so calibration on new persons is part of the result. **Detected** '
      f'with reliability diagrams ({C.ECE_BINS} equal-mass bins, person-bootstrap bands with the bins held fixed), the ECE against its '
      'distribution under perfect calibration, the calibration slope and intercept with person-bootstrap CIs, Brier and log '
      'loss, overall and per platform (`M3_04_calibration`, tables `m3_calibration`, `m3_calibration_platform`). '
      '**Remedies**, fitted inside every outer fold on the cross-fitted logits of the training persons and applied to the '
      'held-out person: Platt, temperature, isotonic, beta and the calibrator chosen inside the fold. **Evidence**: paired '
      'per-person log-loss and Brier differences against the uncalibrated model (family `calib_vs_none`) and against the '
      'pre-specified remedy (family `calib_vs_pre`), Holm-adjusted. The pre-specified remedy (docs/09 D19) is temperature '
      f"for an antisymmetric model and Platt otherwise. Target (docs/09 M4, recommended): ECE at most {N('ece_target')} "
      'and no reliability bin off the diagonal by more than its uncertainty.\n')
    # the bin criterion itself: a problem found in the lane's own check (detected, why, remedy, evidence)
    shipped_m = {r: c['decision']['shipped'] for r, c in cal.items()}
    ev = []
    for r, m in shipped_m.items():
        pk = f'{r}_cal_{m}'
        if N.has(pk + '_bins_off_bootstrap_only'):
            ev.append(f"{ROLE_LABELS[r]} ({M.CALIB_LABELS[m]}): {N(pk + '_bins_off_bootstrap_only')} bin(s) off with the "
                      f"bootstrap band alone ({N(pk + '_bins_off_degenerate')} of them degenerate), {N(pk + '_bins_off')} "
                      f"with both; chance of at least one bin off for a perfectly calibrated model {N(pk + '_bins_off_null_rate')}")
    if ev:
        w('**Which bins count as off (a problem in this lane\'s own check).** *Detected:* judged by its person-bootstrap '
          'band alone, a bin whose trials all have the same hand has an observed share of exactly 0 or 1 and a band of '
          'zero width, so it "excludes" any prediction short of 0 or 1; in the deep finalist the extreme bins were flagged '
          'this way. *Why it matters:* the criterion would then call a perfectly calibrated model miscalibrated and the '
          'docs/09 target missed. *Remedy:* a bin is off only when its person-bootstrap band excludes the bin\'s mean '
          'prediction AND its observed share lies outside the 95% consistency band of a perfectly calibrated model with the '
          'same predictions (y ~ Bernoulli(p); the bootstrap band covers the clustering of trials within persons, the '
          'consistency band covers the binomial noise that a degenerate bootstrap band misses). *Evidence* (shipped '
          'models): ' + '; '.join(ev) + '.\n')
    for role, c in cal.items():
        dec = c['decision']
        w(f"**{ROLE_LABELS[role]} ({c['label']}, `{c['best']}`)**, pre-specified remedy: {c['prespecified']}\n")
        w('| Calibration | Accuracy | ECE (null 95th pct) | Slope [95% CI] | Intercept [95% CI] | Brier | Log loss | '
          'Log-loss diff. vs none [95% CI], Holm p | Brier diff. vs none [95% CI] | Bins off | Target |')
        w('|---|---|---|---|---|---|---|---|---|---|---|')
        for e in c['entries']:
            vs = e.get('vs_none') or {}
            d1 = ('' if e['method'] == 'none' else
                  f"{vs['logloss_diff']:+.4f} [{vs['logloss_lo']:+.4f}, {vs['logloss_hi']:+.4f}], {_pn(vs['p_holm_logloss'])}")
            d2 = '' if e['method'] == 'none' else f"{vs['brier_diff']:+.4f} [{vs['brier_lo']:+.4f}, {vs['brier_hi']:+.4f}]"
            tags = (' (pre-specified)' if e['prespecified'] else '') + (' (shipped)' if e['method'] == dec['shipped'] else '')
            if e['method'] == 'nested' and e.get('nested_choices'):
                tags += ' [' + ', '.join(f'{k} x{v}' for k, v in e['nested_choices'].items()) + ']'
            w(f"| {M.CALIB_LABELS[e['method']]}{tags} | {_p(e['accuracy'])} | {_f(e['ece'])} ({_f(e['ece_null_q95'])}) | "
              f"{_f(e['cal_slope'], 2)} [{_f(e.get('cal_slope_lo'), 2)}, {_f(e.get('cal_slope_hi'), 2)}] | "
              f"{e['cal_intercept']:+.2f} [{_f(e.get('cal_intercept_lo'), 2)}, {_f(e.get('cal_intercept_hi'), 2)}] | "
              f"{_f(e['brier'])} | {_f(e['log_loss'])} | {d1} | {d2} | {e['bins_off']} | {'met' if e['target_met'] else 'missed'} |")
        w('')
        base_e = next(e for e in c['entries'] if e['method'] == 'none')
        sh_e = next((e for e in c['entries'] if e['method'] == dec['shipped']), base_e)
        p0, p1 = f'{role}_cal_none', f"{role}_cal_{dec['shipped']}"
        w('Per platform, uncalibrated: ' + '; '.join(
            f"{pl['platform']}: ECE {N(p0 + '_' + pl['platform'] + '_ece')} (null {N(p0 + '_' + pl['platform'] + '_ece_null_q95')}), "
            f"slope {N(p0 + '_' + pl['platform'] + '_cal_slope')}, Brier {N(p0 + '_' + pl['platform'] + '_brier')}, log loss "
            f"{N(p0 + '_' + pl['platform'] + '_log_loss')} ({N(p0 + '_' + pl['platform'] + '_n')} trials, "
            f"{N(p0 + '_' + pl['platform'] + '_n_persons')} persons)" for pl in base_e['per_platform']) + '.')
        if sh_e is not base_e:
            w(f"Shipped ({M.CALIB_LABELS[dec['shipped']]}): " + '; '.join(
                f"{pl['platform']}: ECE {N(p1 + '_' + pl['platform'] + '_ece')}, log loss {N(p1 + '_' + pl['platform'] + '_log_loss')}"
                for pl in sh_e['per_platform']) + '.')
        why0 = '; '.join(dec['uncalibrated_miscalibration']) or 'none of the miscalibration criteria fires'
        w(f"- **Detected (uncalibrated):** {why0} (ECE {N(p0 + '_ece')} against a perfect-calibration 95th percentile of "
          f"{N(p0 + '_ece_null_q95')}; slope {N(p0 + '_cal_slope')} [{N(p0 + '_cal_slope_lo')}, {N(p0 + '_cal_slope_hi')}]).")
        w(f"- **Decision:** ship {M.CALIB_LABELS[dec['shipped']]} ({dec['reason']})"
          + (f"; candidate improvements for the synthesis's calibration choice (lower log loss than the pre-specified "
             f"remedy after adjustment): {', '.join(dec['candidates'])}" if dec['candidates'] else '') + '.')
        if dec['shipped_target_met']:
            w(f"- **Target:** met by the shipped model (ECE {N(p1 + '_ece')}, no reliability bin off).\n")
        else:
            w(f"- **Target: missed** by the shipped model ({'; '.join(dec['shipped_miscalibration']) or 'ECE above the target'}; "
              f"ECE {N(p1 + '_ece')}, {N(p1 + '_bins_off')} bin(s) off). This finalist does not meet the docs/09 calibration "
              f"target; the lowest-ECE remedy on these folds is {M.CALIB_LABELS.get(dec['lowest_ece_method'], dec['lowest_ece_method'])} "
              '(not adopted: choosing it here would be an outer-fold selection without a detectable log-loss gain).\n')

    # ───────── aggregation ─────────
    rec = agj['recommended']
    w('**Multi-attempt aggregates.** For k attempts of one person and hand (out of fold, the persons with at least '
      f"{max(C.K_ATTEMPTS)} trials per hand), {N('agg_n_compared')} aggregators: the mean P(right) (the harness k-curve), the mean "
      'logit, the sum of logits (attempts treated as independent evidence) and the sum of the raw logits scaled by c_k '
      "(fitted on the training persons' cross-fitted logits of each fold). Over-confidence = mean confidence minus "
      'accuracy (`M3_07_attempts`, table `m3_aggregate`).\n')
    for role, a in agj['roles'].items():
        blk = a.get('calibrated') or a.get('uncalibrated')
        tag = 'c' if a.get('calibrated') else 'u'
        k = rec['k']
        parts = [f"{AGG_LABELS[m]} {N(f'{role}_agg{tag}_{m}_k{k}_oc')} {N.ci(f'{role}_agg{tag}_{m}_k{k}_oc')}"
                 for m in AGG_LABELS if N.has(f'{role}_agg{tag}_{m}_k{k}_oc')]
        ck_keys = [(int(c['k']), f"{role}_agg{tag}_c{int(c['k'])}") for c in blk['c_k']]
        cks = ', '.join(f'c_{kk} = {N(kc)}' for kk, kc in ck_keys)
        w(f"- {ROLE_LABELS[role]} (`{blk['name']}`), k = {k} over-confidence: {'; '.join(parts)}. Median scales {cks}.")
    w('')
    if rec['aggregator']:
        a, kk = rec['aggregator'], rec['k']
        acc_keys = [(r, f'{r}_aggc_{a}_k{kk}_acc') for r in agj['roles']]
        calibrated_all = ', '.join(AGG_LABELS[m] for m in rec['calibrated_for_every_role']) or 'none'
        ks_txt = '; '.join(f"{AGG_LABELS[m]} at k = {N('agg_' + m + '_calibrated_ks')}" for m in AGG_LABELS
                           if N.has('agg_' + m + '_calibrated_ks'))
        w(f"**Decision (aggregation; checked on these outer folds among {N('agg_n_compared')} aggregators, so "
          f"exploratory).** Rule: {rec['rule']}. Calibrated at k = {kk} for every role: {calibrated_all}; recommended: "
          f"**{N('agg_recommended')}** ({rec.get('how', '')}). Calibrated for every role, by number of attempts: {ks_txt}. "
          f"Its accuracy for the shipped finalists at k = {kk}: "
          + '; '.join(f'{ROLE_LABELS[r]} {N(ak)} {N.ci(ak)}' for r, ak in acc_keys if N.has(ak))
          + f" (the headline k = {kk} column averages P(right), the harness definition).\n")

    # ───────── breakdown ─────────
    w(f"## Breakdown of the shipped deep finalist (`{bdj['names'].get('deep', '')}`; platform and conditions only stratify)\n")
    b = bd[bd['role'] == 'deep'] if (bd['role'] == 'deep').any() else bd
    w('| Condition | Level | Trials | Persons | Accuracy [95% CI] | AUC |')
    w('|---|---|---|---|---|---|')
    for _, r in b.iterrows():
        w(f"| {r['condition']} | {r['level']} | {int(r['n'])} | {int(r['n_persons'])} | {_p(r['accuracy'])} "
          f"{_ci(r['accuracy_lo'], r['accuracy_hi']) or '(no CI)'} | {_f(r['roc_auc'])} |")
    w('')
    overall = N('deep_ship_accuracy')
    weak = b[pd.notna(b['accuracy_hi']) & (b['accuracy_hi'] < N.v('deep_ship_accuracy'))]
    w(f"Levels whose CI lies entirely below the overall accuracy ({overall}; exploratory, {len(b)} levels compared "
      'without adjustment): ' + (', '.join(
          f"{r['condition']} {r['level']} ({N('bd_deep_' + r['condition'] + '_' + str(r['level']) + '_acc')}, "
          f"{N('bd_deep_' + r['condition'] + '_' + str(r['level']) + '_n')} trials of "
          f"{N('bd_deep_' + r['condition'] + '_' + str(r['level']) + '_persons')} persons)" for _, r in weak.iterrows())
          or 'none') + f". Levels with fewer than {bdj['min_persons_ci']} persons carry no CI.\n")

    # ───────── grip transfer ─────────
    if gtj:
        gt = pd.DataFrame(gtj['rows'])
        w('## Grip transfer: device frame vs heading-following body frame\n')
        k_all = key('treatment', 'm3_cnn_all', 'm3_cnn')
        w('- **Detected.** The device-frame channels (acceleration, angular velocity and up vector in phone coordinates) '
          f"add accuracy to the gravity-frame CNN ({N.cmp(k_all)}), but their values depend on the grip.")
        w('- **Why it matters.** Production grips are free. Instructed grips are balanced across hands, so the gain is not a '
          'protocol artefact, but evidence tied to phone coordinates may fail on a grip that training never saw.')
        w('- **Test.** `*_upright` trains a configuration only on the trials whose grip, detected from the pre-onset up '
          'vector, is upright, and tests it on every grip (the instructed orientation only stratifies the result). The '
          'statistic is the paired difference on the **rotated** grips, the ones upright-only training never saw (family '
          '`grip`, Holm-adjusted). **Remedy:** the same information in the heading-following body frame (`m3_body`).\n')
        w('| Model | Orientation | Trials | Accuracy [95% CI] |')
        w('|---|---|---|---|')
        for _, r in gt.iterrows():
            w(f"| `{r['name']}` | {r['level']} | {int(r['n'])} | {_p(r['accuracy'])} {_ci(r['accuracy_lo'], r['accuracy_hi'])} |")
        w('')
        for a_, b_, sub in (('m3_cnn_all_upright', 'm3_cnn_all', 'rotated'), ('m3_cnn_body_upright', 'm3_cnn_body', 'rotated'),
                            ('m3_cnn_body_upright', 'm3_cnn_body', 'upright'), ('m3_cnn_body', 'm3_cnn', 'rotated'),
                            ('m3_cnn_body', 'm3_cnn', 'upright'), ('m3_cnn_body_upright', 'm3_cnn', 'rotated')):
            k = key('grip', a_, b_, sub)
            if N.has(k + '_diff'):
                w(f"- `{a_}` vs `{b_}` on the {sub} grips ({N(k + '_n_trials')} trials, {N(k + '_n_persons')} persons): "
                  f"{N(k + '_acc')} vs {N(k + '_base_acc')}, {N.cmp(k)}.")
        sv = gtj.get('sensor_vs_instructed') or {}
        if sv:
            w('- Sensor-detected grip against the instructed orientation (trials): ' + '; '.join(
                f"{i}: " + ', '.join(f'{c} {v}' for c, v in d.items() if v) for i, d in sv.items())
              + f". {N('grip_rotated_detected_upright')} rotated-grip trials were detected as upright, so upright-only "
              f"training never saw a rotated grip; {N('grip_upright_detected_other')} upright trials were not detected as "
              'upright and were left out of that training.')
        w('')

    # ───────── saliency ─────────
    if sal:
        bc = sal['by_channel']
        w("## What the deep finalist's network looks at (integrated gradients, out of fold)\n")
        w(f"`{sal['name']}` ({sal['label']}): each held-out trial is attributed by its own fold's ensemble, with the "
          'normalised zero (mirror-invariant) as the baseline; the calibrators are monotone maps of this logit '
          '(`M3_05_saliency`, `tables/m3/m3_saliency.npz`).\n')
        top = sorted(bc.items(), key=lambda kv: -kv[1]['share_of_all'])[:5]
        w(f"- Largest shares of |IG| ({len(bc)} input channels): " + '; '.join(f"{c} {N('ig_' + c + '_share')}" for c, _ in top)
          + f". The yaw channels ({', '.join(sal['yaw_channels'])}) carry {N('ig_share_yaw')} together.")
        w(f"- The grip-invariant odd (hand-carrying) channels ({', '.join(sal['odd_channels'])}) carry {N('ig_share_odd')} "
          f"of all |IG|, against {N('ig_share_odd_uniform')} if every channel weighed the same.")
        generic = sal['share_first_half_even'] - 0.5                     # front-loading shared by the even channels
        excess = sal['first_half_odd_minus_even']
        lo_ex, hi_ex = sal['first_half_odd_minus_even_lo'], sal['first_half_odd_minus_even_hi']
        how = ('more front-loaded' if lo_ex > 0 else 'less front-loaded' if hi_ex < 0 else 'not detectably more front-loaded')
        reading = ('so most of the early concentration is the generic time profile of the attribution (shared by the even '
                   'channels), not a property of the hand evidence' if generic > excess else
                   'so the early concentration is mostly specific to the hand-carrying channels')
        w(f"- Timing: {N('ig_share_first_half_odd')} of the odd channels' |IG| falls in the first half "
          f"({sal['first_half_definition']}), against {N('ig_share_first_half_even')} for the even channels: the odd channels "
          f"are {how} by {N('ig_first_half_odd_minus_even')} points {N.ci('ig_first_half_odd_minus_even')} (person "
          f'bootstrap), {reading}. Per odd channel: ' + '; '.join(
              f"{c} {N('ig_' + c + '_first_half')} in the first half, peak at {N('ig_' + c + '_peak')} of the window"
              for c in sal['odd_channels']) + '.')
        w(f"- Mirror consistency: corr(IG(mirror x), -IG(x)) = {N('ig_mirror_corr')} (1 for an exactly antisymmetric model). "
          f"IG completeness: median relative error {N('ig_completeness_median_rel_err')} ({sal['ig_steps']} steps); the sign "
          f"of the summed attribution agrees with the logit in {N('ig_ig_sign_agrees_with_logit')} of the trials.")
        yr = bc.get('yaw_rate')
        if yr:
            body_odd = [c for c in sal['odd_channels'] if c not in sal['yaw_channels']]
            s_body = sum(bc[c]['share_of_all'] for c in body_odd)
            lean = ('it uses the early yaw, but leans more on the body-relative motion' if s_body > sal['share_yaw'] else
                    'the yaw channels carry more of its attribution than the body-relative motion')
            w(f"- Comparison with the physics of the descriptive analysis and lane m1 (the first-half yaw about gravity "
              f"separates the hands): the yaw rate carries {N('ig_yaw_rate_share')} of |IG|, {N('ig_yaw_rate_first_half')} of "
              f"it in the first half (peak at {N('ig_yaw_rate_peak')}); the largest single channel is "
              f"{top[0][0]} ({N('ig_' + top[0][0] + '_share')})"
              + (f"; the odd body-frame channels ({', '.join(body_odd)}) carry {N('ig_share_body_odd')} against "
                 f"{N('ig_share_yaw')} for the yaw channels, so {lean}" if body_odd and N.has('ig_share_body_odd') else '')
              + '.')
            even = [c for c in sal['even_channels'] if c in bc]
            if even:
                ce = max(even, key=lambda c: (bc[c]['toward_true_hand'], c))
                w(f"- Interactions: in a mirror-equivariant model an even channel (unchanged by the mirror) can push "
                  f"toward the true hand only through its interaction with the odd channels; {ce} does so with "
                  f"{N('ig_' + ce + '_toward')} of its attribution (sum of IG x hand sign over the sum of |IG|; "
                  f"{sal['odd_channels'][0]}: {N('ig_' + sal['odd_channels'][0] + '_toward')}), so the network combines "
                  'channels rather than reading a single hand-carrying one.')
            w('')

    # ───────── label noise ─────────
    if lnj and lnj['roles']:
        w('## Label-noise sensitivity\n')
        blocks = ', '.join(f"{b_['session_label']} block {b_['block_index']}" for b_ in lnj['blocks']) or 'none'
        parts = []
        for r in lnj['roles']:
            if not N.has(r + '_swapfree_acc'):
                continue
            on = f"; {N(r + '_swap_acc')} on the swap trials" if N.has(r + '_swap_acc') else ''
            parts.append(f"{ROLE_LABELS[r]} {N(r + '_swapfree_acc')} {N.ci(r + '_swapfree_acc')} (all trials "
                         f"{N(r + '_ship_accuracy')}{on})")
        w(f"Blocks with the hand-swap signature (found from the data by `reference.swap_blocks`; labels never changed): "
          f"{blocks} ({N('swap_n_trials')} trials of {N('swap_n_persons')} person(s)). Scored without them: "
          + '; '.join(parts) + '.\n')

    # ───────── figures ─────────
    w('## Figures (`analysis/reports/stage2/figures/m3/`, data in `tables/m3/<name>.csv`)\n')
    caps_p = L.fig_dir / 'captions.json'
    if caps_p.exists():
        for k, v in sorted(_r(caps_p).items()):
            w(f"- `{k}`: {v.get('decision', '')}")
    w('')
    w(COMMANDS)
    w(f"Training cost of a rebuild with an empty fit cache: {N('base_cpu_min')} CPU-minutes for the base experiments, "
      f"about {N('base_wall_min_workers')} minutes of wall-clock time on {N('n_workers')} one-thread workers; the second "
      'phase, the nested choices and the derived files reuse the cached fits.\n')

    # ───────── conclusions ─────────
    w('## Conclusions\n')
    concl: list[str] = []
    d_ship = ship.get('deep')
    if d_ship:
        k_ref, k_m1, k_m2 = (key('engineered', d_ship, c) for c in ENGINEERED)
        concl.append(
            f"**Deep finalist.** {ex.loc[d_ship, 'label']}, shipped with {M.CALIB_LABELS[cal['deep']['decision']['shipped']]} "
            f"calibration (`{d_ship}`; base kept among {N('deep_n_configs')} deep configurations on the outer folds"
            f"{', input designed post hoc' if sel['deep'].get('post_hoc') else ''}): {N('deep_ship_accuracy')} "
            f"{N.ci('deep_ship_accuracy')}, AUC {N('deep_ship_roc_auc')}, log loss {N('deep_ship_log_loss')}, ECE "
            f"{N('deep_ship_ece')}, natural use {N('deep_ship_nat_acc')} {N.ci('deep_ship_nat', '_lo', '_hi')}, k = 3 "
            f"{N('deep_ship_acc_k3')}. Against the reference floor {N.cmp(k_ref)}; against lane m1's pre-specified physics "
            f"logistic {N.cmp(k_m1)}; against lane m2's engineered-feature stack {N.cmp(k_m2)}.")
    k_inv = 'x_deep_nested_inv_accuracy'
    if N.has(k_inv):
        k_m2i = key('engineered', 'm3_deep_nested_inv', 'm2_ens_stack')
        concl.append(
            f"**Nested estimate of the shipping policy** (grip-invariant configuration chosen inside every fold): "
            f"{N(k_inv)} {N.ci(k_inv)}; against lane m2's stack {N.cmp(k_m2i)}. With grip-dependent configurations allowed "
            f"it is {N('x_deep_nested_all_accuracy')} {N.ci('x_deep_nested_all_accuracy')}.")
    eq_ok = row_of('equal_input', 'm3_cnn_body', 'm1_lr_phys6')
    eq_grav = row_of('equal_input', 'm3_cnn', 'm1_lr_phys6')
    if eq_ok is not None:
        m1_lab = "lane m1's 6-feature physics logistic"
        txt = (f"**Input and model class.** On matched information (yaw about gravity plus body-frame roll and lateral "
               f"motion) the CNN is {vphrase(eq_ok['verdict'], m1_lab)} "
               f"({N.cmp(key('equal_input', 'm3_cnn_body', 'm1_lr_phys6'))})")
        if eq_grav is not None:
            txt += (f"; on the gravity frame alone it is {vphrase(eq_grav['verdict'], 'the same logistic')} "
                    f"({N.cmp(key('equal_input', 'm3_cnn', 'm1_lr_phys6'))})")
        if eq_ok['verdict'] == 'better' and (eq_grav is None or eq_grav['verdict'] != 'better'):
            txt += ('. The deep advantage over the engineered linear model appears only with the body-frame channels: it '
                    'comes from the input and the model class together, not from the model class alone')
        concl.append(txt + '.')
    if 'mr' in ship and 'tf' in ship:
        concl.append(
            f"**MiniRocket and time-frequency models** ({ex.loc[ship['mr'], 'label']}: {N('mr_ship_accuracy')} "
            f"{N.ci('mr_ship_accuracy')}; {ex.loc[ship['tf'], 'label']}: {N('tf_ship_accuracy')} {N.ci('tf_ship_accuracy')}) "
            f"against lane m2's stack: {N.cmp(key('engineered', ship['mr'], 'm2_ens_stack'))} and "
            f"{N.cmp(key('engineered', ship['tf'], 'm2_ens_stack'))}; against the reference "
            f"{N.cmp(key('engineered', ship['mr'], 'example_yaw_logreg'))} and "
            f"{N.cmp(key('engineered', ship['tf'], 'example_yaw_logreg'))}.")
    kr_dev = key('grip', 'm3_cnn_all_upright', 'm3_cnn_all', 'rotated')
    kr_body = key('grip', 'm3_cnn_body_upright', 'm3_cnn_body', 'rotated')
    if N.has(kr_dev + '_diff') and N.has(kr_body + '_diff'):
        k_bg_rot, k_bg_up = key('grip', 'm3_cnn_body', 'm3_cnn', 'rotated'), key('grip', 'm3_cnn_body', 'm3_cnn', 'upright')
        vb = N.v(kr_body + '_verdict')
        rb = row_of('grip', 'm3_cnn_body_upright', 'm3_cnn_body', 'rotated')
        vll = rb['verdict_logloss'] if rb is not None else None
        ll_txt = (f"; its per-person log loss on those grips changes by {N(kr_body + '_ll')} {N.ci(kr_body + '_ll')} "
                  + (f"(higher after the Holm adjustment, {N.p(kr_body + '_ll_p_holm')})" if vll == 'higher log loss'
                     else f"({N(kr_body + '_ll_verdict')} after the Holm adjustment)")
                  if rb is not None else '')
        if vb == 'worse':
            resid = 'a residual penalty remains on grips absent from training (accuracy worse after adjustment)' + ll_txt
        elif N.v(kr_body + '_hi') is not None and N.v(kr_body + '_hi') < 0:
            resid = ('a residual accuracy penalty on grips absent from training is suggested by its CI (unadjusted '
                     f"sign-flip {N.p(kr_body + '_p')}) but not established after the Holm adjustment" + ll_txt)
        else:
            resid = 'no residual accuracy penalty on grips absent from training is detectable' + ll_txt
        concl.append(
            f"**Grip.** Trained on upright grips only and tested on the rotated grips it never saw, the device-frame CNN "
            f"changes by {N(kr_dev + '_diff')} points {N.ci(kr_dev)} ({N(kr_dev + '_acc')} accuracy on those grips; "
            f"{N(kr_dev + '_verdict')}), the body-frame CNN by {N(kr_body + '_diff')} points {N.ci(kr_body)} "
            f"({N(kr_body + '_acc')}; Holm {N.p(kr_body + '_p_holm')}). The body frame avoids the device-frame collapse; "
            f"{resid}, so the training data should cover the grips expected in production. Its gain over the gravity "
            f"frame: {N(k_bg_up + '_diff')} points on upright grips ({N(k_bg_up + '_verdict')}), {N(k_bg_rot + '_diff')} "
            f"points {N.ci(k_bg_rot)} on rotated grips ({N(k_bg_rot + '_verdict')}).")
    if sal:
        top1 = sorted(sal['by_channel'].items(), key=lambda kv: -kv[1]['share_of_all'])[0][0]
        concl.append(
            f"**Evidence used.** The network relies on the mirror-odd, grip-invariant channels ({N('ig_share_odd')} of "
            f"|IG| against {N('ig_share_odd_uniform')} uniform), led by {top1} ({N('ig_' + top1 + '_share')}); the yaw "
            f"channels carry {N('ig_share_yaw')}. Its attribution is front-loaded on every channel "
            f"({N('ig_share_first_half_all')} of all |IG| in the first half); the odd channels exceed the even ones by "
            f"{N('ig_first_half_odd_minus_even')} points {N.ci('ig_first_half_odd_minus_even')}. It agrees with the physics "
            'in using early, hand-signed motion, but the evidence is not the early yaw alone; mirror consistency '
            f"{N('ig_mirror_corr')}.")
    if N.has('x_tflog_stft_mag_roc_auc'):
        concl.append(
            f"**Time-frequency inputs must keep the sign**: magnitude-only planes are identical for a trial and its mirror "
            f"image (largest difference {N('tf_max_mag_diff_stft')}) and score AUC {N('x_tflog_stft_mag_roc_auc')}; the "
            f"signed planes reach AUC {N('x_tflog_stft_roc_auc')}.")
    cal_txt = []
    for role, c in cal.items():
        dec = c['decision']
        pm = f"{role}_cal_{dec['shipped']}"
        band = 'inside' if N.v(pm + '_ece') <= N.v(pm + '_ece_null_q95') else 'above'
        target = 'met' if dec['shipped_target_met'] else '**missed**'
        cal_txt.append(f"{ROLE_LABELS[role]}: {M.CALIB_LABELS[dec['shipped']]}, ECE {N(pm + '_ece')} ({band} its "
                       f"perfect-calibration band), slope {N(pm + '_cal_slope')} [{N(pm + '_cal_slope_lo')}, "
                       f"{N(pm + '_cal_slope_hi')}], {N(pm + '_bins_off')} bin(s) off, docs/09 target {target}")
    concl.append(f"**Calibration.** {'; '.join(cal_txt)}.")
    if rec['aggregator']:
        under = 'mean_prob' not in rec['calibrated_for_every_role']
        concl.append(f"**Several attempts** are combined with the {N('agg_recommended')} (calibrated at k = {rec['k']} for "
                     'every role; an exploratory choice among the aggregators compared on these folds)'
                     + ('; averaging P(right) is under-confident.' if under else '.'))
    for i, t in enumerate(concl, 1):
        w(f'{i}. {t}')
    w('')
    # ───────── caveats ─────────
    w('## Caveats and open questions\n')
    w(f"- {N('n_persons')} persons: most paired CIs between configurations overlap, so the ranking of architectures is not "
      'resolved, and the many comparisons are only interpretable with the family-wise adjustment used here.')
    w("- Outer-fold choices: each role's finalist was kept among its role's configurations (`n_configs_tried` on the base "
      'result, its calibration variants, the registry and the finalists file); the body-frame input was designed after '
      'the grip-transfer result of the device-frame CNN had been seen (post hoc), which no nested estimate removes; the '
      'aggregator recommendation is also an outer-fold choice.')
    w('- The natural-use set has a single trial per person, all with the right hand: it measures the recall of the right '
      'hand only, with an exact binomial CI that is wide; a constant "right" would score 100%.')
    w('- The grip-transfer check covers the three protocol grips (screen towards the person); a screen-away grip reverses '
      "the body frame's facing axis and was never recorded.")
    w('- The ECE null band treats trials as independent (person clustering ignored), so it is a lower bound on the noise: '
      'an ECE above it is weaker evidence than it looks, an ECE inside it is strong evidence of calibration.')
    w('- Deviations from a textbook inner cross-fit, kept because they are cheaper and checked by the calibration '
      'statistics: the calibrators are fitted on the cross-fitted logits of single ensemble members but applied to the '
      f"ensemble mean (the uncalibrated deep base has slope {N('deep_cal_none_cal_slope')} "
      f"[{N('deep_cal_none_cal_slope_lo')}, {N('deep_cal_none_cal_slope_hi')}]); each member's early-stopping epoch is "
      'chosen on the rows that also supply its cross-fitted logits (mildly optimistic for the inner choices, never for the '
      'held-out person); docs/09 D19 step 3 (refit on the whole fold) is replaced by the cross-validated ensemble; the '
      'MiniRocket biases are fitted on all outer-training rows, inner-validation persons included (label-free, but '
      'transductive for the inner choice of C and for the calibrator).')
    w('- The body-frame re-checks (`treatment_body`) repeat three treatments only (window, normalisation, symmetry); the '
      'other treatment decisions are inherited from the gravity-frame CNN as an assumption.')
    w('- The deep models are research models: shipping one would need a TypeScript port and the parity tests of docs/09 '
      'D14; the linear models of other lanes are cheaper to deploy.')
    w('- Open: cue-free bounds (detected instead of oracle segments) were not evaluated in this lane; the fixed window is '
      'the representation that needs no end bound.')
    w('- Open: the body frame was given to the 1D CNN only; MiniRocket and the time-frequency models still use the gravity '
      'frame.')
    w("- Open: lane m2's engineered features include device-frame channels; whether its stack transfers to grips unseen in "
      'training was not tested here.')
    comps = sorted(set(ENGINEERED) | {c for _, c, _, _ in EQUAL_INPUT})
    w('- The comparisons with other lanes read their saved results as they stood when the derived files were computed ('
      + ', '.join(f'`{c}`' for c in comps) + '); a re-run in those lanes changes the inputs signature of the derived '
      'files, and a plain `compute --lanes m3` then recomputes them.')
    text = '\n'.join(out) + '\n'
    p = L.summary_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8', newline='\n')
    return text
