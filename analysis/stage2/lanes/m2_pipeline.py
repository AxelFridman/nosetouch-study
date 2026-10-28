"""Lane m2: rich hand-crafted features and classical machine learning. Steps: compute (experiments + derived data),
figures, tables, numbers.

    cd analysis
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes m2                 # missing / stale experiments, then derive
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes m2 --only "m2_lr_*" --force
    .venv/Scripts/python.exe -m stage2.pipeline figures --lanes m2                 # redraw from saved files (seconds)
    .venv/Scripts/python.exe -m stage2.pipeline tables --lanes m2
    .venv/Scripts/python.exe -m stage2.pipeline numbers --lanes m2
    .venv/Scripts/python.exe -m stage2.scripts.run_m2 --only "m2_lgbm*"           # experiments only, no derive (workers)

Every experiment is one leave-one-person-out ``run_experiment`` (lane registry ``results/registry_m2.csv``) defined
in ``experiments()``: a feature spec (``m2_features.FeatureSpec``) and a model pipeline (``m2_models``). Tuning
happens only inside fit_predict on ``fold.inner_splits()``. ``n_configs_tried`` of an experiment is the number of
configurations of the comparison it belongs to that are computed on the outer folds (1 for the pre-specified default,
the nested selection and the learning curve), derived from the experiment list itself.
The derived data every figure / table / number needs is written by ``m2_derive.derive()`` at the end of compute.
"""
from __future__ import annotations

import os

from stage2.pipeline import Lane, limit_threads

# this lane's thread budget is 2; a smaller budget set by the caller (--threads 1, workers) is kept
limit_threads(max(1, min(2, int(os.environ.get('OMP_NUM_THREADS', '2') or 2))))

import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from typing import Any  # noqa: E402

from stage2.lanes.m2_features import DEFAULT_SPEC, FAMILIES, FeatureSpec  # noqa: E402

L = Lane('m2')
ORDER = 120
DEPENDS = ()

#: post-hoc calibrators compared (fitted inside the outer folds on inner cross-fitted predictions), and the models
#: they are compared for: the pre-specified default, the best tree and the nested selection (the honest headline)
CALIBRATORS = ('platt', 'temperature', 'isotonic', 'beta', 'platt_platform')
CALIBRATED = (('m2_lr_full', 'lr_l2'), ('m2_lgbm', 'lgbm'), ('m2_select', 'select'))
#: learning curve: training persons kept per outer fold (all = the default experiment) and repetitions per size
LC_SIZES = (4, 8, 12, 16)
LC_REPS = 5
#: mirror variants run besides augmentation: all four for the linear default; LightGBM only without mirroring (one
#: LightGBM experiment tunes 20 x 8 Optuna configurations; a wrapper variant re-tunes with the wrapper in the loss)
LR_MIRRORS = ('none', 'anti', 'both')
LGBM_MIRRORS = ('none',)
#: the unscaled-feature linear model: lbfgs iteration cap (it does not converge on features whose SDs span orders of
#: magnitude; the non-converged fits are counted and saved with the result)
MAX_ITER_UNSCALED = 300
#: the candidates of the nested model selection (each tuned in the fold exactly as in its own experiment)
SELECT_CANDIDATES = ('lr_l2', 'svm_rbf', 'lgbm', 'ens_soft')
ENSEMBLE = ('lr_l2', 'svm_rbf', 'lgbm')
#: Optuna trials per outer fold; fewer for slow models (one-parameter linear models use an exhaustive grid)
TRIALS = {'svm_rbf': 8, 'knn': 8, 'lgbm': 8, 'hgb': 6, 'lr_en': 6}
#: inner person folds of the nested tuning (cv.inner_splits = fold.inner_splits(INNER)): 3 rather than the default
#: 5 to fit six lanes on eight cores; with 19 training persons each inner validation fold holds 6-7 persons
INNER = 3
WINSOR_SD = 3.0


@dataclass(frozen=True)
class Exp:
    name: str
    group: str
    label: str
    feat: FeatureSpec = DEFAULT_SPEC
    model: str = 'lr_l2'
    mirror: str = 'aug'
    scaler: str = 'z'
    winsor: float | None = None
    calibrator: str = 'none'
    n_trials: int | None = None
    n_splits: int = INNER
    n_train_persons: int | None = None
    rep: int = 0
    max_iter: int | None = None
    ensemble: tuple[str, ...] = ()
    method: str = ''
    select: tuple[str, ...] = ()
    importance: bool = False
    baseline: str = ''
    n_configs_tried: int = 1
    notes: str = ''

    def settings(self) -> dict[str, Any]:
        return {'feat': self.feat.to_dict(), 'model': self.model, 'mirror': self.mirror, 'scaler': self.scaler,
                'winsor': self.winsor, 'calibrator': self.calibrator, 'n_trials': self.n_trials,
                'n_splits': self.n_splits, 'n_train_persons': self.n_train_persons, 'rep': self.rep,
                'max_iter': self.max_iter, 'ensemble': list(self.ensemble), 'method': self.method,
                'select': list(self.select)}


MIRROR_LABELS = {'none': 'none', 'aug': 'augmentation', 'anti': 'antisymmetric wrapper',
                 'both': 'augmentation + wrapper'}
CAL_LABELS = {'none': 'uncalibrated', 'platt': 'Platt', 'temperature': 'temperature', 'isotonic': 'isotonic',
              'beta': 'beta', 'platt_platform': 'Platt per platform'}
MODEL_LABELS = {'lr_l2': 'Logistic regression (L2)', 'lr_l1': 'Logistic regression (L1)',
                'lr_en': 'Logistic regression (elastic net)', 'svm_lin': 'Linear SVM + Platt',
                'svm_rbf': 'RBF SVM + Platt', 'knn': 'k-NN + Platt', 'rf': 'Random forest (fixed hyperparameters)',
                'et': 'Extra trees (fixed hyperparameters)', 'hgb': 'HistGradientBoosting', 'lgbm': 'LightGBM',
                'ens_soft': 'Soft vote (LR + SVM + LGBM)', 'ens_stack': 'Stack (LR + SVM + LGBM)',
                'select': 'Nested selection'}
LINEAR = ('lr_l2', 'lr_l1', 'lr_en', 'svm_lin')
TREES = ('rf', 'et', 'hgb', 'lgbm')


def experiments() -> OrderedDict[str, Exp]:
    """Every experiment of the lane, in run order (most important first). ``n_configs_tried`` of each comparison is
    computed from the configurations listed here (all of them are computed)."""
    D = DEFAULT_SPEC
    E: list[Exp] = []
    base = 'm2_lr_full'
    E.append(Exp(base, 'default', 'L2 logistic regression, default features (pre-specified)', importance=True,
                 notes=f'pre-specified default: approach window, {len(D.families)} families, k={D.n_bins} bins, '
                       'all channels, z-score, mirror augmentation, C tuned on inner person folds'))
    # model comparison (each nested-tuned); n_configs_tried = number of pipelines compared on the outer folds
    models = ('lr_l1', 'lr_en', 'svm_lin', 'svm_rbf', 'knn', 'rf', 'et', 'hgb', 'lgbm')
    ensembles = ('soft', 'stack')
    k_models = 1 + len(models) + len(ensembles)
    for m in models:
        E.append(Exp(f'm2_{m}', 'model', MODEL_LABELS[m], model=m, n_trials=TRIALS.get(m),
                     importance=(m == 'lgbm'), baseline=base, n_configs_tried=k_models))
    for meth in ensembles:
        E.append(Exp(f'm2_ens_{meth}', 'model', MODEL_LABELS[f'ens_{meth}'], model=f'ens_{meth}', ensemble=ENSEMBLE,
                     method=meth, baseline=base, n_configs_tried=k_models))
    E.append(Exp('m2_select', 'select', 'Nested selection (LR / SVM / LGBM / soft vote, by inner log loss)',
                 model='select', select=SELECT_CANDIDATES, baseline=base,
                 notes='per outer fold the candidate with the lowest inner cross-validated log loss is used'))
    # calibration (post-hoc maps fitted on inner cross-fitted predictions) of the default, LightGBM and the selection
    for src, m in CALIBRATED:
        for c in CALIBRATORS:
            E.append(Exp(f'{src}_cal_{c}', 'calibration', f'{MODEL_LABELS[m]} + {CAL_LABELS[c]}', model=m, calibrator=c,
                         n_trials=TRIALS.get(m), select=SELECT_CANDIDATES if m == 'select' else (), baseline=src,
                         n_configs_tried=1 + len(CALIBRATORS)))
    # mirror handling: the four variants for the linear default; LightGBM see LGBM_MIRRORS
    for m, src, variants in (('lr_l2', base, LR_MIRRORS), ('lgbm', 'm2_lgbm', LGBM_MIRRORS)):
        for mir in variants:
            E.append(Exp(f'm2_{"lr" if m == "lr_l2" else m}_mirror_{mir}', 'mirror',
                         f'{MODEL_LABELS[m]}, mirror: {MIRROR_LABELS[mir]}',
                         model=m, mirror=mir, n_trials=TRIALS.get(m), baseline=src, n_configs_tried=1 + len(variants)))
    # representation: window, frame, bins
    windows = (('fixed', 'fixed 3-s window (absolute time)'), ('both', 'approach + fixed windows'))
    for w, wl in windows:
        E.append(Exp(f'm2_lr_win_{w}', 'window', f'window: {wl}', feat=D.with_(window=w), baseline=base,
                     n_configs_tried=1 + len(windows)))
    frames = (('gravity', 'gravity frame only'), ('device', 'device frame only'))
    for f, fl in frames:
        E.append(Exp(f'm2_lr_frame_{f}', 'frame', f'channels: {fl}', feat=D.with_(channels=f), baseline=base,
                     n_configs_tried=1 + len(frames)))
    bins = (4, 16)
    for k in bins:
        E.append(Exp(f'm2_lr_bins{k}', 'bins', f'bins: k = {k} time bins', feat=D.with_(n_bins=k), baseline=base,
                     n_configs_tried=1 + len(bins)))
    # preprocessing: scaling (none, robust), winsorising, per-trial amplitude normalisation
    scal = [Exp('m2_lr_scale_none', 'scaling', f'scaling: none (raw features; lbfgs capped at {MAX_ITER_UNSCALED} '
                'iterations)', scaler='none', max_iter=MAX_ITER_UNSCALED, baseline=base,
                notes='no scaling: the solver is capped and its non-converged fits are counted (info)'),
            Exp('m2_lr_scale_robust', 'scaling', 'scaling: robust (median / IQR)', scaler='robust', baseline=base),
            Exp('m2_lr_winsor3', 'scaling', f'scaling: z-score + winsorising at {WINSOR_SD:g} SD', winsor=WINSOR_SD,
                baseline=base),
            Exp('m2_lr_ampnorm', 'scaling', 'scaling: per-trial amplitude normalisation (RMS)',
                feat=D.with_(amp_norm=True), baseline=base)]
    E += [replace(e, n_configs_tried=1 + len(scal)) for e in scal]
    # feature families: drop one, only one; with mirror augmentation (the default: only the chiral, odd / swapped
    # features can carry weight) and without mirroring (every family can, the even ones included)
    for tag, mir, src, sfx in (('lr', 'aug', base, ''), ('lrnm', 'none', 'm2_lr_mirror_none', '_nomirror')):
        fam_exps = [Exp(f'm2_{tag}_drop_{fam}', f'family_drop{sfx}', f'without {fam}', mirror=mir,
                        feat=D.with_(families=[f for f in FAMILIES if f != fam]), baseline=src) for fam in FAMILIES]
        # one feature (log duration) alone gives every trial P = 0.5 under augmentation (an even feature): run only
        # without mirroring, where it is a one-feature logistic regression
        fam_exps += [Exp(f'm2_{tag}_only_{fam}', f'family_only{sfx}', f'only {fam}', mirror=mir,
                         feat=D.with_(families=[fam]), baseline=src) for fam in FAMILIES
                     if not (fam == 'global' and mir == 'aug')]
        E += [replace(e, n_configs_tried=1 + len(fam_exps)) for e in fam_exps]
    # learning curve: training persons subsampled inside fit_predict
    for n in LC_SIZES:
        for r in range(LC_REPS):
            E.append(Exp(f'm2_lc_n{n:02d}_r{r}', 'learning_curve', f'{n} training persons (rep {r})',
                         n_train_persons=n, rep=r, baseline=base))
    return OrderedDict((e.name, e) for e in E)


def importance_name(name: str) -> str:
    return f'{name}_importance'


# ───────────────────────────── building the fit_predict of an experiment ─────────────────────────────


def make_fp(e: Exp, mi: Any, ms: Any, store: dict | None = None) -> Any:
    from stage2.lanes import m2_models as M
    if e.select:
        return make_select_fp(e.select, mi, ms, calibrator=e.calibrator)
    if e.ensemble:
        return M.make_ensemble_fit_predict(e.ensemble, method=e.method, mirror=e.mirror, mirror_index=mi,
                                           mirror_sign=ms, n_trials=TRIALS, calibrator=e.calibrator, n_splits=INNER)
    return M.make_fit_predict(e.model, mirror=e.mirror, scaler=e.scaler, winsor=e.winsor, calibrator=e.calibrator,
                              n_trials=e.n_trials, mirror_index=mi, mirror_sign=ms, n_train_persons=e.n_train_persons,
                              subsample_rep=e.rep, n_splits=e.n_splits, store=store, max_iter=e.max_iter)


def make_select_fp(candidates: tuple[str, ...], mi: Any, ms: Any, calibrator: str = 'none') -> Any:
    """Nested model selection: per outer fold, every candidate is tuned on the inner person folds (cached: the
    same fits as the candidates' own experiments) and the one with the lowest inner cross-validated log loss
    predicts. Every candidate's inner predictions are out of sample (cross-fitted Platt maps for the SVM), so the
    criterion favours no candidate. ``calibrator``: a post-hoc map fitted on the CHOSEN candidate's inner
    cross-fitted P(right). The nested multi-attempt temperatures come from the same predictions."""
    import numpy as np

    from stage2 import cv
    from stage2.lanes import m2_models as M

    def fit_predict(fold: Any) -> tuple[Any, dict[str, Any]]:
        X = np.asarray(fold.X_train, dtype=np.float64)
        y = np.asarray(fold.y_train).astype(int)
        g = np.asarray(fold.groups_train).astype(str)
        Xte = np.asarray(fold.X_test, dtype=np.float64)
        base = [c for c in candidates if not c.startswith('ens_')]
        need = sorted(set(base) | (set(ENSEMBLE) if 'ens_soft' in candidates else set()))
        tuned = {c: M.tune(M.MODELS[c], X, y, g, fold.seed, mirror='aug', mi=mi, ms=ms, n_trials=TRIALS.get(c),
                           n_splits=INNER) for c in need}
        losses = {c: tuned[c].inner_logloss for c in base}
        oofs = {c: tuned[c].oof_p for c in base}
        if 'ens_soft' in candidates:
            splits = cv.inner_splits(g, INNER)
            oofs['ens_soft'] = np.column_stack([tuned[c].oof_p for c in ENSEMBLE]).mean(1)
            losses['ens_soft'] = float(np.mean([M._logloss(y[va], oofs['ens_soft'][va]) for _, va in splits]))
        best = min(candidates, key=lambda c: (losses[c], candidates.index(c)))
        if best == 'ens_soft':
            p = np.column_stack([M.predict(tuned[c], Xte, mi, ms) for c in ENSEMBLE]).mean(1)
        else:
            p = M.predict(tuned[best], Xte, mi, ms)
        p, oof, cal_params = M.calibrate(calibrator, oofs[best], y, p, fold)
        return p, {'chosen': best, 'inner_logloss': losses, 'calibrator': calibrator, 'calib_params': cal_params,
                   'attempt_tau': M.attempt_tau(oof, y, g, fold.seed)}

    return fit_predict


# ───────────────────────────── compute ─────────────────────────────


def _needs(e: Exp, force: bool, only: list[str] | None) -> bool:
    from stage2.pipeline import file_current
    if L.should_run(e.name, force, only):
        return True
    if e.importance and (only is None or any(__import__('fnmatch').fnmatchcase(e.name, p) for p in only)):
        return not file_current(L.tab_dir / f'{importance_name(e.name)}.json')
    return False


LOCK_DIR = __import__('pathlib').Path(__import__('tempfile').gettempdir()) / 'stage2_m2_locks'
LOCK_STALE_S = 3 * 3600


def _try_lock(name: str) -> bool:
    """Atomic per-experiment lock (outside the repository), so parallel workers never run one experiment twice."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    p = LOCK_DIR / f'{name}.lock'
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - p.stat().st_mtime > LOCK_STALE_S:
                p.unlink()
                return _try_lock(name)
        except OSError:
            pass
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def _unlock(name: str) -> None:
    try:
        (LOCK_DIR / f'{name}.lock').unlink()
    except OSError:
        pass


def _ordered(exps: OrderedDict[str, Exp], only: list[str] | None) -> list[Exp]:
    """Experiments in the order of ``only`` (patterns expand in registry order), else in registry order."""
    if only is None:
        return list(exps.values())
    import fnmatch
    out: list[str] = []
    for pat in only:
        out += [n for n in exps if fnmatch.fnmatchcase(n, pat) and n not in out]
    return [exps[n] for n in out]


def run_experiments(force: bool = False, only: list[str] | None = None) -> list[str]:
    """Run the selected experiments whose results are missing or stale (all with force), in the order of ``only``.
    Each experiment is re-checked and locked right before it runs, so several workers can share one queue.
    No derived data."""
    exps = experiments()
    todo = [e for e in _ordered(exps, only) if _needs(e, force, only)]
    if not todo:
        print('m2: experiment results current; skipped (--force recomputes)', flush=True)
        return []
    from stage2.data import load_cache
    d, nat = load_cache(), load_cache(('natural_use',))
    feats: dict[FeatureSpec, tuple[Any, Any, Any]] = {}
    done = []
    for e in todo:
        if not force and not _needs(e, force, only):
            continue                                   # finished by another worker meanwhile
        if not _try_lock(e.name):
            print(f'  {e.name}: running in another worker; skipped', flush=True)
            continue
        try:
            done += _run_one(e, d, nat, feats)
        finally:
            _unlock(e.name)
        _evict_unneeded(e, todo)
    return done


def _evict_unneeded(e: Exp, todo: list[Exp]) -> None:
    """Free the cached fits of a model that no later experiment of this run re-uses (ensembles, the selection and
    the calibration variants re-use the fits of their components)."""
    from stage2.lanes import m2_models as M
    later = todo[todo.index(e) + 1:]
    needed = set()
    for x in later:
        needed |= ({x.model} | set(x.ensemble) | set(x.select)
                   | (set(ENSEMBLE) if 'ens_soft' in x.select or x.ensemble else set()))
    if e.model in M.MODELS and e.model not in needed:
        M.evict(e.model)


def _run_one(e: Exp, d: Any, nat: Any, feats: dict) -> list[str]:
    """One experiment: features (cached per spec), fit_predict, run_experiment, held-out importance if asked."""
    import numpy as np

    from stage2.lanes import m2_features as F
    from stage2.lanes.m2_derive import clean
    if e.feat not in feats:
        X, reg = F.build_features(d, e.feat)
        Xn, _ = F.build_features(nat, e.feat)
        feats[e.feat] = (X, Xn, reg)
    X, Xn, reg = feats[e.feat]
    mi, ms = F.mirror_map(reg)
    store: dict[int, Any] | None = {} if e.importance else None
    fp = make_fp(e, mi, ms, store)
    t0 = time.time()
    r = L.run_experiment(e.name, fp, X, d.trials, extra=(Xn, nat.trials), model=e.model,
                         params={'settings': e.settings(), 'n_features': int(X.shape[1])},
                         notes=(e.notes or e.label), n_configs_tried=e.n_configs_tried)
    if store is not None:
        tot = np.zeros(X.shape[1])
        n = 0
        for v in store.values():
            tot += v['abs_sum']
            n += v['n']
        L.write_json(importance_name(e.name), clean({
            'experiment': e.name, 'n_rows': n, 'n_folds': len(store),
            'what': 'mean |contribution to the logit| on held-out persons (linear: coef x scaled value; '
                    'LightGBM: TreeSHAP pred_contrib)',
            'features': reg.assign(mean_abs_contrib=tot / max(n, 1))[
                ['name', 'family', 'window', 'channel', 'stat', 'parity', 'mean_abs_contrib']].to_dict(orient='records')}))
    print(f'  {e.name}: accuracy {100 * r.metrics["accuracy"]:.1f} % [{100 * r.ci["accuracy"][0]:.1f}, '
          f'{100 * r.ci["accuracy"][1]:.1f}], log loss {r.metrics["log_loss"]:.3f} ({r.leak_check}; '
          f'{time.time() - t0:.0f} s)', flush=True)
    return [e.name]


def compute(force: bool = False, only: list[str] | None = None) -> None:
    run_experiments(force, only)
    from stage2.lanes import m2_derive
    m2_derive.derive()


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m2_figures
    m2_figures.draw(only)


def tables(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m2_report
    m2_report.tables(only)


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m2_report, m2_summary
    m2_report.numbers()
    m2_summary.write()                                  # reports/stage2/m2_SUMMARY.md, from the same saved files


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
