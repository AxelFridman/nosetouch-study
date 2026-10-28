"""Lane final: calibration statistics and remedies of the product P(right) (task 5; reporting standard F).

* :func:`fit_calibrator` - none, Platt, temperature, isotonic, beta: fitted on logits of the TRAINING persons'
  inner cross-fitted predictions, returned with plain-array parameters (the export needs them).
* :func:`ece_null` - the ECE a perfectly calibrated model of the same sharpness shows on the same rows
  (consistency resampling: y ~ Bernoulli(p)), with its 95 % band and the p value of the observed ECE.
* :func:`slope_intercept` - logistic recalibration y ~ sigmoid(a logit(p) + b) with person-bootstrap CIs.
* :func:`reliability_bands` - equal-mass bins fixed on the data, the observed share per bin with person-bootstrap
  bands (bins held fixed), and the perfect-calibration consistency band of each bin.
* :func:`paired_metric` - paired person-bootstrap difference of a per-trial loss (log loss, Brier) between two
  prediction sets on the same trials, with the exact sign-flip p over persons (``cv._sign_flip_p``).
* :func:`paired_ece` - the ECE difference of two prediction sets on the SAME person resamples (the upward bias of a
  resampled ECE cancels in the difference) and a basic-bootstrap interval.
* :func:`aggregate_attempts` - multi-attempt aggregates of one person and hand (mean P, mean logit, sum of logits,
  sum of logits / tau_k), with tau_k fitted on INNER cross-fitted predictions of the training persons of every
  outer fold.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv

EPS = 1e-15
CAL_LOGIT_EPS = 1e-12


def sigmoid(z: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p: Any, eps: float = CAL_LOGIT_EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def ll_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), np.finfo(float).eps, 1 - np.finfo(float).eps)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(ll_terms(y, p)))


# ───────────────────────────── calibrators ─────────────────────────────


def _logistic_1d(x: np.ndarray, y: np.ndarray, *, slope: bool = True, iters: int = 100) -> tuple[float, float]:
    """Maximum-likelihood (a, b) of y ~ sigmoid(a x + b) by damped Newton (a = 1 fixed when ``slope`` is False)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a, b = 1.0, 0.0
    prev = math.inf
    for _ in range(iters):
        z = a * x + b
        p = sigmoid(z)
        w = np.clip(p * (1 - p), 1e-12, None)
        g_b = np.sum(p - y)
        if slope:
            g_a = np.sum((p - y) * x)
            H = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, np.array([g_a, g_b]))
        else:
            step = np.array([0.0, g_b / np.sum(w)])
        t = 1.0
        f0 = float(np.sum(ll_terms(y, p)))
        improved = False
        while t > 1e-6:
            a2, b2 = a - t * step[0], b - t * step[1]
            f1 = float(np.sum(ll_terms(y, sigmoid(a2 * x + b2))))
            if f1 <= f0 + 1e-12:
                improved = True
                break
            t /= 2
        if not improved:
            break
        a, b = a2, b2
        if abs(prev - f1) < 1e-12 or abs(f0 - f1) < 1e-13:
            break
        prev = f1
    return float(a), float(b)


def fit_calibrator(kind: str, z: np.ndarray, y: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """(apply(logit) -> P(right), params) of a calibrator fitted on logits ``z`` with labels ``y`` (the training
    persons' inner cross-fitted predictions). Parameters are plain numbers / lists (exportable).

    platt: sigmoid(a z + b); temperature: sigmoid(z / T) (keeps P(mirror) = 1 - P); isotonic: a non-decreasing
    piecewise-linear map of z through the pool-adjacent-violators fit (``IsotonicRegression``, clipped to
    [1/(n+2), 1 - 1/(n+2)]); beta (Kull et al. 2017): sigmoid(a ln p - b ln(1 - p) + c), a, b >= 0."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y).astype(int)
    if kind == 'none':
        return sigmoid, {}
    if kind == 'platt':
        a, b = _logistic_1d(z, y, slope=True)
        return (lambda zz: sigmoid(a * np.asarray(zz, dtype=float) + b)), {'a': a, 'b': b}
    if kind == 'temperature':
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(lambda lt: log_loss(y, sigmoid(z / math.exp(lt))), bounds=(-3.0, 3.0), method='bounded',
                            options={'xatol': 1e-8})
        T = float(math.exp(r.x))
        return (lambda zz: sigmoid(np.asarray(zz, dtype=float) / T)), {'T': T}
    if kind == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        n = z.size
        lo, hi = 1.0 / (n + 2), 1.0 - 1.0 / (n + 2)
        iso = IsotonicRegression(y_min=lo, y_max=hi, out_of_bounds='clip', increasing=True).fit(z, y)
        xt = np.asarray(iso.X_thresholds_, dtype=float)
        yt = np.asarray(iso.y_thresholds_, dtype=float)
        return (lambda zz: np.interp(np.asarray(zz, dtype=float), xt, yt)), {'x': xt.tolist(), 'y': yt.tolist()}
    if kind == 'beta':
        from sklearn.linear_model import LogisticRegression

        def feats(zz: np.ndarray) -> np.ndarray:
            zz = np.asarray(zz, dtype=float)
            return np.column_stack([-np.logaddexp(0, -zz), np.logaddexp(0, zz)])     # ln p, -ln(1 - p)
        use = [0, 1]
        for _ in range(2):
            m = LogisticRegression(C=1e6, max_iter=10000, tol=1e-10).fit(feats(z)[:, use], y)
            if np.all(m.coef_[0] >= 0) or len(use) == 1:
                break
            use = [use[int(np.argmax(m.coef_[0]))]]
        coef = np.zeros(2)
        coef[use] = m.coef_[0]
        c = float(m.intercept_[0])
        return (lambda zz: sigmoid(feats(zz) @ coef + c)), {'a': float(coef[0]), 'b': float(coef[1]), 'c': c}
    raise ValueError(f'calibrator {kind!r}')


#: the remedies of the nested ('auto') choice: lane p1's nested set. Isotonic is left out: its flat steps cost AUC in
#: lane p1 and its many parameters overfit the few training persons (it stays a remedy of the final model's analysis)
AUTO_METHODS: tuple[str, ...] = ('none', 'temperature', 'platt', 'beta')
#: fitted parameters of each remedy (the tie-break of every choice between remedies: fewer first)
N_PARAMS: dict[str, int] = {'none': 0, 'temperature': 1, 'platt': 2, 'beta': 3, 'isotonic': 99}


def choose_calibrator(z: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      methods: Sequence[str] = AUTO_METHODS) -> tuple[str, dict[str, float]]:
    """Nested choice of the remedy from the TRAINING persons' cross-fitted logits ``z``: every method is fitted on all
    but one person group of ``cv.inner_splits(groups)`` (the groups of the inner folds) and scored by the log loss of
    the left-out group; the lowest pooled log loss wins (ties within 1e-10: fewer parameters). The held-out person is
    never involved. Returns (method, {method: cross-validated log loss})."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y).astype(int)
    splits = cv.inner_splits(np.asarray(groups).astype(str))
    scores: dict[str, float] = {}
    for m in methods:
        terms = np.empty(z.size)
        for tr, va in splits:
            f, _ = fit_calibrator(m, z[tr], y[tr])
            terms[va] = ll_terms(y[va], f(z[va]))
        scores[m] = float(terms.mean())
    best = min(methods, key=lambda m: (round(scores[m], 10), N_PARAMS.get(m, 99)))
    return best, scores


# ───────────────────────────── statistics ─────────────────────────────


def ece_null(p: np.ndarray, *, n_sim: int = 2000, seed: int = C.SEED, observed: float | None = None,
             y: np.ndarray | None = None) -> dict[str, float]:
    """ECE of perfectly calibrated outcomes on the same predictions (y* ~ Bernoulli(p), trials independent): mean,
    95 % band, 95th percentile, and the p value of the observed ECE (share of null ECEs at least as large)."""
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng([seed, 31])
    sims = np.array([cv.ece((rng.random(p.size) < p).astype(int), p) for _ in range(n_sim)])
    obs = cv.ece(y, p) if observed is None and y is not None else observed
    out = {'null_mean': float(sims.mean()), 'null_lo': float(np.quantile(sims, 0.025)),
           'null_hi': float(np.quantile(sims, 0.975)), 'null_q95': float(np.quantile(sims, 0.95))}
    if obs is not None:
        out['ece'] = float(obs)
        out['p_value'] = float((1 + np.sum(sims >= obs - 1e-15)) / (n_sim + 1))
        out['inside_band'] = bool(obs <= out['null_hi'])
    return out


def _persons_index(persons: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    uniq = np.unique(persons)
    return uniq, [np.flatnonzero(persons == u) for u in uniq]


def person_resamples(persons: np.ndarray, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> list[np.ndarray]:
    """Row indices of ``n_boot`` person-cluster resamples (persons drawn with replacement, all their rows along)."""
    uniq, rows = _persons_index(np.asarray(persons).astype(str))
    rng = np.random.default_rng([seed, 41])
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        out.append(np.concatenate([rows[i] for i in pick]))
    return out


def slope_intercept(y: np.ndarray, p: np.ndarray, persons: np.ndarray, *, n_boot: int = 1000,
                    seed: int = C.SEED) -> dict[str, Any]:
    """Calibration slope and intercept (y ~ sigmoid(a logit(p) + b); perfect: a = 1, b = 0) with person-bootstrap
    percentile CIs; also the calibration-in-the-large (b with a = 1)."""
    y = np.asarray(y).astype(int)
    z = logit(p)
    a, b = _logistic_1d(z, y)
    _, citl = _logistic_1d(z, y, slope=False)
    A, B = [], []
    for idx in person_resamples(persons, n_boot, seed):
        if len(np.unique(y[idx])) < 2:
            continue
        aa, bb = _logistic_1d(z[idx], y[idx], iters=50)
        A.append(aa)
        B.append(bb)
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    return {'slope': a, 'slope_ci': q(A), 'intercept': b, 'intercept_ci': q(B), 'citl': citl,
            'slope_ci_excludes_1': bool(not (q(A)[0] <= 1 <= q(A)[1])),
            'intercept_ci_excludes_0': bool(not (q(B)[0] <= 0 <= q(B)[1]))}


def reliability_bands(y: np.ndarray, p: np.ndarray, persons: np.ndarray, *, n_bins: int = C.ECE_BINS,
                      n_boot: int = 1000, n_sim: int = 2000, seed: int = C.SEED) -> pd.DataFrame:
    """Equal-mass bins of p (fixed on all rows): mean p, observed share, n; the 95 % person-bootstrap band of the
    observed share (bins held fixed) and the 95 % consistency band of a perfectly calibrated model."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    k = np.searchsorted(edges[1:-1], p, side='right')
    bins = [b for b in range(n_bins) if np.any(k == b)]
    boots = {b: [] for b in bins}
    for idx in person_resamples(persons, n_boot, seed):
        kk = k[idx]
        for b in bins:
            m = kk == b
            if m.any():
                boots[b].append(float(y[idx][m].mean()))
    rng = np.random.default_rng([seed, 53])
    rows = []
    for b in bins:
        m = k == b
        sims = [(rng.random(int(m.sum())) < p[m]).mean() for _ in range(n_sim)]
        lo, hi = np.quantile(boots[b], [0.025, 0.975]) if boots[b] else (math.nan, math.nan)
        clo, chi = np.quantile(sims, [0.025, 0.975])
        pm, ym = float(p[m].mean()), float(y[m].mean())
        rows.append({'bin': b, 'p_mean': pm, 'y_mean': ym, 'n': int(m.sum()), 'boot_lo': float(lo),
                     'boot_hi': float(hi), 'cons_lo': float(clo), 'cons_hi': float(chi),
                     'off': bool((pm < lo or pm > hi) and (ym < clo or ym > chi))})
    return pd.DataFrame(rows)


def _per_person_means(v: np.ndarray, persons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(np.asarray(persons).astype(str), return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=uniq.size)
    n = np.bincount(inv, minlength=uniq.size)
    return uniq, s / n


def paired_metric(loss_a: np.ndarray, loss_b: np.ndarray, persons: np.ndarray, *, n_boot: int = C.N_BOOT,
                  seed: int = C.SEED) -> dict[str, Any]:
    """Mean per-person difference of a per-trial loss (A - B) with its paired person-bootstrap 95 % CI and the exact
    sign-flip p over persons (the same estimand and test as ``cv.paired_compare``)."""
    uniq, da = _per_person_means(np.asarray(loss_a, float) - np.asarray(loss_b, float), persons)
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.integers(0, uniq.size, uniq.size) for _ in range(n_boot)])
    boot = da[idx].mean(axis=1)
    return {'diff': float(da.mean()), 'ci': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            'p': float(cv._sign_flip_p(da, seed=seed)), 'n_persons': int(uniq.size),
            'pooled_diff': float(np.mean(np.asarray(loss_a, float) - np.asarray(loss_b, float)))}


def paired_ece(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, persons: np.ndarray, *, n_boot: int = 1000,
               seed: int = C.SEED) -> dict[str, Any]:
    """ECE(A) - ECE(B) on the same rows, with a BASIC bootstrap interval from the same person resamples (2 d - the
    percentile interval: corrects the upward bias of a resampled ECE)."""
    y = np.asarray(y).astype(int)
    d = cv.ece(y, pa) - cv.ece(y, pb)
    boots = [cv.ece(y[i], pa[i]) - cv.ece(y[i], pb[i]) for i in person_resamples(persons, n_boot, seed)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {'diff': float(d), 'ci_basic': [float(2 * d - hi), float(2 * d - lo)], 'interval': 'basic bootstrap'}


def ece_basic_ci(y: np.ndarray, p: np.ndarray, persons: np.ndarray, *, n_boot: int = 1000,
                 seed: int = C.SEED) -> list[float]:
    """Basic (reverse-percentile) person-bootstrap interval of one ECE: 2 ECE - the percentile bounds."""
    y = np.asarray(y).astype(int)
    e = cv.ece(y, p)
    boots = [cv.ece(y[i], p[i]) for i in person_resamples(persons, n_boot, seed)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return [float(max(0.0, 2 * e - hi)), float(2 * e - lo)]


def calibration_block(y: np.ndarray, p: np.ndarray, persons: np.ndarray, *, n_boot: int = 1000,
                      seed: int = C.SEED) -> dict[str, Any]:
    """The reporting standard of one prediction set: ECE with its null band, slope and intercept with CIs, Brier,
    log loss, accuracy, n, and the verdict (miscalibrated when a slope / intercept CI excludes 1 / 0 or the ECE lies
    outside its null band)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    e = float(cv.ece(y, p))
    nul = ece_null(p, observed=e, seed=seed)
    si = slope_intercept(y, p, persons, n_boot=n_boot, seed=seed)
    mis = bool(si['slope_ci_excludes_1'] or si['intercept_ci_excludes_0'] or not nul['inside_band'])
    return {'n': int(y.size), 'n_persons': int(len(np.unique(np.asarray(persons).astype(str)))),
            'accuracy': float(np.mean(cv.credit(y, p))), 'log_loss': log_loss(y, p),
            'brier': float(np.mean((p - y) ** 2)), 'ece': e, 'ece_basic_ci': ece_basic_ci(y, p, persons, seed=seed),
            **{f'ece_{k}': v for k, v in nul.items() if k != 'ece'}, **si, 'miscalibrated': mis}


def platform_difference(y: np.ndarray, p: np.ndarray, persons: np.ndarray, platform: np.ndarray, *,
                        n_boot: int = 1000, seed: int = C.SEED) -> dict[str, Any]:
    """Calibration of one platform minus the other (alphabetical: android - ios): calibration slope, calibration-in-
    the-large (intercept with the slope fixed at 1) and log loss, with a person-bootstrap 95 % CI that resamples persons
    WITHIN each platform (the platforms hold different persons) and a two-sided bootstrap p (twice the share of
    resampled differences on the far side of 0). The test of a platform difference; overlapping per-platform CIs are
    not one."""
    y = np.asarray(y).astype(int)
    z = logit(p)
    plat = np.asarray(platform).astype(str)
    persons = np.asarray(persons).astype(str)
    a, b = sorted(np.unique(plat))[:2]

    def stats(idx: np.ndarray) -> np.ndarray:
        s, _ = _logistic_1d(z[idx], y[idx], iters=50)
        _, citl = _logistic_1d(z[idx], y[idx], slope=False, iters=50)
        return np.array([s, citl, float(np.mean(ll_terms(y[idx], p[idx])))])
    ia, ib = np.flatnonzero(plat == a), np.flatnonzero(plat == b)
    point = stats(ia) - stats(ib)
    ra = person_resamples(persons[ia], n_boot, seed)
    rb = person_resamples(persons[ib], n_boot, seed + 1)
    boots = np.array([stats(ia[i]) - stats(ib[j]) for i, j in zip(ra, rb)])
    out: dict[str, Any] = {'a': str(a), 'b': str(b), 'n_persons_a': int(len(np.unique(persons[ia]))),
                           'n_persons_b': int(len(np.unique(persons[ib])))}
    for j, k in enumerate(('slope', 'citl', 'log_loss')):
        v = boots[:, j]
        out[k] = {'diff': float(point[j]), 'ci': [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
                  'p': float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0))))}
    return out


def natural_calibration(nat: pd.DataFrame, prim: pd.DataFrame) -> dict[str, Any]:
    """Calibration-in-the-large on the free-choice trials (every reported hand so far is the right one, so only the
    mean P of the reported hand can be checked): the mean P(reported hand) on the free-choice trials against the mean
    P(right) of the SAME persons' instructed right-hand trials (per-person difference, sign-flip p over persons) and
    against the observed share (1.0 when every free-choice trial is a right-hand one)."""
    e = nat.copy()
    e['p_rep'] = np.where(e['y'] == 1, e['p_right'], 1 - e['p_right'])
    rows = []
    for pid, g in e.groupby('person_id'):
        same = prim[(prim['person_id'] == pid) & (prim['y'].isin(g['y'].unique()))]
        if not len(same):
            continue
        ps = np.where(same['y'] == 1, same['p_right'], 1 - same['p_right'])
        rows.append({'person_id': pid, 'nat': float(g['p_rep'].mean()), 'primary': float(np.mean(ps))})
    df = pd.DataFrame(rows)
    d = (df['nat'] - df['primary']).to_numpy()
    rng = np.random.default_rng([C.SEED, 61])
    boots = [float(d[rng.integers(0, d.size, d.size)].mean()) for _ in range(C.N_BOOT)]
    return {'n': int(len(e)), 'mean_p_reported': float(e['p_rep'].mean()), 'observed_share': 1.0,
            'mean_p_same_persons_primary': float(df['primary'].mean()), 'diff': float(d.mean()),
            'diff_ci': [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            'p': float(cv._sign_flip_p(d)), 'n_persons': int(d.size)}


# ───────────────────────────── multi-attempt aggregates ─────────────────────────────

AGG_KS: tuple[int, ...] = (1, 2, 3, 4, 5)
AGG_RULES: tuple[str, ...] = ('mean_p', 'mean_logit', 'sum_logit', 'sum_tau')


def k_subsets(z: np.ndarray, persons: np.ndarray, y: np.ndarray, k: int, n_draws: int, seed: int,
              cells: set[tuple[str, int]] | None = None) -> pd.DataFrame:
    """Random k-subsets of the trials of one person and hand: per draw the sum of logits, the mean P and the cell."""
    rng = np.random.default_rng([seed, 59, k])
    df = pd.DataFrame({'person': np.asarray(persons).astype(str), 'y': np.asarray(y).astype(int), 'z': z})
    rows = []
    for (pid, yy), g in df.groupby(['person', 'y'], sort=True):
        if cells is not None and (pid, int(yy)) not in cells:
            continue
        zz = g['z'].to_numpy()
        if zz.size < k:
            continue
        import itertools
        if math.comb(zz.size, k) <= n_draws:
            subs = np.array(list(itertools.combinations(range(zz.size), k)))
        else:
            subs = np.array([rng.choice(zz.size, size=k, replace=False) for _ in range(n_draws)])
        zs = zz[subs]
        rows.append(pd.DataFrame({'person': pid, 'y': int(yy), 'sum_z': zs.sum(1),
                                  'mean_p': sigmoid(zs).mean(1), 'k': k}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['person', 'y', 'sum_z', 'mean_p', 'k'])


def fit_tau(sum_z: np.ndarray, y: np.ndarray) -> float:
    """Temperature tau of the summed logits minimising the log loss of sigmoid(sum / tau) (bounded search)."""
    from scipy.optimize import minimize_scalar
    if sum_z.size == 0:
        return 1.0
    r = minimize_scalar(lambda lt: log_loss(y, sigmoid(sum_z / math.exp(lt))), bounds=(-3.0, 3.0), method='bounded',
                        options={'xatol': 1e-6})
    return float(math.exp(r.x))


__all__ = ['fit_calibrator', 'choose_calibrator', 'AUTO_METHODS', 'N_PARAMS', 'ece_null', 'slope_intercept', 'reliability_bands', 'paired_metric', 'paired_ece',
           'ece_basic_ci', 'calibration_block', 'k_subsets', 'fit_tau', 'sigmoid', 'logit', 'log_loss', 'll_terms',
           'person_resamples', 'AGG_KS', 'AGG_RULES']
