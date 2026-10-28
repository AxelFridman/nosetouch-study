"""Lane m2 finalists, importable by the synthesis step (``results/finalists_m2.json`` names these functions).

    from stage2.lanes import m2_finalists as F
    X = F.build_x(d)                                  # per-trial features of a Stage2Data, no label
    fp = F.factory(model='lr_l2', mirror='aug')       # -> fit_predict(fold) for cv.run_experiment
    res = run_experiment('m2_lr_full_again', fp, X, d.trials, extra=(F.build_x(nat), nat.trials))

Both run as-is from ``analysis/`` and are deterministic (seeded with ``fold.seed``). The mirror map of the
features is computed from the feature registry (no data), so ``factory`` takes JSON-able parameters only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from stage2.lanes import m2_features as F
from stage2.lanes import m2_models as M


def build_x(d: Any, **spec_kw: Any) -> np.ndarray:
    """Features of the default spec (m2_features.DEFAULT_SPEC: approach window, every channel and family): per trial, no label."""
    return F.build_features(d, F.DEFAULT_SPEC.with_(**spec_kw) if spec_kw else F.DEFAULT_SPEC)[0]


def _mirror(spec_kw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    reg = F.registry(F.DEFAULT_SPEC.with_(**spec_kw) if spec_kw else F.DEFAULT_SPEC)
    return F.mirror_map(reg)


def factory(model: str = 'lr_l2', mirror: str = 'aug', scaler: str = 'z', winsor: float | None = None,
            calibrator: str = 'none', n_trials: int | None = None, n_splits: int = 3, spec: dict[str, Any] | None = None,
            **_: Any) -> Any:
    """fit_predict of one m2 pipeline on ``build_x`` features (``spec``: FeatureSpec overrides, default none)."""
    mi, ms = _mirror(spec or {})
    return M.make_fit_predict(model, mirror=mirror, scaler=scaler, winsor=winsor, calibrator=calibrator,
                              n_trials=n_trials, mirror_index=mi, mirror_sign=ms, n_splits=n_splits)


def factory_ensemble(components: tuple[str, ...] | list[str] = ('lr_l2', 'svm_rbf', 'lgbm'), method: str = 'stack',
                     mirror: str = 'aug', n_trials: dict[str, int] | None = None, n_splits: int = 3, **_: Any) -> Any:
    """fit_predict of the soft-vote / stacked ensemble of components tuned inside the fold."""
    mi, ms = _mirror({})
    return M.make_ensemble_fit_predict(tuple(components), method=method, mirror=mirror, mirror_index=mi, mirror_sign=ms,
                                       n_trials=n_trials, n_splits=n_splits)


def factory_select(candidates: tuple[str, ...] | list[str] = ('lr_l2', 'svm_rbf', 'lgbm', 'ens_soft'),
                   **_: Any) -> Any:
    """fit_predict of the nested model selection (inner log loss) among the candidates."""
    from stage2.lanes import m2_pipeline as P
    mi, ms = _mirror({})
    return P.make_select_fp(tuple(candidates), mi, ms)
