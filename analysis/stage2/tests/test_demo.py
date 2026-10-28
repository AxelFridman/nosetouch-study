"""Lane demo (demographics add-on test): fast checks of the second stage and of the saved outputs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C
from stage2 import cv
from stage2.lanes import demo_models as DM


def _fold(rng: np.random.Generator, garbage: float) -> cv.Fold:
    persons = [f'P{i:02d}' for i in range(1, 9)]
    test = persons[0]
    train = persons[1:]
    n_per = 20
    g = np.repeat(train, n_per)
    y = np.tile(np.r_[np.zeros(n_per // 2), np.ones(n_per // 2)], len(train)).astype(int)
    z = (2 * y - 1) * 1.5 + rng.normal(0, 1, y.size)
    Z = np.full((y.size, len(persons)), garbage)       # other folds' columns: must never be read
    Z[:, persons.index(test)] = z
    cov = rng.normal(0, 1, (len(persons), len(DM.COVARIATES)))
    cov_rows = cov[[persons.index(p) for p in g]]
    n_te = 10
    X_tr = {'inner_logit_by_fold': Z, 'outer_logit': rng.normal(0, 1, y.size), 'person_covariates': cov_rows}
    X_te = {'inner_logit_by_fold': np.full((n_te, len(persons)), np.nan),
            'outer_logit': np.linspace(-3, 3, n_te), 'person_covariates': np.repeat(cov[:1], n_te, axis=0)}
    meta_tr = pd.DataFrame({'person_id': g})
    meta_te = pd.DataFrame({'row': np.arange(n_te), 'person_id': test})
    return cv.Fold(0, test, X_tr, y, g, meta_tr, X_te, meta_te, C.SEED)


def test_second_stage_reads_only_its_own_fold_column_and_the_outer_logit():
    """The stack fits on the inner logits of ITS fold's column only (other columns hold other folds' training rows)
    and predicts the held-out person from the saved outer logit; the covariates-only floor ignores the logit."""
    for which in DM.FEATURE_SETS:
        fp = DM.stack_fit_predict(which, [f'P{i:02d}' for i in range(1, 9)])
        p1, info1 = fp(_fold(np.random.default_rng(0), garbage=0.0))
        p2, info2 = fp(_fold(np.random.default_rng(0), garbage=1e6))
        np.testing.assert_allclose(p1, p2, atol=1e-12)
        assert info1['C'] in DM.C_GRID
        if which == 'cov_only':
            assert np.ptp(p1) < 1e-9                      # one test person: one covariate row, one P
        else:
            assert np.all(np.diff(p1) > 0)                # monotone in the held-out person's logit


def test_saved_outputs_are_labelled_and_finalists_empty():
    from stage2.lanes.demo_pipeline import EXPERIMENTS, L
    if not L.finalists_path.exists():
        pytest.skip('lane demo has not been run')
    assert json.loads(L.finalists_path.read_text(encoding='utf-8')) == []
    for name in EXPERIMENTS:
        r = L.load_result(name)
        notes = json.dumps(r.info, default=str)
        assert 'SANCTIONED DEMOGRAPHICS ADD-ON TEST' in notes, name
    nat = L.load_result('demo_rule_right').extra_predictions
    assert (nat['y'] == 1).all()                          # the base-rate note: every natural-use trial is right-hand
