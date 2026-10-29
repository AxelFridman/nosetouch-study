"""numpy / scipy values for app/src/predict/pipeline/numeric.test.ts (synthetic inputs only).

Run from analysis/:
    PYTHONPATH=. .venv/Scripts/python.exe ../app/src/predict/pipeline/testdata/make_numpy_reference.py
        > ../app/src/predict/pipeline/testdata/numpy_reference.json
Generated with numpy 2.4.6 and scipy 1.17.1.
"""
import json, math
import numpy as np
from motionlab.preprocess import fill_gaps_linear, lowpass
out = {}
xp = np.array([0.0, 1.0, 2.5, 4.0]); fp = np.array([1.0, 3.0, -2.0, 0.1])
xs = np.array([-1.0, 0.0, 0.3, 1.0, 1.7, 2.5, 3.99, 4.0, 5.0, np.nan])
out['interp'] = {'xp': xp.tolist(), 'fp': fp.tolist(), 'x': [None if math.isnan(v) else v for v in xs],
                 'y': [None if math.isnan(v) else v for v in np.interp(xs, xp, fp)]}
out['interp1'] = {'xp': [2.0], 'fp': [7.5], 'x': [1.0, 2.0, 3.0], 'y': np.interp([1.0, 2.0, 3.0], [2.0], [7.5]).tolist()}
out['linspace'] = [{'a': a, 'b': b, 'n': n, 'y': np.linspace(a, b, n).tolist()} for a, b, n in
                   [(1438.9042375187205, 2105.570904185387, 64), (0.1, 0.7, 64), (-3.0, 5.5, 7), (2.0, 3.0, 1)]]
rng = np.random.default_rng(7)
cases = []
for n in (3, 4, 9, 16, 17, 40, 100):
    x = np.cumsum(rng.normal(size=n)) * 3 + 9.8
    cases.append({'x': x.tolist(), 'y': lowpass(x, 60.0, 15.0, order=4).tolist()})
out['lowpass'] = cases
# a column with NaN runs: lengths 2, 5, 20
x = np.cumsum(rng.normal(size=30))
x[2] = np.nan; x[8] = np.nan; x[9] = np.nan
out['lowpass_runs'] = {'x': [None if not np.isfinite(v) else v for v in x],
                       'y': [None if not np.isfinite(v) else v for v in lowpass(x, 60.0, 15.0, order=4)]}
v = rng.normal(size=(40, 3)) * 9.8
out['norm_dot'] = {'v': v.tolist(), 'n': [float(np.linalg.norm(r)) for r in v]}
t = np.arange(12) * 10.0
X = rng.normal(size=(12, 3)); X[0, 0] = np.nan; X[5:7, 1] = np.nan; X[11, 2] = np.nan
out['fill'] = {'t': t.tolist(), 'x': [[None if not np.isfinite(c) else c for c in r] for r in X],
               'y': [[None if not np.isfinite(c) else c for c in r] for r in fill_gaps_linear(t, X)]}
print(json.dumps(out))
