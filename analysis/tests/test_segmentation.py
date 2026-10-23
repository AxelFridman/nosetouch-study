"""online_v1: parity with the app's TypeScript implementation, known-truth streams, the literal readings
of the spec; offline_v1 against the synthetic ground truth."""
from __future__ import annotations

import math

import numpy as np
import pytest
from _mlhelpers import condition

from motionlab import segment, synth
from motionlab.conventions import DEFAULT_SEGMENTATION_PARAMS

P = dict(DEFAULT_SEGMENTATION_PARAMS)


def _close(a, b, rel=1e-12):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _close(a[k], b[k], rel)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y, rel)
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-15), (a, b)
    else:
        assert a == b, (a, b)


# ───────────────────────────── parity with app/src/core/segmentation.ts ─────────────────────────────


def test_rerun_reproduces_every_ts_record(ts_reference):
    """Records written by the app's TrialController: the port re-runs them to the stored indices."""
    agree = segment.port_agreement(ts_reference['records'])
    assert agree['n'] == len(ts_reference['records']) >= 10
    assert agree['agreement'] == 1.0, agree['mismatches']
    statuses = {r['outcome']['status'] for r in ts_reference['records']}
    assert statuses == {'complete', 'no_movement', 'too_long'}
    assert {r['segmentation']['acc_source'] for r in ts_reference['records']} == {'linear', 'highpass'}
    for rec in ts_reference['records']:
        run = segment.rerun_online_v1(rec, keep_trace=False)
        # floating values agree to the last bits (Math.exp differs between JS engines and libm)
        _close(run.result, rec['segmentation'], rel=1e-12)
        on = rec['segmentation']['onset_index']
        if on is not None:
            assert rec['streams']['motion']['t'][on] == rec['timing']['onset_ms']


def test_scenarios_match_ts_events_exactly(ts_reference):
    """tau = 0, GO between two samples and automatic GO: same events, GO time and indices as the TS."""
    for sc in ts_reference['scenarios']:
        run = segment.run_online_v1(sc['motion'], sc['params'], sc['go_t'], sc['foreperiod_ms'], keep_trace=False)
        ts = sc['ts']
        assert run.events == ts['events'], sc['id']
        assert run.go_t == ts['goT'], sc['id']
        assert (run.result['onset_index'], run.result['end_index']) == (ts['result']['onset_index'], ts['result']['end_index'])
        _close(run.result, ts['result'], rel=1e-12)


# ───────────────────────────── known truth ─────────────────────────────


def _stream(hz=60.0, dur_ms=5000.0, move=(2000.0, 2500.0), rate=100.0, acc_null=False):
    t = np.arange(0.0, dur_ms, 1000.0 / hz)
    moving = (t >= move[0]) & (t < move[1])
    rr = np.zeros((t.size, 3))
    rr[moving, 1] = rate
    acc = np.zeros((t.size, 3))
    accg = np.tile([0.0, 9.80665, 0.0], (t.size, 1))
    m = {'t': t.tolist()}
    for j, a in enumerate('xyz'):
        m[f'acc_{a}'] = [None] * t.size if acc_null else acc[:, j].tolist()
        m[f'accg_{a}'] = accg[:, j].tolist()
    for j, k in enumerate(('alpha', 'beta', 'gamma')):
        m[f'rr_{k}'] = rr[:, j].tolist()
    return m, t, moving


def test_known_truth_step_stream():
    """Without smoothing the onset is the first moving sample and the end the first still one."""
    p = {**P, 'smoothing_ms': 0}
    m, t, moving = _stream()
    run = segment.run_online_v1(m, p, go_t=1000.0)
    first_move = int(np.flatnonzero(moving)[0])
    first_still_after = int(np.flatnonzero(~moving & (t > t[first_move]))[0])
    assert run.status == 'complete'
    assert run.onset_index == first_move
    assert run.end_index == first_still_after
    assert run.result['duration_ms'] == pytest.approx(t[first_still_after] - t[first_move])
    assert run.result['peak_gyro_dps'] == pytest.approx(100.0)
    assert [e['type'] for e in run.events] == ['still', 'onset', 'end']
    end = run.event('end')
    assert end['detected_t'] - end['t'] >= p['end_sustain_ms']
    assert run.result['baseline']['gravity'] == pytest.approx([0.0, 9.80665, 0.0])
    assert run.result['acc_source'] == 'linear'


def test_onset_is_backdated_to_the_last_still_sample():
    """A slow ramp: C_on is reached late, the onset goes back to the first sample that was not still."""
    m, t, _ = _stream(dur_ms=8000)
    rate = np.clip((t - 1100.0) * 0.05, 0.0, 100.0) * (t < 4000.0)  # 0 -> 100 deg/s over 2 s
    m['rr_beta'] = rate.tolist()
    run = segment.run_online_v1(m, {**P, 'smoothing_ms': 0}, go_t=1000.0)
    assert run.status == 'complete'
    assert run.onset_index == int(np.flatnonzero(rate >= P['still_gyro_dps'])[0])
    j0 = int(np.flatnonzero(rate > P['onset_gyro_min_dps'])[0])
    assert t[j0] - t[run.onset_index] > 250  # back-dated by ~300 ms
    # a movement that starts before GO is a false start: GO is not given
    m2, _, _ = _stream(move=(900.0, 1600.0))
    early = segment.run_online_v1(m2, P, go_t=1000.0)
    assert early.status == 'no_go' and early.event('false_start') is not None


def test_no_movement_and_too_long():
    m, _, _ = _stream(move=(1e9, 1e9 + 1), dur_ms=8000)
    assert segment.run_online_v1(m, P, go_t=1000.0).status == 'no_movement'
    m, _, _ = _stream(move=(2000.0, 1e9), dur_ms=12000)
    assert segment.run_online_v1(m, P, go_t=1000.0).status == 'too_long'


def test_highpass_when_acc_missing_and_gyro_gaps():
    m, t, _ = _stream(acc_null=True)
    run = segment.run_online_v1(m, P, go_t=1000.0)
    assert run.result['acc_source'] == 'highpass' and run.status == 'complete'
    m['rr_beta'][130] = None  # a missing rate reuses the previous raw value
    run2 = segment.run_online_v1(m, P, go_t=1000.0)
    assert run2.onset_index == run.onset_index


def test_literal_readings():
    seg = segment.OnlineSegmenterV1({**P, 'smoothing_ms': 0})
    assert seg.mark_go(10.0) is False and seg.state == 'waiting_still'  # reading i: ignored outside 'still'
    assert seg.push(float('nan'), None, [0, 9.8, 0], [0, 0, 0]) == []  # reading j: non-finite t ignored
    assert seg.trace()['t'].size == 0
    seg.push(0.0, None, [0, 9.8, 0], [3.0, 4.0, 0.0], index=7)
    seg.push(-5.0, None, [0, 9.8, 0], [0.0, 0.0, 0.0], index=2.0)  # decreasing t: dt = 0
    seg.push(20.0, None, [0, 9.8, 0], [0.0, 0.0, 0.0], index=-1)
    tr = seg.trace()
    assert tr['gyro_s'][0] == 5.0  # tau = 0: S_i = x_i exactly
    assert list(tr['index']) == [7, 2, 2]  # reading a: own index if a non-negative integer, else position
    assert seg.acc_source == 'highpass'  # decided by the first sample: acc missing, accg present
    assert segment._ema_step(1.0, 2.0, 16.0, 0.0) == 2.0
    assert segment._ema_step(1.0, 2.0, 0.0, 40.0) == 1.0  # equal times: k = 0


def test_automatic_go_after_foreperiod():
    m, t, _ = _stream()
    run = segment.run_online_v1(m, P, go_t=None, foreperiod_ms=0.0)
    still = run.event('still')
    assert run.go_t == still['t']  # foreperiod 0: GO at the 'still' sample itself
    run = segment.run_online_v1(m, P, go_t=None, foreperiod_ms=250.0)
    assert run.go_t == min(x for x in t if x >= run.event('still')['t'] + 250.0)


def test_go_not_given_when_not_still():
    m, _, _ = _stream(move=(0.0, 5000.0))  # never still
    run = segment.run_online_v1(m, P, go_t=500.0)
    assert not run.go_given and run.status == 'no_go'
    assert run.result['baseline'] is None


def test_online_v1_against_synthetic_truth(symmetric_participant):
    # absolute thresholds detect slow movements late (docs/05 section 8.1, segmentation bias)
    late_ms = {'fast': 150, 'natural': 250, 'slow': 800}
    for speed in ('fast', 'natural', 'slow'):
        plat = synth.default_platform('android', np.random.default_rng(2))
        rec, truth = synth.simulate_trial(symmetric_participant, plat, condition(speed=speed), seed=13)
        run = segment.rerun_online_v1(rec)
        assert run.status == 'complete' and run.onset_index == rec['segmentation']['onset_index']
        t = rec['streams']['motion']['t']
        assert truth.onset_ms - 100 <= t[run.onset_index] <= truth.onset_ms + late_ms[speed], speed
        assert t[run.end_index] > t[run.onset_index] + 150
        assert run.go_t == rec['timing']['go_ms']
    for behavior, status in (('no_move', 'no_movement'), ('too_slow', 'too_long')):
        rec, _ = synth.simulate_trial(symmetric_participant, plat, condition(), seed=3, behavior=behavior)
        assert rec['outcome']['status'] == status
        assert segment.rerun_online_v1(rec).status == status


# ───────────────────────────── offline_v1 ─────────────────────────────


def test_offline_v1_against_synthetic_truth(symmetric_participant):
    errs_on, errs_end = [], []
    for k, speed in enumerate(('fast', 'natural', 'natural', 'slow')):
        plat = synth.default_platform('ios' if k % 2 else 'android', np.random.default_rng(k))
        rec, truth = synth.simulate_trial(symmetric_participant, plat, condition(speed=speed), seed=40 + k)
        off = segment.offline_v1_from_record(rec)
        assert off.ok, off.reason
        errs_on.append(off.onset_ms - truth.onset_ms)
        errs_end.append(off.end_ms - truth.end_ms)
        assert off.peak_gyro_dps > 10 and off.duration_ms > 200
    # a %-of-peak threshold cuts a fixed fraction of the duration (largest for the 3 s slow trial)
    assert max(abs(e) for e in errs_on) < 250, errs_on
    assert max(abs(e) for e in errs_end) < 350, errs_end
    rec, _ = synth.simulate_trial(symmetric_participant, plat, condition(), seed=3, behavior='no_move')
    assert not segment.offline_v1_from_record(rec).ok


def test_compare_segmentations_and_bounds(tiny_dataset):
    rows = segment.compare_segmentations(tiny_dataset.trials[:6])
    assert len(rows) == 6 and all(r['all_match'] for r in rows)
    ok = [r for r in rows if r['offline_ok']]
    assert ok and all(abs(r['offline_minus_online_onset_ms']) < 500 for r in ok)
    rec = tiny_dataset.trials[0]
    for method in ('stored', 'online', 'offline', 'auto'):
        on, end, src = segment.segment_bounds_ms(rec, method)
        assert on is not None and end > on, method
