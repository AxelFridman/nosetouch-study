import { describe, expect, it } from 'vitest';
import { DEFAULT_SEGMENTATION_PARAMS } from '../src/core/constants';
import {
  MotionSignals,
  OnlineSegmenter,
  runOnlineSegmentation,
  type RelMotionSample,
  type SegmenterEvent,
} from '../src/core/segmentation';
import type { SegmentationParams, Speed } from '../src/core/types';
import { allNumbersFinite, segParamsForSpeed } from './helpers/segFixtures';
import { synthSamples, type SegScenario } from './helpers/segSignals';

const P = (over: Partial<SegmentationParams> = {}): SegmentationParams => ({ ...DEFAULT_SEGMENTATION_PARAMS, ...over });

/** τ = 0 parameters with round numbers, so every smoothed value equals the raw one. */
const EXACT = P({
  smoothing_ms: 0,
  still_gyro_dps: 15,
  still_acc_ms2: 0.35,
  still_hold_ms: 100,
  onset_k_sd: 6,
  onset_gyro_min_dps: 30,
  onset_acc_min_ms2: 0.6,
  onset_sustain_ms: 20,
  end_gyro_dps: 20,
  end_acc_ms2: 0.45,
  end_sustain_ms: 50,
  min_movement_ms: 30,
  max_wait_ms: 1000,
  max_movement_ms: 2000,
});

/** Sample i at t = 10·i ms with |ω| = gyro (about one axis) and |a| = acc. */
function S(i: number, gyro: number, acc = 0, extra: Partial<RelMotionSample> = {}): RelMotionSample {
  return {
    t: 10 * i,
    ts: null,
    acc: [0, 0, acc],
    accg: [0, 9.8, acc],
    rr: { alpha: 0, beta: 0, gamma: gyro },
    interval: 16,
    index: i,
    ...extra,
  };
}

function feed(seg: OnlineSegmenter, samples: RelMotionSample[], goAt?: number): SegmenterEvent[] {
  const out: SegmenterEvent[] = [];
  for (const s of samples) {
    out.push(...seg.push(s));
    if (goAt !== undefined && s.t === goAt) seg.markGo(goAt);
  }
  return out;
}

/** gyro per index for the canonical exact scenario (see the first state-machine test). */
function canonicalGyro(i: number): number {
  if (i <= 15) return i % 2 === 0 ? 4 : 6;
  if (i <= 19) return 10;
  const ramp: Record<number, number> = { 20: 20, 21: 40, 22: 50, 23: 60, 24: 60, 25: 70, 26: 60, 27: 50, 28: 40, 29: 30 };
  if (i in ramp) return ramp[i];
  if (i <= 35) return 5;
  return 100; // after the end: must be ignored
}

describe('online_v1 signals (spec 1)', () => {
  it('smooths |ω| and |a| with S_0 = x_0 and k_i = 1 − exp(−dt_i/τ); missing rr/acc reuse the previous raw value', () => {
    const seg = new OnlineSegmenter(P({ smoothing_ms: 40 }));
    seg.push({ ...S(0, 0), rr: { alpha: 3, beta: 4, gamma: 0 }, acc: [0, 0, 2] });
    expect(seg.live.gyro).toBe(5);
    expect(seg.live.acc).toBe(2);
    seg.push({ ...S(1, 12, 0), acc: [1, 2, 2] }); // |a| = 3
    const k1 = 1 - Math.exp(-10 / 40);
    const g1 = 5 + k1 * (12 - 5);
    const a1 = 2 + k1 * (3 - 2);
    expect(seg.live.gyro).toBe(g1);
    expect(seg.live.acc).toBe(a1);
    seg.push({ ...S(2, 0), t: 25, rr: null, acc: [Number.NaN, 0, 0] }); // both missing → reuse 12 and 3
    const k2 = 1 - Math.exp(-15 / 40);
    expect(seg.live.gyro).toBe(g1 + k2 * (12 - g1));
    expect(seg.live.acc).toBe(a1 + k2 * (3 - a1));
    expect(seg.result().acc_source).toBe('linear');
  });

  it('uses 0 for gyro until the first finite rotation rate, and τ = 0 disables smoothing', () => {
    const seg = new OnlineSegmenter(P({ smoothing_ms: 0 }));
    seg.push({ ...S(0, 0), rr: null });
    expect(seg.live.gyro).toBe(0);
    seg.push({ ...S(1, 0), rr: { alpha: 2, beta: 3, gamma: 6 } });
    expect(seg.live.gyro).toBe(7);
    seg.push({ ...S(2, 0), rr: { alpha: 1, beta: Number.POSITIVE_INFINITY, gamma: 0 } });
    expect(seg.live.gyro).toBe(7);
  });

  it("'highpass': G starts at accg_0, is updated BEFORE subtracting, and holds when accg is missing", () => {
    const seg = new OnlineSegmenter(P({ smoothing_ms: 0, gravity_tau_ms: 600 }));
    seg.push({ ...S(0, 0), acc: null, accg: [0, 9.8, 0] });
    expect(seg.result().acc_source).toBe('highpass');
    expect(seg.live.acc).toBe(0);
    seg.push({ ...S(1, 0), acc: null, accg: [1, 9.8, 0] });
    const gx1 = 0 + (1 - Math.exp(-10 / 600)) * (1 - 0);
    expect(seg.live.acc).toBe(Math.sqrt((1 - gx1) * (1 - gx1)));
    seg.push({ ...S(2, 0), acc: null, accg: null }); // reuse previous a, G unchanged
    expect(seg.live.acc).toBe(Math.sqrt((1 - gx1) * (1 - gx1)));
    seg.push({ ...S(3, 0), t: 35, acc: null, accg: [1, 9.8, 0] }); // dt = 15 to the previous sample
    const gx3 = gx1 + (1 - Math.exp(-15 / 600)) * (1 - gx1);
    expect(seg.live.acc).toBe(Math.sqrt((1 - gx3) * (1 - gx3)));
  });

  it('acc_source is decided by the FIRST sample only', () => {
    const hp = new OnlineSegmenter(P({ smoothing_ms: 0 }));
    hp.push({ ...S(0, 0), acc: [0, 0, Number.NaN], accg: [0, 9.8, 0] });
    hp.push({ ...S(1, 0), acc: [0, 0, 5], accg: [0, 9.8, 0] });
    expect(hp.result().acc_source).toBe('highpass');
    expect(hp.live.acc).toBe(0); // the later linear acc is ignored

    const none = new OnlineSegmenter(P({ smoothing_ms: 0 }));
    none.push({ ...S(0, 0), acc: null, accg: null });
    none.push({ ...S(1, 0), acc: [0, 0, 5], accg: [0, 9.8, 5] });
    expect(none.result().acc_source).toBe('none');
    expect(none.live.acc).toBe(0);

    expect(new OnlineSegmenter(P()).result().acc_source).toBe('none'); // before any sample
  });

  it('MotionSignals gives the segmenter\'s smoothed values sample by sample, and keeps going after the end', () => {
    const scenarios: SegScenario[] = [
      { seed: 3, durationMs: 5000, moves: [{ startMs: 1200, durMs: 900, distM: 0.45, angleDeg: 60 }] },
      { seed: 4, durationMs: 5000, acc: 'null', moves: [{ startMs: 1200, durMs: 900, distM: 0.45, angleDeg: 60 }] }, // highpass
      { seed: 5, durationMs: 5000, rr: 'intermittent', jitterMs: 4, dropProb: 0.1, moves: [{ startMs: 1500, durMs: 700, distM: 0.4, angleDeg: 50 }] },
      { seed: 6, durationMs: 2000, acc: 'null', accg: 'null' }, // acc_source 'none'
    ];
    for (const sc of scenarios) {
      const samples = synthSamples(sc);
      const params = P();
      const seg = new OnlineSegmenter(params);
      const sig = new MotionSignals(params);
      let goT: number | null = null;
      let afterEnd = 0;
      for (const s of samples) {
        const ended = seg.state === 'ended' || seg.state === 'failed';
        const events = seg.push(s);
        const v = sig.push(s);
        expect(v).toEqual({ gyro: sig.gyro, acc: sig.acc });
        if (!ended) {
          expect(sig.gyro).toBe(seg.live.gyro);
          expect(sig.acc).toBe(seg.live.acc);
        } else {
          afterEnd++;
        }
        if (events.some((e) => e.type === 'still') && goT === null) {
          goT = s.t + 500;
        }
        if (goT !== null && seg.state === 'still' && s.t >= goT) seg.markGo(s.t);
      }
      if (sc.moves) {
        expect(seg.result().end_index, `seed ${sc.seed}`).not.toBeNull();
        expect(afterEnd, `seed ${sc.seed}`).toBeGreaterThan(30);
        expect(sig.gyro).not.toBe(seg.live.gyro); // the segmenter's live values froze at the end
      }
    }
    // a non-finite t changes nothing
    const sig = new MotionSignals(P());
    sig.push(S(0, 10, 1));
    expect(sig.push({ ...S(1, 50, 5), t: Number.NaN })).toEqual({ gyro: 10, acc: 1 });
  });
});

describe('online_v1 state machine (exact, τ = 0)', () => {
  it('still → GO → back-dated onset → back-dated end, with the baseline over the still buffer', () => {
    const seg = new OnlineSegmenter(EXACT);
    const samples = Array.from({ length: 41 }, (_, i) => S(i, canonicalGyro(i)));
    const events = feed(seg, samples, 150);
    expect(events).toEqual([
      { type: 'still', t: 100 },
      { type: 'onset', t: 200, index: 20 },
      { type: 'end', t: 300, index: 30, detected_t: 350 },
    ]);
    expect(seg.state).toBe('ended');
    const r = seg.result();
    expect(r).toEqual({
      method: 'online_v1',
      params: EXACT,
      acc_source: 'linear',
      onset_index: 20,
      end_index: 30,
      duration_ms: 100,
      peak_gyro_dps: 70,
      peak_acc_ms2: 0,
      // buffer = samples 0..15 (the GO sample included): gyro 4,6,4,6,… → mean 5, population sd 1
      baseline: { n: 16, gyro_mean: 5, gyro_sd: 1, acc_mean: 0, acc_sd: 0, gravity: [0, 9.8, 0] },
    });
    expect(seg.thresholds).toEqual({ gyro: 30, acc: 0.6 });
    expect(seg.goTime).toBe(150);
  });

  it('the adaptive threshold uses baseline_mean + k·sd when it exceeds the floor', () => {
    // gyro alternates 2 / 12 while still (mean 7, sd 5) → thr = max(30, 7 + 6·5) = 37
    const seg = new OnlineSegmenter(EXACT);
    const s = Array.from({ length: 16 }, (_, i) => S(i, i % 2 === 0 ? 2 : 12));
    feed(seg, s, 150);
    expect(seg.thresholds?.gyro).toBe(37);
    expect(feed(seg, [S(16, 36), S(17, 36), S(18, 36), S(19, 36)])).toEqual([]); // 36 < 37: no onset
    expect(seg.state).toBe('armed');
    expect(feed(seg, [S(20, 38), S(21, 38), S(22, 38)])).toEqual([{ type: 'onset', t: 160, index: 16 }]);
  });

  it("emits 'still_lost' below the onset floors and 'false_start' above them; the still run restarts", () => {
    const seg = new OnlineSegmenter(EXACT);
    const gyro = (i: number): number => (i === 11 ? 20 : i === 23 ? 35 : 5);
    const progress: number[] = [];
    const events: SegmenterEvent[] = [];
    for (let i = 0; i <= 34; i++) {
      events.push(...seg.push(S(i, gyro(i))));
      progress[i] = seg.live.stillProgress;
    }
    expect(events).toEqual([
      { type: 'still', t: 100 },
      { type: 'still_lost', t: 110 },
      { type: 'still', t: 220 },
      { type: 'false_start', t: 230 },
      { type: 'still', t: 340 },
    ]);
    expect(progress[0]).toBe(0);
    expect(progress[5]).toBe(0.5);
    expect(progress[10]).toBe(1);
    expect(progress[11]).toBe(0);
    expect(progress[12]).toBe(0); // new run starts here
    expect(progress[17]).toBe(0.5);
    expect(progress[23]).toBe(0);
    expect(seg.state).toBe('still');
  });

  it("an acceleration above onset_acc_min_ms2 is a false start; between the still and onset limits it is 'still_lost'", () => {
    const a = new OnlineSegmenter(EXACT);
    feed(a, Array.from({ length: 11 }, (_, i) => S(i, 5)));
    expect(a.push(S(11, 5, 0.7))).toEqual([{ type: 'false_start', t: 110 }]);
    const b = new OnlineSegmenter(EXACT);
    feed(b, Array.from({ length: 11 }, (_, i) => S(i, 5)));
    expect(b.push(S(11, 5, 0.5))).toEqual([{ type: 'still_lost', t: 110 }]);
  });

  it('confirms an onset before checking max_wait on the same sample', () => {
    const p = { ...EXACT, max_wait_ms: 95 };
    const onsetFirst = new OnlineSegmenter(p);
    const g1 = (i: number): number => (i >= 23 ? 40 : 5);
    const ev1 = feed(onsetFirst, Array.from({ length: 26 }, (_, i) => S(i, g1(i))), 150);
    // t = 250: 250 − 150 > 95, but the C_on run 230..250 reaches 20 ms on this very sample
    expect(ev1.slice(1)).toEqual([{ type: 'onset', t: 230, index: 23 }]);

    const late = new OnlineSegmenter(p);
    const g2 = (i: number): number => (i >= 24 ? 40 : 5);
    const ev2 = feed(late, Array.from({ length: 27 }, (_, i) => S(i, g2(i))), 150);
    expect(ev2.slice(1)).toEqual([{ type: 'no_movement', t: 250 }]);
    expect(late.state).toBe('failed');
    expect(late.result()).toMatchObject({ onset_index: null, end_index: null, duration_ms: null, peak_gyro_dps: null });
  });

  it("'no_movement' fires at the first sample with t − go_t > max_wait_ms", () => {
    const seg = new OnlineSegmenter(EXACT);
    const ev = feed(seg, Array.from({ length: 130 }, (_, i) => S(i, 5)), 150);
    expect(ev).toEqual([{ type: 'still', t: 100 }, { type: 'no_movement', t: 1160 }]);
  });

  it("'too_long' fires at the first sample with t − t_onset > max_movement_ms", () => {
    const seg = new OnlineSegmenter(EXACT);
    const ev = feed(seg, Array.from({ length: 240 }, (_, i) => S(i, i <= 19 ? 5 : 60)), 150);
    expect(ev).toEqual([
      { type: 'still', t: 100 },
      { type: 'onset', t: 200, index: 20 },
      { type: 'too_long', t: 2210 },
    ]);
    expect(seg.result()).toMatchObject({ onset_index: 20, end_index: null, duration_ms: null, peak_gyro_dps: 60 });
  });

  it('the end needs min_movement_ms after the onset and end_sustain_ms of continuous C_end', () => {
    const seg = new OnlineSegmenter({ ...EXACT, onset_sustain_ms: 0 });
    const g = (i: number): number => (i <= 19 ? 5 : i === 20 ? 20 : i === 21 ? 40 : i === 27 ? 25 : 5);
    const ev = feed(seg, Array.from({ length: 40 }, (_, i) => S(i, g(i))), 150);
    // onset confirmed at t = 210 (sustain 0) and back-dated to 200. C_end needs t ≥ 230; the run
    // 230..260 is broken at 270 (25 dps), restarts at 280 and is confirmed at 330.
    expect(ev).toEqual([
      { type: 'still', t: 100 },
      { type: 'onset', t: 200, index: 20 },
      { type: 'end', t: 280, index: 28, detected_t: 330 },
    ]);
    expect(seg.result().peak_gyro_dps).toBe(40);
  });

  it('starts the C_end run at the sample AFTER the onset confirmation (one event per sample)', () => {
    // C_on via acc (0.8 > 0.6) and C_end (gyro < 20, acc < 1.0) are both true on the same samples.
    const seg = new OnlineSegmenter({ ...EXACT, onset_sustain_ms: 0, end_sustain_ms: 0, end_acc_ms2: 1, min_movement_ms: 0 });
    feed(seg, Array.from({ length: 16 }, (_, i) => S(i, 5)), 150);
    expect(seg.push(S(16, 5, 0.8))).toEqual([{ type: 'onset', t: 160, index: 16 }]);
    expect(seg.push(S(17, 5, 0.8))).toEqual([{ type: 'end', t: 170, index: 17, detected_t: 170 }]);
  });

  it('ignores every sample after ENDED or FAILED, and markGo outside STILL', () => {
    const seg = new OnlineSegmenter(EXACT);
    feed(seg, Array.from({ length: 41 }, (_, i) => S(i, canonicalGyro(i))), 150);
    const before = seg.result();
    const liveBefore = seg.live;
    for (let i = 41; i < 300; i++) expect(seg.push(S(i, i % 7 === 0 ? 300 : 3, i % 5 === 0 ? 4 : 0))).toEqual([]);
    seg.markGo(5000);
    expect(seg.state).toBe('ended');
    expect(seg.result()).toEqual(before);
    expect(seg.live).toEqual(liveBefore);

    const failed = new OnlineSegmenter(EXACT);
    feed(failed, Array.from({ length: 130 }, (_, i) => S(i, 5)), 150);
    expect(failed.state).toBe('failed');
    expect(failed.push(S(130, 200))).toEqual([]);

    const waiting = new OnlineSegmenter(EXACT);
    waiting.push(S(0, 50));
    waiting.markGo(0);
    expect(waiting.state).toBe('waiting_still');
    expect(waiting.baseline).toBeNull();
    expect(waiting.result().baseline).toBeNull();

    const armed = new OnlineSegmenter(EXACT);
    feed(armed, Array.from({ length: 16 }, (_, i) => S(i, 5)), 150);
    armed.markGo(Number.NaN);
    armed.markGo(9999); // already armed: ignored
    expect(armed.goTime).toBe(150);
  });

  it('back-dates the onset to just after the last still sample, even when stillness flickers after GO', () => {
    // After GO the signal leaves the still band (15) twice before the sustained C_on run.
    const g = (i: number): number => {
      if (i <= 15) return 5;
      const tail: Record<number, number> = { 16: 18, 17: 8, 18: 16, 19: 22, 20: 25, 21: 31, 22: 33, 23: 35 };
      return tail[i] ?? 40;
    };
    const seg = new OnlineSegmenter(EXACT);
    const ev = feed(seg, Array.from({ length: 25 }, (_, i) => S(i, g(i))), 150);
    // j0 = 21 (first C_on), confirmed at 23; the largest k ≤ 20 with C_still is 17 → onset 18
    expect(ev).toContainEqual({ type: 'onset', t: 180, index: 18 });
  });

  it('reports `index` from the samples (not the push position) and survives out-of-spec input', () => {
    const seg = new OnlineSegmenter(EXACT);
    const shifted = Array.from({ length: 41 }, (_, i) => ({ ...S(i, canonicalGyro(i)), index: 1000 + i }));
    feed(seg, shifted, 150);
    expect(seg.result()).toMatchObject({ onset_index: 1020, end_index: 1030 });

    const odd = new OnlineSegmenter(EXACT);
    expect(odd.push({ ...S(0, 5), t: Number.NaN })).toEqual([]);
    odd.push(S(1, 5));
    odd.push({ ...S(2, 5), t: 5 }); // goes back in time: dt treated as 0
    expect(allNumbersFinite(odd.live)).toEqual([]);
  });
});

describe('runOnlineSegmentation', () => {
  const samples = Array.from({ length: 41 }, (_, i) => S(i, canonicalGyro(i)));

  it('with an explicit goT gives the same result as driving the segmenter by hand', () => {
    const manual = new OnlineSegmenter(EXACT);
    const events = feed(manual, samples, 150);
    const run = runOnlineSegmentation(samples, EXACT, 150);
    expect(run.events).toEqual(events);
    expect(run.result).toEqual(manual.result());
    expect(run.goT).toBe(150);
  });

  it('applies an explicit goT at the first sample with t ≥ goT (with that sample’s t), once, and only in STILL', () => {
    const between = runOnlineSegmentation(samples, EXACT, 145); // first t ≥ 145 is 150
    expect(between.goT).toBe(150);
    expect(between.result).toEqual(runOnlineSegmentation(samples, EXACT, 150).result);
    expect(between.result.baseline?.n).toBe(16);
    const tooEarly = runOnlineSegmentation(samples, EXACT, 50); // still not reached at t = 50
    expect(tooEarly.goT).toBeNull();
    expect(tooEarly.result.onset_index).toBeNull();
    expect(tooEarly.result.baseline).toBeNull();
    expect(runOnlineSegmentation(samples, EXACT, 99999).goT).toBeNull();
  });

  it('gives GO automatically foreperiodMs after the still event (default foreperiod_min_ms)', () => {
    const auto = runOnlineSegmentation(samples, { ...EXACT, foreperiod_min_ms: 50 }, null);
    expect(auto.goT).toBe(150);
    expect(auto.result.onset_index).toBe(20);
    const custom = runOnlineSegmentation(samples, EXACT, null, 30);
    expect(custom.goT).toBe(130);
    const zero = runOnlineSegmentation(samples, EXACT, null, 0);
    expect(zero.goT).toBe(100); // the 'still' sample itself
  });

  it('a false start cancels the automatic GO; the next still event schedules it again', () => {
    const g = (i: number): number => (i === 13 ? 40 : i >= 40 && i <= 45 ? 60 : 5);
    const s = Array.from({ length: 70 }, (_, i) => S(i, g(i)));
    const run = runOnlineSegmentation(s, EXACT, null, 50);
    expect(run.events.slice(0, 3)).toEqual([
      { type: 'still', t: 100 },
      { type: 'false_start', t: 130 },
      { type: 'still', t: 240 },
    ]);
    expect(run.goT).toBe(290);
    expect(run.result.onset_index).toBe(40);
  });
});

// ─────────────────────────── synthetic minimum-jerk movements ───────────────────────────

const GO_T = 1100;
const REACTION = 250;
const TRUE_ONSET = GO_T + REACTION;

interface Accuracy {
  onsetErr: number[];
  endErr: number[];
  failures: string[];
}

function runSynth(sc: Omit<SegScenario, 'seed' | 'durationMs'>, params: SegmentationParams, seeds: number, T: number): Accuracy {
  const acc: Accuracy = { onsetErr: [], endErr: [], failures: [] };
  for (let seed = 1; seed <= seeds; seed++) {
    const samples = synthSamples({
      seed,
      durationMs: TRUE_ONSET + T + 2500,
      moves: [{ startMs: TRUE_ONSET, durMs: T, distM: 0.45, angleDeg: 60 }],
      ...sc,
    });
    const run = runOnlineSegmentation(samples, params, GO_T);
    const ends = run.events.filter((e) => e.type === 'end');
    const onset = run.events.find((e) => e.type === 'onset');
    if (run.goT === null || !onset || ends.length !== 1) {
      acc.failures.push(`seed ${seed}: ${JSON.stringify(run.events.map((e) => e.type))}`);
      continue;
    }
    expect(allNumbersFinite(run.result)).toEqual([]);
    const r = run.result;
    expect(samples[r.onset_index as number].t).toBe(onset.t);
    acc.onsetErr.push(onset.t - TRUE_ONSET);
    acc.endErr.push(samples[r.end_index as number].t - (TRUE_ONSET + T));
  }
  return acc;
}

function summary(xs: number[]): { min: number; mean: number; max: number } {
  return { min: Math.min(...xs), mean: xs.reduce((a, b) => a + b, 0) / xs.length, max: Math.max(...xs) };
}

const report: string[] = [];
function log(label: string, a: Accuracy): void {
  const o = summary(a.onsetErr);
  const e = summary(a.endErr);
  report.push(
    `${label.padEnd(34)} onset err ms min/mean/max ${o.min.toFixed(0)}/${o.mean.toFixed(0)}/${o.max.toFixed(0)}` +
      `   end err ${e.min.toFixed(0)}/${e.mean.toFixed(0)}/${e.max.toFixed(0)}   (n=${a.onsetErr.length})`,
  );
}

describe('online_v1 on synthetic minimum-jerk movements (60 Hz, noise + tremor)', () => {
  const cases: Array<{ T: number; speed: Speed }> = [
    { T: 500, speed: 'fast' },
    { T: 1000, speed: 'natural' },
    { T: 2500, speed: 'slow' },
  ];
  for (const { T, speed } of cases) {
    it(`T = ${T} ms (${speed} parameters): onset and end close to the true times`, () => {
      const a = runSynth({}, segParamsForSpeed(speed), 25, T);
      log(`T=${T} ${speed}`, a);
      expect(a.failures).toEqual([]);
      // Protocol 2.0.0: on a slow ramp the looser still thresholds (20 / 0.5) stop the back-dating
      // later and the slow onset floors are higher (25 / 0.55), so slow onsets come ~280 ms later than
      // with 1.0.0 (median 567 vs 283 ms here; reported in app/scripts/out/segmentation_eval.md).
      const onsetBound = speed === 'slow' ? 0.25 * T + 60 : 0.2 * T + 60;
      for (const x of a.onsetErr) expect(x).toBeGreaterThanOrEqual(0);
      for (const x of a.onsetErr) expect(x).toBeLessThanOrEqual(onsetBound);
      for (const x of a.endErr) expect(Math.abs(x)).toBeLessThanOrEqual(0.12 * T + 80);
    });
  }

  it('T = 2500 ms with the DEFAULT parameters still segments (later onset, earlier end)', () => {
    const a = runSynth({}, P(), 25, 2500);
    log('T=2500 default params', a);
    expect(a.failures).toEqual([]);
  });

  it("acc null → 'highpass' (gravity removed by the EMA high-pass) still finds onset and end", () => {
    const samples = synthSamples({
      seed: 3,
      durationMs: 6000,
      acc: 'null',
      moves: [{ startMs: TRUE_ONSET, durMs: 1000, distM: 0.45, angleDeg: 60 }],
    });
    const run = runOnlineSegmentation(samples, P(), GO_T);
    expect(run.result.acc_source).toBe('highpass');
    const a = runSynth({ acc: 'null' }, P(), 25, 1000);
    log('T=1000 acc null (highpass)', a);
    expect(a.failures).toEqual([]);
    for (const x of a.onsetErr) expect(x).toBeLessThanOrEqual(260);
    // the high-pass leaves a decaying gravity residue after a 60° turn: the end comes late
    for (const x of a.endErr) expect(x).toBeLessThanOrEqual(2500);
  });

  it('rr null → the acceleration alone drives the detector', () => {
    const a = runSynth({ rr: 'null' }, P(), 25, 1000);
    log('T=1000 rr null', a);
    expect(a.failures).toEqual([]);
    for (const x of a.onsetErr) expect(x).toBeLessThanOrEqual(260);
    for (const x of a.endErr) expect(Math.abs(x)).toBeLessThanOrEqual(260);
  });

  it('intermittently missing rr (30 %) is bridged by reusing the previous value', () => {
    const a = runSynth({ rr: 'intermittent' }, P(), 25, 1000);
    log('T=1000 rr 30% missing', a);
    expect(a.failures).toEqual([]);
    for (const x of a.onsetErr) expect(x).toBeLessThanOrEqual(260);
    for (const x of a.endErr) expect(Math.abs(x)).toBeLessThanOrEqual(260);
  });

  it('jittered (±5 ms), lossy (10 % drops) and gappy (120 ms holes) sampling', () => {
    const a = runSynth(
      {
        jitterMs: 5,
        dropProb: 0.1,
        gaps: [
          [700, 820],
          [TRUE_ONSET + 400, TRUE_ONSET + 520],
        ],
      },
      P(),
      25,
      1000,
    );
    log('T=1000 jitter+drops+gaps', a);
    expect(a.failures).toEqual([]);
    for (const x of a.onsetErr) expect(x).toBeLessThanOrEqual(260);
    for (const x of a.endErr) expect(Math.abs(x)).toBeLessThanOrEqual(260);
  });

  it('works at 30 Hz and 100 Hz sampling', () => {
    for (const hz of [30, 100]) {
      const a = runSynth({ hz }, P(), 15, 1000);
      log(`T=1000 at ${hz} Hz`, a);
      expect(a.failures).toEqual([]);
      for (const x of a.onsetErr) expect(x).toBeLessThanOrEqual(260);
      for (const x of a.endErr) expect(Math.abs(x)).toBeLessThanOrEqual(260);
    }
  });

  it('a jerk before GO is a false start; GO is rescheduled and the real movement is found', () => {
    for (let seed = 1; seed <= 10; seed++) {
      const samples = synthSamples({
        seed,
        durationMs: 7000,
        moves: [
          { startMs: 700, durMs: 200, distM: 0.05, angleDeg: 25 },
          { startMs: 900, durMs: 200, distM: -0.05, angleDeg: -25 },
          { startMs: 3500, durMs: 1000, distM: 0.45, angleDeg: 60 },
        ],
      });
      const run = runOnlineSegmentation(samples, P(), null, 600);
      const types = run.events.map((e) => e.type);
      expect(types).toEqual(['still', 'false_start', 'still', 'onset', 'end']);
      expect(run.goT).not.toBeNull();
      expect(run.goT as number).toBeGreaterThan(1100);
      expect(run.goT as number).toBeLessThan(3500);
      const onset = run.events.find((e) => e.type === 'onset') as { t: number };
      expect(onset.t - 3500).toBeGreaterThanOrEqual(0);
      expect(onset.t - 3500).toBeLessThanOrEqual(260);
    }
  });

  it("holding still after GO → 'no_movement' at the first sample past max_wait_ms", () => {
    const samples = synthSamples({ seed: 9, durationMs: 9000 });
    const run = runOnlineSegmentation(samples, P(), GO_T);
    expect(run.events.map((e) => e.type)).toEqual(['still', 'no_movement']);
    const nm = run.events[1];
    const first = samples.find((s) => s.t - GO_T > DEFAULT_SEGMENTATION_PARAMS.max_wait_ms);
    expect(nm.t).toBe(first?.t);
    expect(run.result.baseline).not.toBeNull();
  });

  it("a rotation that never stops → 'too_long' max_movement_ms after the onset", () => {
    const samples = synthSamples({ seed: 4, durationMs: 12000, spin: { fromMs: TRUE_ONSET, dps: 70 } });
    const run = runOnlineSegmentation(samples, P(), GO_T);
    expect(run.events.map((e) => e.type)).toEqual(['still', 'onset', 'too_long']);
    const onset = run.events[1] as { t: number };
    const tooLong = run.events[2];
    expect(tooLong.t - onset.t).toBeGreaterThan(DEFAULT_SEGMENTATION_PARAMS.max_movement_ms);
    expect(tooLong.t - onset.t).toBeLessThanOrEqual(DEFAULT_SEGMENTATION_PARAMS.max_movement_ms + 1000 / 60 + 1e-9);
    expect(run.result.end_index).toBeNull();
  });

  it('prints the accuracy table', () => {
    // eslint-disable-next-line no-console
    console.log(`\nonline_v1 accuracy vs ground truth (detected − true):\n${report.join('\n')}\n`);
    expect(report.length).toBeGreaterThan(0);
  });
});
