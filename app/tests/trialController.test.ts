import { describe, expect, it } from 'vitest';
import { DEFAULT_SEGMENTATION_PARAMS, SCHEMA_VERSION } from '../src/core/constants';
import { runOnlineSegmentation } from '../src/core/segmentation';
import { createTrialController, type TrialController } from '../src/core/trialController';
import type {
  CalibrationResult,
  MotionSample,
  SensorListener,
  SensorSource,
  TrialControllerOptions,
  TrialLiveState,
  TrialPhase,
  TrialRecord,
} from '../src/core/types';
import { expectEqualColumnLengths, expectValid } from './helpers/schema';
import {
  SEG_IDS,
  allNumbersFinite,
  segCondition,
  segCueSink,
  segParamsForSpeed,
  storedMotionSamples,
} from './helpers/segFixtures';
import { segRng } from './helpers/segSignals';
import { SEG_BASE_EPOCH, createSegSource, type SegFakeSource, type SegSourceOptions } from './helpers/segSource';

const PARAMS = { ...DEFAULT_SEGMENTATION_PARAMS };
const START = 1000; // virtual clock of the offline / async fake sources at start()

interface Harness {
  source: SegFakeSource;
  cues: ReturnType<typeof segCueSink>;
  ctrl: TrialController;
  lives: TrialLiveState[];
}

/** Builds a controller around a fake source; `hook` runs after every delivered motion sample. */
function harness(
  src: Partial<SegSourceOptions> = {},
  opts: Partial<TrialControllerOptions> = {},
  hook?: (h: Harness, t: number, n: number) => void,
): Harness {
  const h = {} as Harness;
  h.source = createSegSource({
    mode: 'offline',
    orientation: true,
    quaternion: true,
    startTime: START,
    onTick: hook ? (t, n) => hook(h, t, n) : undefined,
    ...src,
  });
  h.cues = segCueSink();
  h.lives = [];
  h.ctrl = createTrialController({
    source: h.source,
    cues: h.cues,
    ids: SEG_IDS,
    condition: segCondition(),
    params: PARAMS,
    calibration: null,
    rng: segRng(7),
    appVersion: '0.1.0',
    simulated: true,
    onLive: (s) => h.lives.push(s),
    ...opts,
  });
  return h;
}

async function runTrial(h: Harness): Promise<TrialRecord> {
  await h.ctrl.start();
  return h.ctrl.result;
}

const cols = (v: unknown): Record<string, unknown> | null => v as Record<string, unknown> | null;

/** Checks that hold for EVERY record, whatever its outcome. */
function checkRecord(rec: TrialRecord): void {
  expectValid('TrialRecord', rec);
  expectEqualColumnLengths(cols(rec.streams.motion));
  expectEqualColumnLengths(cols(rec.streams.orientation));
  expectEqualColumnLengths(cols(rec.streams.quaternion));
  expect(allNumbersFinite(rec)).toEqual([]);
  expect(JSON.parse(JSON.stringify(rec))).toEqual(rec);
  expect(rec.outcome.accepted).toBe(rec.outcome.status === 'complete');
  const tm = rec.timing;
  const order = [tm.still_ms, tm.go_ms, tm.onset_ms, tm.end_ms, tm.end_detected_ms, tm.stop_ms].filter(
    (v): v is number => v !== null,
  );
  for (let i = 1; i < order.length; i++) expect(order[i]).toBeGreaterThanOrEqual(order[i - 1]);
  const m = rec.streams.motion;
  for (const t of [...m.t, ...(rec.streams.orientation?.t ?? []), ...rec.streams.cues.map((c) => c.t)]) {
    expect(t).toBeLessThanOrEqual(tm.stop_ms);
  }
  for (let i = 1; i < m.t.length; i++) expect(m.t[i]).toBeGreaterThanOrEqual(m.t[i - 1]);
  const seg = rec.segmentation;
  expect(seg.onset_index === null).toBe(tm.onset_ms === null);
  expect(seg.end_index === null).toBe(tm.end_ms === null);
  if (seg.onset_index !== null) expect(m.t[seg.onset_index]).toBe(tm.onset_ms);
  if (seg.end_index !== null) expect(m.t[seg.end_index]).toBe(tm.end_ms);
  if (tm.go_ms !== null) {
    expect(rec.streams.cues.filter((c) => c.kind === 'go')).toEqual([{ t: tm.go_ms, kind: 'go' }]);
    expect(m.t).toContain(tm.go_ms); // GO is given at a sample
  }
  // 'ready' opens every started trial (an abort before start() has no cues at all)
  if (rec.streams.cues.length > 0) expect(rec.streams.cues[0]).toEqual({ t: 0, kind: 'ready' });
  expect(rec.quality.n_motion).toBe(m.t.length);
  expect(rec.quality.n_orientation).toBe(rec.streams.orientation?.t.length ?? 0);
  expect(rec.quality.n_quaternion).toBe(rec.streams.quaternion?.t.length ?? 0);
}

/** Distinct consecutive phases seen by onLive. */
function phases(lives: TrialLiveState[]): TrialPhase[] {
  const out: TrialPhase[] = [];
  for (const l of lives) if (out[out.length - 1] !== l.phase) out.push(l.phase);
  return out;
}

describe('TrialController with an offline source (all samples delivered inside start())', () => {
  it('records a complete, schema-valid trial', async () => {
    const h = harness();
    const rec = await runTrial(h);
    checkRecord(rec);

    expect(rec).toMatchObject({
      record_type: 'trial',
      schema_version: SCHEMA_VERSION,
      app_version: '0.1.0',
      trial_id: SEG_IDS.trial_id,
      session_id: SEG_IDS.session_id,
      participant_id: SEG_IDS.participant_id,
      study_code: SEG_IDS.study_code,
      block_index: 0,
      trial_index: 3,
      trial_in_block: 3,
      attempt: 1,
      simulated: true,
      condition: segCondition(),
      outcome: { status: 'complete', accepted: true, participant_flag: null, note: null },
    });

    const tm = rec.timing;
    expect(tm.t0_perf_ms).toBe(START);
    expect(tm.t0_epoch_ms).toBe(SEG_BASE_EPOCH + START);
    expect(rec.created_at).toBe(new Date(SEG_BASE_EPOCH + START + tm.stop_ms).toISOString());
    expect(tm.false_starts).toBe(0);
    expect(tm.first_touch_ms).toBeNull();
    for (const k of ['still_ms', 'go_ms', 'onset_ms', 'end_ms', 'end_detected_ms', 'foreperiod_ms'] as const) {
      expect(tm[k], k).not.toBeNull();
    }
    const fp = tm.foreperiod_ms as number;
    expect(fp).toBeGreaterThanOrEqual(PARAMS.foreperiod_min_ms);
    expect(fp).toBeLessThanOrEqual(PARAMS.foreperiod_max_ms);
    // GO = first sample with t ≥ still + foreperiod
    const goGap = (tm.go_ms as number) - ((tm.still_ms as number) + fp);
    expect(goGap).toBeGreaterThanOrEqual(0);
    expect(goGap).toBeLessThan(1000 / 60);
    expect(tm.stop_ms - (tm.end_detected_ms as number)).toBeGreaterThanOrEqual(PARAMS.post_roll_ms);
    expect(tm.stop_ms - (tm.end_detected_ms as number)).toBeLessThan(PARAMS.post_roll_ms + 1000 / 60);

    // cues: played, logged in the stream, and passed to the simulated participant
    expect(h.cues.played).toEqual(['ready', 'go', 'end']);
    expect(rec.streams.cues).toEqual([
      { t: 0, kind: 'ready' },
      { t: tm.go_ms, kind: 'go' },
      { t: tm.end_detected_ms, kind: 'end' },
    ]);
    expect(h.source.cueLog.map((c) => c.cue)).toEqual(['go', 'end']);
    expect(h.source.cueLog[0].t - START).toBeCloseTo(tm.go_ms as number, 2);
    expect(h.source.stopCalls).toBe(1);
    expect(h.source.deliveredAfterStop).toBe(0);
    expect(rec.streams.motion.t.length).toBe(h.source.delivered);

    // accuracy against the simulated participant's ground truth
    const onsetErr = (tm.onset_ms as number) - ((h.source.truth.onset_t as number) - START);
    const endErr = (tm.end_ms as number) - ((h.source.truth.end_t as number) - START);
    expect(onsetErr).toBeGreaterThanOrEqual(0);
    expect(onsetErr).toBeLessThan(120);
    expect(Math.abs(endErr)).toBeLessThan(120);

    const seg = rec.segmentation;
    expect(seg.method).toBe('online_v1');
    expect(seg.params).toEqual(PARAMS);
    expect(seg.acc_source).toBe('linear');
    expect(seg.duration_ms).toBeCloseTo((tm.end_ms as number) - (tm.onset_ms as number), 9);
    expect(seg.peak_gyro_dps as number).toBeGreaterThan(80); // 60° in 1 s → 112 dps peak
    expect(seg.peak_acc_ms2 as number).toBeGreaterThan(1);
    expect(seg.baseline).not.toBeNull();
    expect(seg.baseline?.gravity?.[1] as number).toBeGreaterThan(9.5);

    expect(rec.detected.rotation_start).toBe('upright');
    expect(rec.detected.orientation_match).toBe(true);
    // a 60° turn about (1, 0.35, 0.2) tilts gravity by ≈ 56.5°
    expect(rec.detected.tilt_change_deg as number).toBeGreaterThan(50);
    expect(rec.detected.tilt_change_deg as number).toBeLessThan(63);
    expect(rec.detected.rotation_end).toBe('unknown');

    expect(rec.quality).toMatchObject({
      acc_available: true,
      rotation_available: true,
      hidden_during: false,
      gaps_over_50ms: 0,
      warnings: [],
    });
    expect(rec.quality.motion_hz as number).toBeCloseTo(60, 1);
    expect(rec.screen).toEqual({ type_start: null, angle_start: null, type_end: null, angle_end: null });
    expect(rec.streams.quaternion?.frequency_hz).toBe(60);

    expect(phases(h.lives)).toEqual(['waiting_still', 'foreperiod', 'armed', 'moving', 'post_roll', 'done']);
    const last = h.lives[h.lives.length - 1];
    expect(last).toMatchObject({ phase: 'done', message: 'done', t: tm.stop_ms });
    expect(h.ctrl.live).toBe(last);
    expect(h.lives.find((l) => l.phase === 'armed')?.message).toBe('go');
    expect(h.lives.find((l) => l.phase === 'post_roll')?.message).toBe('hold_at_nose');
    expect(h.ctrl.lastSampleAt).toBe(START + (rec.streams.motion.t.at(-1) as number));
  });

  it('re-running online_v1 on the STORED record (GO at timing.go_ms) reproduces the online result exactly', async () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const src of [{}, { acc: 'null' as const }, { rr: 'intermittent' as const, jitterMs: 4 }]) {
        const h = harness({ seed, ...src });
        const rec = await runTrial(h);
        checkRecord(rec);
        const rerun = runOnlineSegmentation(storedMotionSamples(rec), rec.segmentation.params, rec.timing.go_ms);
        expect(rerun.result).toEqual(rec.segmentation);
        expect(rerun.goT).toBe(rec.timing.go_ms);
        const byType = Object.fromEntries(rerun.events.map((e) => [e.type, e]));
        expect(byType.onset?.t).toBe(rec.timing.onset_ms);
        expect((byType.end as { detected_t: number } | undefined)?.detected_t).toBe(rec.timing.end_detected_ms);
      }
    }
  });

  it('is deterministic for the same source seed and rng', async () => {
    const a = await runTrial(harness({ seed: 9 }));
    const b = await runTrial(harness({ seed: 9 }));
    expect(b).toEqual(a);
  });

  it('a false start plays an error cue, counts it, and the trial still completes after a new GO', async () => {
    const h = harness({ behavior: 'false_start', seed: 5 });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    expect(rec.timing.false_starts).toBe(1);
    expect(h.cues.played).toEqual(['ready', 'error', 'go', 'end']);
    expect(rec.streams.cues.map((c) => c.kind)).toEqual(['ready', 'error', 'go', 'end']);
    const errT = rec.streams.cues[1].t;
    expect(errT).toBeLessThan(rec.timing.go_ms as number);
    expect(rec.timing.still_ms as number).toBeGreaterThan(errT); // still_ms = the hold that led to GO
    expect(h.lives.some((l) => l.message === 'too_early' && l.phase === 'waiting_still')).toBe(true);
    expect(phases(h.lives).slice(0, 3)).toEqual(['waiting_still', 'foreperiod', 'waiting_still']);
  });

  it("no movement after GO → 'no_movement', not accepted, error cue", async () => {
    const h = harness({ behavior: 'no_move' });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome).toEqual({ status: 'no_movement', accepted: false, participant_flag: null, note: null });
    expect(rec.timing.onset_ms).toBeNull();
    expect(rec.timing.end_ms).toBeNull();
    expect(rec.timing.end_detected_ms).toBeNull();
    expect(rec.timing.stop_ms - (rec.timing.go_ms as number)).toBeGreaterThan(PARAMS.max_wait_ms);
    expect(rec.timing.stop_ms - (rec.timing.go_ms as number)).toBeLessThan(PARAMS.max_wait_ms + 1000 / 60);
    expect(h.cues.played).toEqual(['ready', 'go', 'error']);
    expect(rec.streams.cues.map((c) => c.kind)).toEqual(['ready', 'go', 'error']);
    expect(rec.streams.cues[2].t).toBe(rec.timing.stop_ms);
    expect(rec.segmentation.baseline).not.toBeNull();
    expect(rec.segmentation.peak_gyro_dps).toBeNull();
    expect(rec.detected.rotation_start).toBe('upright');
    expect(h.ctrl.live).toMatchObject({ phase: 'done', message: 'no_movement' });
    expect(h.source.stopCalls).toBe(1);
  });

  it("a movement that never ends → 'too_long'", async () => {
    const h = harness({ behavior: 'endless' });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('too_long');
    expect(rec.outcome.accepted).toBe(false);
    expect(rec.timing.onset_ms).not.toBeNull();
    expect(rec.timing.end_ms).toBeNull();
    expect(rec.timing.stop_ms - (rec.timing.onset_ms as number)).toBeGreaterThan(PARAMS.max_movement_ms);
    expect(h.cues.played).toEqual(['ready', 'go', 'error']);
    expect(rec.segmentation.end_index).toBeNull();
    expect(rec.segmentation.peak_gyro_dps as number).toBeGreaterThan(60);
    expect(h.ctrl.live).toMatchObject({ phase: 'done', message: 'too_long' });
  });

  it("abort('user') from a listener callback mid-movement → 'aborted', start() and result resolve", async () => {
    const h = harness({}, {}, (hh, t) => {
      const onset = hh.source.truth.onset_t;
      if (onset !== null && t > onset + 300) hh.ctrl.abort();
    });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome).toEqual({ status: 'aborted', accepted: false, participant_flag: null, note: null });
    expect(rec.timing.onset_ms).not.toBeNull();
    expect(rec.timing.end_ms).toBeNull();
    expect(h.cues.played).toEqual(['ready', 'go']);
    expect(h.source.stopCalls).toBe(1);
    expect(h.source.deliveredAfterStop).toBe(0);
    expect(h.ctrl.live.phase).toBe('done');
    h.ctrl.abort('interrupted'); // no effect after finishing
    expect((await h.ctrl.result).outcome.status).toBe('aborted');
  });

  it("abort('interrupted') (e.g. page hidden) → 'interrupted' with hidden_during", async () => {
    let hiddenNow = false;
    const h = harness({}, { isHidden: () => hiddenNow }, (hh, t, n) => {
      if (n === 40) hiddenNow = true;
      if (n === 41) hh.ctrl.abort('interrupted');
    });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('interrupted');
    expect(rec.outcome.accepted).toBe(false);
    expect(rec.quality.hidden_during).toBe(true);
    expect(rec.streams.motion.t.length).toBe(42);
  });

  it('abort() before start() → an empty aborted record; start() resolves without starting the source', async () => {
    const h = harness();
    h.ctrl.abort();
    const rec = await h.ctrl.result;
    await h.ctrl.start();
    checkRecord(rec);
    expect(rec.outcome.status).toBe('aborted');
    expect(rec.streams.motion.t).toEqual([]);
    expect(rec.streams.orientation).toBeNull();
    expect(rec.streams.cues).toEqual([]);
    expect(rec.segmentation).toMatchObject({ acc_source: 'none', onset_index: null, baseline: null });
    expect(rec.detected).toEqual({ rotation_start: 'unknown', rotation_end: 'unknown', tilt_change_deg: null, orientation_match: null });
    expect(rec.quality.warnings).toContain('no_motion_samples');
    expect(h.source.delivered).toBe(0);
    expect(h.source.stopCalls).toBe(0);
  });

  it("an offline source that runs out before the trial ends → 'interrupted' + 'source_ended'", async () => {
    const h = harness({ behavior: 'never_still', maxDurationMs: 3000 });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('interrupted');
    expect(rec.quality.warnings).toContain('source_ended');
    expect(rec.timing.still_ms).toBeNull();
    expect(rec.timing.go_ms).toBeNull();
    expect(rec.segmentation.baseline).toBeNull();
    expect(h.cues.played).toEqual(['ready']);
    expect(h.lives.every((l) => l.phase === 'waiting_still' || l.phase === 'done')).toBe(true);
  });

  it('source.start() rejecting or throwing → start() rejects, result resolves as interrupted', async () => {
    for (const failStart of ['reject', 'throw'] as const) {
      const h = harness({ failStart });
      await expect(h.ctrl.start()).rejects.toThrow();
      const rec = await h.ctrl.result;
      checkRecord(rec);
      expect(rec.outcome.status).toBe('interrupted');
      expect(rec.quality.warnings.some((w) => w.startsWith('source_start_failed: '))).toBe(true);
      expect(h.source.stopCalls).toBe(1);
    }
  });

  it('start() is idempotent', async () => {
    const h = harness();
    const p1 = h.ctrl.start();
    const p2 = h.ctrl.start();
    expect(p2).toBe(p1);
    await p1;
    expect((await h.ctrl.result).outcome.status).toBe('complete');
    expect(h.source.delivered).toBe((await h.ctrl.result).streams.motion.t.length);
  });

  it('ignores samples a misbehaving source keeps delivering after stop()', async () => {
    const h = harness({ ignoreStop: true, maxDurationMs: 9000 });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    expect(h.source.deliveredAfterStop).toBeGreaterThan(50);
    expect(rec.streams.motion.t.length).toBe(h.source.delivered - h.source.deliveredAfterStop);
    expect(h.cues.played).toEqual(['ready', 'go', 'end']);
    expect(h.lives.at(-1)?.phase).toBe('done');
  });

  it('records touches; first_touch_ms only counts touchstarts at or after GO', async () => {
    const late = { at: null as number | null };
    const h = harness({}, {}, (hh, t, n) => {
      if (n === 10) {
        hh.ctrl.addTouch('start', 1, 100.123, 200.456); // before GO: ignored for first_touch_ms
        hh.ctrl.addTouch('end', 0, null, null);
      }
      const go = hh.source.truth.go_t;
      if (go !== null && late.at === null && t > go + 200) {
        hh.ctrl.addTouch('start', 1, 10, 20, go - 50); // after GO, but stamped before it
        hh.ctrl.addTouch('end', 0, 10, 20); // not a touchstart
        late.at = t;
        hh.ctrl.addTouch('start', 2, 150, 300); // the nose
      }
    });
    const rec = await runTrial(h);
    checkRecord(rec);
    const t10 = rec.streams.motion.t[10];
    const goAbs = h.source.truth.go_t as number;
    expect(rec.streams.touches).toEqual([
      { t: t10, type: 'start', n: 1, x: 100.12, y: 200.46 },
      { t: t10, type: 'end', n: 0, x: null, y: null },
      { t: Math.round((goAbs - 50 - START) * 100) / 100, type: 'start', n: 1, x: 10, y: 20 },
      { t: Math.round(((late.at as number) - START) * 100) / 100, type: 'end', n: 0, x: 10, y: 20 },
      { t: Math.round(((late.at as number) - START) * 100) / 100, type: 'start', n: 2, x: 150, y: 300 },
    ]);
    expect(rec.timing.first_touch_ms).toBe(Math.round(((late.at as number) - START) * 100) / 100);
    h.ctrl.addTouch('start', 1, 0, 0); // after finishing: ignored
    expect((await h.ctrl.result).streams.touches).toHaveLength(5);
  });

  it('uses the per-trial params it was given (segmentation.params, post_roll_ms)', async () => {
    const slow = { ...segParamsForSpeed('slow'), post_roll_ms: 300 };
    const h = harness({ movementMs: 2500 }, { params: slow, condition: segCondition({ speed: 'slow' }) });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    expect(rec.segmentation.params).toEqual(slow);
    expect(rec.segmentation.params).not.toEqual(PARAMS);
    expect(rec.timing.stop_ms - (rec.timing.end_detected_ms as number)).toBeGreaterThanOrEqual(300);
    expect(rec.timing.stop_ms - (rec.timing.end_detected_ms as number)).toBeLessThan(300 + 1000 / 60);
  });

  it("acc null → 'highpass' segmentation; rr null → acceleration only; both flagged in quality", async () => {
    const noAcc = await runTrial(harness({ acc: 'null' }));
    checkRecord(noAcc);
    expect(noAcc.outcome.status).toBe('complete');
    expect(noAcc.segmentation.acc_source).toBe('highpass');
    expect(noAcc.quality.acc_available).toBe(false);
    expect(noAcc.quality.warnings).toEqual(['acc_unavailable']);

    const noRr = await runTrial(harness({ rr: 'null' }));
    checkRecord(noRr);
    expect(noRr.outcome.status).toBe('complete');
    expect(noRr.quality.rotation_available).toBe(false);
    expect(noRr.quality.warnings).toEqual(['rotation_unavailable']);
    expect(noRr.streams.motion.rr_alpha.every((v) => v === null)).toBe(true);
  });

  it('detects the start rotation with the calibrated gravity sign and compares it with the instruction', async () => {
    const calib = (sign: 1 | -1): CalibrationResult => ({
      performed_at: '2026-10-23T10:00:00.000Z',
      flat: null,
      upright: null,
      gravity_sign: sign,
      gravity_magnitude: 9.81,
      upright_check: 'ok',
      gyro_bias_dps: null,
      notes: [],
    });
    const ios = await runTrial(harness({ gravitySign: -1 }, { calibration: calib(-1) }));
    checkRecord(ios);
    expect(ios.detected).toMatchObject({ rotation_start: 'upright', orientation_match: true });

    const uncalibrated = await runTrial(harness({ gravitySign: -1 }));
    expect(uncalibrated.detected).toMatchObject({ rotation_start: 'upside_down', orientation_match: false });

    const landscape = await runTrial(
      harness({ up: [1, 0, 0] }, { condition: segCondition({ orientation: 'top_left_toward' }) }),
    );
    checkRecord(landscape);
    expect(landscape.detected).toMatchObject({ rotation_start: 'top_left', orientation_match: true });

    const wrong = await runTrial(harness({ up: [-1, 0, 0] }, { condition: segCondition({ orientation: 'top_left_toward' }) }));
    expect(wrong.detected).toMatchObject({ rotation_start: 'top_right', orientation_match: false });

    const free = await runTrial(harness({}, { condition: segCondition({ orientation: 'free' }) }));
    checkRecord(free);
    expect(free.detected.rotation_start).toBe('upright');
    expect(free.detected.orientation_match).toBeNull();
  });

  it('reads the injected screen orientation at start and finish; throwing callbacks never break a trial', async () => {
    let calls = 0;
    const h = harness(
      {},
      {
        getScreenOrientation: () => (calls++ === 0 ? { type: 'portrait-primary', angle: 0 } : { type: 'landscape-primary', angle: 90 }),
      },
    );
    const rec = await runTrial(h);
    expect(rec.screen).toEqual({ type_start: 'portrait-primary', angle_start: 0, type_end: 'landscape-primary', angle_end: 90 });

    const boom = (): never => {
      throw new Error('boom');
    };
    const hostile = harness(
      {},
      {
        cues: { play: boom },
        onLive: boom,
        getScreenOrientation: boom,
        isHidden: boom,
      },
    );
    const rec2 = await runTrial(hostile);
    checkRecord(rec2);
    expect(rec2.outcome.status).toBe('complete');
    expect(rec2.screen).toEqual({ type_start: null, angle_start: null, type_end: null, angle_end: null });
    expect(rec2.quality.hidden_during).toBe(false);
  });

  it('drops motion samples whose time is not finite or goes backwards, and says so in quality.warnings', async () => {
    // A hand-made offline source: the regular fake participant, with two corrupt samples injected.
    const inner = createSegSource({ mode: 'offline', startTime: START });
    const corrupt: SensorSource = {
      kind: 'simulated',
      clock: inner.clock,
      notifyCue: (c, t) => inner.notifyCue?.(c, t),
      stop: () => inner.stop(),
      start(listener: SensorListener) {
        let n = 0;
        return inner.start({
          ...listener,
          onMotion(s: MotionSample) {
            n++;
            if (n === 20) listener.onMotion({ ...s, t: Number.NaN });
            if (n === 21) listener.onMotion({ ...s, t: s.t - 40 });
            listener.onMotion(s);
          },
        });
      },
    };
    const ctrl = createTrialController({
      source: corrupt,
      cues: segCueSink(),
      ids: SEG_IDS,
      condition: segCondition(),
      params: PARAMS,
      calibration: null,
      rng: segRng(1),
      appVersion: '0.1.0',
      simulated: true,
    });
    await ctrl.start();
    const rec = await ctrl.result;
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    expect(rec.quality.warnings).toEqual(['motion_t_invalid', 'motion_t_not_monotonic']);
    expect(rec.streams.motion.t.length).toBe(inner.delivered);
  });

  it('sensor errors become warnings; no_events shows sensor_stalled', async () => {
    const h = harness();
    // The fake source has no error channel: wrap start() to emit errors through the listener.
    const original = h.source.start.bind(h.source);
    (h.source as { start: SensorSource['start'] }).start = (l: SensorListener) => {
      l.onError?.({ code: 'no_events', message: 'no devicemotion yet' });
      l.onError?.({ code: 'sensor_error', message: 'quaternion sensor failed' });
      return original(l);
    };
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.quality.warnings).toEqual(['sensor_error:no_events', 'sensor_error:sensor_error']);
    expect(h.lives.some((l) => l.message === 'sensor_stalled')).toBe(true);
    expect(rec.outcome.status).toBe('complete');
  });

  it('an abort from inside the cue sink (re-entrant, while GO plays) still yields a consistent record', async () => {
    const played: string[] = [];
    let ctrl: TrialController | null = null;
    const h = harness(
      {},
      {
        cues: {
          play(kind) {
            played.push(kind);
            if (kind === 'go') ctrl?.abort();
          },
        },
      },
    );
    ctrl = h.ctrl;
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('aborted');
    expect(played).toEqual(['ready', 'go']);
    expect(rec.timing.go_ms).not.toBeNull();
    expect(rec.streams.cues.map((c) => c.kind)).toEqual(['ready', 'go']);
    expect(rec.segmentation.baseline).not.toBeNull(); // markGo ran before the callback
    expect(h.source.cueLog).toEqual([]); // the participant was never told to move
    expect(h.lives.at(-1)?.phase).toBe('done');
    expect(h.lives.filter((l) => l.phase === 'done')).toHaveLength(1);
    expect(h.source.stopCalls).toBe(1);
  });
});

describe('TrialController with asynchronous sources', () => {
  it('async delivery (virtual clock): start() resolves before any sample, result resolves later', async () => {
    const h = harness({ mode: 'async', seed: 3 });
    await h.ctrl.start();
    expect(h.source.delivered).toBe(0);
    const rec = await h.ctrl.result;
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    expect(h.source.stopCalls).toBe(1);
    await h.source.drained;
    expect(h.source.deliveredAfterStop).toBe(0);
  });

  it('async delivery with a false start, then an abort from outside a callback', async () => {
    const h = harness({ mode: 'async', behavior: 'false_start', seed: 2 });
    await h.ctrl.start();
    const rec = await h.ctrl.result;
    checkRecord(rec);
    expect(rec.timing.false_starts).toBe(1);

    const h2 = harness({ mode: 'async', behavior: 'no_move' });
    await h2.ctrl.start();
    await new Promise((r) => setTimeout(r, 20)); // let a few batches arrive
    h2.ctrl.abort();
    const rec2 = await h2.ctrl.result;
    checkRecord(rec2);
    expect(rec2.outcome.status).toBe('aborted');
    expect(rec2.timing.stop_ms).toBeGreaterThanOrEqual(rec2.streams.motion.t.at(-1) as number);
  });

  it(
    'a real-time source (setInterval, performance.now) runs a complete trial',
    // Real timers: when the whole suite runs in parallel the CPU can starve setInterval, which
    // delays samples. The trial logic itself is covered deterministically by the offline tests.
    { timeout: 20000, retry: 2 },
    async () => {
      const h = harness({ mode: 'realtime', hz: 60 });
      const before = performance.now();
      await h.ctrl.start();
      const rec = await h.ctrl.result;
      checkRecord(rec);
      expect(rec.outcome.status).toBe('complete');
      expect(rec.timing.t0_perf_ms).toBeGreaterThanOrEqual(Math.round(before * 100) / 100 - 0.01);
      expect(Math.abs(rec.timing.t0_epoch_ms - (performance.timeOrigin + rec.timing.t0_perf_ms))).toBeLessThan(0.02);
      expect(rec.quality.motion_hz as number).toBeGreaterThan(30);
      const onsetErr = (rec.timing.onset_ms as number) - ((h.source.truth.onset_t as number) - rec.timing.t0_perf_ms);
      expect(onsetErr).toBeGreaterThan(-40);
      expect(onsetErr).toBeLessThan(200);
    },
  );
});
