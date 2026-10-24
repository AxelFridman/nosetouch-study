import { describe, expect, it } from 'vitest';
import { DEFAULT_SEGMENTATION_PARAMS, SCHEMA_VERSION } from '../src/core/constants';
import { runOnlineSegmentation } from '../src/core/segmentation';
import {
  FALSE_START_WATCH_MS,
  GRIP_WINDOW_MS,
  HIDDEN_AFTER_END,
  RETURN_CUT_SHORT,
  SCREEN_ROTATED_DURING_MOVEMENT,
  WRONG_GRIP_AT_STOP,
  createTrialController,
  resetSessionWarnings,
  returnMissed,
  signVerified,
  type TrialController,
} from '../src/core/trialController';
import type {
  CalibrationResult,
  MotionSample,
  SegmentationParams,
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
const FRAME = 1000 / 60;

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

/** A calibration whose gravity sign a still upright hold confirms (signVerified). */
function verifiedCal(sign: 1 | -1): CalibrationResult {
  return {
    method: 'implicit_v1',
    performed_at: '2026-10-23T10:00:00.000Z',
    flat: null,
    upright: {
      n: 30,
      duration_ms: 480,
      accg_mean: [0.1, 9.7 * sign, 0.3],
      accg_sd: [0.02, 0.02, 0.02],
      acc_mean: null,
      rr_mean: null,
      rr_sd: null,
      still: true,
    },
    gravity_sign: sign,
    gravity_magnitude: 9.71,
    upright_check: 'ok',
    gyro_bias_dps: null,
    notes: [],
    axis_check: null,
  };
}

/** Checks that hold for EVERY record, whatever its outcome. */
function checkRecord(rec: TrialRecord, o: { ready?: boolean } = {}): void {
  expectValid('TrialRecord', rec);
  const s = rec.streams;
  for (const stream of [s.motion, s.orientation, s.quaternion, s.orientation_abs, s.quaternion_abs, s.magnetometer]) {
    expectEqualColumnLengths(cols(stream));
  }
  expect(allNumbersFinite(rec)).toEqual([]);
  expect(JSON.parse(JSON.stringify(rec))).toEqual(rec);
  expect(rec.outcome.accepted).toBe(rec.outcome.status === 'complete');
  const tm = rec.timing;
  // the v2 app always writes the schema 1.2 timing fields (null when not observed)
  expect(tm.return_onset_ms).not.toBeUndefined();
  expect(tm.return_end_ms).not.toBeUndefined();
  const order = [tm.still_ms, tm.go_ms, tm.onset_ms, tm.end_ms, tm.end_detected_ms, tm.return_onset_ms ?? null, tm.return_end_ms ?? null, tm.stop_ms].filter(
    (v): v is number => v !== null,
  );
  for (let i = 1; i < order.length; i++) expect(order[i]).toBeGreaterThanOrEqual(order[i - 1]);
  if (tm.return_end_ms !== null) expect(tm.return_onset_ms).not.toBeNull();
  const m = s.motion;
  const times = [
    ...m.t,
    ...(s.orientation?.t ?? []),
    ...(s.orientation_abs?.t ?? []),
    ...(s.quaternion_abs?.t ?? []),
    ...(s.magnetometer?.t ?? []),
    ...(s.screen_events ?? []).map((e) => e.t),
    ...s.cues.map((c) => c.t),
  ];
  for (const t of times) expect(t).toBeLessThanOrEqual(tm.stop_ms);
  for (let i = 1; i < m.t.length; i++) expect(m.t[i]).toBeGreaterThanOrEqual(m.t[i - 1]);
  const seg = rec.segmentation;
  expect(seg.onset_index === null).toBe(tm.onset_ms === null);
  expect(seg.end_index === null).toBe(tm.end_ms === null);
  if (seg.onset_index !== null) expect(m.t[seg.onset_index]).toBe(tm.onset_ms);
  if (seg.end_index !== null) expect(m.t[seg.end_index]).toBe(tm.end_ms);
  if (tm.go_ms !== null) {
    expect(s.cues.filter((c) => c.kind === 'go')).toEqual([{ t: tm.go_ms, kind: 'go' }]);
    expect(m.t).toContain(tm.go_ms); // GO is given at a sample
  }
  // 'ready' opens every started trial (an abort before start() has no cues at all) unless readyCue: false
  if (o.ready === false) expect(s.cues.some((c) => c.kind === 'ready')).toBe(false);
  else if (s.cues.length > 0) expect(s.cues[0]).toEqual({ t: 0, kind: 'ready' });
  expect(rec.quality.n_motion).toBe(m.t.length);
  expect(rec.quality.n_orientation).toBe(s.orientation?.t.length ?? 0);
  expect(rec.quality.n_quaternion).toBe(s.quaternion?.t.length ?? 0);
  expect(rec.quality.n_orientation_abs).toBe(s.orientation_abs?.t.length ?? 0);
  expect(rec.quality.n_quaternion_abs).toBe(s.quaternion_abs?.t.length ?? 0);
  expect(Array.isArray(s.screen_events)).toBe(true);
}

/** Distinct consecutive phases seen by onLive. */
function phases(lives: TrialLiveState[]): TrialPhase[] {
  const out: TrialPhase[] = [];
  for (const l of lives) if (out[out.length - 1] !== l.phase) out.push(l.phase);
  return out;
}

/** Parameters as in protocol 1.0.0: no return_cap_ms (the trial ends after the post-roll). */
function withoutReturnCap(p: SegmentationParams): SegmentationParams {
  const { return_cap_ms: _cap, ...rest } = p;
  return rest;
}

describe('TrialController with an offline source (all samples delivered inside start())', () => {
  it('records a complete, schema-valid trial, the return of the arm included', async () => {
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
    for (const k of ['still_ms', 'go_ms', 'onset_ms', 'end_ms', 'end_detected_ms', 'foreperiod_ms', 'return_onset_ms', 'return_end_ms'] as const) {
      expect(tm[k], k).not.toBeNull();
    }
    const fp = tm.foreperiod_ms as number;
    expect(fp).toBeGreaterThanOrEqual(PARAMS.foreperiod_min_ms);
    expect(fp).toBeLessThanOrEqual(PARAMS.foreperiod_max_ms);
    // GO = first sample with t ≥ still + foreperiod
    const goGap = (tm.go_ms as number) - ((tm.still_ms as number) + fp);
    expect(goGap).toBeGreaterThanOrEqual(0);
    expect(goGap).toBeLessThan(FRAME);
    // the post-roll lasts at least post_roll_ms and then follows the return until the arm is still again
    const endDet = tm.end_detected_ms as number;
    const retOnset = tm.return_onset_ms as number;
    const retEnd = tm.return_end_ms as number;
    expect(tm.stop_ms - endDet).toBeGreaterThanOrEqual(PARAMS.post_roll_ms);
    expect(tm.stop_ms - endDet).toBeLessThan(PARAMS.return_cap_ms as number);
    const trueReturn = (h.source.truth.return_t as number) - START;
    expect(retOnset).toBeGreaterThan(trueReturn - 20);
    expect(retOnset).toBeLessThan(trueReturn + 250);
    expect(retEnd - retOnset).toBeGreaterThan(400);
    expect(retEnd - trueReturn).toBeLessThan(900 + 100); // the 900 ms return, give or take its slow tail
    expect(tm.stop_ms - retEnd).toBeGreaterThanOrEqual(PARAMS.still_hold_ms);
    expect(tm.stop_ms - retEnd).toBeLessThan(PARAMS.still_hold_ms + FRAME);
    expect(returnMissed(rec)).toBe(false);

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
    // the end pose is the hold AT THE NOSE (end_ms … end_detected_ms), not the arm-out pose after the
    // return: a 60° turn about (1, 0.35, 0.2) tilts gravity by ≈ 56.5°
    expect(rec.detected.tilt_change_deg as number).toBeGreaterThan(50);
    expect(rec.detected.tilt_change_deg as number).toBeLessThan(63);
    expect(rec.detected.rotation_end).toBe('unknown');

    expect(rec.quality).toMatchObject({
      acc_available: true,
      rotation_available: true,
      hidden_during: false,
      gaps_over_50ms: 0,
      warnings: [],
      n_quaternion_abs: 0,
      n_orientation_abs: 0,
    });
    expect(rec.quality.motion_hz as number).toBeCloseTo(60, 1);
    expect(rec.screen).toEqual({ type_start: null, angle_start: null, type_end: null, angle_end: null });
    expect(rec.streams.quaternion?.frequency_hz).toBe(60);
    expect(rec.streams).toMatchObject({ quaternion_abs: null, orientation_abs: null, magnetometer: null, screen_events: [] });

    expect(phases(h.lives)).toEqual(['waiting_still', 'foreperiod', 'armed', 'moving', 'post_roll', 'returning', 'done']);
    const last = h.lives[h.lives.length - 1];
    expect(last).toMatchObject({ phase: 'done', message: 'done', t: tm.stop_ms });
    expect(h.ctrl.live).toBe(last);
    expect(h.lives.find((l) => l.phase === 'armed')?.message).toBe('go');
    expect(h.lives.find((l) => l.phase === 'post_roll')?.message).toBe('return_arm');
    expect(h.lives.find((l) => l.phase === 'returning')?.message).toBe('return_arm');
    // the live signals keep moving during the return (the segmenter's are frozen after the end)
    const returning = h.lives.filter((l) => l.phase === 'returning');
    expect(Math.max(...returning.map((l) => l.gyro_dps))).toBeGreaterThan(45);
    // stored times are rounded to 0.01 ms (ROUNDING.time); lastSampleAt is the raw source time
    expect(h.ctrl.lastSampleAt).toBeCloseTo(START + (rec.streams.motion.t.at(-1) as number), 2);
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
    expect(rec.timing.return_onset_ms).toBeNull();
    expect(rec.timing.return_end_ms).toBeNull();
    expect(returnMissed(rec)).toBe(false); // only a complete trial can leave the arm at the nose
    expect(rec.timing.stop_ms - (rec.timing.go_ms as number)).toBeGreaterThan(PARAMS.max_wait_ms);
    expect(rec.timing.stop_ms - (rec.timing.go_ms as number)).toBeLessThan(PARAMS.max_wait_ms + FRAME);
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

  it('records touches (schema 1.2 contact details null when not given); first_touch_ms only counts touchstarts at or after GO', async () => {
    const late = { at: null as number | null };
    const none = { radius_x: null, radius_y: null, force: null, rotation_angle: null, x_norm: null, y_norm: null };
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
        hh.ctrl.addTouch('start', 2, 150, 300, undefined, {
          radius_x: 12.345,
          radius_y: 9.999,
          force: 0.512345,
          rotation_angle: 33.33333,
          x_norm: 1.25, // clamped to the viewport
          y_norm: 0.123456,
        }); // the nose
      }
    });
    const rec = await runTrial(h);
    checkRecord(rec);
    const t10 = rec.streams.motion.t[10];
    const goAbs = h.source.truth.go_t as number;
    const lateT = Math.round(((late.at as number) - START) * 100) / 100;
    expect(rec.streams.touches).toEqual([
      { t: t10, type: 'start', n: 1, x: 100.12, y: 200.46, ...none },
      { t: t10, type: 'end', n: 0, x: null, y: null, ...none },
      { t: Math.round((goAbs - 50 - START) * 100) / 100, type: 'start', n: 1, x: 10, y: 20, ...none },
      { t: lateT, type: 'end', n: 0, x: 10, y: 20, ...none },
      { t: lateT, type: 'start', n: 2, x: 150, y: 300, radius_x: 12.35, radius_y: 10, force: 0.5123, rotation_angle: 33.333, x_norm: 1, y_norm: 0.1235 },
    ]);
    expect(rec.timing.first_touch_ms).toBe(lateT);
    h.ctrl.addTouch('start', 1, 0, 0); // after finishing: ignored
    expect((await h.ctrl.result).streams.touches).toHaveLength(5);
  });

  it('uses the per-trial params it was given; without return_cap_ms (protocol 1.0.0) the trial ends after post_roll_ms', async () => {
    const slow = withoutReturnCap({ ...segParamsForSpeed('slow'), post_roll_ms: 300 });
    const h = harness({ movementMs: 2500 }, { params: slow, condition: segCondition({ speed: 'slow' }) });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    expect(rec.segmentation.params).toEqual(slow);
    expect(rec.segmentation.params).not.toEqual(PARAMS);
    expect(rec.timing.stop_ms - (rec.timing.end_detected_ms as number)).toBeGreaterThanOrEqual(300);
    expect(rec.timing.stop_ms - (rec.timing.end_detected_ms as number)).toBeLessThan(300 + FRAME);
    // the return started 350 ms after the end cue, after the recording had stopped
    expect(rec.timing.return_onset_ms).toBeNull();
    expect(phases(h.lives)).not.toContain('returning');
  });

  it("acc null → 'highpass' segmentation; rr null → acceleration only; both flagged in quality", async () => {
    const noAcc = await runTrial(harness({ acc: 'null' }));
    checkRecord(noAcc);
    expect(noAcc.outcome.status).toBe('complete');
    expect(noAcc.segmentation.acc_source).toBe('highpass');
    expect(noAcc.quality.acc_available).toBe(false);
    expect(noAcc.quality.warnings).toEqual(['acc_unavailable']);
    expect(noAcc.timing.return_end_ms).not.toBeNull(); // the high-pass signal sees the return too

    const noRr = await runTrial(harness({ rr: 'null' }));
    checkRecord(noRr);
    expect(noRr.outcome.status).toBe('complete');
    expect(noRr.quality.rotation_available).toBe(false);
    expect(noRr.quality.warnings).toEqual(['rotation_unavailable']);
    expect(noRr.streams.motion.rr_alpha.every((v) => v === null)).toBe(true);
    expect(noRr.timing.return_onset_ms).not.toBeNull(); // > 1.5 m/s² during the 0.45 m return
  });

  it('detects the start rotation with the gravity sign and compares it with the instruction (null when only an unverified sign separates them)', async () => {
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
    expect(signVerified(calib(1))).toBe(false); // an engine prior: no measured hold behind it
    expect(signVerified(verifiedCal(-1))).toBe(true);
    expect(signVerified({ ...verifiedCal(1), gravity_sign: -1 })).toBe(false); // the hold contradicts the sign
    expect(signVerified(null)).toBe(false);

    const ios = await runTrial(harness({ gravitySign: -1 }, { calibration: calib(-1) }));
    checkRecord(ios);
    expect(ios.detected).toMatchObject({ rotation_start: 'upright', orientation_match: true });

    // no calibration: the W3C sign is assumed, so iOS-signed data READS upside down; flipping the
    // unverified sign explains the mismatch → undecidable, never a rejection
    const uncalibrated = await runTrial(harness({ gravitySign: -1 }));
    expect(uncalibrated.detected).toMatchObject({ rotation_start: 'upside_down', orientation_match: null });
    // the same reading under a VERIFIED sign is a real mismatch
    const verified = await runTrial(harness({ gravitySign: -1 }, { calibration: verifiedCal(1) }));
    expect(verified.detected).toMatchObject({ rotation_start: 'upside_down', orientation_match: false });

    const landscape = await runTrial(
      harness({ up: [1, 0, 0] }, { condition: segCondition({ orientation: 'top_left_toward' }) }),
    );
    checkRecord(landscape);
    expect(landscape.detected).toMatchObject({ rotation_start: 'top_left', orientation_match: true });

    const wrongSide = (calibration: CalibrationResult | null) =>
      runTrial(harness({ up: [-1, 0, 0] }, { condition: segCondition({ orientation: 'top_left_toward' }), calibration }));
    expect((await wrongSide(null)).detected).toMatchObject({ rotation_start: 'top_right', orientation_match: null });
    expect((await wrongSide(verifiedCal(1))).detected).toMatchObject({ rotation_start: 'top_right', orientation_match: false });
    // an axis mismatch (sideways held, upright instructed) survives any sign
    const sideways = await runTrial(harness({ up: [1, 0, 0] }));
    expect(sideways.detected).toMatchObject({ rotation_start: 'top_left', orientation_match: false });

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

describe('TrialController v2: the return of the arm (post-roll with the return)', () => {
  it('a participant who stays at the nose: recording ends return_cap_ms after the end was confirmed; returnMissed', async () => {
    const h = harness({ returnAfterEndMs: null });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    const endDet = rec.timing.end_detected_ms as number;
    const cap = PARAMS.return_cap_ms as number;
    expect(rec.timing.stop_ms - endDet).toBeGreaterThanOrEqual(cap);
    expect(rec.timing.stop_ms - endDet).toBeLessThan(cap + FRAME);
    expect(rec.timing.return_onset_ms).toBeNull();
    expect(rec.timing.return_end_ms).toBeNull();
    expect(returnMissed(rec)).toBe(true);
    expect(phases(h.lives)).toEqual(['waiting_still', 'foreperiod', 'armed', 'moving', 'post_roll', 'returning', 'done']);
  });

  it('a return seen but not settled before the cap keeps its onset; return_end_ms null', async () => {
    const h = harness({ returnAfterEndMs: 2200, returnMs: 1500 });
    const rec = await runTrial(h);
    checkRecord(rec);
    const endDet = rec.timing.end_detected_ms as number;
    expect(rec.timing.return_onset_ms as number).toBeGreaterThan(endDet + 2200 - 1);
    expect(rec.timing.return_end_ms).toBeNull();
    expect(rec.timing.stop_ms - endDet).toBeLessThan((PARAMS.return_cap_ms as number) + FRAME);
    expect(returnMissed(rec)).toBe(false); // the arm is on its way out: the next trial need not wait
  });

  it("an abort in phase 'returning' keeps the trial complete (warning return_cut_short); during the post-roll minimum it is an abort", async () => {
    const inReturn = harness({}, {}, (hh) => {
      if (hh.ctrl.live.phase === 'returning') hh.ctrl.abort('interrupted');
    });
    const rec = await runTrial(inReturn);
    checkRecord(rec);
    expect(rec.outcome).toMatchObject({ status: 'complete', accepted: true });
    expect(rec.quality.warnings).toEqual([RETURN_CUT_SHORT]);
    expect(rec.timing.return_end_ms).toBeNull();
    expect(inReturn.ctrl.live).toMatchObject({ phase: 'done', message: 'done' });

    const inPostRoll = harness({}, {}, (hh) => {
      if (hh.ctrl.live.phase === 'post_roll') hh.ctrl.abort('interrupted');
    });
    const rec2 = await runTrial(inPostRoll);
    checkRecord(rec2);
    expect(rec2.outcome.status).toBe('interrupted');
    expect(rec2.quality.warnings).not.toContain(RETURN_CUT_SHORT);
  });

  it('an offline source that ends during the return → complete with source_ended and return_cut_short', async () => {
    // GO ≈ 0.8–1.3 s, movement 1 s after a 250 ms reaction, end confirmed ≈ +0.3 s; the source stops
    // at 4 s, inside the 3 s cap of a participant who never returns
    const h = harness({ returnAfterEndMs: null, maxDurationMs: 4000 });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    expect(rec.quality.warnings).toEqual(['source_ended', RETURN_CUT_SHORT]);
  });
});

describe('TrialController v2: gates before GO, the ready cue, the false-start policy', () => {
  it('readyCue: false (a chained trial) plays and logs no ready cue', async () => {
    const h = harness({}, { readyCue: false });
    const rec = await runTrial(h);
    checkRecord(rec, { ready: false });
    expect(h.cues.played).toEqual(['go', 'end']);
    expect(rec.streams.cues.map((c) => c.kind)).toEqual(['go', 'end']);
  });

  it('awaitReturn: stillness at the nose does not count; GO follows the return movement and a new hold', async () => {
    // the participant starts at the nose and brings the arm out 600 ms later (900 ms)
    const back = { startMs: 600, durMs: 900, distM: -0.45, angleDeg: -60 };
    const h = harness({ preMoves: [back] }, { awaitReturn: true });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    // first sample is at START + 0; the arm is out at ≈ 1500 ms
    expect(rec.timing.still_ms as number).toBeGreaterThan(back.startMs + back.durMs - 200);
    expect(rec.timing.false_starts).toBe(0); // moving while GO is withheld is no false start
    expect(h.cues.played).toEqual(['ready', 'go', 'end']);
    const before = h.lives.filter((l) => l.t < back.startMs);
    expect(before.length).toBeGreaterThan(10);
    expect(before.every((l) => l.phase === 'waiting_still' && l.message === 'return_arm')).toBe(true);
    // the hold at the nose was long enough for the segmenter's 'still', but no GO was scheduled
    expect(h.lives.some((l) => l.t < back.startMs && l.still_progress === 1)).toBe(true);
  });

  it('awaitReturn without a return: never GO (the session layer pauses after its idle timeout)', async () => {
    const h = harness({ maxDurationMs: 5000 }, { awaitReturn: true });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('interrupted');
    expect(rec.timing.go_ms).toBeNull();
    expect(rec.timing.still_ms).toBeNull();
    expect(h.lives.filter((l) => l.phase !== 'done').every((l) => l.message === 'return_arm')).toBe(true);
  });

  it('grip gate: a confidently wrong grip withholds GO (wrong_grip) until the phone is held as instructed', async () => {
    // held sideways (top_left) while upright is instructed; turned upright 2 s after the start (0.8 s)
    const fix = { startMs: 2000, durMs: 800, distM: 0, angleDeg: -90 };
    const h = harness({ up: [1, 0, 0], axis: [0, 0, 1], preMoves: [fix] }, { gripGate: true, calibration: verifiedCal(1) });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    const blocked = h.lives.filter((l) => l.t < fix.startMs && l.message === 'wrong_grip');
    expect(blocked.length).toBeGreaterThan(30);
    expect(blocked.every((l) => l.phase === 'waiting_still')).toBe(true);
    expect(rec.timing.still_ms as number).toBeGreaterThan(fix.startMs + fix.durMs - 200);
    expect(rec.timing.false_starts).toBe(0);
    expect(rec.detected).toMatchObject({ rotation_start: 'upright', orientation_match: true });
  });

  it('grip gate: a gate that keeps failing never gives GO; free trials and an unverified sign are never blocked by a flip', async () => {
    const stuck = harness({ up: [1, 0, 0], maxDurationMs: 4000 }, { gripGate: true });
    const rec = await runTrial(stuck);
    checkRecord(rec);
    expect(rec.timing.go_ms).toBeNull(); // sideways vs upright: an axis mismatch blocks even without a verified sign
    expect(stuck.lives.filter((l) => l.phase !== 'done').at(-1)?.message).toBe('wrong_grip');

    // iOS-signed data with no verified sign reads upside down: a sign flip explains it → not blocked
    const ios = await runTrial(harness({ gravitySign: -1 }, { gripGate: true }));
    checkRecord(ios);
    expect(ios.outcome.status).toBe('complete');
    expect(ios.detected.orientation_match).toBeNull();

    // with a verified (here: wrong) sign the same reading blocks
    const verified = await runTrial(harness({ gravitySign: -1, maxDurationMs: 4000 }, { gripGate: true, calibration: verifiedCal(1) }));
    expect(verified.timing.go_ms).toBeNull();

    const free = await runTrial(harness({ up: [1, 0, 0] }, { gripGate: true, condition: segCondition({ orientation: 'free' }) }));
    expect(free.outcome.status).toBe('complete');
    expect(free.timing.go_ms).not.toBeNull();

    // gripGate off (the default, e.g. tap mode): GO whatever the grip
    const off = await runTrial(harness({ up: [1, 0, 0] }));
    expect(off.timing.go_ms).not.toBeNull();
    expect(off.detected.orientation_match).toBe(false);
  });

  // the hold is met at 400 ms and GO comes ≥ 400 ms later: a sway starting at 450 ms breaks the hold
  // inside the foreperiod (a 7.5° sway reaches 20 deg/s ≈ 180 ms after it starts)
  const SWAY_AT = 450;

  it('a stillness loss before GO that stays ≤ 30 deg/s and ≤ 0.6 m/s² is silent, even when the onset floors call it a false start', async () => {
    // protocol 1.0.0's slow floors (15 / 0.3) lie below the still thresholds (20 / 0.5), so the
    // segmenter calls every loss of stillness a 'false_start' (7 of the 8 pilot false starts)
    const v1Floors = { ...segParamsForSpeed('slow'), onset_gyro_min_dps: 15, onset_acc_min_ms2: 0.3 };
    // 7.5° over 500 ms: smoothed |ω| peaks ≈ 27 deg/s
    const h = harness({ behavior: 'sway', falseStartAtMs: SWAY_AT, movementMs: 2500 }, { params: v1Floors, condition: segCondition({ speed: 'slow' }) });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    // the segmenter's own reading of the stored samples saw a 'false_start'...
    const rerun = runOnlineSegmentation(storedMotionSamples(rec), rec.segmentation.params, rec.timing.go_ms);
    expect(rerun.events.filter((e) => e.type === 'false_start' || e.type === 'still_lost').map((e) => e.type)).toEqual(['false_start']);
    // ... the controller restarted the hold silently
    expect(rec.timing.false_starts).toBe(0);
    expect(h.cues.played).toEqual(['ready', 'go', 'end']);
    expect(h.lives.some((l) => l.message === 'too_early')).toBe(false);
    expect(phases(h.lives).slice(0, 3)).toEqual(['waiting_still', 'foreperiod', 'waiting_still']);
    expect(Math.max(...h.lives.filter((l) => l.t < 1100).map((l) => l.gyro_dps))).toBeGreaterThan(24); // a real sway
  });

  it('a stillness loss that grows past 30 deg/s within FALSE_START_WATCH_MS is a counted false start (error cue)', async () => {
    // 12° over 500 ms (peak ≈ 43 deg/s): its first non-still sample is below the slow floor (the
    // segmenter says 'still_lost'), the motion passes 30 deg/s a few samples later
    const slow = segParamsForSpeed('slow');
    const h = harness({ behavior: 'sway', swayDeg: 12, falseStartAtMs: SWAY_AT, movementMs: 2500 }, { params: slow, condition: segCondition({ speed: 'slow' }) });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    const rerun = runOnlineSegmentation(storedMotionSamples(rec), rec.segmentation.params, rec.timing.go_ms);
    const loss = rerun.events.find((e) => e.type === 'false_start' || e.type === 'still_lost');
    expect(loss?.type).toBe('still_lost');
    expect(rec.timing.false_starts).toBe(1);
    expect(h.cues.played).toEqual(['ready', 'error', 'go', 'end']);
    const errorT = rec.streams.cues.find((c) => c.kind === 'error')?.t as number;
    expect(errorT).toBeGreaterThan(loss!.t);
    expect(errorT - loss!.t).toBeLessThanOrEqual(FALSE_START_WATCH_MS);
    expect(h.lives.find((l) => l.t === errorT)?.message).toBe('too_early');

    // an abrupt jerk is counted at the loss itself (the default false_start behaviour of the fake source)
    const jerk = harness({ behavior: 'false_start', falseStartAtMs: SWAY_AT });
    const rec2 = await runTrial(jerk);
    const loss2 = runOnlineSegmentation(storedMotionSamples(rec2), rec2.segmentation.params, rec2.timing.go_ms).events.find(
      (e) => e.type === 'false_start' || e.type === 'still_lost',
    );
    expect(rec2.timing.false_starts).toBe(1);
    expect(rec2.streams.cues.find((c) => c.kind === 'error')?.t).toBe(loss2?.t);
  });

  it('motion that grows only after FALSE_START_WATCH_MS is not a false start (no GO was pending any more)', async () => {
    // a 7.5° sway breaks the hold silently; a big movement 600 ms later comes after the watch window
    // and before the hold is met again
    const big = { startMs: SWAY_AT + 600, durMs: 400, distM: 0.05, angleDeg: 25 };
    const back = { startMs: SWAY_AT + 1000, durMs: 400, distM: -0.05, angleDeg: -25 };
    const h = harness({ behavior: 'sway', falseStartAtMs: SWAY_AT, preMoves: [big, back] });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    const events = runOnlineSegmentation(storedMotionSamples(rec), rec.segmentation.params, rec.timing.go_ms).events;
    const loss = events.find((e) => e.type === 'still_lost' || e.type === 'false_start');
    expect(big.startMs).toBeGreaterThan((loss?.t as number) + FALSE_START_WATCH_MS);
    expect(events.filter((e) => e.type === 'still').map((e) => e.t > big.startMs)).toEqual([false, true]);
    expect(rec.timing.false_starts).toBe(0);
    expect(h.cues.played).toEqual(['ready', 'go', 'end']);
    expect(rec.timing.still_ms as number).toBeGreaterThan(back.startMs + back.durMs);
  });
});

describe('TrialController v2: schema 1.2 streams and once-per-session warnings', () => {
  it('records the absolute orientation / quaternion, magnetometer and screen orientation streams', async () => {
    const h = harness({ extraStreams: true });
    const rec = await runTrial(h);
    checkRecord(rec);
    const s = rec.streams;
    expect(s.orientation_abs?.absolute).toBe(true);
    // one event per tick, sent after the motion sample: the tick whose motion sample ends the trial has none
    expect(s.orientation_abs?.t.length).toBe(s.orientation?.t.length);
    expect(s.orientation_abs?.t.length).toBe(rec.streams.motion.t.length - 1);
    expect(s.orientation?.compass_accuracy?.every((v) => v === null)).toBe(true);
    expect(s.quaternion_abs).toMatchObject({ sensor: 'AbsoluteOrientationSensor', frequency_hz: 60 });
    expect(s.magnetometer).toMatchObject({ frequency_hz: 60 });
    expect(s.magnetometer?.y.every((v) => v === -30)).toBe(true);
    expect(s.screen_events).toEqual([{ t: Math.round((rec.streams.motion.t[5] + 0.3) * 100) / 100, type: 'landscape-primary', angle: 90 }]);
    expect(rec.quality.n_orientation_abs).toBe(s.orientation_abs?.t.length);
    expect(rec.quality.n_quaternion_abs).toBe(s.quaternion_abs?.t.length);
  });

  it('optional-stream warnings are written once per session, by a trial that reached GO', async () => {
    resetSessionWarnings();
    const ids = (session_id: string) => ({ ...SEG_IDS, session_id });
    const w = ['abs_orientation_unavailable', 'magnetometer_unavailable'];
    const first = await runTrial(harness({ warnings: w }, { ids: ids('S-once-1') }));
    expect(first.quality.warnings).toEqual(w);
    const second = await runTrial(harness({ warnings: w }, { ids: ids('S-once-1') }));
    expect(second.quality.warnings).toEqual([]);
    const otherSession = await runTrial(harness({ warnings: w }, { ids: ids('S-once-2') }));
    expect(otherSession.quality.warnings).toEqual(w);

    // an attempt that never reached GO (discarded by the session layer) does not use up the report
    const noGo = harness({ warnings: w }, { ids: ids('S-once-3') }, (hh, _t, n) => {
      if (n === 5) hh.ctrl.abort();
    });
    expect((await runTrial(noGo)).quality.warnings).toEqual(w);
    expect((await runTrial(harness({ warnings: w }, { ids: ids('S-once-3') }))).quality.warnings).toEqual(w);
    expect((await runTrial(harness({ warnings: w }, { ids: ids('S-once-3') }))).quality.warnings).toEqual([]);

    resetSessionWarnings('S-once-1');
    expect((await runTrial(harness({ warnings: w }, { ids: ids('S-once-1') }))).quality.warnings).toEqual(w);
  });

  it(`the grip gate averages gravity over the last ${GRIP_WINDOW_MS} ms only`, async () => {
    // upright for 2 s, then a quick turn to sideways that stays: the gate must judge the CURRENT grip
    const turn = { startMs: 300, durMs: 300, distM: 0, angleDeg: 90 };
    const h = harness({ axis: [0, 0, 1], preMoves: [turn], maxDurationMs: 4000 }, { gripGate: true, calibration: verifiedCal(1) });
    const rec = await runTrial(h);
    expect(rec.timing.go_ms).toBeNull();
    expect(h.lives.some((l) => l.message === 'wrong_grip')).toBe(true);
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

describe('TrialController v2 review fixes: reading time, hidden timing, rotation and grip warnings', () => {
  it('readMs: no GO before the reading time ("read" shows meanwhile); still_ms is the first still sample after it', async () => {
    const plain = await runTrial(harness());
    const READ = 2500;
    const h = harness({}, { readMs: READ });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    // without it GO comes ≈ 0.8–1.3 s after the start (the hold + the foreperiod)
    expect(plain.timing.go_ms as number).toBeLessThan(READ);
    expect(rec.timing.still_ms as number).toBeGreaterThanOrEqual(READ);
    expect(rec.timing.still_ms as number).toBeLessThan(READ + FRAME);
    expect(rec.timing.go_ms as number).toBeGreaterThanOrEqual(READ + PARAMS.foreperiod_min_ms);
    expect(rec.timing.false_starts).toBe(0);
    // the status says "read" until the time is up, then the usual hold
    const reading = h.lives.filter((l) => l.t < READ);
    expect(reading.length).toBeGreaterThan(100);
    expect(reading.every((l) => l.phase === 'waiting_still' && l.message === 'read')).toBe(true);
    expect(h.lives.find((l) => l.t >= READ)?.message).toBe('hold_still');
    expect(h.ctrl.live.phase).toBe('done');
    // the ring of the hold is still filled during the reading (the segmenter was still all along)
    expect(reading.some((l) => l.still_progress === 1)).toBe(true);
  });

  it('readMs never hides a more urgent message: the arm must come back out first (awaitReturn)', async () => {
    const back = { startMs: 600, durMs: 900, distM: -0.45, angleDeg: -60 };
    const h = harness({ preMoves: [back] }, { awaitReturn: true, readMs: 1000 });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('complete');
    expect(h.lives.filter((l) => l.t < back.startMs).every((l) => l.message === 'return_arm')).toBe(true);
  });

  it('a page hidden during the return of the arm: complete, hidden_during false, warning hidden_after_end (T4 keeps it)', async () => {
    let hiddenNow = false;
    const h = harness({}, { isHidden: () => hiddenNow }, (hh) => {
      // the home button / auto-lock while the arm goes back out: SessionScreen aborts on visibilitychange
      if (hh.ctrl.live.phase === 'returning' && !hiddenNow) {
        hiddenNow = true;
        hh.ctrl.abort('interrupted');
      }
    });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome).toMatchObject({ status: 'complete', accepted: true });
    expect(rec.quality.hidden_during).toBe(false);
    expect(rec.quality.warnings).toEqual([RETURN_CUT_SHORT, HIDDEN_AFTER_END]);
    expect(rec.timing.stop_ms).toBeGreaterThanOrEqual((rec.timing.end_detected_ms as number) + PARAMS.post_roll_ms);
  });

  it('a page hidden before the end of the movement window (up to end_detected + post_roll) is hidden_during', async () => {
    let hiddenNow = false;
    const h = harness({}, { isHidden: () => hiddenNow }, (hh) => {
      if (hh.ctrl.live.phase === 'post_roll' && !hiddenNow) {
        hiddenNow = true;
        hh.ctrl.abort('interrupted');
      }
    });
    const rec = await runTrial(h);
    checkRecord(rec);
    expect(rec.outcome.status).toBe('interrupted');
    expect(rec.quality.hidden_during).toBe(true);
    expect(rec.quality.warnings).not.toContain(HIDDEN_AFTER_END);
  });

  it('a screen orientation change between GO and the end is flagged (screen_rotated_during_movement)', async () => {
    const run = async (when: 'moving' | 'returning') => {
      let listener: SensorListener | null = null;
      let sent = false;
      const h = harness({}, {}, (hh, t) => {
        if (!sent && listener && hh.ctrl.live.phase === when) {
          sent = true;
          listener.onScreen?.({ t: t + 0.5, type: 'landscape-primary', angle: 90 });
        }
      });
      const inner = h.source;
      const source: SensorSource = {
        kind: inner.kind,
        clock: inner.clock,
        start: (l) => {
          listener = l;
          return inner.start(l);
        },
        stop: () => inner.stop(),
        notifyCue: (c, t) => inner.notifyCue?.(c, t),
      };
      const ctrl = createTrialController({
        source,
        cues: h.cues,
        ids: SEG_IDS,
        condition: segCondition(),
        params: PARAMS,
        calibration: null,
        rng: segRng(7),
        appVersion: '0.2.0',
        simulated: true,
        onLive: (s) => {
          h.lives.push(s);
          (h as { ctrl: TrialController }).ctrl = ctrl;
        },
      });
      h.ctrl = ctrl;
      await ctrl.start();
      return ctrl.result;
    };
    const during = await run('moving');
    checkRecord(during);
    expect(during.streams.screen_events).toHaveLength(1);
    expect(during.quality.warnings).toContain(SCREEN_ROTATED_DURING_MOVEMENT);
    // after the end (the return of the arm): the movement itself was not affected
    const after = await run('returning');
    expect(after.streams.screen_events).toHaveLength(1);
    expect(after.quality.warnings).not.toContain(SCREEN_ROTATED_DURING_MOVEMENT);
  });

  it('stopped before GO while the grip gate held GO back: warning wrong_grip_at_stop; the live state says how it was held', async () => {
    const stuck = harness({ up: [1, 0, 0] }, { gripGate: true, calibration: verifiedCal(1) }, (hh, _t, n) => {
      if (n === 150) hh.ctrl.abort('interrupted'); // the idle limit of the session layer
    });
    const rec = await runTrial(stuck);
    checkRecord(rec);
    expect(rec.timing.go_ms).toBeNull();
    expect(rec.quality.warnings).toContain(WRONG_GRIP_AT_STOP);
    const blocked = stuck.lives.filter((l) => l.message === 'wrong_grip');
    expect(blocked.length).toBeGreaterThan(10);
    expect(new Set(blocked.map((l) => l.grip_seen))).toEqual(new Set(['top_left']));
    // a normal trial: no gate reading without the gate, no warning
    const plain = await runTrial(harness());
    expect(plain.quality.warnings).not.toContain(WRONG_GRIP_AT_STOP);
    expect(plain.timing.go_ms).not.toBeNull();
  });
});
