/**
 * Test doubles for the session UI agent's tests (sessionFlow, screens' pure helpers).
 * They stand in for modules other agents implement (protocol, util, config, sensors, storage,
 * uploader, services), so these tests never depend on code that may still be a stub.
 * Use with vi.mock(path, async () => (await import('./helpers/sessionFakes')).xxxModule).
 */
import { vi } from 'vitest';
import { DEFAULT_SEGMENTATION_PARAMS } from '../../src/core/constants';
import type {
  BlockKind,
  CustomProtocolSpec,
  DeviceInfo,
  ExportBundle,
  Hand,
  MotionStream,
  OutboxItem,
  ParticipantProfile,
  PlannedBlock,
  PlannedTrial,
  ProtocolOptions,
  SegmentationParams,
  SensorCapabilities,
  OrientationId,
  SessionRecord,
  Speed,
  StudyConfig,
  TrialCondition,
  TrialRecord,
  TrialTiming,
  Variant,
} from '../../src/core/types';
import type { LocalStore, OutboxCounts } from '../../src/core/storage';
import type { FlushReport, UploadQueue } from '../../src/core/uploader';

// ───────────────────────────── protocol (simple, deterministic) ─────────────────────────────

export const FAKE_DEFAULT_CUSTOM_SPEC: CustomProtocolSpec = {
  orientations: ['upright_toward', 'top_left_toward', 'top_right_toward'],
  speeds: ['natural', 'fast', 'slow'],
  start_positions: ['natural'],
  reps: 1,
  odd_variants: [],
  include_free: false,
};

const PRESET_SPEC: CustomProtocolSpec = {
  orientations: ['upright_toward', 'top_left_toward'],
  speeds: ['natural', 'fast'],
  start_positions: ['natural'],
  reps: 1,
  odd_variants: [],
  include_free: false,
};

function fakeFirstHand(opts: ProtocolOptions): Hand {
  return opts.firstHand ?? (opts.seed & 1 ? 'left' : 'right');
}

/**
 * preset 'street': a small plan with the street STRUCTURE (A = first hand, B = the other):
 *   block 0 tier 0 choice A free         natural (hand_source 'reported')      trial 0
 *   block 1 tier 1 main   A upright      natural, fast                         trials 1-2
 *   block 2 tier 1 main   B upright      natural, slow                         trials 3-4
 *   block 3 tier 2 main   B top_left     natural                               trial 5
 *   block 4 tier 3 odd    A upright      tilted_touch                          trial 6
 *   block 5 tier 4 main   A upright      natural   (optional)                  trial 7
 *   block 6 tier 4 main   B upright      fast      (optional)                  trial 8
 */
export const FAKE_STREET = { trials: 9, required: 7, firstOptionalTrial: 7, firstOptionalBlock: 5 } as const;

function fakeStreetPlan(opts: ProtocolOptions): PlannedBlock[] {
  const a = fakeFirstHand(opts);
  const b: Hand = a === 'left' ? 'right' : 'left';
  const specs: { kind: BlockKind; tier: number; hand: Hand; orientation: OrientationId; cells: [Speed, Variant][] }[] = [
    { kind: 'choice', tier: 0, hand: a, orientation: 'free', cells: [['natural', 'standard']] },
    { kind: 'main', tier: 1, hand: a, orientation: 'upright_toward', cells: [['natural', 'standard'], ['fast', 'standard']] },
    { kind: 'main', tier: 1, hand: b, orientation: 'upright_toward', cells: [['natural', 'standard'], ['slow', 'standard']] },
    { kind: 'main', tier: 2, hand: b, orientation: 'top_left_toward', cells: [['natural', 'standard']] },
    { kind: 'odd', tier: 3, hand: a, orientation: 'upright_toward', cells: [['natural', 'tilted_touch']] },
    { kind: 'main', tier: 4, hand: a, orientation: 'upright_toward', cells: [['natural', 'standard']] },
    { kind: 'main', tier: 4, hand: b, orientation: 'upright_toward', cells: [['fast', 'standard']] },
  ];
  let trialIndex = 0;
  return specs.map((spec, blockIndex) => ({
    block_index: blockIndex,
    kind: spec.kind,
    tier: spec.tier,
    optional: spec.tier === 4,
    hand: spec.hand,
    orientation: spec.orientation,
    trials: spec.cells.map(([speed, variant], trialInBlock) => ({
      trial_index: trialIndex++,
      block_index: blockIndex,
      trial_in_block: trialInBlock,
      condition: {
        hand: spec.hand,
        orientation: spec.orientation,
        speed,
        start_position: 'natural',
        variant,
        posture: opts.posture,
        rep: 0,
        hand_source: spec.kind === 'choice' ? 'reported' : 'instructed',
      },
    })),
  }));
}

/** orientations × [A, B] blocks, each with speeds × start positions × reps trials (no shuffling). */
function fakeBuildPlan(opts: ProtocolOptions): PlannedBlock[] {
  if (opts.preset === 'street') return fakeStreetPlan(opts);
  const spec = opts.preset === 'custom' ? (opts.custom ?? FAKE_DEFAULT_CUSTOM_SPEC) : PRESET_SPEC;
  const a = fakeFirstHand(opts);
  const b: Hand = a === 'left' ? 'right' : 'left';
  const blocks: PlannedBlock[] = [];
  let trialIndex = 0;
  for (const orientation of spec.orientations) {
    for (const hand of [a, b]) {
      const block: PlannedBlock = { block_index: blocks.length, kind: 'main', hand, orientation, trials: [] };
      for (const speed of spec.speeds) {
        for (const start_position of spec.start_positions) {
          for (let rep = 0; rep < spec.reps; rep++) {
            const condition: TrialCondition = {
              hand,
              orientation,
              speed,
              start_position,
              variant: 'standard',
              posture: opts.posture,
              rep,
            };
            block.trials.push({
              trial_index: trialIndex++,
              block_index: block.block_index,
              trial_in_block: block.trials.length,
              condition,
            });
          }
        }
      }
      blocks.push(block);
    }
  }
  return blocks;
}

export const protocolModule = {
  buildPlan: vi.fn(fakeBuildPlan),
  flattenPlan: vi.fn((plan: PlannedBlock[]): PlannedTrial[] => plan.flatMap((b) => b.trials)),
  estimateMinutes: vi.fn((plan: PlannedBlock[]) => Math.round(plan.flatMap((b) => b.trials).length * 0.15 * 2) / 2 + 1),
  resolveFirstHand: vi.fn(fakeFirstHand),
  resolveParams: vi.fn(
    (base: SegmentationParams, c: TrialCondition): SegmentationParams =>
      c.speed === 'fast' ? { ...base, onset_sustain_ms: 40 } : { ...base },
  ),
  presetTrialCount: vi.fn(() => 8),
  isOptionalBlock: vi.fn((block: PlannedBlock) => block.optional === true),
  DEFAULT_CUSTOM_SPEC: FAKE_DEFAULT_CUSTOM_SPEC,
};

// ───────────────────────────── util ─────────────────────────────

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const FAKE_RANDOM_SEED = 4242;

export const utilModule = {
  makeRng: vi.fn(mulberry32),
  randomSeed: vi.fn(() => FAKE_RANDOM_SEED),
  newId: vi.fn(() => crypto.randomUUID()),
  isoTime: vi.fn((ms?: number) => new Date(ms ?? Date.now()).toISOString()),
  roundOrNull: vi.fn((v: number | null | undefined, d: number) =>
    typeof v === 'number' && Number.isFinite(v) ? Number(v.toFixed(d)) : null,
  ),
  shuffle: vi.fn(<T>(items: readonly T[]) => [...items]),
  newParticipantId: vi.fn(() => 'P-ABC234'),
  scoreEhi: vi.fn(() => ({ lq: 100, cls: 'right' as const })),
};

// ───────────────────────────── config / sensors ─────────────────────────────

export const configModule = {
  redactProfile: vi.fn(
    (p: ParticipantProfile, cfg: StudyConfig): ParticipantProfile =>
      cfg.upload_nickname ? { ...p } : { ...p, nickname: null },
  ),
  redactSession: vi.fn((s: SessionRecord) => s),
  effectiveUpload: vi.fn((cfg: StudyConfig) => cfg.upload),
  sanitizeStudyConfig: vi.fn(),
  loadStudyConfig: vi.fn(),
};

export function makeDevice(): DeviceInfo {
  return {
    user_agent: 'vitest',
    platform: 'android',
    browser: 'Chrome',
    os_version: '15',
    model_hint: null,
    standalone: false,
    screen_w: 412,
    screen_h: 915,
    dpr: 2.625,
    language: 'en-US',
    hardware_concurrency: 8,
    device_memory_gb: 8,
    time_origin_ms: 1_760_000_000_000,
  };
}

export function makeCapabilities(): SensorCapabilities {
  return {
    secure_context: true,
    device_motion: true,
    device_orientation: true,
    permission_api: false,
    permission_state: 'not_required',
    generic_relative_orientation: true,
    wake_lock: true,
    vibrate: true,
    share_files: false,
    measured: null,
  };
}

export const sensorsModule = {
  getDeviceInfo: vi.fn(async () => makeDevice()),
  detectCapabilities: vi.fn(() => makeCapabilities()),
  getScreenOrientation: vi.fn(() => ({ type: 'portrait-primary', angle: 0 })),
  requestMotionPermission: vi.fn(async () => 'not_required' as const),
  createBrowserSensorSource: vi.fn(),
  measureRates: vi.fn(),
  readTouch: vi.fn(() => ({
    x: null,
    y: null,
    details: { radius_x: null, radius_y: null, force: null, rotation_angle: null, x_norm: null, y_norm: null },
  })),
};

// ───────────────────────────── storage / uploader / services ─────────────────────────────

export interface MemoryStore extends LocalStore {
  readonly sessions: Map<string, SessionRecord>;
  readonly trials: Map<string, TrialRecord>;
  readonly outbox: Map<string, OutboxItem>;
}

/** In-memory LocalStore with IndexedDB copy semantics (values are cloned in and out). */
export function createMemoryStore(): MemoryStore {
  const participants = new Map<string, ParticipantProfile>();
  const sessions = new Map<string, SessionRecord>();
  const trials = new Map<string, TrialRecord>();
  const outbox = new Map<string, OutboxItem>();
  const clone = <T>(v: T): T => structuredClone(v);
  const counts = (): OutboxCounts => {
    const c = { pending: 0, sent: 0, failed: 0 };
    for (const o of outbox.values()) c[o.status]++;
    return c;
  };
  const sessionTrials = (sid: string) =>
    [...trials.values()]
      .filter((t) => t.session_id === sid)
      .sort((a, b) => a.trial_index - b.trial_index || a.attempt - b.attempt);
  const putPending = (kind: 'session' | 'trial', recordId: string, sessionId: string): void => {
    const id = `${kind}:${recordId}`;
    outbox.set(id, {
      id,
      kind,
      record_id: recordId,
      session_id: sessionId,
      status: 'pending',
      attempts: 0,
      last_error: null,
      last_attempt_at: null,
      sent_at: null,
      confirmed: false,
    });
  };
  return {
    sessions,
    trials,
    outbox,
    async saveParticipant(p) {
      participants.set(p.participant_id, clone(p));
    },
    async getParticipant(id) {
      const p = participants.get(id);
      return p && clone(p);
    },
    async listParticipants() {
      return [...participants.values()].map(clone);
    },
    async deleteParticipant(id) {
      participants.delete(id);
      for (const s of [...sessions.values()]) if (s.participant_id === id) await this.deleteSession(s.session_id);
    },
    async saveSession(s) {
      sessions.set(s.session_id, clone(s));
    },
    async getSession(id) {
      const s = sessions.get(id);
      return s && clone(s);
    },
    async listSessions(pid) {
      return [...sessions.values()]
        .filter((s) => !pid || s.participant_id === pid)
        .sort((a, b) => b.started_at.localeCompare(a.started_at))
        .map(clone);
    },
    async findResumableSession(pid, maxAgeMs = 24 * 3600 * 1000) {
      const now = Date.now();
      const s = (await this.listSessions(pid)).find(
        (x) => x.status === 'in_progress' && now - Date.parse(x.updated_at) <= maxAgeMs,
      );
      return s;
    },
    async deleteSession(id) {
      sessions.delete(id);
      for (const t of [...trials.values()]) if (t.session_id === id) trials.delete(t.trial_id);
      for (const o of [...outbox.values()]) if (o.session_id === id) outbox.delete(o.id);
    },
    async saveTrial(t) {
      trials.set(t.trial_id, clone(t));
    },
    async getTrial(id) {
      const t = trials.get(id);
      return t && clone(t);
    },
    async listTrials(sid) {
      return sessionTrials(sid).map(clone);
    },
    async enqueue(kind, recordId, sessionId) {
      putPending(kind, recordId, sessionId);
    },
    // one atomic write, like the IndexedDB transaction: nothing can fail half-way in memory
    async commit({ session, trial }) {
      if (trial) trials.set(trial.trial_id, clone(trial));
      if (session) sessions.set(session.session_id, clone(session));
      if (session) putPending('session', session.session_id, session.session_id);
      if (trial) putPending('trial', trial.trial_id, trial.session_id);
    },
    async enqueueMissing() {
      let added = 0;
      for (const t of trials.values()) {
        if (outbox.has(`trial:${t.trial_id}`)) continue;
        putPending('trial', t.trial_id, t.session_id);
        added++;
      }
      return added;
    },
    async listOutbox(status) {
      return [...outbox.values()].filter((o) => !status || o.status === status).map(clone);
    },
    async updateOutbox(item) {
      outbox.set(item.id, clone(item));
    },
    async outboxCounts() {
      return counts();
    },
    async exportBundle(opts): Promise<ExportBundle> {
      const ss = [...sessions.values()].filter((s) => !opts.sessionIds || opts.sessionIds.includes(s.session_id));
      return {
        format: 'nose-touch-export',
        schema_version: '1.0.0',
        app_version: '0.1.0',
        exported_at: new Date().toISOString(),
        participants: [...participants.values()].map((p) => opts.redact(clone(p))),
        sessions: ss.map((s) => ({ ...clone(s), participant: opts.redact(clone(s.participant)) })),
        trials: ss.flatMap((s) => sessionTrials(s.session_id)).map(clone),
      };
    },
    async clearAll() {
      participants.clear();
      sessions.clear();
      trials.clear();
      outbox.clear();
    },
    async requestPersistence() {
      return true;
    },
    async estimate() {
      return null;
    },
  };
}

export function createFakeUploader(store: LocalStore): UploadQueue & { flush: ReturnType<typeof vi.fn> } {
  return {
    flush: vi.fn(async (): Promise<FlushReport> => ({ attempted: 0, sent: 0, failed: 0, counts: await store.outboxCounts() })),
    flushOnLeave: vi.fn(async () => 0),
    start: vi.fn(),
    stop: vi.fn(),
    onChange: vi.fn(() => () => undefined),
  };
}

export interface FakeServices {
  store: MemoryStore;
  uploader: ReturnType<typeof createFakeUploader>;
  cues: { play: ReturnType<typeof vi.fn>; say: ReturnType<typeof vi.fn>; unlock: ReturnType<typeof vi.fn>; setOptions: ReturnType<typeof vi.fn> };
  wakeLock: { acquire: ReturnType<typeof vi.fn>; release: ReturnType<typeof vi.fn>; active: boolean };
  createSource: ReturnType<typeof vi.fn>;
}

const holder: { services: FakeServices | null } = { services: null };

/** Fresh fake services (memory store, spy uploader/cues), installed for the mocked getServices(). */
export function installFakeServices(): FakeServices {
  const store = createMemoryStore();
  const services: FakeServices = {
    store,
    uploader: createFakeUploader(store),
    cues: { play: vi.fn(), say: vi.fn(), unlock: vi.fn(async () => undefined), setOptions: vi.fn() },
    wakeLock: { acquire: vi.fn(async () => true), release: vi.fn(async () => undefined), active: false },
    createSource: vi.fn(),
  };
  holder.services = services;
  return services;
}

export const servicesModule = {
  getServices: () => {
    if (!holder.services) throw new Error('fake services not installed');
    return holder.services;
  },
  initServices: vi.fn(),
  DEFAULT_CONDITION: {
    hand: 'right',
    orientation: 'upright_toward',
    speed: 'natural',
    start_position: 'natural',
    variant: 'standard',
    posture: 'standing',
    rep: 0,
  } as TrialCondition,
};

// ───────────────────────────── fixtures ─────────────────────────────

export function makeProfile(overrides: Partial<ParticipantProfile> = {}): ParticipantProfile {
  return {
    participant_id: 'P-ABC234',
    nickname: 'Dana',
    age_years: 30,
    sex: 'female',
    handedness_self: 'right',
    ehi_sf: { writing: 'always_right', throwing: 'always_right', toothbrush: 'always_right', spoon: 'usually_right' },
    ehi_lq: 87.5,
    ehi_class: 'right',
    phone_hand: 'right',
    height_cm: 168,
    arm_issue: false,
    phone_model: null,
    phone_case: true,
    created_at: '2026-10-23T10:00:00.000Z',
    updated_at: '2026-10-23T10:00:00.000Z',
    ...overrides,
  };
}

/** A motion stream: still, a rotation/acceleration burst between 1000 and 1600 ms, still again. */
export function makeMotionStream(n = 150, opts: { linear?: boolean; dtMs?: number } = {}): MotionStream {
  const dt = opts.dtMs ?? 16;
  const linear = opts.linear ?? true;
  const m: MotionStream = {
    t: [],
    ts: [],
    acc_x: [],
    acc_y: [],
    acc_z: [],
    accg_x: [],
    accg_y: [],
    accg_z: [],
    rr_alpha: [],
    rr_beta: [],
    rr_gamma: [],
    interval_raw: 16,
    ts_first_raw: 123456.7,
  };
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    const moving = t >= 1000 && t <= 1600;
    const a = moving ? 3 * Math.sin(((t - 1000) / 600) * Math.PI) : 0;
    m.t.push(t);
    m.ts.push(t);
    m.acc_x.push(linear ? 0 : null);
    m.acc_y.push(linear ? 0 : null);
    m.acc_z.push(linear ? a : null);
    m.accg_x.push(0);
    m.accg_y.push(9.81);
    m.accg_z.push(a);
    m.rr_alpha.push(moving ? 120 : 1);
    m.rr_beta.push(moving ? 50 : 0.5);
    m.rr_gamma.push(0);
  }
  return m;
}

/**
 * A schema-valid TrialRecord for a planned trial of a session. The phone rests still and upright
 * (accg = (0, 9.81, 0)) from 0 to 1000 ms, i.e. through the pre-GO hold (still_ms - still_hold_ms ...
 * go_ms = 100 ... 900 ms), so it can fill an implicit calibration; `accgY` changes that resting value.
 */
export function makeTrialRecord(
  session: SessionRecord,
  planned: PlannedTrial,
  attempt: number,
  overrides: {
    status?: TrialRecord['outcome']['status'];
    accepted?: boolean;
    trial_id?: string;
    orientation_match?: boolean | null;
    simulated?: boolean;
    timing?: Partial<TrialTiming>;
    accgY?: number;
    created_at?: string;
  } = {},
): TrialRecord {
  const status = overrides.status ?? 'complete';
  const motion = makeMotionStream(120);
  const accgY = overrides.accgY;
  if (accgY !== undefined) motion.accg_y = motion.accg_y.map(() => accgY);
  return {
    record_type: 'trial',
    schema_version: '1.0.0',
    app_version: '0.1.0',
    trial_id: overrides.trial_id ?? crypto.randomUUID(),
    session_id: session.session_id,
    participant_id: session.participant_id,
    study_code: session.study_code,
    created_at: overrides.created_at ?? new Date().toISOString(),
    block_index: planned.block_index,
    trial_index: planned.trial_index,
    trial_in_block: planned.trial_in_block,
    attempt,
    simulated: overrides.simulated ?? true,
    condition: planned.condition,
    outcome: { status, accepted: overrides.accepted ?? status === 'complete', participant_flag: null, note: null },
    timing: {
      t0_perf_ms: 5000,
      t0_epoch_ms: 1_760_000_005_000,
      still_ms: 500,
      go_ms: 900,
      onset_ms: status === 'complete' ? 1000 : null,
      end_ms: status === 'complete' ? 1600 : null,
      end_detected_ms: status === 'complete' ? 1900 : null,
      stop_ms: 1904,
      first_touch_ms: null,
      foreperiod_ms: 400,
      false_starts: 0,
      ...overrides.timing,
    },
    segmentation: {
      method: 'online_v1',
      params: { ...DEFAULT_SEGMENTATION_PARAMS },
      acc_source: 'linear',
      onset_index: status === 'complete' ? 62 : null,
      end_index: status === 'complete' ? 100 : null,
      duration_ms: status === 'complete' ? 600 : null,
      peak_gyro_dps: status === 'complete' ? 130 : null,
      peak_acc_ms2: status === 'complete' ? 3 : null,
      baseline: { n: 30, gyro_mean: 1.1, gyro_sd: 0.2, acc_mean: 0.05, acc_sd: 0.01, gravity: [0, 9.81, 0] },
    },
    detected: {
      rotation_start: 'upright',
      rotation_end: 'upright',
      tilt_change_deg: 4,
      orientation_match: overrides.orientation_match === undefined ? true : overrides.orientation_match,
    },
    quality: {
      n_motion: motion.t.length,
      n_orientation: 0,
      n_quaternion: 0,
      motion_hz: 62.5,
      motion_dt_ms_median: 16,
      motion_dt_ms_max: 16,
      gaps_over_50ms: 0,
      acc_available: true,
      rotation_available: true,
      hidden_during: false,
      warnings: [],
    },
    screen: { type_start: 'portrait-primary', angle_start: 0, type_end: 'portrait-primary', angle_end: 0 },
    streams: { motion, orientation: null, quaternion: null, touches: [], cues: [{ t: 900, kind: 'go' }] },
  };
}
