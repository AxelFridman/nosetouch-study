/**
 * The screens outside the session (spec v2 §1), their pure parts: the short profile (validateDraft,
 * buildProfile), where the profile's Continue leads (routeAfterProfile), the background sensor check
 * started by the welcome screen's Start tap (sensorCheck.ts), the done screen's upload state, the
 * researchers' setup preview, and the consent copy.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  buildProfile,
  cleanText,
  draftFromProfile,
  emptyDraft,
  HEIGHT_MAX_CM,
  HEIGHT_MIN_CM,
  MAX_AGE,
  NAME_MAX,
  normalizeCode,
  validateDraft,
  type ProfileDraft,
} from '../src/app/screens/ProfileScreen';
import { doneStatusText, doneUploadState, uploadConfigured } from '../src/app/screens/DoneScreen';
import { planPreview, PRESETS as SETUP_PRESETS } from '../src/app/screens/SetupScreen';
import { firstName, routeAfterProfile, setupRequestedIn } from '../src/app/startFlow';
import {
  awaitSensorCheck,
  beginSensorCheck,
  getRepairNext,
  lastSensorCheck,
  MEASURE_MS,
  MIN_MOTION_HZ,
  needsSensorRepair,
  resetSensorCheck,
  runSensorCheck,
  sensorVerdict,
  setRepairNext,
  type SensorCheckDeps,
  type SensorCheckOutcome,
} from '../src/app/sensorCheck';
import { PRESETS as CONFIG_PRESETS } from '../src/core/config';
import { buildPlan, estimateMinutes, flattenPlan, isOptionalBlock } from '../src/core/protocol';
import type { MeasuredRates, ParticipantProfile, PermissionState, SensorCapabilities, SessionRecord } from '../src/core/types';
import { en } from '../src/i18n/en';
import { he } from '../src/i18n/he';
import { makeProfile } from './helpers/dataFixtures';
import { expectValid } from './helpers/schema';

const NOW = '2026-10-24T09:30:00.000Z';

function draft(over: Partial<ProfileDraft> = {}): ProfileDraft {
  return { name: 'Dana Cohen', age: 24, sex: 'female', handedness: 'right', phoneHand: 'both', height: 168, ...over };
}

function rates(motionHz: number | null, over: Partial<MeasuredRates> = {}): MeasuredRates {
  return {
    duration_ms: 2500,
    motion_hz: motionHz,
    orientation_hz: motionHz,
    quaternion_hz: null,
    motion_dt_ms_median: motionHz ? 1000 / motionHz : null,
    motion_dt_ms_p95: motionHz ? 1000 / motionHz : null,
    acc_available: motionHz !== null,
    accg_available: motionHz !== null,
    rotation_available: motionHz !== null,
    interval_raw: 16,
    ...over,
  };
}

// ───────────────────────────── profile ─────────────────────────────

describe('validateDraft (short profile)', () => {
  it('accepts a complete draft; name and height are optional', () => {
    expect(validateDraft(draft(), 18)).toEqual({});
    expect(validateDraft(draft({ name: '', height: null }), 18)).toEqual({});
  });

  it('asks for age, sex, dominant hand and phone hand, and nothing else', () => {
    expect(validateDraft(emptyDraft(), 18)).toEqual({
      age: 'age_required',
      sex: 'required',
      handedness: 'required',
      phoneHand: 'required',
    });
  });

  it('age: whole years from min_age to MAX_AGE', () => {
    expect(validateDraft(draft({ age: 18 }), 18).age).toBeUndefined();
    expect(validateDraft(draft({ age: 17 }), 18).age).toBe('age_too_young');
    expect(validateDraft(draft({ age: 20 }), 21).age).toBe('age_too_young'); // config.min_age is respected
    expect(validateDraft(draft({ age: MAX_AGE }), 18).age).toBeUndefined();
    for (const bad of [MAX_AGE + 1, 30.5, -1, Number.NaN]) expect(validateDraft(draft({ age: bad }), 18).age).toBe('age_range');
  });

  it(`height: empty, or ${HEIGHT_MIN_CM}–${HEIGHT_MAX_CM} cm`, () => {
    for (const ok of [null, HEIGHT_MIN_CM, 172.5, HEIGHT_MAX_CM]) expect(validateDraft(draft({ height: ok }), 18).height).toBeUndefined();
    for (const bad of [HEIGHT_MIN_CM - 0.1, HEIGHT_MAX_CM + 0.1, 0, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(validateDraft(draft({ height: bad }), 18).height).toBe('height_range');
    }
  });

  it('a returning code is optional, but a typed one must be a valid participant code', () => {
    expect(validateDraft(draft(), 18, '').code).toBeUndefined();
    expect(validateDraft(draft(), 18, '   ').code).toBeUndefined();
    expect(validateDraft(draft(), 18, 'abc234').code).toBeUndefined();
    expect(validateDraft(draft(), 18, 'P-ABC23').code).toBe('code_invalid');
    expect(validateDraft(draft(), 18, 'P-ABCO23').code).toBe('code_invalid'); // O is not in the alphabet
  });

  it('normalizeCode: case, spaces, any dash and a missing "P-" do not matter', () => {
    for (const raw of [' p-abc234 ', 'pabc234', 'abc234', 'P–ABC234', 'P_ABC234', 'P- ABC 234']) {
      expect(normalizeCode(raw)).toBe('P-ABC234');
    }
  });
});

describe('buildProfile (short profile → ParticipantProfile)', () => {
  it('a new participant: the v1-only answers are null, arm_issue false, and the record is schema-valid', () => {
    const p = buildProfile(draft(), 'P-ABC234', null, NOW);
    expect(p).toEqual<ParticipantProfile>({
      participant_id: 'P-ABC234',
      nickname: 'Dana Cohen',
      age_years: 24,
      sex: 'female',
      handedness_self: 'right',
      ehi_sf: null,
      ehi_lq: null,
      ehi_class: null,
      phone_hand: 'both',
      height_cm: 168,
      arm_issue: false,
      phone_model: null,
      phone_case: null,
      created_at: NOW,
      updated_at: NOW,
    });
    expectValid('ParticipantProfile', p);
    // every key of the schema is present (the loader skips records with a missing key)
    expect(Object.keys(p).sort()).toEqual(Object.keys(makeProfile()).sort());
  });

  it(`the full name is trimmed, inner spaces collapsed, cut to ${NAME_MAX} characters; empty → null`, () => {
    expect(buildProfile(draft({ name: '  Dana   Cohen ' }), 'P-ABC234', null, NOW).nickname).toBe('Dana Cohen');
    expect(buildProfile(draft({ name: '' }), 'P-ABC234', null, NOW).nickname).toBeNull();
    expect(buildProfile(draft({ name: '   ' }), 'P-ABC234', null, NOW).nickname).toBeNull();
    const long = buildProfile(draft({ name: 'x'.repeat(NAME_MAX + 15) }), 'P-ABC234', null, NOW).nickname;
    expect(long).toHaveLength(NAME_MAX);
    expect(cleanText(' a  b ', 2)).toBe('a');
    expect(buildProfile(draft({ height: null }), 'P-ABC234', null, NOW).height_cm).toBeNull();
  });

  it("a returning participant's stored EHI, phone model and case are carried over, never erased", () => {
    const stored = makeProfile({
      participant_id: 'P-ABC234',
      phone_model: 'Pixel 8',
      phone_case: true,
      arm_issue: true,
      created_at: '2026-01-02T03:04:05.000Z',
    });
    const p = buildProfile(draft({ age: 32, name: 'Dana Levi' }), 'P-ABC234', stored, NOW);
    expect(p.ehi_sf).toEqual(stored.ehi_sf);
    expect(p.ehi_sf).not.toBe(stored.ehi_sf); // a copy: editing one never changes the other
    expect(p.ehi_lq).toBe(stored.ehi_lq);
    expect(p.ehi_class).toBe(stored.ehi_class);
    expect(p.phone_model).toBe('Pixel 8');
    expect(p.phone_case).toBe(true);
    expect(p.created_at).toBe('2026-01-02T03:04:05.000Z');
    expect(p.updated_at).toBe(NOW);
    // this visit's answers win; the consent checkbox confirms no arm or hand injury
    expect(p).toMatchObject({ age_years: 32, nickname: 'Dana Levi', handedness_self: 'right', phone_hand: 'both', arm_issue: false });
    expectValid('ParticipantProfile', p);
  });

  it('a stored v2 profile (nulls) stays null, and the draft of a stored profile saves the same answers', () => {
    const v2 = buildProfile(draft(), 'P-ABC234', null, '2026-10-01T00:00:00.000Z');
    const again = buildProfile(draftFromProfile(v2), 'P-ABC234', v2, NOW);
    expect(again).toEqual({ ...v2, updated_at: NOW });
  });

  it('throws when a required answer is missing (the screen validates first)', () => {
    for (const missing of [{ age: null }, { sex: null }, { handedness: null }, { phoneHand: null }] as Partial<ProfileDraft>[]) {
      expect(() => buildProfile(draft(missing), 'P-ABC234', null, NOW)).toThrow(/incomplete/);
    }
  });

  it('draftFromProfile shows only the short-profile answers', () => {
    expect(draftFromProfile(makeProfile({ nickname: null, height_cm: null }))).toEqual({
      name: '',
      age: 31,
      sex: 'female',
      handedness: 'right',
      phoneHand: 'right',
      height: null,
    });
    expect(emptyDraft()).toEqual({ name: '', age: null, sex: null, handedness: null, phoneHand: null, height: null });
  });
});

// ───────────────────────────── where Continue leads ─────────────────────────────

describe('routeAfterProfile', () => {
  const ok: SensorCheckOutcome = { permission: 'granted', measured: rates(60), failed: false };
  const base = { consented: true, setupRequested: false, simulate: false, secureContext: true, check: ok };

  it('normally straight into the session; researchers (?setup=1) to the setup screen', () => {
    expect(routeAfterProfile(base)).toBe('session');
    expect(routeAfterProfile({ ...base, setupRequested: true })).toBe('setup');
  });

  it('without consent in this app run: back to the welcome screen, whatever else', () => {
    expect(routeAfterProfile({ ...base, consented: false })).toBe('welcome');
    expect(routeAfterProfile({ ...base, consented: false, simulate: true, setupRequested: true })).toBe('welcome');
  });

  it('the repair screen only when the sensors cannot record (also before the setup screen)', () => {
    const broken: SensorCheckOutcome[] = [
      { permission: 'denied', measured: null, failed: false },
      { permission: 'granted', measured: null, failed: true },
      { permission: 'granted', measured: rates(null), failed: false },
      { permission: 'not_required', measured: rates(0), failed: false },
      { permission: 'granted', measured: rates(MIN_MOTION_HZ - 0.5), failed: false },
    ];
    for (const check of broken) {
      expect(routeAfterProfile({ ...base, check })).toBe('sensors');
      expect(routeAfterProfile({ ...base, check, setupRequested: true })).toBe('sensors');
    }
    expect(routeAfterProfile({ ...base, secureContext: false })).toBe('sensors');
  });

  it('a slow but usable rate, a missing gyroscope or a check still running never block', () => {
    expect(routeAfterProfile({ ...base, check: { ...ok, measured: rates(MIN_MOTION_HZ) } })).toBe('session');
    expect(routeAfterProfile({ ...base, check: { ...ok, measured: rates(30, { rotation_available: false }) } })).toBe('session');
    expect(routeAfterProfile({ ...base, check: null })).toBe('session');
    expect(routeAfterProfile({ ...base, check: { permission: 'unknown', measured: null, failed: false } })).toBe('session');
  });

  it('the simulator never needs a repair', () => {
    const denied: SensorCheckOutcome = { permission: 'denied', measured: null, failed: true };
    expect(routeAfterProfile({ ...base, simulate: true, secureContext: false, check: denied })).toBe('session');
    expect(needsSensorRepair({ simulate: true, secureContext: false, check: denied })).toBe(false);
  });
});

describe('launch flags and names', () => {
  it('?setup=1 is parsed like ?sim=1 (exactly "1")', () => {
    expect(setupRequestedIn('?setup=1')).toBe(true);
    expect(setupRequestedIn('?sim=1&setup=1&lang=he')).toBe(true);
    for (const no of ['', '?setup=0', '?setup=true', '?setup', '?sim=1', '?setup=11']) expect(setupRequestedIn(no)).toBe(false);
  });

  it('firstName: the first word of the full name, or nothing', () => {
    expect(firstName('Dana Cohen')).toBe('Dana');
    expect(firstName('  Dana  ')).toBe('Dana');
    expect(firstName('דנה כהן')).toBe('דנה');
    for (const none of ['', '   ', null, undefined]) expect(firstName(none)).toBe('');
  });
});

// ───────────────────────────── sensor check ─────────────────────────────

describe('sensorVerdict (repair screen)', () => {
  const v = (over: Partial<Parameters<typeof sensorVerdict>[0]>) =>
    sensorVerdict({ simulate: false, secureContext: true, permission: 'granted', measured: rates(60), failed: false, ...over });

  it('60 Hz with a gyroscope: continue, no issues', () => {
    expect(v({})).toEqual({ canContinue: true, issues: [], hz: 60 });
  });

  it('blocks on a denial, no events and too low a rate; warns on a slow rate and a missing gyroscope', () => {
    expect(v({ permission: 'denied', measured: null })).toMatchObject({ canContinue: false, issues: ['denied'] });
    expect(v({ measured: rates(null) })).toMatchObject({ canContinue: false, issues: ['no_events'] });
    expect(v({ measured: rates(12) })).toMatchObject({ canContinue: false, issues: ['too_slow'], hz: 12 });
    expect(v({ measured: rates(30) })).toMatchObject({ canContinue: true, issues: ['slow'] });
    expect(v({ measured: rates(60, { rotation_available: false }) })).toMatchObject({ canContinue: true, issues: ['no_gyro'] });
    expect(v({ secureContext: false, measured: null, permission: null }).issues).toEqual(['insecure']);
  });

  it('the simulator always continues', () => {
    expect(v({ simulate: true, permission: 'denied', measured: null, failed: true })).toEqual({ canContinue: true, issues: [], hz: null });
  });
});

describe('background sensor check (welcome Start → profile Continue)', () => {
  afterEach(() => {
    resetSensorCheck();
    vi.useRealTimers();
  });

  function deps(measure: (ms: number) => Promise<MeasuredRates>, simulate = false) {
    const patches: Partial<SensorCapabilities>[] = [];
    const d: SensorCheckDeps = { simulate: () => simulate, measure: vi.fn(measure), patch: (p) => void patches.push(p) };
    return { d, patches };
  }

  it('a denial is stored and nothing is measured', async () => {
    const { d, patches } = deps(async () => rates(60));
    const out = await runSensorCheck(Promise.resolve<PermissionState>('denied'), d);
    expect(out).toEqual({ permission: 'denied', measured: null, failed: false });
    expect(d.measure).not.toHaveBeenCalled();
    expect(patches).toEqual([{ permission_state: 'denied', measured: null }]);
  });

  it(`after the permission, the rates are measured for ${MEASURE_MS} ms into the capabilities`, async () => {
    const { d, patches } = deps(async () => rates(60));
    const out = await runSensorCheck(Promise.resolve<PermissionState>('granted'), d);
    expect(d.measure).toHaveBeenCalledWith(MEASURE_MS);
    expect(out).toEqual({ permission: 'granted', measured: rates(60), failed: false });
    expect(patches).toEqual([
      { permission_state: 'granted', measured: null },
      { permission_state: 'granted', measured: rates(60) },
    ]);
  });

  it('a failed measurement or request never throws; the simulator counts as granted', async () => {
    const failing = deps(async () => {
      throw new Error('boom');
    });
    expect(await runSensorCheck(Promise.resolve<PermissionState>('not_required'), failing.d)).toEqual({
      permission: 'not_required',
      measured: null,
      failed: true,
    });
    const rejected = deps(async () => rates(60));
    expect((await runSensorCheck(Promise.reject(new Error('no')), rejected.d)).permission).toBe('unknown');
    const sim = deps(async () => rates(60), true);
    expect((await runSensorCheck(Promise.resolve<PermissionState>('denied'), sim.d)).permission).toBe('granted');
    expect(sim.d.measure).toHaveBeenCalled();
  });

  it('awaitSensorCheck: null before any check, the outcome once done, null after the wait limit', async () => {
    expect(await awaitSensorCheck(10)).toBeNull();
    const done = deps(async () => rates(58));
    const outcome = await beginSensorCheck(Promise.resolve<PermissionState>('granted'), done.d);
    expect(lastSensorCheck()).toEqual(outcome);
    expect(await awaitSensorCheck(0)).toEqual(outcome);

    const stuck = deps(() => new Promise<MeasuredRates>(() => undefined)); // never answers
    void beginSensorCheck(Promise.resolve<PermissionState>('granted'), stuck.d);
    expect(lastSensorCheck()).toBeNull();
    const t0 = Date.now();
    expect(await awaitSensorCheck(30)).toBeNull();
    expect(Date.now() - t0).toBeLessThan(1000);
  });

  it('the latest check wins: an older one that ends later never overwrites it', async () => {
    let finishOld: (r: MeasuredRates) => void = () => undefined;
    const old = deps(() => new Promise<MeasuredRates>((resolve) => (finishOld = resolve)));
    const oldRun = beginSensorCheck(Promise.resolve<PermissionState>('granted'), old.d);
    await Promise.resolve();
    await Promise.resolve();
    const fresh = deps(async () => rates(60));
    const freshOutcome = await beginSensorCheck(Promise.resolve<PermissionState>('granted'), fresh.d);
    finishOld(rates(5));
    expect((await oldRun).measured?.motion_hz).toBe(5);
    expect(lastSensorCheck()).toEqual(freshOutcome);
    expect(old.patches.some((p) => p.measured?.motion_hz === 5)).toBe(false);
  });

  it('resetSensorCheck forgets the result and where the repair screen goes', async () => {
    const d = deps(async () => rates(60));
    await beginSensorCheck(Promise.resolve<PermissionState>('granted'), d.d);
    setRepairNext('welcome');
    expect(getRepairNext()).toBe('welcome');
    resetSensorCheck();
    expect(lastSensorCheck()).toBeNull();
    expect(await awaitSensorCheck(0)).toBeNull();
    expect(getRepairNext()).toBe('start');
  });
});

// ───────────────────────────── done screen ─────────────────────────────

describe('done screen', () => {
  it('upload state: off, all sent, sending (keep the page open), offline', () => {
    const s = (over: Partial<Parameters<typeof doneUploadState>[0]>) =>
      doneUploadState({ configured: true, pending: 0, failed: 0, online: true, ...over });
    expect(s({ configured: false, pending: 4 })).toEqual({ kind: 'off' });
    expect(s({})).toEqual({ kind: 'all_sent' });
    expect(s({ pending: 3, failed: 1 })).toEqual({ kind: 'sending', left: 4, failed: 1 });
    expect(s({ pending: 2, online: false })).toEqual({ kind: 'offline', left: 2 });
    expect(s({ online: false })).toEqual({ kind: 'all_sent' }); // nothing left: offline does not matter
    expect(s({ pending: -1, failed: 0 })).toEqual({ kind: 'all_sent' });
  });

  it('uploads count as configured only with a mode and an endpoint', () => {
    expect(uploadConfigured({ mode: 'json_post', endpoint: 'https://x.example/ingest', study_key: 'k' })).toBe(true);
    expect(uploadConfigured({ mode: 'none', endpoint: 'https://x.example/ingest', study_key: 'k' })).toBe(false);
    expect(uploadConfigured({ mode: 'json_post', endpoint: '  ', study_key: 'k' })).toBe(false);
    expect(uploadConfigured(null)).toBe(false);
  });

  it('the status line follows the session status (ended_early included), in both languages', () => {
    for (const strings of [en, he]) {
      const t = strings.onboarding.done;
      const of = (status: SessionRecord['status'] | null) =>
        doneStatusText(status ? ({ status } as SessionRecord) : null, t);
      expect(of('completed')).toBe(t.completed);
      expect(of('ended_early')).toBe(t.endedEarly);
      expect(of('abandoned')).toBe(t.abandoned);
      expect(of(null)).toBe(t.noSession);
      expect(t.thanks('Dana')).toContain('Dana');
      expect(t.thanks('')).not.toMatch(/,\s*!/);
      expect(t.sending(12)).toContain('12');
    }
  });
});

// ───────────────────────────── setup screen (researchers) ─────────────────────────────

describe('setup screen (researchers only)', () => {
  it('offers every preset of the config, the study session first', () => {
    expect([...SETUP_PRESETS].sort()).toEqual([...CONFIG_PRESETS].sort());
    expect(SETUP_PRESETS[0]).toBe('street');
  });

  it('previews required and bonus movements separately, and minutes for the required part', () => {
    for (const preset of SETUP_PRESETS.filter((p) => p !== 'custom')) {
      const plan = buildPlan({ preset, seed: 1, posture: 'standing' });
      const bonus = plan.filter(isOptionalBlock).reduce((n, b) => n + b.trials.length, 0);
      const pv = planPreview(preset, 'standing', null);
      expect(pv).toEqual({ trials: flattenPlan(plan).length - bonus, bonus, minutes: estimateMinutes(plan) });
    }
    const street = planPreview('street', 'standing', null)!;
    expect(street.bonus).toBeGreaterThan(0); // the checkpoint's bonus round
    expect(street.minutes).toBeGreaterThanOrEqual(4);
    expect(street.minutes).toBeLessThanOrEqual(5.5);
    expect(planPreview('quick', 'standing', null)!.bonus).toBe(0);
  });

  it('the preview line names the bonus round only when there is one', () => {
    expect(en.onboarding.setup.presetMeta(29, 5, 10)).toBe('29 movements (+10 bonus) · about 5 min');
    expect(en.onboarding.setup.presetMeta(18, 3.5, 0)).toBe('18 movements · about 3.5 min');
    expect(he.onboarding.setup.presetMeta(29, 5, 10)).toContain('10');
    expect(en.onboarding.setup.summary(29, 5, 10)).toBe(
      '29 movements, about 5 minutes including short breaks, plus an optional bonus round of 10 movements.',
    );
    expect(en.onboarding.setup.summary(2, 1, 0)).toBe('2 movements, about 1 minute including short breaks.');
  });
});

// ───────────────────────────── consent copy ─────────────────────────────

describe('consent copy (v2: about 5 minutes, names stored, no away grips, no table)', () => {
  it('the stored-data line says what is uploaded and whether the name is', () => {
    const t = en.onboarding.welcome;
    expect(t.summaryStored(true, true)).toMatch(/name, age, sex, dominant hand and height .* private research server/);
    expect(t.summaryStored(true, false)).toMatch(/Your name stays on this phone/);
    expect(t.summaryStored(false, true)).toMatch(/on this phone/);
    for (const strings of [en, he]) {
      const w = strings.onboarding.welcome;
      expect(new Set([w.summaryStored(true, true), w.summaryStored(true, false), w.summaryStored(false, true)]).size).toBe(3);
    }
  });

  it('the checkbox covers age and arm/hand injuries; the region clause only when configured', () => {
    expect(en.onboarding.welcome.agree(18, '')).toBe(
      "I agree, I'm 18 or older, and I have no arm or hand injury that makes this uncomfortable.",
    );
    expect(en.onboarding.welcome.agree(21, 'Israel')).toContain("I'm taking part from Israel");
    expect(he.onboarding.welcome.agree(18, '')).toContain('18');
    expect(he.onboarding.welcome.agree(18, 'ישראל')).toContain('ישראל');
  });

  it('the full information matches the v2 visit', () => {
    const c = en.onboarding.consent;
    expect(c.duration).toMatch(/About 5 minutes/);
    expect(c.procedure).not.toMatch(/away from you|upside down|unusual/i); // never a screen-away grip
    expect(c.procedure).toMatch(/No table/);
    expect(c.procedure).toMatch(/Any part of the phone/);
    expect(c.collected.join(' ')).toMatch(/your name/);
    expect(c.retention('')).not.toMatch(/ethics committee/);
    expect(en.onboarding.welcome.summaryDuration).toMatch(/about 40 times — around 5 minutes/);
    expect(he.onboarding.consent.duration).toContain('5');
  });

  it('the numbers in the copy match the street plan: about 30 movements (40 with the bonus), about 5 minutes', () => {
    const plan = buildPlan({ preset: 'street', seed: 1, posture: 'standing' });
    const total = flattenPlan(plan).length;
    const required = total - plan.filter(isOptionalBlock).reduce((n, b) => n + b.trials.length, 0);
    const near = (n: number, target: number) => Math.abs(n - target) <= 3;
    const counts = (text: string, re: RegExp) => (text.match(re) ?? []).slice(1).map(Number);
    const [enRequired, enTotal] = counts(en.onboarding.consent.procedure, /about (\d+) times \(about (\d+) if/);
    const [heRequired, heTotal] = counts(he.onboarding.consent.procedure, /בערך (\d+) פעמים \(ובערך (\d+) /);
    for (const n of [enRequired, heRequired]) expect(near(n, required), `${n} vs ${required} required`).toBe(true);
    for (const n of [enTotal, heTotal]) expect(near(n, total), `${n} vs ${total} with the bonus`).toBe(true);
    // the key facts give the total and the main part's time
    expect(near(counts(en.onboarding.welcome.summaryDuration, /about (\d+) times/)[0], total)).toBe(true);
    expect(near(counts(he.onboarding.welcome.summaryDuration, /בערך (\d+) פעמים/)[0], total)).toBe(true);
    expect(Math.abs(estimateMinutes(plan) - 5)).toBeLessThanOrEqual(0.5);
    for (const text of [en.onboarding.welcome.summaryDuration, en.onboarding.consent.duration, en.onboarding.setup.presets.street.desc]) {
      expect(text).toMatch(/\b5[ -]minute/i);
    }
    for (const text of [he.onboarding.welcome.summaryDuration, he.onboarding.consent.duration, he.onboarding.setup.presets.street.desc]) {
      expect(text).toMatch(/5 דקות/);
    }
  });

  it('the calibration screen, the EHI questions and the arm/model/case questions are gone', () => {
    expect(Object.keys(en.onboarding)).not.toContain('calibration');
    expect(Object.keys(he.onboarding)).not.toContain('calibration');
    for (const gone of ['ehiTitle', 'ehiItems', 'armIssue', 'phoneModel', 'phoneCase']) {
      expect(Object.keys(en.onboarding.profile)).not.toContain(gone);
    }
  });
});
