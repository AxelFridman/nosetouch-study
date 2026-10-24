/**
 * Session setup, for RESEARCHERS only (spec v2 §1): participants never see it; their profile's
 * Continue starts config.default_preset. It opens after the profile when the app was loaded with
 * ?setup=1 or the switch in Settings is on (state.setupRequested): preset (with trial count, bonus
 * and estimated minutes from the real plan), posture, the custom builder (only grips that are
 * collected: never *_away), the safety note, then createSession() → the session screen.
 * The safety note sits in the sticky footer above the start button, so it is on screen whatever the
 * phone size. Taps within TAP_GUARD_MS of the screen appearing are ignored: the profile's Continue
 * sits where "Start recording" appears, and a double tap must not start a session.
 */
import { useMemo, useState } from 'preact/hooks';
import { COLLECTED_ORIENTATIONS, ODD_VARIANTS } from '../../core/constants';
import {
  buildPlan,
  DEFAULT_CUSTOM_SPEC,
  estimateMinutes,
  isOptionalBlock,
  normalizeCustomSpec,
  SPEEDS,
  START_POSITIONS,
} from '../../core/protocol';
import type { CustomProtocolSpec, OrientationId, Posture, ProtocolPreset, Speed, StartPosition, Variant } from '../../core/types';
import { useStrings } from '../../i18n';
import { Button, Card, ChoiceGroup, Field, MultiChoice, Notice, NumberInput, Screen, Toggle } from '../../ui/components';
import '../../ui/onboarding.css';
import { createSession } from '../sessionFlow';
import { unlockInTap } from '../startFlow';
import { createTapGuard } from '../state';
import { getState, navigate, showToast, useAppState } from '../store';

/** Display order: the study session first. */
export const PRESETS: ProtocolPreset[] = ['street', 'quick', 'standard', 'full', 'custom'];
export const POSTURES: Posture[] = ['standing', 'sitting'];
export const MIN_REPS = 1;
export const MAX_REPS = 5;
/** Any fixed seed: trial counts and durations do not depend on the seed, only the order does. */
const PREVIEW_SEED = 1;

export interface PlanPreview {
  /** trials of the required blocks */
  trials: number;
  /** trials of the optional bonus blocks (PlannedBlock.optional), offered at the checkpoint */
  bonus: number;
  /** estimateMinutes: the required blocks only */
  minutes: number;
}

/** Trial counts and estimated duration of what the participant would get; null if the plan cannot be built. */
export function planPreview(preset: ProtocolPreset, posture: Posture, custom: CustomProtocolSpec | null): PlanPreview | null {
  try {
    const plan = buildPlan({ preset, seed: PREVIEW_SEED, posture, custom: preset === 'custom' ? custom : null });
    let trials = 0;
    let bonus = 0;
    for (const block of plan) {
      if (isOptionalBlock(block)) bonus += block.trials.length;
      else trials += block.trials.length;
    }
    return { trials, bonus, minutes: estimateMinutes(plan) };
  } catch (e) {
    console.warn('plan preview failed', e);
    return null;
  }
}

export type CustomError = 'orientation' | 'speed' | 'start' | 'reps';

export function validateCustom(spec: CustomProtocolSpec): CustomError[] {
  const errs: CustomError[] = [];
  if (spec.orientations.length === 0) errs.push('orientation');
  if (spec.speeds.length === 0) errs.push('speed');
  if (spec.start_positions.length === 0) errs.push('start');
  if (!Number.isInteger(spec.reps) || spec.reps < MIN_REPS || spec.reps > MAX_REPS) errs.push('reps');
  return errs;
}

function cloneSpec(spec: CustomProtocolSpec): CustomProtocolSpec {
  return {
    ...spec,
    orientations: [...spec.orientations],
    speeds: [...spec.speeds],
    start_positions: [...spec.start_positions],
    odd_variants: [...spec.odd_variants],
  };
}

export function SetupScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.onboarding.setup;
  const [preset, setPreset] = useState<ProtocolPreset>(st.config.default_preset);
  const [posture, setPosture] = useState<Posture>('standing');
  const [custom, setCustom] = useState<CustomProtocolSpec>(() => cloneSpec(DEFAULT_CUSTOM_SPEC));
  /** reps as typed (null while the field is empty), validated separately */
  const [reps, setReps] = useState<number | null>(DEFAULT_CUSTOM_SPEC.reps);
  const [starting, setStarting] = useState(false);
  const [tap] = useState(createTapGuard);
  tap.shown('setup');

  const customSpec: CustomProtocolSpec = { ...custom, reps: reps ?? Number.NaN };
  const customErrors = preset === 'custom' ? validateCustom(customSpec) : [];
  const valid = customErrors.length === 0;

  const presetPreviews = useMemo(() => {
    const out: Partial<Record<ProtocolPreset, PlanPreview | null>> = {};
    for (const p of PRESETS) if (p !== 'custom') out[p] = planPreview(p, 'standing', null);
    return out;
  }, []);
  const customPreview = useMemo(
    () => (validateCustom(customSpec).length === 0 ? planPreview('custom', posture, normalizeCustomSpec(customSpec)) : null),
    [custom, reps, posture],
  );
  const preview = preset === 'custom' ? customPreview : (presetPreviews[preset] ?? null);

  const onStart = async () => {
    if (tap.tooSoon()) return; // the second tap of a double tap on the previous screen's button
    // Inside the tap: unlock the audio and keep the screen on for the whole session.
    unlockInTap();
    if (starting || !valid) return;
    const now = getState();
    if (!now.participant) {
      navigate('profile'); // first: navigate() clears the toast
      showToast(t.needProfile, 'warn');
      return;
    }
    if (!now.consentAcceptedAt) {
      navigate('welcome'); // the consent is on the welcome screen
      showToast(t.needConsent, 'warn');
      return;
    }
    setStarting(true);
    try {
      await createSession({ preset, posture, custom: preset === 'custom' ? normalizeCustomSpec(customSpec) : null });
      navigate('session');
    } catch (e) {
      console.error('creating the session failed', e);
      showToast(s.common.errorGeneric, 'error');
      setStarting(false);
    }
  };

  const has = (e: CustomError) => customErrors.includes(e);

  return (
    <Screen
      title={t.title}
      onBack={() => navigate('profile')}
      testId="screen-setup"
      footer={
        <>
          <div class="onb-footer-note">
            <Notice kind="warn" testId="safety-note">
              <strong>{t.safetyTitle}:</strong> {s.common.safety}
            </Notice>
          </div>
          <Button testId="btn-start-session" size="lg" block disabled={starting || !valid} onClick={() => void onStart()}>
            {starting ? t.starting : t.start}
          </Button>
        </>
      }
    >
      <p class="onb-lead">{t.intro}</p>

      <Field label={t.presetLabel}>
        <ChoiceGroup<ProtocolPreset>
          name="preset"
          value={preset}
          columns={1}
          options={PRESETS.map((p) => {
            const pv = p === 'custom' ? customPreview : presetPreviews[p];
            return {
              value: p,
              label: (
                <span class="onb-preset">
                  <span class="onb-preset-name">{t.presets[p].label}</span>
                  {pv && <span class="onb-preset-meta">{t.presetMeta(pv.trials, pv.minutes, pv.bonus)}</span>}
                </span>
              ),
              hint: t.presets[p].desc,
            };
          })}
          onChange={setPreset}
          testIdPrefix="opt-preset"
          ariaLabel={t.presetLabel}
        />
      </Field>

      <Field label={t.posture} hint={t.postureHint}>
        <ChoiceGroup<Posture>
          name="posture"
          value={posture}
          options={POSTURES.map((p) => ({ value: p, label: s.common.postures[p] }))}
          onChange={setPosture}
          testIdPrefix="opt-posture"
          ariaLabel={t.posture}
        />
      </Field>

      {preset === 'custom' && (
        <Card title={t.customTitle} testId="custom-builder">
          <div class="stack">
            <Field label={t.orientations} error={has('orientation') ? t.errOrientation : null}>
              <MultiChoice<OrientationId>
                values={custom.orientations}
                columns={1}
                options={COLLECTED_ORIENTATIONS.map((o) => ({ value: o, label: s.common.orientations[o].label }))}
                onChange={(orientations) => setCustom((c) => ({ ...c, orientations }))}
                testIdPrefix="opt-orient"
              />
            </Field>
            <Field label={t.speeds} error={has('speed') ? t.errSpeed : null}>
              <MultiChoice<Speed>
                values={custom.speeds}
                columns={3}
                options={SPEEDS.map((v) => ({ value: v, label: s.common.speeds[v].label }))}
                onChange={(speeds) => setCustom((c) => ({ ...c, speeds }))}
                testIdPrefix="opt-speed"
              />
            </Field>
            <Field label={t.starts} error={has('start') ? t.errStart : null}>
              <MultiChoice<StartPosition>
                values={custom.start_positions}
                columns={1}
                options={START_POSITIONS.map((v) => ({ value: v, label: s.common.startPositions[v].label }))}
                onChange={(start_positions) => setCustom((c) => ({ ...c, start_positions }))}
                testIdPrefix="opt-start"
              />
            </Field>
            <Field label={t.odd}>
              <MultiChoice<Variant>
                values={custom.odd_variants}
                columns={2}
                options={ODD_VARIANTS.map((v) => ({ value: v, label: s.common.variants[v].label }))}
                onChange={(odd_variants) => setCustom((c) => ({ ...c, odd_variants }))}
                testIdPrefix="opt-odd"
              />
            </Field>
            <Field label={t.reps} htmlFor="in-reps" hint={t.repsHint} error={has('reps') ? t.errReps : null}>
              <NumberInput id="in-reps" testId="in-reps" value={reps} min={MIN_REPS} max={MAX_REPS} onChange={setReps} />
            </Field>
            <Toggle
              testId="chk-free"
              checked={custom.include_free}
              onChange={(include_free) => setCustom((c) => ({ ...c, include_free }))}
              label={t.free}
              hint={t.freeHint}
            />
          </div>
        </Card>
      )}

      <Card title={t.summaryTitle} testId="setup-summary">
        <div class="stack-sm">
          {st.participant && (
            <p>
              {t.participant}: <bdi class="mono">{st.participant.participant_id}</bdi>
            </p>
          )}
          {preview && <p data-testid="setup-estimate">{t.summary(preview.trials, preview.minutes, preview.bonus)}</p>}
          <p class="small muted">{t.bothHands}</p>
        </div>
      </Card>

      <Card title={t.checklistTitle}>
        <ul class="onb-checklist">
          {t.checklist.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </Card>
    </Screen>
  );
}
