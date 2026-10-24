/**
 * "About you" (spec v2 §1, screen 2): the short participant profile (ParticipantProfile in the schema):
 * full name (optional, stored in `nickname`), age, sex, dominant hand, usual phone hand and height
 * (optional). Nothing else is asked; the fields v1 asked (EHI-SF, arm issue, phone model, phone case)
 * are written as null — or carried over from a returning participant's stored profile, never erased —
 * and arm_issue is false (the consent checkbox confirms no arm or hand injury).
 *
 * New participants get a random code (shown small, and on the done screen); returning participants
 * type theirs inside "Returning? Enter your code": a profile stored on this phone under that code is
 * loaded, otherwise the code is kept as the participant_id. Data-quality rules for the code field
 * (see useEffect below):
 *  - the form is REPLACED when the active code matches a stored profile (it is loaded), including
 *    going back to the known participant's own code;
 *  - answers loaded from someone else's stored profile are never saved under a different code
 *    (the form is reset instead), while answers a new participant typed themselves are kept;
 *  - Continue re-checks the store, so a code typed just before tapping it cannot overwrite a stored
 *    profile with answers that were not loaded from it.
 * Validation errors appear inline only after the first Continue. Continue then saves the profile and
 * goes on (startFlow.continueToSession): the session with the study's default preset, the repair
 * screen when the sensors cannot record, or the researchers' setup screen.
 */
import { useEffect, useRef, useState } from 'preact/hooks';
import type { HandednessSelf, ParticipantProfile, PhoneHand, Sex } from '../../core/types';
import { isParticipantId, isoTime, newParticipantId } from '../../core/util';
import { useStrings } from '../../i18n';
import { Button, ChoiceGroup, Field, Notice, NumberInput, Screen, TextInput } from '../../ui/components';
import '../../ui/onboarding.css';
import { getServices } from '../services';
import { continueToSession, unlockInTap } from '../startFlow';
import { createTapGuard } from '../state';
import { navigate, setState, showToast, updateSettings, useAppState } from '../store';

export const MAX_AGE = 110;
export const HEIGHT_MIN_CM = 50;
export const HEIGHT_MAX_CM = 250;
/** the schema's limit for ParticipantProfile.nickname, which holds the full name */
export const NAME_MAX = 60;

export const SEX_OPTIONS: Sex[] = ['female', 'male', 'other', 'prefer_not_to_say'];
export const HANDEDNESS_OPTIONS: HandednessSelf[] = ['right', 'left', 'ambidextrous'];
export const PHONE_HAND_OPTIONS: PhoneHand[] = ['right', 'left', 'both'];

/** Form state; empty strings / nulls mean "not answered". */
export interface ProfileDraft {
  /** full name (optional), saved as ParticipantProfile.nickname */
  name: string;
  age: number | null;
  sex: Sex | null;
  handedness: HandednessSelf | null;
  phoneHand: PhoneHand | null;
  height: number | null;
}

export type ProfileField = 'code' | 'age' | 'sex' | 'handedness' | 'phoneHand' | 'height';
export type ProfileErrorCode = 'code_invalid' | 'age_required' | 'age_range' | 'age_too_young' | 'required' | 'height_range';
export type ProfileErrors = Partial<Record<ProfileField, ProfileErrorCode>>;
/** Display order of the fields that can have errors (first one gets scrolled into view). */
export const PROFILE_FIELDS: ProfileField[] = ['age', 'sex', 'handedness', 'phoneHand', 'height', 'code'];

export function emptyDraft(): ProfileDraft {
  return { name: '', age: null, sex: null, handedness: null, phoneHand: null, height: null };
}

export function draftFromProfile(p: ParticipantProfile): ProfileDraft {
  return {
    name: p.nickname ?? '',
    age: p.age_years,
    sex: p.sex,
    handedness: p.handedness_self,
    phoneHand: p.phone_hand,
    height: p.height_cm,
  };
}

/**
 * Canonical form of a typed participant code: upper case, no spaces, any dash → '-', and the "P-"
 * prefix added to a bare 6-character code ("abc234" or "PABC234" → "P-ABC234"). The result may still
 * be invalid; check it with isParticipantId().
 */
export function normalizeCode(raw: string): string {
  const c = raw
    .trim()
    .toUpperCase()
    .replace(/[‐-―−_]/g, '-')
    .replace(/\s+/g, '');
  if (/^[A-Z0-9]{6}$/.test(c)) return `P-${c}`;
  if (/^P[A-Z0-9]{6}$/.test(c)) return `P-${c.slice(1)}`;
  return c;
}

export function validateDraft(d: ProfileDraft, minAge: number, codeText = ''): ProfileErrors {
  const e: ProfileErrors = {};
  if (codeText.trim() !== '' && !isParticipantId(normalizeCode(codeText))) e.code = 'code_invalid';
  if (d.age === null) e.age = 'age_required';
  else if (!Number.isInteger(d.age) || d.age < 0 || d.age > MAX_AGE) e.age = 'age_range';
  else if (d.age < minAge) e.age = 'age_too_young';
  if (!d.sex) e.sex = 'required';
  if (!d.handedness) e.handedness = 'required';
  if (!d.phoneHand) e.phoneHand = 'required';
  if (d.height !== null && !(Number.isFinite(d.height) && d.height >= HEIGHT_MIN_CM && d.height <= HEIGHT_MAX_CM)) {
    e.height = 'height_range';
  }
  return e;
}

/** Trimmed, inner whitespace collapsed, at most `max` characters; null when empty. */
export function cleanText(v: string, max: number): string | null {
  const s = v.trim().replace(/\s+/g, ' ').slice(0, max).trim();
  return s === '' ? null : s;
}

/**
 * Builds the schema-valid profile from a VALIDATED draft (throws if a required answer is missing).
 * `stored`: the profile already stored under this participant_id (a returning participant), or null.
 * The v1-only answers (EHI-SF and its score, phone model, phone case) are kept from `stored` instead
 * of being erased, and are null for a new participant; arm_issue is false (the consent checkbox
 * confirms no arm or hand injury); created_at is kept from `stored`.
 */
export function buildProfile(
  d: ProfileDraft,
  participantId: string,
  stored: ParticipantProfile | null,
  now: string,
): ParticipantProfile {
  if (d.age === null || !d.sex || !d.handedness || !d.phoneHand) throw new Error('profile incomplete');
  return {
    participant_id: participantId,
    nickname: cleanText(d.name, NAME_MAX),
    age_years: d.age,
    sex: d.sex,
    handedness_self: d.handedness,
    ehi_sf: stored?.ehi_sf ? { ...stored.ehi_sf } : null,
    ehi_lq: stored?.ehi_lq ?? null,
    ehi_class: stored?.ehi_class ?? null,
    phone_hand: d.phoneHand,
    height_cm: d.height,
    arm_issue: false,
    phone_model: stored?.phone_model ?? null,
    phone_case: stored?.phone_case ?? null,
    created_at: stored?.created_at ?? now,
    updated_at: now,
  };
}

type CodeStatus = 'idle' | 'checking' | 'found' | 'new';

export function ProfileScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.onboarding.profile;
  const minAge = st.config.min_age;

  // Captured once: the participant who was active when the screen opened (null = new participant).
  const [baseProfile] = useState<ParticipantProfile | null>(st.participant);
  const [baseId] = useState<string>(() => st.participant?.participant_id ?? newParticipantId());
  const [draft, setDraft] = useState<ProfileDraft>(() => (st.participant ? draftFromProfile(st.participant) : emptyDraft()));
  /** The stored profile the form content was loaded from (null = typed by the participant). */
  const [formSource, setFormSource] = useState<ParticipantProfile | null>(st.participant);
  const [codeText, setCodeText] = useState('');
  const [codeStatus, setCodeStatus] = useState<CodeStatus>('idle');
  const [submitted, setSubmitted] = useState(false);
  const [saving, setSaving] = useState(false);
  /** "Returning? Enter your code" is folded away unless opened, typed in, or wrong */
  const [codeOpen, setCodeOpen] = useState(false);
  const [tap] = useState(createTapGuard);
  tap.shown('profile');
  const lookupSeq = useRef(0);
  const fieldRefs = {
    code: useRef<HTMLDivElement>(null),
    age: useRef<HTMLDivElement>(null),
    sex: useRef<HTMLDivElement>(null),
    handedness: useRef<HTMLDivElement>(null),
    phoneHand: useRef<HTMLDivElement>(null),
    height: useRef<HTMLDivElement>(null),
  };

  const normalized = normalizeCode(codeText);
  const codeValid = codeText.trim() !== '' && isParticipantId(normalized);
  const activeId = codeValid ? normalized : baseId;

  const load = (p: ParticipantProfile | null) => {
    setDraft(p ? draftFromProfile(p) : emptyDraft());
    setFormSource(p);
  };

  // The identity changed (a valid code was typed or removed): load what belongs to it.
  useEffect(() => {
    const seq = ++lookupSeq.current;
    if (activeId === baseId) {
      setCodeStatus('idle');
      if (baseProfile) {
        // back to the known participant: their stored answers, not what was typed for another code
        if (formSource?.participant_id !== baseId) load(baseProfile);
      } else if (formSource) {
        load(null); // answers of a stored profile must not move to the new participant's code
      }
      return;
    }
    if (formSource?.participant_id === activeId) {
      setCodeStatus('found');
      return;
    }
    setCodeStatus('checking');
    void (async () => {
      let found: ParticipantProfile | undefined;
      try {
        found = await getServices().store.getParticipant(activeId);
      } catch (e) {
        console.warn('participant lookup failed', e);
      }
      if (seq !== lookupSeq.current) return; // superseded by a newer code
      if (found) {
        load(found);
        setCodeStatus('found');
      } else {
        // Answers loaded from another stored profile must not be saved under this code.
        if (formSource) load(null);
        setCodeStatus('new');
      }
    })();
  }, [activeId]);

  const set = (patch: Partial<ProfileDraft>) => setDraft((d) => ({ ...d, ...patch }));
  const errors: ProfileErrors = submitted ? validateDraft(draft, minAge, codeText) : {};

  const errText = (code: ProfileErrorCode | undefined): string | null => {
    switch (code) {
      case 'code_invalid':
        return t.codeFormat;
      case 'age_required':
        return t.errors.ageRequired;
      case 'age_range':
        return t.errors.ageRange;
      case 'age_too_young':
        return t.errors.ageTooYoung(minAge);
      case 'required':
        return t.errors.required;
      case 'height_range':
        return t.errors.heightRange;
      default:
        return null;
    }
  };

  const onSave = async () => {
    if (tap.tooSoon()) return; // the second tap of a double tap on the welcome screen's Start
    // Inside the tap, before any await: keep the screen on and the audio unlocked (iOS needs a gesture).
    unlockInTap();
    if (saving) return;
    setSubmitted(true);
    const errs = validateDraft(draft, minAge, codeText);
    const first = PROFILE_FIELDS.find((f) => errs[f]);
    if (first) {
      if (first === 'code') setCodeOpen(true);
      fieldRefs[first].current?.scrollIntoView?.({ block: 'center', behavior: 'smooth' });
      return;
    }
    setSaving(true);
    try {
      const store = getServices().store;
      const id = activeId;
      const loaded = formSource?.participant_id === id ? formSource : null;
      const stored = loaded ?? (await store.getParticipant(id)) ?? null;
      if (stored && !loaded) {
        // Typed a known code and tapped Continue before the lookup finished: show the stored answers first.
        load(stored);
        setCodeStatus('found');
        showToast(t.codeFound, 'info');
        setSaving(false);
        return;
      }
      const profile = buildProfile(draft, id, stored, isoTime());
      await store.saveParticipant(profile);
      updateSettings({ last_participant_id: profile.participant_id });
      setState({ participant: profile });
      await continueToSession(); // navigates away (session / sensor repair / setup / welcome)
    } catch (e) {
      console.error('saving the profile or starting the session failed', e);
      showToast(s.common.errorGeneric, 'error');
      setSaving(false);
    }
  };

  const codeHint =
    codeText.trim() === ''
      ? t.codeHint
      : !codeValid
        ? t.codeFormat
        : codeStatus === 'checking'
          ? t.codeChecking
          : codeStatus === 'found'
            ? t.codeFound
            : codeStatus === 'new'
              ? t.codeNew
              : t.codeHint;
  const nameHint = st.config.upload_nickname ? t.nameHintUploaded : t.nameHintLocal;

  return (
    <Screen
      title={t.title}
      onBack={() => navigate('welcome')}
      testId="screen-profile"
      footer={
        <Button testId="btn-save-profile" size="lg" block disabled={saving} onClick={() => void onSave()}>
          {saving ? t.saving : t.save}
        </Button>
      }
    >
      <Field label={t.name} htmlFor="in-name" hint={nameHint}>
        <TextInput
          id="in-name"
          testId="in-name"
          value={draft.name}
          placeholder={t.namePlaceholder}
          maxLength={NAME_MAX}
          autoCapitalize="words"
          // the phone offers its owner's name in one tap
          autoComplete="name"
          enterKeyHint="next"
          onInput={(name) => set({ name })}
        />
      </Field>

      <div ref={fieldRefs.age}>
        <Field label={t.age} htmlFor="in-age" error={errText(errors.age)}>
          <NumberInput
            id="in-age"
            testId="in-age"
            value={draft.age}
            min={minAge}
            max={MAX_AGE}
            inputMode="numeric"
            onChange={(age) => set({ age })}
          />
        </Field>
      </div>

      <div ref={fieldRefs.sex}>
        <Field label={t.sex} error={errText(errors.sex)}>
          <ChoiceGroup<Sex>
            name="sex"
            value={draft.sex}
            columns={2}
            options={SEX_OPTIONS.map((v) => ({ value: v, label: t.sexes[v] }))}
            onChange={(sex) => set({ sex })}
            testIdPrefix="opt-sex"
            ariaLabel={t.sex}
          />
        </Field>
      </div>

      <div ref={fieldRefs.handedness}>
        <Field label={t.handedness} error={errText(errors.handedness)}>
          <ChoiceGroup<HandednessSelf>
            name="handedness"
            value={draft.handedness}
            columns={3}
            options={HANDEDNESS_OPTIONS.map((v) => ({ value: v, label: t.handednessOptions[v] }))}
            onChange={(handedness) => set({ handedness })}
            testIdPrefix="opt-handed"
            ariaLabel={t.handedness}
          />
        </Field>
      </div>

      <div ref={fieldRefs.phoneHand}>
        <Field label={t.phoneHand} error={errText(errors.phoneHand)}>
          <ChoiceGroup<PhoneHand>
            name="phone-hand"
            value={draft.phoneHand}
            columns={3}
            options={PHONE_HAND_OPTIONS.map((v) => ({ value: v, label: t.phoneHandOptions[v] }))}
            onChange={(phoneHand) => set({ phoneHand })}
            testIdPrefix="opt-phonehand"
            ariaLabel={t.phoneHand}
          />
        </Field>
      </div>

      <div ref={fieldRefs.height}>
        <Field label={t.height} htmlFor="in-height" error={errText(errors.height)}>
          <NumberInput
            id="in-height"
            testId="in-height"
            value={draft.height}
            min={HEIGHT_MIN_CM}
            max={HEIGHT_MAX_CM}
            onChange={(height) => set({ height })}
          />
        </Field>
      </div>

      <div ref={fieldRefs.code}>
        <details
          class="onb-details onb-code-details"
          data-testid="returning-details"
          open={codeOpen || codeText.trim() !== '' || !!errors.code}
          onToggle={(e) => setCodeOpen((e.currentTarget as HTMLDetailsElement).open)}
        >
          <summary>{baseProfile ? t.notYou : t.returning}</summary>
          <div class="stack-sm onb-details-body">
            <p class="small muted" data-testid="participant-code-card">
              {t.codeYours}{' '}
              <bdi class="mono onb-code-inline" data-testid="participant-code">
                {activeId}
              </bdi>
            </p>
            <Field label={t.codeInput} htmlFor="in-code" hint={errors.code ? undefined : codeHint} error={errText(errors.code)}>
              <div dir="ltr">
                <TextInput
                  id="in-code"
                  testId="in-code"
                  value={codeText}
                  placeholder="P-XXXXXX"
                  maxLength={16}
                  autoCapitalize="characters"
                  onInput={setCodeText}
                />
              </div>
            </Field>
          </div>
        </details>
      </div>

      {submitted && Object.keys(errors).length > 0 && <Notice kind="error">{t.errors.fixBelow}</Notice>}
    </Screen>
  );
}
