/**
 * Participant profile (ParticipantProfile in the schema). New participants get a random code that is
 * shown prominently before saving; returning participants can type their code: a profile stored on
 * this phone under that code is loaded, otherwise the code is kept as the participant_id.
 *
 * Data-quality rules for the code field (see useEffect below):
 *  - the form is REPLACED when the active code matches a stored profile (it is loaded), including
 *    going back to the known participant's own code;
 *  - answers loaded from someone else's stored profile are never saved under a different code
 *    (the form is reset instead), while answers a new participant typed themselves are kept;
 *  - Save re-checks the store, so a code typed just before tapping Save cannot overwrite a stored
 *    profile with answers that were not loaded from it.
 */
import { useEffect, useRef, useState } from 'preact/hooks';
import type {
  EhiAnswer,
  EhiShortForm,
  HandednessSelf,
  ParticipantProfile,
  PhoneHand,
  Sex,
} from '../../core/types';
import { EHI_ITEMS, isParticipantId, isoTime, newParticipantId, scoreEhi } from '../../core/util';
import { useStrings } from '../../i18n';
import { Button, Card, ChoiceGroup, Field, Notice, NumberInput, Screen, TextInput } from '../../ui/components';
import '../../ui/onboarding.css';
import { getServices } from '../services';
import { navigate, setState, showToast, updateSettings, useAppState } from '../store';

export const MAX_AGE = 110;
export const HEIGHT_MIN_CM = 50;
export const HEIGHT_MAX_CM = 250;
export const NICKNAME_MAX = 60;
export const PHONE_MODEL_MAX = 80;

export const SEX_OPTIONS: Sex[] = ['female', 'male', 'other', 'prefer_not_to_say'];
export const HANDEDNESS_OPTIONS: HandednessSelf[] = ['right', 'left', 'ambidextrous'];
export const PHONE_HAND_OPTIONS: PhoneHand[] = ['right', 'left', 'both'];
/** EHI-SF answer scale in questionnaire order (CSS lays it out with the left answers on the left). */
export const EHI_ANSWERS: EhiAnswer[] = ['always_right', 'usually_right', 'both', 'usually_left', 'always_left'];

/** Form state; empty strings / nulls mean "not answered". */
export interface ProfileDraft {
  nickname: string;
  age: number | null;
  sex: Sex | null;
  handedness: HandednessSelf | null;
  ehi: Partial<EhiShortForm>;
  phoneHand: PhoneHand | null;
  height: number | null;
  armIssue: boolean | null;
  phoneModel: string;
  phoneCase: boolean | null;
}

export type ProfileField = 'code' | 'age' | 'sex' | 'handedness' | 'phoneHand' | 'height';
export type ProfileErrorCode = 'code_invalid' | 'age_required' | 'age_range' | 'age_too_young' | 'required' | 'height_range';
export type ProfileErrors = Partial<Record<ProfileField, ProfileErrorCode>>;
/** Display order of the fields that can have errors (first one gets scrolled into view). */
export const PROFILE_FIELDS: ProfileField[] = ['code', 'age', 'sex', 'handedness', 'phoneHand', 'height'];

export function emptyDraft(phoneModelHint: string | null = null): ProfileDraft {
  return {
    nickname: '',
    age: null,
    sex: null,
    handedness: null,
    ehi: {},
    phoneHand: null,
    height: null,
    armIssue: null,
    phoneModel: (phoneModelHint ?? '').slice(0, PHONE_MODEL_MAX),
    phoneCase: null,
  };
}

export function draftFromProfile(p: ParticipantProfile): ProfileDraft {
  return {
    nickname: p.nickname ?? '',
    age: p.age_years,
    sex: p.sex,
    handedness: p.handedness_self,
    ehi: p.ehi_sf ? { ...p.ehi_sf } : {},
    phoneHand: p.phone_hand,
    height: p.height_cm,
    armIssue: p.arm_issue,
    phoneModel: p.phone_model ?? '',
    phoneCase: p.phone_case,
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

function cleanText(v: string, max: number): string | null {
  const s = v.trim().replace(/\s+/g, ' ').slice(0, max).trim();
  return s === '' ? null : s;
}

/** The EHI-SF answers, only when all four items were answered. */
export function completeEhi(ehi: Partial<EhiShortForm>): EhiShortForm | null {
  const { writing, throwing, toothbrush, spoon } = ehi;
  if (!writing || !throwing || !toothbrush || !spoon) return null;
  return { writing, throwing, toothbrush, spoon };
}

/**
 * Builds the schema-valid profile from a VALIDATED draft (throws if a required answer is missing).
 * EHI fields stay null unless all four items were answered.
 */
export function buildProfile(d: ProfileDraft, participantId: string, createdAt: string | null, now: string): ParticipantProfile {
  if (d.age === null || !d.sex || !d.handedness || !d.phoneHand) throw new Error('profile incomplete');
  const ehi = completeEhi(d.ehi);
  const score = ehi ? scoreEhi(ehi) : null;
  return {
    participant_id: participantId,
    nickname: cleanText(d.nickname, NICKNAME_MAX),
    age_years: d.age,
    sex: d.sex,
    handedness_self: d.handedness,
    ehi_sf: ehi,
    ehi_lq: score ? score.lq : null,
    ehi_class: score ? score.cls : null,
    phone_hand: d.phoneHand,
    height_cm: d.height,
    arm_issue: d.armIssue,
    phone_model: cleanText(d.phoneModel, PHONE_MODEL_MAX),
    phone_case: d.phoneCase,
    created_at: createdAt ?? now,
    updated_at: now,
  };
}

type CodeStatus = 'idle' | 'checking' | 'found' | 'new';
type YesNo = 'yes' | 'no';

function toYesNo(v: boolean | null): YesNo | null {
  return v === null ? null : v ? 'yes' : 'no';
}

export function ProfileScreen() {
  const st = useAppState();
  const s = useStrings();
  const t = s.onboarding.profile;
  const minAge = st.config.min_age;
  const modelHint = st.device?.model_hint ?? null;

  // Captured once: the participant who was active when the screen opened (null = new participant).
  const [baseProfile] = useState<ParticipantProfile | null>(st.participant);
  const [baseId] = useState<string>(() => st.participant?.participant_id ?? newParticipantId());
  const [draft, setDraft] = useState<ProfileDraft>(() =>
    st.participant ? draftFromProfile(st.participant) : emptyDraft(modelHint),
  );
  /** The stored profile the form content was loaded from (null = typed by the participant). */
  const [formSource, setFormSource] = useState<ParticipantProfile | null>(st.participant);
  const [codeText, setCodeText] = useState('');
  const [codeStatus, setCodeStatus] = useState<CodeStatus>('idle');
  const [submitted, setSubmitted] = useState(false);
  const [saving, setSaving] = useState(false);
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
    setDraft(p ? draftFromProfile(p) : emptyDraft(modelHint));
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
    if (saving) return;
    setSubmitted(true);
    const errs = validateDraft(draft, minAge, codeText);
    const first = PROFILE_FIELDS.find((f) => errs[f]);
    if (first) {
      fieldRefs[first].current?.scrollIntoView?.({ block: 'center', behavior: 'smooth' });
      showToast(t.errors.fixBelow, 'warn');
      return;
    }
    setSaving(true);
    try {
      const store = getServices().store;
      const id = activeId;
      const loaded = formSource?.participant_id === id ? formSource : null;
      const stored = loaded ?? (await store.getParticipant(id)) ?? null;
      if (stored && !loaded) {
        // Typed a known code and tapped Save before the lookup finished: show the stored answers first.
        load(stored);
        setCodeStatus('found');
        showToast(t.codeFound, 'info');
        setSaving(false);
        return;
      }
      const profile = buildProfile(draft, id, stored?.created_at ?? null, isoTime());
      await store.saveParticipant(profile);
      updateSettings({ last_participant_id: profile.participant_id });
      setState({ participant: profile });
      navigate('sensors');
    } catch (e) {
      console.error('saving the profile failed', e);
      showToast(s.common.errorGeneric, 'error');
      setSaving(false);
    }
  };

  const codeHint =
    codeText.trim() === ''
      ? t.haveCodeHint
      : !codeValid
        ? t.codeFormat
        : codeStatus === 'checking'
          ? t.codeChecking
          : codeStatus === 'found'
            ? t.codeFound
            : codeStatus === 'new'
              ? t.codeNew
              : t.haveCodeHint;
  const nicknameHint = st.config.upload_nickname ? t.nicknameHintUploaded : t.nicknameHintLocal;
  const yesNo = [
    { value: 'yes' as const, label: s.common.yes },
    { value: 'no' as const, label: s.common.no },
  ];

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
      <p class="onb-lead">{t.intro}</p>

      <Card tone="info" testId="participant-code-card">
        <div class="onb-code-label">{t.codeTitle}</div>
        <div class="onb-code" data-testid="participant-code">
          <bdi>{activeId}</bdi>
        </div>
        <p class="small">{t.codeHint}</p>
      </Card>

      <div ref={fieldRefs.code}>
        <Field
          label={baseProfile ? t.otherCodeTitle : t.haveCodeTitle}
          htmlFor="in-code"
          hint={errors.code ? undefined : codeHint}
          error={errText(errors.code)}
        >
          <div dir="ltr">
            <TextInput
              id="in-code"
              testId="in-code"
              value={codeText}
              placeholder="P-XXXXXX"
              maxLength={16}
              onInput={setCodeText}
            />
          </div>
        </Field>
      </div>

      <Field label={`${t.nickname} (${t.optional})`} htmlFor="in-nickname" hint={nicknameHint}>
        <TextInput
          id="in-nickname"
          testId="in-nickname"
          value={draft.nickname}
          maxLength={NICKNAME_MAX}
          autoComplete="nickname"
          onInput={(nickname) => set({ nickname })}
        />
      </Field>

      <div ref={fieldRefs.age}>
        <Field label={t.age} htmlFor="in-age" hint={t.ageHint} error={errText(errors.age)}>
          <NumberInput id="in-age" testId="in-age" value={draft.age} min={minAge} max={MAX_AGE} onChange={(age) => set({ age })} />
        </Field>
      </div>

      <div ref={fieldRefs.sex}>
        <Field label={t.sex} error={errText(errors.sex)}>
          <ChoiceGroup<Sex>
            name="sex"
            value={draft.sex}
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
            columns={1}
            options={HANDEDNESS_OPTIONS.map((v) => ({ value: v, label: t.handednessOptions[v] }))}
            onChange={(handedness) => set({ handedness })}
            testIdPrefix="opt-handed"
            ariaLabel={t.handedness}
          />
        </Field>
      </div>

      <Card title={t.ehiTitle} testId="ehi-card">
        <div class="stack">
          <p class="small muted">
            {t.ehiHint} ({t.optional})
          </p>
          {EHI_ITEMS.map((item) => (
            <div class="onb-ehi" key={item}>
              <div class="field-label">{t.ehiItems[item]}</div>
              <ChoiceGroup<EhiAnswer>
                name={`ehi-${item}`}
                value={draft.ehi[item] ?? null}
                columns={2}
                options={EHI_ANSWERS.map((a) => ({ value: a, label: t.ehiAnswers[a] }))}
                onChange={(answer) => setDraft((d) => ({ ...d, ehi: { ...d.ehi, [item]: answer } }))}
                testIdPrefix={`opt-ehi-${item}`}
                ariaLabel={t.ehiItems[item]}
              />
            </div>
          ))}
        </div>
      </Card>

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
        <Field label={`${t.height} (${t.optional})`} htmlFor="in-height" error={errText(errors.height)}>
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

      <Field label={`${t.armIssue} (${t.optional})`}>
        <ChoiceGroup<YesNo>
          name="arm-issue"
          value={toYesNo(draft.armIssue)}
          options={yesNo}
          onChange={(v) => set({ armIssue: v === 'yes' })}
          testIdPrefix="opt-armissue"
          ariaLabel={t.armIssue}
        />
      </Field>

      <Field label={`${t.phoneModel} (${t.optional})`} htmlFor="in-phonemodel" hint={t.phoneModelHint}>
        <TextInput
          id="in-phonemodel"
          testId="in-phonemodel"
          value={draft.phoneModel}
          maxLength={PHONE_MODEL_MAX}
          onInput={(phoneModel) => set({ phoneModel })}
        />
      </Field>

      <Field label={`${t.phoneCase} (${t.optional})`}>
        <ChoiceGroup<YesNo>
          name="phone-case"
          value={toYesNo(draft.phoneCase)}
          options={yesNo}
          onChange={(v) => set({ phoneCase: v === 'yes' })}
          testIdPrefix="opt-case"
          ariaLabel={t.phoneCase}
        />
      </Field>

      {submitted && Object.keys(errors).length > 0 && <Notice kind="error">{t.errors.fixBelow}</Notice>}
    </Screen>
  );
}
