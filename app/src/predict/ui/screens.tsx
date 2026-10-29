/**
 * The seven screens of predict mode (types.ts PredictScreen; docs/stage3/APP_SPEC.md section 5.2) and the
 * pure rule that picks one. Each screen's root carries data-testid="pm-<screen>". Pure components: the
 * state lives in PredictApp.
 */
import type { Strings } from '../i18n/strings';
import type { EngineSnapshot, ModelId, Preflight, PredictScreen, SourceFailure } from '../types';
import { AlertIcon, AttemptDots, CompareTable, Orb, PhoneIcon, ProbabilityDisplay, TurnIcon } from './components';

/** where PredictApp is: before the tap, waiting for the permission prompt, or sensing */
export type Stage = 'start' | 'permission' | 'running';

export interface ScreenInput {
  readonly stage: Stage;
  readonly failure: SourceFailure | null;
  readonly snapshot: EngineSnapshot | null;
}

export function screenOf({ stage, failure, snapshot }: ScreenInput): PredictScreen {
  if (failure) return 'unsupported';
  if (stage === 'start') return 'start';
  if (stage === 'permission') return 'permission';
  if (!snapshot) return 'watching';
  if (snapshot.phase === 'grip' || snapshot.decision.status === 'grip') return 'grip';
  if (snapshot.phase === 'final') return snapshot.decision.answered ? 'answer' : 'abstain';
  return 'watching';
}

export type FailureAction = 'demo' | 'copy' | 'reload' | null;

/** One message and one button per SourceFailure (APP_SPEC 5.2 'unsupported'). */
export function failureView(
  failure: SourceFailure,
  preflight: Pick<Preflight, 'engine' | 'desktop'> | null,
): { text: Parameters<Strings['t']>[0]; action: FailureAction } {
  const desktop = preflight?.desktop ?? false;
  switch (failure) {
    case 'insecure_context':
      return { text: 'u_https', action: 'demo' };
    case 'not_supported':
      // a phone without DeviceMotionEvent is an unusual browser (a WebView): Safari or Chrome has it
      return preflight !== null && !desktop ? { text: 'u_inapp', action: 'copy' } : { text: 'u_desktop', action: 'demo' };
    case 'engine_error':
      return { text: 'u_error', action: 'reload' };
    case 'in_app_browser':
      return { text: 'u_inapp', action: 'copy' };
    case 'permission_denied':
      // iOS keeps a refusal until Safari is quit: the link is copied for after the restart
      return preflight?.engine === 'webkit' ? { text: 'denied_ios', action: 'copy' } : { text: 'denied_other', action: 'reload' };
    case 'no_gyroscope':
      return { text: 'u_gyro', action: 'demo' };
    case 'no_accelerometer':
      return { text: 'u_acc', action: 'demo' };
    case 'no_events':
    default:
      if (desktop) return { text: 'u_desktop', action: 'demo' };
      return { text: preflight?.engine === 'blink' ? 'u_events_android' : 'u_events', action: 'reload' };
  }
}

// ───────────────────────────── screens ─────────────────────────────

export interface StartProps {
  readonly s: Strings;
  readonly onStart: () => void;
  /** a link to the demo, offered on a desktop */
  readonly demoHref: string | null;
  /** false until the preflight is known (a tap before it would start without the wake lock) */
  readonly ready?: boolean;
  /** the desktop demo: a simulated person moves */
  readonly sim?: boolean;
}

export function StartScreen({ s, onStart, demoHref, ready = true, sim = false }: StartProps) {
  return (
    <main class="pm-main pm-center" data-testid="pm-start">
      <PhoneIcon />
      <h1 class="pm-title">{s.t('title')}</h1>
      <p class="pm-lead">{s.t(sim ? 'watch_sim' : 'instruct')}</p>
      <button type="button" class="btn btn-primary btn-lg btn-block pm-cta" data-testid="pm-start-btn" disabled={!ready} onClick={onStart}>
        {s.t('start')}
      </button>
      {demoHref && (
        <a class="pm-link" href={demoHref} data-testid="pm-demo-link">
          {s.t('try_demo')}
        </a>
      )}
    </main>
  );
}

export function PermissionScreen({ s }: { readonly s: Strings }) {
  return (
    <main class="pm-main pm-center" data-testid="pm-permission">
      <div class="pm-spinner" aria-hidden="true" />
      <p class="pm-lead" role="status">
        {s.t('permission')}
      </p>
    </main>
  );
}

export interface WatchingProps {
  readonly s: Strings;
  readonly snap: EngineSnapshot | null;
  readonly compare: boolean;
  readonly onSelectModel: (id: ModelId) => void;
  readonly onStartOver: () => void;
  /** the desktop demo (the CNN row is not applicable there) */
  readonly sim?: boolean;
}

export function WatchingScreen({ s, snap, compare, onSelectModel, onStartOver, sim = false }: WatchingProps) {
  const kMax = snap?.stopping.kMax ?? 5;
  const k = snap ? snap.attempts.length : 0;
  const hint = snap?.screenAwayHint === true;
  const d = snap?.decision;
  // the last movement was seen but not counted (another movement, or a doubtful approach)
  const ev = snap?.lastEvent;
  const notCounted = ev?.type === 'ignored' && ev.reason !== 'return'; // a return follows every touch: not worth a line
  return (
    <main class="pm-main pm-center" data-testid="pm-watching">
      {k === 0 || !d ? (
        <>
          <Orb still={snap ? snap.still : true} />
          <p class="pm-lead">{s.t(sim ? 'watch_sim' : 'watch')}</p>
          <AttemptDots k={0} kMax={kMax} />
        </>
      ) : (
        <>
          <ProbabilityDisplay s={s} pRight={d.pRight} percentRight={d.percentRight} mode="provisional" k={d.nUsed} hand={d.hand} />
          <div class="pm-again">
            <AttemptDots k={k} kMax={kMax} />
            <span>{s.t('again_hint')}</span>
          </div>
        </>
      )}
      {hint ? (
        <p class="pm-hint" data-testid="pm-hint" role="status">
          {s.t('grip')}
        </p>
      ) : (
        notCounted && (
          <p class="pm-hint pm-hint-soft" data-testid="pm-not-counted" role="status">
            {s.t('not_counted')}
          </p>
        )
      )}
      {compare && snap && k > 0 && (
        <CompareTable s={s} rows={snap.compare} driver={snap.driver} k={snap.decision.nUsed} onSelect={onSelectModel} sim={sim} />
      )}
      <button type="button" class="btn btn-ghost pm-quiet" data-testid="pm-start-over" onClick={onStartOver}>
        {s.t('start_over')}
      </button>
    </main>
  );
}

export interface AnswerProps {
  readonly s: Strings;
  readonly snap: EngineSnapshot;
  readonly compare: boolean;
  readonly onAgain: () => void;
  readonly onToggleCompare: () => void;
  readonly onSelectModel: (id: ModelId) => void;
  /** the desktop demo (the CNN row is not applicable there) */
  readonly sim?: boolean;
}

/** a final answer that reached p_stop; a forced one (k_max, or the end of the stream) is shown as a leaning */
export function isConfident(snap: EngineSnapshot): boolean {
  const d = snap.decision;
  return d.status === 'confident' || Math.max(d.pRight, 1 - d.pRight) >= snap.stopping.pStop;
}

export function AnswerScreen({ s, snap, compare, onAgain, onToggleCompare, onSelectModel, sim = false }: AnswerProps) {
  const d = snap.decision;
  return (
    <main class="pm-main pm-center" data-testid="pm-answer">
      <ProbabilityDisplay
        s={s}
        pRight={d.pRight}
        percentRight={d.percentRight}
        mode="final"
        k={d.nUsed}
        hand={d.hand}
        confident={isConfident(snap)}
      />
      <button type="button" class="btn btn-primary btn-lg btn-block pm-cta" data-testid="pm-again" onClick={onAgain}>
        {s.t('again')}
      </button>
      <button type="button" class="pm-link" aria-expanded={compare} data-testid="pm-compare-toggle" onClick={onToggleCompare}>
        {s.t('compare')} <span aria-hidden="true">{compare ? '▾' : s.lang === 'he' ? '◂' : '▸'}</span>
      </button>
      {compare && <CompareTable s={s} rows={snap.compare} driver={snap.driver} k={d.nUsed} onSelect={onSelectModel} sim={sim} />}
    </main>
  );
}

export function AbstainScreen({ s, snap, onAgain }: { readonly s: Strings; readonly snap: EngineSnapshot; readonly onAgain: () => void }) {
  const d = snap.decision;
  return (
    <main class="pm-main pm-center" data-testid="pm-abstain">
      <AlertIcon />
      <h2 class="pm-big" role="status">
        {s.t('abstain')}
      </h2>
      {d.nUsed > 0 && <ProbabilityDisplay s={s} pRight={d.pRight} percentRight={d.percentRight} mode="muted" k={d.nUsed} hand={d.hand} />}
      <button type="button" class="btn btn-primary btn-lg btn-block pm-cta" data-testid="pm-again" onClick={onAgain}>
        {s.t('again')}
      </button>
    </main>
  );
}

export function GripScreen({ s, onStartOver }: { readonly s: Strings; readonly onStartOver: () => void }) {
  return (
    <main class="pm-main pm-center" data-testid="pm-grip">
      <TurnIcon />
      <h2 class="pm-big" role="status">
        {s.t('grip')}
      </h2>
      <button type="button" class="btn btn-primary btn-lg btn-block pm-cta" data-testid="pm-start-over" onClick={onStartOver}>
        {s.t('start_over')}
      </button>
    </main>
  );
}

export interface UnsupportedProps {
  readonly s: Strings;
  readonly failure: SourceFailure;
  readonly preflight: Pick<Preflight, 'engine' | 'desktop'> | null;
  readonly copied: boolean;
  readonly onAction: (action: Exclude<FailureAction, null>) => void;
  /** offer the desktop demo as a second, quiet link (not in the demo itself) */
  readonly demoLink?: boolean;
}

export function UnsupportedScreen({ s, failure, preflight, copied, onAction, demoLink = false }: UnsupportedProps) {
  const view = failureView(failure, preflight);
  const label = view.action === 'demo' ? s.t('try_demo') : view.action === 'copy' ? (copied ? s.t('copied') : s.t('copy_link')) : s.t('reload');
  return (
    <main class="pm-main pm-center" data-testid="pm-unsupported" data-failure={failure}>
      <AlertIcon />
      <p class="pm-msg" role="alert">
        {s.t(view.text)}
      </p>
      {view.action && (
        <button type="button" class="btn btn-primary btn-lg btn-block pm-cta" data-testid={`pm-action-${view.action}`} onClick={() => onAction(view.action!)}>
          {label}
        </button>
      )}
      {demoLink && view.action !== 'demo' && (
        <button type="button" class="pm-link" data-testid="pm-action-demo-link" onClick={() => onAction('demo')}>
          {s.t('try_demo')}
        </button>
      )}
    </main>
  );
}
