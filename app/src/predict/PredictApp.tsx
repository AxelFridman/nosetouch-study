/**
 * Predict mode's page (docs/stage3/APP_SPEC.md section 5): preflight -> Start -> (permission) -> the
 * engine's snapshots. Cue-free: after one tap the page watches the stream; every detected approach is an
 * attempt, P(right) updates after each one and the driver model's stopping rule decides when it is final.
 *
 * Rules kept from the collector (docs/08): requestMotionPermission() is the FIRST statement of the Start
 * tap (iOS shows the prompt only inside the gesture), the wake lock is requested in the same tap, nothing
 * is stored or uploaded, and the page never makes a sound (one 30 ms vibration per attempt on phones
 * that have it, never under automation or in the demo).
 */
import { Component, type ComponentChildren } from 'preact';
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { APP_VERSION } from '../core/constants';
import { createWakeLockGuard, isAutomatedBrowser, type WakeLockGuard } from '../core/cues';
import { requestMotionPermission } from '../core/sensors';
import { mixSeed, randomSeed } from '../core/util';
import { strings } from './i18n/strings';
import { createBrowserLiveSource } from './live/browserSource';
import { createDemoSource } from './live/demoSource';
import type { EngineBinding } from './live/engine';
import { demoPreflight, runPreflight } from './live/preflight';
import { createLiveSession, type LiveSession } from './live/session';
import type { EngineSnapshot, Hand, LiveSource, ModelId, Preflight, PredictParams, SourceFailure } from './types';
import { DebugPanel, Footer, InfoSheet, ModelSheet, TopBar } from './ui/components';
import { demoSearch, withParam } from './ui/params';
import {
  AbstainScreen,
  AnswerScreen,
  GripScreen,
  PermissionScreen,
  screenOf,
  StartScreen,
  UnsupportedScreen,
  WatchingScreen,
  type FailureAction,
  type Stage,
} from './ui/screens';

/** A fixed state to render instead of the live page (dev-only screenshots: ?predict=1&preview=<screen>). */
export interface PreviewState {
  readonly stage: Stage;
  readonly failure: SourceFailure | null;
  readonly snapshot: EngineSnapshot | null;
  readonly preflight?: Preflight | null;
  readonly demoHand?: Hand | null;
}

export interface PredictAppProps {
  readonly params: PredictParams;
  readonly binding: EngineBinding;
  readonly preview?: PreviewState | null;
}

type Sheet = 'none' | 'models' | 'info';

/** the build's short git commit (vite.config.ts define; '' in tests and when git is not available) */
const BUILD_ID: string = typeof __PREDICT_BUILD__ === 'string' ? __PREDICT_BUILD__ : '';

/** A render error shows one line and Reload instead of a blank page. */
export class ErrorBoundary extends Component<{ readonly message: string; readonly reload: string; readonly children?: ComponentChildren }, { failed: boolean }> {
  state = { failed: false };
  componentDidCatch(): void {
    this.setState({ failed: true });
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main class="pm-main pm-center" data-testid="pm-error">
        <p class="pm-msg" role="alert">
          {this.props.message}
        </p>
        <button type="button" class="btn btn-primary btn-lg btn-block pm-cta" onClick={() => location.reload()}>
          {this.props.reload}
        </button>
      </main>
    );
  }
}

function vibrateOnce(): void {
  try {
    navigator.vibrate?.(30);
  } catch {
    // not supported: nothing to do
  }
}

export function PredictApp({ params, binding, preview = null }: PredictAppProps) {
  const s = useMemo(() => strings(params.lang), [params.lang]);
  const [stage, setStage] = useState<Stage>(preview?.stage ?? 'start');
  const [failure, setFailure] = useState<SourceFailure | null>(preview?.failure ?? null);
  const [preflight, setPreflight] = useState<Preflight | null>(preview?.preflight ?? (params.sim ? demoPreflight('unknown') : null));
  const [snap, setSnap] = useState<EngineSnapshot | null>(preview?.snapshot ?? null);
  const [driver, setDriver] = useState<ModelId>(preview?.snapshot?.driver ?? params.model);
  const [compare, setCompare] = useState<boolean>(params.compare);
  const [sheet, setSheet] = useState<Sheet>('none');
  const [restarted, setRestarted] = useState(false);
  const [copied, setCopied] = useState(false);
  const [demoHand, setDemoHand] = useState<Hand | null>(preview?.demoHand ?? params.simHand);
  const session = useRef<LiveSession | null>(null);
  const wake = useRef<WakeLockGuard | null>(null);
  const preflightRef = useRef<Preflight | null>(preflight);
  preflightRef.current = preflight;

  // the preflight blocks Start on a page that cannot sense (the demo never needs it)
  useEffect(() => {
    if (preview || params.sim) return undefined;
    let alive = true;
    void runPreflight().then((p) => {
      if (!alive) return;
      setPreflight(p);
      if (!p.ok && p.failure) setFailure(p.failure);
    });
    return () => {
      alive = false;
    };
  }, []);

  // the session and the wake lock end with the page
  useEffect(
    () => () => {
      session.current?.dispose();
      session.current = null;
      void wake.current?.release();
    },
    [],
  );

  const fail = (reason: SourceFailure): void => {
    session.current?.dispose();
    session.current = null;
    void wake.current?.release();
    setFailure(reason);
  };

  const makeSource = (run: number): LiveSource => {
    if (params.sim) {
      const seed = params.simSeed === null ? randomSeed() : run === 0 ? params.simSeed : mixSeed(params.simSeed, run);
      const src = createDemoSource({ seed, hand: params.simHand });
      setDemoHand(src.hand);
      return src;
    }
    return createBrowserLiveSource(preflightRef.current?.engine ?? 'unknown');
  };

  const begin = (): void => {
    session.current?.dispose();
    const live = createLiveSession({
      binding,
      driver,
      makeSource,
      // a new source per run: the sensors are stopped once an answer is shown and started again by Again
      restartSourceOnReset: true,
      onSnapshot: setSnap,
      onFailure: fail,
      onAttempt: () => {
        setRestarted(false);
        if (!params.sim && preflightRef.current?.vibrate && !isAutomatedBrowser()) vibrateOnce();
      },
      onRestarted: () => setRestarted(true),
    });
    session.current = live;
    setSnap(live.engine.snapshot());
    setStage('running');
    void live.start().catch(() => fail('not_supported'));
  };

  const onStart = (): void => {
    // FIRST statement of the tap: iOS shows the motion prompt only inside the user gesture (no await above)
    const permission = params.sim ? null : requestMotionPermission();
    if (!params.sim && preflightRef.current?.wakeLock) {
      wake.current = wake.current ?? createWakeLockGuard();
      void wake.current.acquire(); // in the same gesture: Safari refuses it later
    }
    if (permission === null) {
      begin();
      return;
    }
    if (preflightRef.current?.needsPermission) setStage('permission');
    void permission.then((state) => {
      if (state === 'denied') fail('permission_denied');
      else begin();
    });
  };

  const onReset = (): void => {
    setRestarted(false);
    if (!params.sim && preflightRef.current?.wakeLock && wake.current) void wake.current.acquire(); // in the tap
    session.current?.reset();
  };

  const selectModel = (id: ModelId): void => {
    setDriver(id);
    session.current?.setDriver(id);
    try {
      history.replaceState(history.state, '', `${location.pathname}${withParam(location.search, 'model', id)}${location.hash}`);
    } catch {
      // a sandboxed frame may refuse it: the selection still applies
    }
  };

  const onAction = (action: Exclude<FailureAction, null>): void => {
    if (action === 'demo') {
      location.assign(`${location.pathname}${demoSearch(location.search)}`);
    } else if (action === 'reload') {
      location.reload();
    } else {
      const url = location.href;
      const done = (): void => setCopied(true);
      try {
        void navigator.clipboard.writeText(url).then(done, () => window.prompt(s.t('copy_link'), url));
      } catch {
        window.prompt(s.t('copy_link'), url);
      }
    }
  };

  const screen = screenOf({ stage, failure, snapshot: snap });

  // an answer (or a refusal) is final: stop the sensors and let the screen sleep; Again starts both again
  const done = screen === 'answer' || screen === 'abstain' || screen === 'grip';
  useEffect(() => {
    if (!done) return;
    session.current?.stopSource();
    void wake.current?.release();
  }, [done]);
  const specs = binding.specs;
  const shownDriver = snap?.driver ?? driver;
  const driverSpec = specs.find((m) => m.id === shownDriver) ?? specs[0];
  const desktopDemo = !params.sim && preflight?.desktop ? `${location.pathname}${demoSearch(location.search)}` : null;

  let body;
  switch (screen) {
    case 'start':
      body = <StartScreen s={s} onStart={onStart} demoHref={desktopDemo} ready={params.sim || preview !== null || preflight !== null} sim={params.sim} />;
      break;
    case 'permission':
      body = <PermissionScreen s={s} />;
      break;
    case 'answer':
      body = (
        <AnswerScreen
          s={s}
          snap={snap!}
          compare={compare}
          onAgain={onReset}
          onToggleCompare={() => setCompare(!compare)}
          onSelectModel={selectModel}
          sim={params.sim}
        />
      );
      break;
    case 'abstain':
      body = <AbstainScreen s={s} snap={snap!} onAgain={onReset} />;
      break;
    case 'grip':
      body = <GripScreen s={s} onStartOver={onReset} />;
      break;
    case 'unsupported':
      body = <UnsupportedScreen s={s} failure={failure!} preflight={preflight} copied={copied} onAction={onAction} demoLink={!params.sim} />;
      break;
    case 'watching':
    default:
      body = <WatchingScreen s={s} snap={snap} compare={compare} onSelectModel={selectModel} onStartOver={onReset} sim={params.sim} />;
  }

  return (
    <div class="pm" data-engine={binding.kind} data-screen={screen} data-sim={params.sim ? '1' : '0'}>
      <TopBar s={s} driver={shownDriver} onModels={() => setSheet('models')} />
      {params.sim && (
        <div class="pm-badge" data-testid="pm-demo-badge">
          {demoHand ? s.t('demo_badge', { HAND: s.hand(demoHand) }) : s.t('demo')}
        </div>
      )}
      {params.sim && shownDriver === 'cnn' && (
        <div class="pm-notice" role="note" data-testid="pm-sim-cnn">
          {s.t('sim_cnn')}
        </div>
      )}
      {restarted && screen === 'watching' && (
        <div class="pm-notice" role="status" data-testid="pm-restarted">
          {s.t('restarted')}
        </div>
      )}
      <ErrorBoundary message={s.t('u_error')} reload={s.t('reload')}>
        {body}
      </ErrorBoundary>
      {params.debug && snap && (
        <DebugPanel
          snap={snap}
          ids={specs.map((m) => m.id)}
          engineKind={binding.kind}
          ignoredParams={params.ignored}
          timing={session.current?.timing() ?? null}
        />
      )}
      <Footer
        s={s}
        driver={shownDriver}
        sha256={driverSpec?.sha256 ?? ''}
        version={APP_VERSION}
        build={BUILD_ID}
        demo={params.sim}
        onInfo={() => setSheet('info')}
      />
      {sheet === 'models' && (
        <ModelSheet
          s={s}
          specs={specs}
          driver={shownDriver}
          compare={compare}
          onSelect={(id) => {
            selectModel(id);
            setSheet('none');
          }}
          onCompare={setCompare}
          onClose={() => setSheet('none')}
          sim={params.sim}
        />
      )}
      {sheet === 'info' && <InfoSheet s={s} onClose={() => setSheet('none')} />}
    </div>
  );
}
