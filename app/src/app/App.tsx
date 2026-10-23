import { useEffect, useState } from 'preact/hooks';
import type { FunctionComponent } from 'preact';
import { fetchStudyConfig, refetchConfigWhenOnline } from '../core/config';
import { detectCapabilities, getDeviceInfo } from '../core/sensors';
import type { Language } from '../core/types';
import { applyDocumentLanguage, stringsFor, useStrings } from '../i18n';
import { Toast } from '../ui/components';
import { getServices, initServices } from './services';
import { loadSettings, persistSettings, type ScreenId } from './state';
import { installHistory, navigate, setState, useAppState } from './store';
import { WelcomeScreen } from './screens/WelcomeScreen';
import { ConsentScreen } from './screens/ConsentScreen';
import { ProfileScreen } from './screens/ProfileScreen';
import { SensorCheckScreen } from './screens/SensorCheckScreen';
import { CalibrationScreen } from './screens/CalibrationScreen';
import { SetupScreen } from './screens/SetupScreen';
import { SessionScreen } from './screens/SessionScreen';
import { DoneScreen } from './screens/DoneScreen';
import { DataScreen } from './screens/DataScreen';
import { SettingsScreen } from './screens/SettingsScreen';
import { SandboxScreen } from './screens/SandboxScreen';
import { AboutScreen } from './screens/AboutScreen';

function LoadingScreen() {
  const s = useStrings();
  return (
    <div class="screen">
      <main class="screen-body center" style={{ justifyContent: 'center' }}>
        <div class="spinner" role="status" aria-label={s.common.loading} />
      </main>
    </div>
  );
}

const SCREENS: Record<ScreenId, FunctionComponent> = {
  loading: LoadingScreen,
  welcome: WelcomeScreen,
  consent: ConsentScreen,
  profile: ProfileScreen,
  sensors: SensorCheckScreen,
  calibration: CalibrationScreen,
  setup: SetupScreen,
  session: SessionScreen,
  done: DoneScreen,
  data: DataScreen,
  settings: SettingsScreen,
  sandbox: SandboxScreen,
  about: AboutScreen,
};

/** Private-use character that marks where a translated sentence puts an interpolated value. */
const SLOT = '\uE000';

/**
 * `template(SLOT)` cut around the slot, so the caller can put an element (e.g. a <bdi>) where the
 * translation puts the value. A translation without the slot gets the value at the end.
 */
export function aroundSlot(template: (value: string) => string): [string, string] {
  const text = template(SLOT);
  const i = text.indexOf(SLOT);
  return i < 0 ? [`${text} `, ''] : [text.slice(0, i), text.slice(i + SLOT.length)];
}

/**
 * Boot fell back to the phone's saved copy or to the built-in defaults (offline start): load the
 * deployed config.json once the phone is online again, so uploads and the study settings recover
 * without a restart, and send what is waiting.
 */
function refreshConfigWhenOnline(): void {
  if (typeof window === 'undefined') return;
  refetchConfigWhenOnline(window, (config) => {
    setState({ config });
    try {
      getServices()
        .uploader.flush()
        .then(
          (r) => setState({ outbox: r.counts }),
          () => undefined,
        );
    } catch {
      // services not ready: the upload queue's own triggers send it later
    }
  });
}

/**
 * URL parameters (all optional):
 *   ?lang=en|he   UI language (remembered)
 *   ?sim=1        use the built-in sensor simulator for this run only (never remembered)
 */
async function boot(): Promise<void> {
  const { config, source } = await fetchStudyConfig();
  if (source !== 'network') refreshConfigWhenOnline();
  const settings = loadSettings(config.default_language);
  const params = new URLSearchParams(location.search);
  const lang = params.get('lang');
  if (lang === 'en' || lang === 'he') settings.language = lang as Language;
  persistSettings(settings);
  if (params.get('sim') === '1') settings.simulate = true;
  applyDocumentLanguage(settings.language);
  setState({ config, settings });

  const services = await initServices();
  const [device, outbox] = await Promise.all([getDeviceInfo(), services.store.outboxCounts()]);
  const participant = settings.last_participant_id
    ? ((await services.store.getParticipant(settings.last_participant_id)) ?? null)
    : null;
  setState({ device, capabilities: detectCapabilities(), participant, outbox });
  void services.store.requestPersistence();
  navigate('welcome');
}

export function App() {
  const st = useAppState();
  const [bootError, setBootError] = useState<string | null>(null);

  useEffect(() => {
    const uninstall = installHistory(window);
    boot().catch((e: unknown) => {
      console.error(e);
      setBootError(e instanceof Error ? e.message : String(e));
    });
    return uninstall;
  }, []);

  useEffect(() => applyDocumentLanguage(st.settings.language), [st.settings.language]);

  if (bootError !== null) {
    // storage blocked (Safari "Block All Cookies", a WebView without storage): the language is
    // already known, the browser's own message stays left-to-right inside a Hebrew sentence
    const c = stringsFor(st.settings.language).common;
    const [before, after] = aroundSlot(c.bootFailed);
    return (
      <div class="screen" data-testid="screen-boot-failed">
        <main class="screen-body">
          <h1>{c.appName}</h1>
          <div class="notice notice-error" role="alert">
            {before}
            <bdi dir="ltr">{bootError || '?'}</bdi>
            {after}
          </div>
        </main>
      </div>
    );
  }

  const Current = SCREENS[st.screen] ?? LoadingScreen;
  return (
    <>
      <Current />
      <Toast />
    </>
  );
}
