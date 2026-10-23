/**
 * Long-lived services created once at boot. Screens get them with getServices().
 */
import { openLocalStore, type LocalStore } from '../core/storage';
import { createUploadQueue, type UploadQueue } from '../core/uploader';
import { createCueSink, createWakeLockGuard, type AppCueSink, type WakeLockGuard } from '../core/cues';
import { createBrowserSensorSource } from '../core/sensors';
import { effectiveUpload, redactSession } from '../core/config';
import { makeRng, randomSeed } from '../core/util';
import type { SensorSource, TrialCondition } from '../core/types';
import { createSimulatedSource, type SimBehavior } from '../sim/simulatedSource';
import { defaultSimPlatform, simParticipantFromProfile } from '../sim/model';
import { getState, setState } from './store';

export interface SourceRequest {
  /** condition the simulated participant should perform (ignored by real sensors) */
  condition?: TrialCondition;
  /** simulated behaviour, e.g. 'static' for the phone-on-a-table calibration (ignored by real sensors) */
  behavior?: SimBehavior;
}

export interface Services {
  store: LocalStore;
  uploader: UploadQueue;
  cues: AppCueSink;
  wakeLock: WakeLockGuard;
  /** A fresh sensor source: real sensors, or the simulator when settings.simulate is on. */
  createSource(req?: SourceRequest): SensorSource;
}

export const DEFAULT_CONDITION: TrialCondition = {
  hand: 'right',
  orientation: 'upright_toward',
  speed: 'natural',
  start_position: 'natural',
  variant: 'standard',
  posture: 'standing',
  rep: 0,
};

let services: Services | null = null;

export function getServices(): Services {
  if (!services) throw new Error('services not initialised');
  return services;
}

/** Stable per-participant seed so the simulated person keeps the same "style" across trials. */
function seedFromId(id: string | undefined): number {
  let h = 2166136261;
  for (const ch of id ?? 'anonymous') h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return h >>> 0;
}

export async function initServices(): Promise<Services> {
  const st = getState();
  const store = await openLocalStore();
  const cues = createCueSink({ audio: st.settings.audio, vibrate: st.settings.vibrate, speech: st.settings.speech });
  const uploader = createUploadQueue(
    store,
    () => effectiveUpload(getState().config, getState().settings.upload_override),
    { redactSession: (s) => redactSession(s, getState().config) },
  );
  uploader.onChange((outbox) => setState({ outbox }));
  const wakeLock = createWakeLockGuard();

  const createSource = (req: SourceRequest = {}): SensorSource => {
    const now = getState();
    if (!now.settings.simulate) return createBrowserSensorSource({ quaternion: true, frequencyHz: 60 });
    const participantSeed = seedFromId(now.participant?.participant_id);
    const seed = randomSeed();
    return createSimulatedSource({
      mode: 'realtime',
      participant: simParticipantFromProfile(now.participant, participantSeed),
      platform: defaultSimPlatform('android', makeRng(participantSeed)),
      condition: req.condition ?? DEFAULT_CONDITION,
      seed,
      behavior: req.behavior ?? 'normal',
    });
  };

  services = { store, uploader, cues, wakeLock, createSource };
  uploader.start();
  return services;
}
