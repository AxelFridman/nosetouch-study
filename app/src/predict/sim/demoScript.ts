/**
 * The desktop demo's script (?predict=1&sim=1, docs/stage3/APP_SPEC.md section 3): a simulated person
 * who holds the phone out, still, and touches the nose six times. Generated synchronously at Start from
 * one seed, so &seed=<n> replays the same samples. It is a known-answer check of the page, not evidence
 * of accuracy: the samples come from app/src/sim (a kinematic model), never from a real recording.
 *
 * Each touch is its own offline SimulatedSource that starts where the previous one stopped:
 *  - GO once the phone has been still for 1.2 to 2.0 s (the person decides to move; nobody cues them),
 *  - END 0.4 to 0.9 s after the phone reached the nose (the person brings it back out),
 *  - stop 1.0 to 1.5 s after the return reached arm's length.
 * Orientation events are dropped: predict mode reads devicemotion only.
 *
 * The grip: the simulator's 'upright_toward' holds the screen exactly vertical (rest up_z about 0, half of
 * the seeds slightly negative), which lane m4's grip gate (rest up_z < 0, m4_config.GRIP_GATE_UPZ) would
 * refuse. People hold the phone tilted back, screen towards the face and a little up, so the demo person
 * holds it tilted back by 15 to 35 deg: one constant grip rotation Rx(tilt) applied to every device-frame
 * vector (acc, accg and the rotation rate, which transform alike under a constant rotation), re-rounded on
 * Chromium's grids on Android like the simulator's own readings.
 */
import type { Euler, Hand, MotionSample, OrientationSample, TrialCondition, Vec3 } from '../../core/types';
import { CHROMIUM_ROUNDING, createSimulatedSource, roundToMultiple } from '../../sim/simulatedSource';
import { defaultSimPlatform, deriveSeed, randomSimParticipant, simRng } from '../../sim/model';
import type { BrowserEngine } from '../types';

export const DEMO_TOUCHES = 6;
/** where the first touch's virtual clock starts (ms); the replay rebases t onto performance.now() */
export const DEMO_START_T = 1000;
/** range of the demo person's backward tilt of the phone (deg; see the module doc) */
export const DEMO_TILT_DEG: readonly [number, number] = [15, 35];

/** A constant grip rotation Rx(tiltDeg) of every device-frame vector of a sample (rounded like Chrome when `quantize`). */
export function tiltSample(s: MotionSample, tiltDeg: number, quantize: boolean): MotionSample {
  const a = (tiltDeg * Math.PI) / 180;
  const c = Math.cos(a);
  const sn = Math.sin(a);
  const accStep = CHROMIUM_ROUNDING.acc_ms2;
  const gyroStep = CHROMIUM_ROUNDING.gyro_rad_s;
  const qa = (x: number): number => (quantize ? roundToMultiple(x, accStep) : x);
  const qg = (x: number): number => (quantize ? roundToMultiple(x * (Math.PI / 180), gyroStep) * (180 / Math.PI) : x);
  const rot = (v: Vec3 | null): Vec3 | null => (v === null ? null : [qa(v[0]), qa(c * v[1] - sn * v[2]), qa(sn * v[1] + c * v[2])]);
  const rr = (e: Euler | null): Euler | null =>
    e === null ? null : { alpha: qg(e.alpha), beta: qg(c * e.beta - sn * e.gamma), gamma: qg(sn * e.beta + c * e.gamma) };
  return { ...s, acc: rot(s.acc), accg: rot(s.accg), rr: rr(s.rr) };
}

export interface DemoTouch {
  goT: number;
  onsetT: number | null;
  endT: number | null;
  returnEndT: number | null;
  stopT: number;
}

export interface DemoScript {
  readonly seed: number;
  readonly hand: Hand;
  readonly platform: 'ios' | 'android';
  /** the engine whose accelerometer sign the samples carry (ios: webkit, gravity sign -1) */
  readonly engine: BrowserEngine;
  /** devicemotion samples on the virtual clock, strictly increasing t */
  readonly samples: readonly MotionSample[];
  readonly touches: readonly DemoTouch[];
}

/** A demo person: the platform, the hand and the timing are drawn once; touches are made on demand. */
export interface DemoGenerator {
  readonly seed: number;
  readonly hand: Hand;
  readonly platform: 'ios' | 'android';
  readonly engine: BrowserEngine;
  /** the backward tilt of the phone in the hand (deg) */
  readonly tiltDeg: number;
  /** the samples of the next touch (strictly increasing t, after every earlier sample); null after `maxTouches` */
  nextTouch(): { samples: MotionSample[]; touch: DemoTouch } | null;
}

/**
 * The demo person of one run. `hand` null: drawn from the seed (the URL's &hand wins). The first
 * DEMO_TOUCHES touches are the script of the spec; the replay asks for more (up to `maxTouches`, about
 * three minutes) only when the engine has not decided by then.
 */
export function createDemoGenerator(seed: number, hand: Hand | null, maxTouches = 30): DemoGenerator {
  const rng = simRng(seed >>> 0);
  const handDraw = rng();
  const platformDraw = rng();
  const H: Hand = hand ?? (handDraw < 0.5 ? 'left' : 'right');
  const platformName: 'ios' | 'android' = platformDraw < 0.5 ? 'ios' : 'android';
  const platform = defaultSimPlatform(platformName, simRng(deriveSeed(seed, 1)));
  const participant = randomSimParticipant(simRng(deriveSeed(seed, 2)));
  const timing = simRng(deriveSeed(seed, 3));
  const tiltDeg = DEMO_TILT_DEG[0] + (DEMO_TILT_DEG[1] - DEMO_TILT_DEG[0]) * simRng(deriveSeed(seed, 4))();
  const quantize = platform.quantize === true;
  const condition: TrialCondition = {
    hand: H,
    orientation: 'upright_toward',
    speed: 'natural',
    start_position: 'natural',
    variant: 'standard',
    posture: 'standing',
    rep: 0,
  };
  let tStart = DEMO_START_T;
  let lastT = -Infinity;
  let i = 0;

  function nextTouch(): { samples: MotionSample[]; touch: DemoTouch } | null {
    if (i >= maxTouches) return null;
    const stillMs = 1200 + 800 * timing();
    const endDelayMs = 400 + 500 * timing();
    const tailMs = 1000 + 500 * timing();
    const source = createSimulatedSource({
      mode: 'offline',
      participant,
      platform,
      condition: { ...condition, rep: i },
      seed: deriveSeed(seed, 100 + i),
      startTime: tStart,
      maxDurationMs: 30_000,
    });
    const samples: MotionSample[] = [];
    let goT: number | null = null;
    let endCue = false;
    let stopT = tStart;
    void source.start({
      onMotion(s: MotionSample): void {
        if (!(s.t > lastT)) return; // the replay needs strictly increasing times (junctions)
        samples.push(tiltSample(s, tiltDeg, quantize));
        lastT = s.t;
        stopT = s.t;
        const truth = source.truth;
        if (goT === null && s.t >= tStart + stillMs) {
          goT = s.t;
          source.notifyCue?.('go', s.t);
        } else if (goT !== null && !endCue && truth.end_t !== null && s.t >= truth.end_t + endDelayMs) {
          endCue = true;
          source.notifyCue?.('end', s.t);
        } else if (endCue && typeof truth.return_end_t === 'number' && s.t >= truth.return_end_t + tailMs) {
          source.stop();
        }
      },
      onOrientation(_s: OrientationSample): void {
        // dropped: predict mode reads devicemotion only
      },
    });
    source.stop(); // offline: start() already ran to the stop (or to maxDurationMs)
    const truth = source.truth;
    i += 1;
    tStart = stopT + 1000 / 60;
    return {
      samples,
      touch: {
        goT: goT ?? stopT,
        onsetT: truth.onset_t,
        endT: truth.end_t,
        returnEndT: typeof truth.return_end_t === 'number' ? truth.return_end_t : null,
        stopT,
      },
    };
  }

  return {
    seed: seed >>> 0,
    hand: H,
    platform: platformName,
    engine: platformName === 'ios' ? 'webkit' : 'blink',
    tiltDeg,
    nextTouch,
  };
}

/** One run of the demo as a whole script (tests, fixtures): the first `touches` touches. */
export function makeDemoScript(seed: number, hand: Hand | null, touches = DEMO_TOUCHES): DemoScript {
  const gen = createDemoGenerator(seed, hand, touches);
  const samples: MotionSample[] = [];
  const out: DemoTouch[] = [];
  for (let next = gen.nextTouch(); next; next = gen.nextTouch()) {
    for (const s of next.samples) samples.push(s);
    out.push(next.touch);
  }
  return { seed: gen.seed, hand: gen.hand, platform: gen.platform, engine: gen.engine, samples, touches: out };
}
