import { describe, expect, it } from 'vitest';
import { createDemoGenerator, DEMO_TILT_DEG, DEMO_TOUCHES, makeDemoScript, tiltSample } from './demoScript';
import { createDemoSource } from '../live/demoSource';
import type { MotionSample } from '../../core/types';

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
};

describe('demo script (?predict=1&sim=1)', () => {
  it('is reproducible from its seed and keeps the requested hand', () => {
    const a = makeDemoScript(7, 'right');
    const b = makeDemoScript(7, 'right');
    expect(a.samples.length).toBeGreaterThan(1500);
    expect(b.samples).toEqual(a.samples);
    expect(a.hand).toBe('right');
    expect(makeDemoScript(7, 'left').hand).toBe('left');
    expect(a.touches).toHaveLength(DEMO_TOUCHES);
  });

  it('draws the hand from the seed when none is given, and both hands occur', () => {
    const hands = new Set(Array.from({ length: 12 }, (_, i) => createDemoGenerator(i + 1, null).hand));
    expect(hands).toEqual(new Set(['left', 'right']));
  });

  it('delivers devicemotion samples only, strictly increasing in time, one touch after the other', () => {
    const s = makeDemoScript(3, null);
    for (let i = 1; i < s.samples.length; i++) expect(s.samples[i].t).toBeGreaterThan(s.samples[i - 1].t);
    for (let i = 1; i < s.touches.length; i++) expect(s.touches[i].goT).toBeGreaterThan(s.touches[i - 1].stopT);
    for (const t of s.touches) {
      expect(t.onsetT).not.toBeNull();
      expect(t.endT).not.toBeNull();
      // the person is still for 1.2-2.0 s before moving (nobody cues them)
      expect(t.onsetT!).toBeGreaterThan(t.goT);
    }
    expect(s.engine).toBe(s.platform === 'ios' ? 'webkit' : 'blink');
  });

  it('holds the phone tilted back, screen towards the face: rest up_z well above the grip gate (0)', () => {
    for (let seed = 1; seed <= 16; seed++) {
      const s = makeDemoScript(seed, null, 1);
      const gs = s.engine === 'webkit' ? -1 : 1;
      const t0 = s.samples[0].t;
      const upz = s.samples
        .filter((x) => x.t - t0 < 1000 && x.accg)
        .map((x) => (gs * x.accg![2]) / Math.hypot(x.accg![0], x.accg![1], x.accg![2]));
      expect(median(upz), `seed ${seed}`).toBeGreaterThan(0.15);
    }
  });

  it('the tilt is one rotation of every device vector: lengths and the yaw about gravity are kept', () => {
    const s: MotionSample = { t: 0, ts: 0, acc: [0.1, 0.2, 0.3], accg: [0, 9.8, 0], rr: { alpha: 10, beta: 20, gamma: 30 }, interval: 16 };
    const r = tiltSample(s, 30, false);
    expect(Math.hypot(...r.accg!)).toBeCloseTo(9.8, 12);
    expect(r.accg![2]).toBeCloseTo(9.8 * Math.sin(Math.PI / 6), 12); // up_z = sin(tilt) > 0
    const dot = (g: number[], w: number[]): number => g[0] * w[0] + g[1] * w[1] + g[2] * w[2];
    const w0 = [10, 20, 30];
    const w1 = [r.rr!.alpha, r.rr!.beta, r.rr!.gamma];
    expect(dot(r.accg!, w1)).toBeCloseTo(dot(s.accg!, w0), 9);
    expect(Math.hypot(...w1)).toBeCloseTo(Math.hypot(...w0), 12);
    const q = tiltSample(s, 30, true);
    expect(Math.abs(q.accg![1] * 10 - Math.round(q.accg![1] * 10))).toBeLessThan(1e-9); // Chrome's 0.1 m/s^2 grid
    expect(DEMO_TILT_DEG[0]).toBeGreaterThan(0);
  });
});

describe('demo source replay', () => {
  it('emits the script in real time, rebased onto the wall clock, through the LiveSource interface', async () => {
    let wall = 5000;
    const ticks: (() => void)[] = [];
    const realSetInterval = globalThis.setInterval;
    const realClearInterval = globalThis.clearInterval;
    // a manual timer: each tick() is one interval period
    (globalThis as { setInterval: unknown }).setInterval = (fn: () => void) => {
      ticks.push(fn);
      return 1 as unknown as ReturnType<typeof setInterval>;
    };
    (globalThis as { clearInterval: unknown }).clearInterval = () => {
      ticks.length = 0;
    };
    try {
      const src = createDemoSource({ seed: 7, hand: 'left', now: () => wall });
      expect(src.kind).toBe('demo');
      expect(src.hand).toBe('left');
      const got: MotionSample[] = [];
      await src.start((s) => got.push(s), () => undefined);
      ticks[0]();
      expect(got).toHaveLength(1); // the first sample, at the start time
      expect(got[0].t).toBe(5000);
      wall += 1000;
      ticks[0]();
      const n1 = got.length;
      expect(n1).toBeGreaterThan(50);
      expect(n1).toBeLessThan(70);
      expect(got[n1 - 1].t).toBeLessThanOrEqual(6000);
      for (let i = 1; i < got.length; i++) expect(got[i].t).toBeGreaterThan(got[i - 1].t);
      const script = makeDemoScript(7, 'left', 1);
      expect(got[1].t - got[0].t).toBeCloseTo(script.samples[1].t - script.samples[0].t, 9);
      expect(got[1].accg).toEqual(script.samples[1].accg);
      src.stop();
      expect(ticks).toHaveLength(0);
    } finally {
      globalThis.setInterval = realSetInterval;
      globalThis.clearInterval = realClearInterval;
    }
  });
});
