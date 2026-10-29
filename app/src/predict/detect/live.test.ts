/** The live helpers on the committed synthetic fixtures: the freeze rule per tick and the screen-away hint. */
import { describe, expect, it } from 'vitest';
import type { Bout, Stream } from '../types';
import { cropStream } from './canonical';
import { liveBouts, sameBout, screenAwayHint } from './live';
import { SYNTH_DIR, loadDetector, loadFixtures, streamOf } from './testSupport';

const det = loadDetector();
const fixtures = loadFixtures(SYNTH_DIR);

/** the first n samples of a stream (the full recording's values: a stand-in for a settled prefix) */
const prefix = (s: Stream, n: number): Stream => cropStream(s, -Infinity, s.t[n - 1]);

describe('liveBouts', () => {
  it('flags exactly the bouts the reference freezes at the end of each recording', () => {
    for (const fx of fixtures) {
      const lb = liveBouts(streamOf(fx), det, fx.live.settle_ms);
      expect(lb.frozen, fx.id).toEqual(fx.reference.segments.map((s) => s.live_frozen));
      expect(lb.nLook, fx.id).toBe(fx.reference.detector.n_look);
    }
  });

  it('a bout, once frozen on a prefix, stays frozen and keeps its bounds on every longer prefix', () => {
    for (const fx of fixtures) {
      const s = streamOf(fx);
      const seen: Bout[] = [];
      const ns: number[] = [];
      for (let n = 30; n < s.t.length; n += 6) ns.push(n);
      ns.push(s.t.length);
      for (const n of ns) {
        const lb = liveBouts(prefix(s, n), det);
        lb.bouts.forEach((b, k) => {
          if (!lb.frozen[k]) return;
          const prev = seen.find((x) => sameBout(x, b));
          if (prev) expect([b.onsetMs, b.endMs, b.decidedMs, b.iOn, b.iEnd], fx.id).toEqual([prev.onsetMs, prev.endMs, prev.decidedMs, prev.iOn, prev.iEnd]);
          else seen.push(b);
        });
      }
      expect(seen.length, fx.id).toBe(fx.reference.segments.filter((x) => x.live_frozen).length);
    }
  });
});

describe('screenAwayHint', () => {
  it('is set at rest with the screen away (syn03), not with the screen towards the face (syn01)', () => {
    const away = fixtures.find((f) => f.id.startsWith('syn03'))!;
    const toward = fixtures.find((f) => f.id.startsWith('syn01'))!;
    const at = (fx: (typeof fixtures)[number], ms: number): boolean => {
      const s = streamOf(fx);
      const n = s.t.findIndex((t) => t >= ms);
      const p = prefix(s, n);
      return screenAwayHint(p, liveBouts(p, det).activity, det);
    };
    // 1.2 s: still at arm's length before the first movement in both recordings
    expect(at(away, 1200)).toBe(true);
    expect(at(toward, 1200)).toBe(false);
  });
});
