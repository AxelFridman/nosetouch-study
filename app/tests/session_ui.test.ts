import { describe, expect, it, vi } from 'vitest';
import { ALL_ORIENTATIONS, DEFAULT_SEGMENTATION_PARAMS, ORIENTATIONS } from '../src/core/constants';
import type {
  MotionSample,
  SegmentationResult,
  TrialMessage,
  TrialPhase,
  TrialRecord,
} from '../src/core/types';
import type { SegmenterEvent, SegmenterLive, SegmenterState, RelMotionSample } from '../src/core/segmentation';
import {
  drawSignalPlot,
  ema,
  finiteMax,
  motionMagnitudes,
  niceCeil,
  niceStep,
  niceTicks,
  plotRanges,
  stepDecimals,
  type SignalPlotProps,
} from '../src/ui/SignalPlot';
import { phoneFigureSpec } from '../src/ui/PhoneFigure';
import {
  failureKind,
  panelTone,
  permissionDeniedText,
  stopAllowed,
  uiAbortWarning,
  type FailureKind,
} from '../src/app/screens/SessionScreen';
import { deniedContext, deniedText, iosBrowserName, issueText } from '../src/app/screens/SensorCheckScreen';
import { SandboxDetector, type SegmenterLike } from '../src/app/screens/SandboxScreen';
import { aroundSlot } from '../src/app/App';
import type { DeviceInfo } from '../src/core/types';
import { session as enSession } from '../src/i18n/en/session';
import { session as heSession } from '../src/i18n/he/session';
import { en } from '../src/i18n/en';
import { he } from '../src/i18n/he';
import { makeDevice, makeMotionStream } from './helpers/sessionFakes';

// ───────────────────────────── SignalPlot helpers ─────────────────────────────

describe('plot scales', () => {
  it('niceCeil rounds axis limits up on a fine grid, niceStep tick steps on a coarse one', () => {
    expect(niceCeil(0.7)).toBe(0.8);
    expect(niceCeil(1)).toBe(1);
    expect(niceCeil(1.3)).toBe(1.5);
    expect(niceCeil(2.2)).toBe(2.5);
    expect(niceCeil(3.1)).toBe(4);
    expect(niceCeil(7)).toBe(8);
    expect(niceCeil(9.81 * 1.05)).toBe(12); // gravity gets a tight axis
    expect(niceCeil(230)).toBe(250);
    expect(niceCeil(0)).toBe(1);
    expect(niceCeil(Number.NaN)).toBe(1);
    expect(niceStep(0.3)).toBe(0.5);
    expect(niceStep(3)).toBe(5);
    expect(niceStep(6)).toBe(10);
  });

  it('niceTicks are evenly spaced, inside the range and without float noise', () => {
    expect(niceTicks(0, 10, 4)).toEqual([0, 2.5, 5, 7.5, 10]);
    expect(niceTicks(0, 1, 4)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(niceTicks(-10, 10, 4)).toEqual([-10, -5, 0, 5, 10]);
    expect(niceTicks(0.1, 0.7, 3)).toEqual([0.2, 0.4, 0.6]);
    expect(niceTicks(5, 5)).toEqual([]);
    expect(niceTicks(Number.NaN, 1)).toEqual([]);
    for (const v of niceTicks(12.3, 17.9, 5)) expect(v).toBeGreaterThanOrEqual(12.3);
    expect(stepDecimals(2.5)).toBe(1);
    expect(stepDecimals(0.25)).toBe(2);
    expect(stepDecimals(50)).toBe(0);
  });

  it('plotRanges: zero-based by default, includes negatives and thresholds, honours fixed ranges', () => {
    const base: SignalPlotProps = { t: [0, 100, 200], series: [{ values: [1, 3, 2], color: 'red', label: 'a' }] };
    expect(plotRanges(base)).toEqual({ x: [0, 200], y: [0, 4], hasData: true });
    expect(plotRanges({ ...base, series: [{ values: [-9.6, 0, 9.8], color: 'red', label: 'g' }] }).y).toEqual([-12, 12]);
    expect(plotRanges({ ...base, hLines: [{ y: 30 }] }).y).toEqual([0, 40]);
    expect(plotRanges({ ...base, yMin: 0, yMax: 1.08 }).y).toEqual([0, 1.08]);
    expect(plotRanges({ ...base, xRange: [100, 200] }).x).toEqual([100, 200]);
    // values outside a fixed time window do not stretch the y axis
    expect(plotRanges({ ...base, xRange: [150, 250] }).y).toEqual([0, 2.5]);
  });

  it('plotRanges never degenerates on empty or non-finite data', () => {
    const empty = plotRanges({ t: [], series: [] });
    expect(empty.hasData).toBe(false);
    expect(empty.x[1]).toBeGreaterThan(empty.x[0]);
    expect(empty.y[1]).toBeGreaterThan(empty.y[0]);
    const nan = plotRanges({ t: [0, 1], series: [{ values: [Number.NaN, Number.POSITIVE_INFINITY], color: 'x', label: 'x' }] });
    expect(nan.hasData).toBe(false);
    expect(nan.y).toEqual([0, 1]);
    const flat = plotRanges({ t: [5, 5], series: [{ values: [0, 0], color: 'x', label: 'x' }] });
    expect(flat.x).toEqual([5, 6]);
  });

  it('finiteMax ignores non-finite values', () => {
    expect(finiteMax([1, Number.NaN, 7], [3])).toBe(7);
    expect(finiteMax([Number.NaN])).toBeNull();
  });
});

describe('ema and motionMagnitudes (online_v1 signal rules)', () => {
  it('ema starts at x0, follows k = 1 − exp(−dt/τ), and τ = 0 disables smoothing', () => {
    const t = [0, 10, 20, 30];
    expect(ema(t, [5, 5, 5, 5], 40)).toEqual([5, 5, 5, 5]);
    expect(ema(t, [0, 10, 10, 10], 0)).toEqual([0, 10, 10, 10]);
    const s = ema(t, [0, 10, 10, 10], 40);
    expect(s[1]).toBeCloseTo(10 * (1 - Math.exp(-10 / 40)), 10);
    expect(s[1]).toBeLessThan(s[2]);
    expect(s[3]).toBeLessThan(10);
  });

  it('uses linear acceleration when the first sample has it', () => {
    const m = makeMotionStream(150, { linear: true });
    const r = motionMagnitudes(m, { smoothingMs: 0, gravityTauMs: 600 });
    expect(r.accSource).toBe('linear');
    expect(r.t).toEqual(m.t);
    expect(r.gyro[0]).toBeCloseTo(Math.hypot(1, 0.5, 0), 10);
    const iMove = m.t.indexOf(1200);
    expect(r.gyro[iMove]).toBeCloseTo(130, 10);
    expect(r.acc[0]).toBe(0);
    expect(r.acc[iMove]).toBeGreaterThan(2);
  });

  it('falls back to a gravity high-pass when only accelerationIncludingGravity exists', () => {
    const m = makeMotionStream(150, { linear: false });
    const r = motionMagnitudes(m, { smoothingMs: 40, gravityTauMs: 600 });
    expect(r.accSource).toBe('highpass');
    // constant gravity is removed
    expect(r.acc[40]).toBeLessThan(1e-9);
    expect(Math.max(...r.acc)).toBeGreaterThan(0.5);
    for (const v of [...r.acc, ...r.gyro]) expect(Number.isFinite(v)).toBe(true);
  });

  it('reuses the previous raw value for a missing reading and handles empty streams', () => {
    const m = makeMotionStream(10);
    m.rr_alpha[5] = null;
    m.acc_z[5] = null;
    const r = motionMagnitudes(m, { smoothingMs: 0, gravityTauMs: 600 });
    expect(r.gyro[5]).toBe(r.gyro[4]);
    expect(r.acc[5]).toBe(r.acc[4]);
    const none = makeMotionStream(0);
    expect(motionMagnitudes(none, { smoothingMs: 40, gravityTauMs: 600 })).toEqual({ t: [], gyro: [], acc: [], accSource: 'none' });
    const blind = makeMotionStream(5, { linear: false });
    blind.accg_x.fill(null);
    expect(motionMagnitudes(blind, { smoothingMs: 40, gravityTauMs: 600 }).acc).toEqual([0, 0, 0, 0, 0]);
  });
});

// a 2D context that records every call
function fakeContext() {
  const calls: { name: string; args: unknown[] }[] = [];
  const props: Record<string | symbol, unknown> = {};
  const ctx = new Proxy(props, {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (prop === 'measureText') return (text: string) => ({ width: String(text).length * 6 });
      return (...args: unknown[]) => {
        calls.push({ name: String(prop), args });
      };
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
}

describe('drawSignalPlot', () => {
  const props: SignalPlotProps = {
    t: [0, 100, 200, 300, 400],
    series: [{ values: [0, 1, Number.NaN, 2, 3], color: 'var(--primary)', label: 'gyro' }],
    markers: [
      { t: 150, label: 'GO' },
      { t: 9999, label: 'outside' },
    ],
    hLines: [{ y: 1.5, label: 'still' }],
  };

  it('passes only finite coordinates to the canvas', () => {
    const { ctx, calls } = fakeContext();
    drawSignalPlot(ctx, 320, 160, props);
    for (const c of calls) {
      for (const a of c.args) if (typeof a === 'number') expect(Number.isFinite(a), `${c.name}(${c.args.join(',')})`).toBe(true);
    }
  });

  it('breaks the line at non-finite values and clips the series', () => {
    const { ctx, calls } = fakeContext();
    drawSignalPlot(ctx, 320, 160, props);
    const clip = calls.findIndex((c) => c.name === 'clip');
    const restore = calls.findIndex((c, i) => i > clip && c.name === 'restore');
    const series = calls.slice(clip, restore);
    expect(series.filter((c) => c.name === 'moveTo')).toHaveLength(2);
    expect(series.filter((c) => c.name === 'lineTo')).toHaveLength(2);
  });

  it('draws markers inside the time range only, and resolves colors through the callback', () => {
    const { ctx, calls } = fakeContext();
    const color = vi.fn((c: string) => (c.startsWith('var(') ? '#123456' : c));
    drawSignalPlot(ctx, 320, 160, props, undefined, color);
    const texts = calls.filter((c) => c.name === 'fillText').map((c) => c.args[0]);
    expect(texts).toContain('GO');
    expect(texts).toContain('still');
    expect(texts).not.toContain('outside');
    expect(color).toHaveBeenCalledWith('var(--primary)');
  });

  it('shows the empty label without data', () => {
    const { ctx, calls } = fakeContext();
    drawSignalPlot(ctx, 200, 100, { t: [], series: [{ values: [], color: 'red', label: 'x' }], emptyLabel: 'no data' });
    expect(calls.filter((c) => c.name === 'fillText').map((c) => c.args[0])).toContain('no data');
  });
});

// ───────────────────────────── PhoneFigure ─────────────────────────────

describe('phoneFigureSpec', () => {
  it('draws each orientation as the participant sees it', () => {
    expect(phoneFigureSpec('upright_toward')).toEqual({ side: 'screen', rotationDeg: 0 });
    expect(phoneFigureSpec('upright_away')).toEqual({ side: 'back', rotationDeg: 0 });
    expect(phoneFigureSpec('top_left_toward')).toEqual({ side: 'screen', rotationDeg: -90 });
    expect(phoneFigureSpec('top_right_toward')).toEqual({ side: 'screen', rotationDeg: 90 });
    expect(phoneFigureSpec('upside_down_toward')).toEqual({ side: 'screen', rotationDeg: 180 });
    expect(phoneFigureSpec('upside_down_away')).toEqual({ side: 'back', rotationDeg: 180 });
    expect(phoneFigureSpec('top_left_away')).toEqual({ side: 'back', rotationDeg: -90 });
    expect(phoneFigureSpec('top_right_away')).toEqual({ side: 'back', rotationDeg: 90 });
    expect(phoneFigureSpec('flat_up')).toEqual({ side: 'flat', rotationDeg: 0 });
    expect(phoneFigureSpec('free')).toEqual({ side: 'free', rotationDeg: 0 });
  });

  it('the mirrored orientation shows the same side with the opposite sideways rotation', () => {
    for (const o of ALL_ORIENTATIONS) {
      const a = phoneFigureSpec(o);
      const b = phoneFigureSpec(ORIENTATIONS[o].mirror);
      expect(b.side).toBe(a.side);
      expect(b.rotationDeg).toBe(Math.abs(a.rotationDeg) === 90 ? -a.rotationDeg : a.rotationDeg);
    }
  });
});

// ───────────────────────────── SessionScreen helpers ─────────────────────────────

describe('session screen helpers', () => {
  it('panelTone colours the live panel by phase, errors first', () => {
    const cases: [TrialPhase | null, TrialMessage | null, string][] = [
      [null, null, 'wait'],
      ['waiting_still', 'hold_still', 'wait'],
      ['foreperiod', 'hold_still', 'wait'],
      ['waiting_still', 'too_early', 'error'],
      ['armed', 'go', 'go'],
      ['moving', 'moving', 'move'],
      ['post_roll', 'hold_at_nose', 'done'],
      ['done', 'done', 'done'],
      ['done', 'no_movement', 'error'],
      ['moving', 'too_long', 'error'],
      ['armed', 'sensor_stalled', 'error'],
    ];
    for (const [phase, message, tone] of cases) expect(panelTone(phase, message), `${phase}/${message}`).toBe(tone);
  });

  it('failureKind and uiAbortWarning', () => {
    const rec = (status: TrialRecord['outcome']['status']) => ({ outcome: { status } }) as TrialRecord;
    expect(failureKind(null, 'stalled')).toBe('sensor_stalled');
    expect(failureKind(rec('interrupted'), 'stalled')).toBe('sensor_stalled');
    expect(failureKind(null, null)).toBe('error');
    expect(failureKind(rec('no_movement'), null)).toBe('no_movement');
    expect(failureKind(rec('too_long'), null)).toBe('too_long');
    expect(failureKind(rec('aborted'), 'user')).toBe('aborted');
    expect(failureKind(rec('interrupted'), 'hidden')).toBe('interrupted');
    expect(uiAbortWarning('hidden')).toBe('ui_abort:page_hidden');
    expect(uiAbortWarning('stalled')).toBe('ui_abort:sensor_stall');
    expect(uiAbortWarning('user')).toBe('ui_abort:user');
    expect(uiAbortWarning(null)).toBeNull();
  });

  it('the Stop control is only live before the movement', () => {
    expect(stopAllowed(null)).toBe(true);
    expect(stopAllowed('waiting_still')).toBe(true);
    expect(stopAllowed('foreperiod')).toBe(true);
    expect(stopAllowed('armed')).toBe(true);
    expect(stopAllowed('moving')).toBe(false);
    expect(stopAllowed('post_roll')).toBe(false);
    expect(stopAllowed('done')).toBe(false);
  });
});

// ───────────────────────────── denied motion permission ─────────────────────────────

describe('denied motion permission: the advice depends on where the page runs', () => {
  const iphone = (o: Partial<DeviceInfo>): DeviceInfo => ({
    ...makeDevice(),
    platform: 'ios',
    engine: 'webkit',
    browser: 'Safari 26.0',
    in_app_browser: null,
    ...o,
  });
  const contexts: [string, DeviceInfo | null][] = [
    ['safari', iphone({})],
    ['home screen', iphone({ standalone: true, browser: 'Safari' })],
    ['chrome', iphone({ browser: 'Chrome iOS 141.0.7390.96' })],
    ['instagram', iphone({ in_app_browser: 'instagram', browser: 'Instagram 360.0' })],
    ['webview', iphone({ in_app_browser: 'ios_webview', browser: 'iOS WebView' })],
    ['android', { ...makeDevice(), engine: 'blink' }],
    ['desktop', { ...makeDevice(), platform: 'other', engine: 'gecko' }],
    ['unknown', null],
  ];

  it('tells Safari tabs, Home Screen apps, other browsers and in-app browsers apart', () => {
    expect(contexts.map(([, d]) => deniedContext(d))).toEqual([
      { kind: 'ios_safari' },
      { kind: 'ios_home_screen' },
      { kind: 'ios_browser', browser: 'Chrome' },
      { kind: 'ios_in_app' },
      { kind: 'ios_in_app' },
      { kind: 'android' },
      { kind: 'other' },
      { kind: 'other' },
    ]);
    expect(iosBrowserName('Firefox iOS 145.1')).toBe('Firefox');
    expect(iosBrowserName('Edge iOS 141.0.3537.85')).toBe('Edge');
    expect(iosBrowserName('Brave 1.84')).toBe('Brave');
    for (const safari of ['Safari 18.6', 'Safari', 'unknown', '', null, undefined]) expect(iosBrowserName(safari)).toBeNull();
  });

  it('says what to close in each context, in both languages, and never suggests a Private tab', () => {
    for (const strings of [en, he]) {
      const sensors = contexts.map(([, d]) => deniedText(strings.onboarding.sensors, d));
      const ready = contexts.map(([, d]) => permissionDeniedText(strings.session.ready.permissionDenied, d));
      expect(new Set(sensors.slice(0, 4)).size).toBe(4); // four different iOS texts
      expect(new Set(ready.slice(0, 4)).size).toBe(4);
      for (const text of [...sensors, ...ready]) {
        expect(text).not.toMatch(/private|פרטית/i); // new storage: a new participant code mid-session
        expect(text.trim().length).toBeGreaterThan(20);
      }
      expect(sensors[2]).toContain('Chrome');
      expect(ready[2]).toContain('Chrome');
      expect(sensors[3]).toContain('Safari'); // in-app browser → open it in Safari
      expect(issueText('denied', strings.onboarding.sensors, contexts[1][1], null)).toBe(sensors[1]);
    }
    expect(deniedText(en.onboarding.sensors, contexts[1][1])).toMatch(/Home Screen/);
    expect(permissionDeniedText(en.session.ready.permissionDenied, contexts[0][1])).toMatch(/Resume session/);
    expect(permissionDeniedText(he.session.ready.permissionDenied, contexts[0][1])).toContain(he.onboarding.welcome.resume);
    expect(permissionDeniedText(en.session.ready.permissionDenied, contexts[5][1])).toMatch(/Motion sensors/);
  });
});

describe('boot failure text', () => {
  it('puts the browser message where each translation wants it, exactly once', () => {
    for (const strings of [en, he]) {
      const [before, after] = aroundSlot(strings.common.bootFailed);
      expect(before.length).toBeGreaterThan(10);
      expect(after.length).toBeGreaterThan(10);
      expect(`${before}X${after}`).toBe(strings.common.bootFailed('X'));
    }
    expect(aroundSlot(he.common.bootFailed)[0]).toMatch(/[֐-׿]/);
    // a translation that forgot the value still shows it, at the end
    expect(aroundSlot(() => 'Broken.')).toEqual(['Broken. ', '']);
  });
});

// ───────────────────────────── SandboxDetector ─────────────────────────────

/** Segmenter that emits scripted events at given sample times and tracks its state like the real one. */
class ScriptedSegmenter implements SegmenterLike {
  state: SegmenterState = 'waiting_still';
  live: SegmenterLive = { gyro: 1, acc: 0.1, stillProgress: 0 };
  readonly pushed: RelMotionSample[] = [];
  goAt: number | null = null;
  constructor(private readonly script: Record<number, SegmenterEvent[]>) {}
  push(s: RelMotionSample): SegmenterEvent[] {
    this.pushed.push(s);
    const events = this.script[s.t] ?? [];
    for (const e of events) {
      if (e.type === 'still') this.state = 'still';
      if (e.type === 'still_lost' || e.type === 'false_start') this.state = 'waiting_still';
      if (e.type === 'onset') this.state = 'moving';
      if (e.type === 'end') this.state = 'ended';
      if (e.type === 'no_movement' || e.type === 'too_long') this.state = 'failed';
    }
    return events;
  }
  markGo(t: number): void {
    this.goAt = t;
    this.state = 'armed';
  }
  result(): SegmentationResult {
    return {
      method: 'online_v1',
      params: DEFAULT_SEGMENTATION_PARAMS,
      acc_source: 'linear',
      onset_index: 1,
      end_index: 2,
      duration_ms: 500,
      peak_gyro_dps: 210,
      peak_acc_ms2: 3,
      baseline: null,
    };
  }
}

function samplesEvery(stepMs: number, untilMs: number): MotionSample[] {
  const out: MotionSample[] = [];
  for (let t = 0; t <= untilMs; t += stepMs) out.push({ t, ts: t, acc: [0, 0, 0], accg: [0, 9.81, 0], rr: { alpha: 0, beta: 0, gamma: 0 }, interval: 16 });
  return out;
}

describe('SandboxDetector', () => {
  it('gives GO after stillness + 600 ms, measures the movement and re-arms after the post-roll', () => {
    const made: ScriptedSegmenter[] = [];
    const scripts: Record<number, SegmenterEvent[]>[] = [
      {
        500: [{ type: 'still', t: 500 }],
        1300: [{ type: 'onset', t: 1250, index: 25 }],
        1800: [{ type: 'end', t: 1750, index: 35, detected_t: 1800 }],
      },
      {},
    ];
    const onCue = vi.fn();
    const d = new SandboxDetector(DEFAULT_SEGMENTATION_PARAMS, {
      goDelayMs: 600,
      onCue,
      makeSegmenter: () => {
        const seg = new ScriptedSegmenter(scripts[made.length] ?? {});
        made.push(seg);
        return seg;
      },
    });
    for (const s of samplesEvery(50, 3000)) d.push(s);

    expect(made[0].goAt).toBe(1100);
    expect(d.movements).toHaveLength(1);
    expect(d.movements[0]).toEqual({
      n: 1,
      go_t: 1100,
      onset_t: 1250,
      end_t: 1750,
      duration_ms: 500,
      peak_gyro_dps: 210,
      outcome: 'complete',
    });
    expect(d.markers.map((m) => m.kind)).toEqual(['go', 'onset', 'end']);
    expect(onCue.mock.calls).toEqual([
      ['go', 1100],
      ['end', 1800],
    ]);
    // re-armed with a fresh segmenter at detected end + post_roll
    expect(made).toHaveLength(2);
    expect(made[1].pushed[0].t).toBe(1800 + DEFAULT_SEGMENTATION_PARAMS.post_roll_ms);
    expect(made[1].pushed[0].index).toBe(0);
    expect(made[0].pushed.map((s) => s.index)).toEqual(made[0].pushed.map((_, i) => i));
  });

  it('cancels a pending GO when stillness is lost', () => {
    const seg = new ScriptedSegmenter({ 500: [{ type: 'still', t: 500 }], 800: [{ type: 'still_lost', t: 800 }] });
    const d = new SandboxDetector(DEFAULT_SEGMENTATION_PARAMS, { makeSegmenter: () => seg });
    for (const s of samplesEvery(50, 2000)) d.push(s);
    expect(seg.goAt).toBeNull();
    expect(d.movements).toHaveLength(0);
  });

  it('reports a failed movement and re-arms on the next sample', () => {
    const made: ScriptedSegmenter[] = [];
    const d = new SandboxDetector(DEFAULT_SEGMENTATION_PARAMS, {
      makeSegmenter: () => {
        const seg = new ScriptedSegmenter(
          made.length === 0 ? { 0: [{ type: 'still', t: 0 }], 1000: [{ type: 'no_movement', t: 1000 }] } : {},
        );
        made.push(seg);
        return seg;
      },
      onCue: vi.fn(),
    });
    for (const s of samplesEvery(100, 1500)) d.push(s);
    expect(d.movements[0]).toMatchObject({ go_t: 600, outcome: 'no_movement', duration_ms: null });
    expect(made).toHaveLength(2);
    expect(made[1].pushed[0].t).toBe(1100);
  });
});

// ───────────────────────────── i18n ─────────────────────────────

type Shape = string | Shape[] | { [k: string]: Shape };
function shape(v: unknown): Shape {
  if (typeof v === 'function') return `function/${v.length}`;
  if (Array.isArray(v)) return `array/${v.length}`;
  if (v && typeof v === 'object') {
    const o = v as Record<string, unknown>;
    return Object.fromEntries(Object.keys(o).sort().map((k) => [k, shape(o[k])]));
  }
  return typeof v;
}

function leaves(v: unknown, path = ''): [string, string][] {
  if (typeof v === 'string') return [[path, v]];
  if (typeof v === 'function') {
    const out = (v as (...a: unknown[]) => unknown)(...Array.from({ length: v.length }, (_, i) => i + 2));
    return [[`${path}()`, String(out)]];
  }
  if (Array.isArray(v)) return v.flatMap((x, i) => leaves(x, `${path}[${i}]`));
  if (v && typeof v === 'object') return Object.entries(v).flatMap(([k, x]) => leaves(x, path ? `${path}.${k}` : k));
  return [];
}

describe('session strings', () => {
  it('Hebrew has exactly the English keys, function arities and array lengths', () => {
    expect(shape(heSession)).toEqual(shape(enSession));
  });

  it('no string is empty, and Hebrew strings are actually Hebrew', () => {
    const en = leaves(enSession);
    const he = leaves(heSession);
    for (const [path, text] of [...en, ...he]) expect(text.trim().length, path).toBeGreaterThan(0);
    const hebrew = he.filter(([, text]) => /[֐-׿]/.test(text)).length;
    expect(hebrew / he.length).toBeGreaterThan(0.9);
  });

  it('every failure reason and live message has a text', () => {
    const kinds: FailureKind[] = ['no_movement', 'too_long', 'aborted', 'interrupted', 'sensor_stalled', 'error'];
    for (const k of kinds) {
      expect(enSession.failed.reasons[k]).toBeTruthy();
      expect(heSession.failed.reasons[k]).toBeTruthy();
    }
    const messages: TrialMessage[] = ['hold_still', 'too_early', 'go', 'moving', 'hold_at_nose', 'done', 'no_movement', 'too_long', 'sensor_stalled'];
    for (const m of messages) expect(heSession.live.messages[m]).toBeTruthy();
  });
});
