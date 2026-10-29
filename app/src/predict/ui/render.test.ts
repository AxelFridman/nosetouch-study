/**
 * Render tests of the predict screens without a DOM: the components are pure functions of their props
 * (no hooks), so a VNode tree can be expanded and searched in Node. The live page is covered by the
 * Playwright specs (e2e/predict*.spec.ts).
 */
import { h, type VNode } from 'preact';
import { describe, expect, it, vi } from 'vitest';
import { strings } from '../i18n/strings';
import { binding } from '../live/engine';
import type { EngineSnapshot, ModelComparison, ModelSpec } from '../types';
import { CompareTable, DebugPanel, Footer, HandBar, ModelSheet, ProbabilityDisplay } from './components';
import { NBSP } from './format';
import { previewState, PREVIEW_NAMES } from './preview';
import {
  AbstainScreen,
  AnswerScreen,
  failureView,
  GripScreen,
  PermissionScreen,
  screenOf,
  StartScreen,
  UnsupportedScreen,
  WatchingScreen,
} from './screens';

interface El {
  type: string;
  props: Record<string, unknown>;
  children: Node[];
}
type Node = El | string;

function expand(v: unknown): Node[] {
  if (v === null || v === undefined || typeof v === 'boolean') return [];
  if (typeof v === 'string' || typeof v === 'number') return [String(v)];
  if (Array.isArray(v)) return v.flatMap(expand);
  const vn = v as VNode<Record<string, unknown>>;
  if (typeof vn.type === 'function') return expand((vn.type as (p: unknown) => unknown)(vn.props));
  return [{ type: vn.type as string, props: vn.props, children: expand(vn.props.children) }];
}

function render(v: unknown): El {
  const out = expand(v).filter((n): n is El => typeof n !== 'string');
  expect(out.length).toBe(1);
  return out[0];
}

function text(n: Node): string {
  return typeof n === 'string' ? n : n.children.map(text).join('');
}

function findAll(n: Node, pred: (e: El) => boolean): El[] {
  if (typeof n === 'string') return [];
  return [...(pred(n) ? [n] : []), ...n.children.flatMap((c) => findAll(c, pred))];
}

const byTestId = (n: Node, id: string): El[] => findAll(n, (e) => e.props['data-testid'] === id);
const en = strings('en');
const he = strings('he');
const models = binding.models();
const snap = (name: string): EngineSnapshot => previewState(name, models, 'cnn')!.snapshot!;

describe('ProbabilityDisplay', () => {
  it('final: P(right) in big type, what it means, the bar and the named hand', () => {
    const root = render(h(ProbabilityDisplay, { s: en, pRight: 0.87, percentRight: 87, mode: 'final', k: 2, hand: 'right' }));
    expect(root.props['aria-live']).toBe('polite');
    const t = text(root);
    expect(t).toContain(`P(right) = 87${NBSP}%`);
    expect(t).toContain('probability that the RIGHT hand was used');
    expect(text(byTestId(root, 'pm-verdict')[0])).toBe('RIGHT hand · 2 touches');
    const bar = byTestId(root, 'pm-bar')[0];
    expect(bar.props.role).toBe('img');
    expect(bar.props['aria-label']).toBe(`LEFT 13${NBSP}%, RIGHT 87${NBSP}%`);
    const [left, right] = bar.children as El[];
    expect((left.props.style as { width: string }).width).toBe('13%');
    expect((right.props.style as { width: string }).width).toBe('87%');
    expect(findAll(root, (e) => String(e.props.class).includes('pm-swatch-right'))).toHaveLength(1);
  });

  it('says LEFT when P(right) is low, and still shows P(right) (never P(left))', () => {
    const root = render(h(ProbabilityDisplay, { s: en, pRight: 0.03, percentRight: 3, mode: 'final', k: 1, hand: 'left' }));
    expect(text(root)).toContain(`P(right) = 3${NBSP}%`);
    expect(text(byTestId(root, 'pm-verdict')[0])).toBe('LEFT hand · 1 touch');
  });

  it('clamps the extremes', () => {
    const root = render(h(ProbabilityDisplay, { s: en, pRight: 0.9996, percentRight: 100, mode: 'final', k: 3, hand: 'right' }));
    expect(text(byTestId(root, 'pm-percent')[0])).toBe(`>99${NBSP}%`);
    expect(text(root)).not.toContain('100');
  });

  it('provisional: muted, with "after k · not final" and no verdict', () => {
    const root = render(h(ProbabilityDisplay, { s: en, pRight: 0.64, percentRight: 64, mode: 'provisional', k: 1, hand: 'right' }));
    expect(String(root.props.class)).toContain('pm-prob-provisional');
    expect(text(byTestId(root, 'pm-provisional')[0])).toBe('after 1 touch · not final');
    expect(byTestId(root, 'pm-verdict')).toHaveLength(0);
  });

  it('Hebrew: the bar stays left-to-right with LEFT on the left; P(right) is isolated as LTR', () => {
    const bar = render(h(HandBar, { s: he, pRight: 0.87, percentRight: 87 }));
    expect(bar.props.dir).toBe('ltr');
    const words = findAll(bar, (e) => e.props.class === 'pm-bar-word').map(text);
    expect(words).toEqual(['שמאל', 'ימין']);
    const root = render(h(ProbabilityDisplay, { s: he, pRight: 0.87, percentRight: 87, mode: 'final', k: 2, hand: 'right' }));
    const bdi = findAll(root, (e) => e.type === 'bdi')[0];
    expect(bdi.props.dir).toBe('ltr');
    expect(text(bdi)).toContain('P(right) =');
  });
});

describe('compare view', () => {
  const rows: ModelComparison[] = [
    { modelId: 'cnn', pRight: 0.93, percentRight: 93, k: 2, confident: true },
    { modelId: 'm4', pRight: 0.8, percentRight: 80, k: 2, confident: false },
    { modelId: 'phys6', pRight: 0.5, percentRight: 50, k: 0, confident: false },
    { modelId: 'yaw', pRight: 0.04, percentRight: 4, k: 2, confident: true },
  ];

  it('one row per model with its P(right), a mini bar and a check where its own rule is confident', () => {
    const onSelect = vi.fn();
    const root = render(h(CompareTable, { s: en, rows, driver: 'm4', k: 2, onSelect }));
    const buttons = findAll(root, (e) => e.type === 'button');
    expect(buttons).toHaveLength(4);
    expect(text(root)).toContain('after 2 touches');
    const m4 = byTestId(root, 'pm-compare-m4')[0];
    expect(m4.props['aria-pressed']).toBe(true);
    expect(String(m4.props.class)).toContain('is-driver');
    expect(text(byTestId(root, 'pm-compare-cnn')[0])).toBe(`Final CNN93${NBSP}%✓`);
    expect(text(byTestId(root, 'pm-compare-phys6')[0])).toBe('Physics logisticn/a');
    expect(text(byTestId(root, 'pm-compare-yaw')[0])).toContain(`4${NBSP}%✓`);
    (byTestId(root, 'pm-compare-yaw')[0].props.onClick as () => void)();
    expect(onSelect).toHaveBeenCalledWith('yaw');
  });
});

describe('screens', () => {
  it('screenOf follows the preflight, the stage and the engine phase', () => {
    expect(screenOf({ stage: 'start', failure: null, snapshot: null })).toBe('start');
    expect(screenOf({ stage: 'start', failure: 'in_app_browser', snapshot: null })).toBe('unsupported');
    expect(screenOf({ stage: 'permission', failure: null, snapshot: null })).toBe('permission');
    expect(screenOf({ stage: 'running', failure: null, snapshot: null })).toBe('watching');
    expect(screenOf({ stage: 'running', failure: null, snapshot: snap('provisional') })).toBe('watching');
    expect(screenOf({ stage: 'running', failure: null, snapshot: snap('answer') })).toBe('answer');
    expect(screenOf({ stage: 'running', failure: null, snapshot: snap('abstain') })).toBe('abstain');
    expect(screenOf({ stage: 'running', failure: null, snapshot: snap('grip') })).toBe('grip');
    expect(screenOf({ stage: 'running', failure: 'no_events', snapshot: snap('answer') })).toBe('unsupported');
  });

  it('every screen root carries data-testid pm-<screen>', () => {
    const noop = (): void => undefined;
    const cases: [unknown, string][] = [
      [h(StartScreen, { s: en, onStart: noop, demoHref: null }), 'pm-start'],
      [h(PermissionScreen, { s: en }), 'pm-permission'],
      [h(WatchingScreen, { s: en, snap: snap('watching'), compare: false, onSelectModel: noop, onStartOver: noop }), 'pm-watching'],
      [h(AnswerScreen, { s: en, snap: snap('answer'), compare: true, onAgain: noop, onToggleCompare: noop, onSelectModel: noop }), 'pm-answer'],
      [h(AbstainScreen, { s: en, snap: snap('abstain'), onAgain: noop }), 'pm-abstain'],
      [h(GripScreen, { s: en, onStartOver: noop }), 'pm-grip'],
      [h(UnsupportedScreen, { s: en, failure: 'no_events', preflight: null, copied: false, onAction: noop }), 'pm-unsupported'],
    ];
    for (const [vnode, id] of cases) expect(render(vnode).props['data-testid']).toBe(id);
  });

  it('start: one primary button, the instruction, a demo link only when given', () => {
    const onStart = vi.fn();
    const root = render(h(StartScreen, { s: en, onStart, demoHref: null }));
    expect(text(root)).toBe('Which hand?Hold the phone out, screen towards you. Bring it to your nose, then back.Start');
    (byTestId(root, 'pm-start-btn')[0].props.onClick as () => void)();
    expect(onStart).toHaveBeenCalledOnce();
    expect(byTestId(render(h(StartScreen, { s: en, onStart, demoHref: '?predict=1&sim=1' })), 'pm-demo-link')).toHaveLength(1);
  });

  it('watching: the instruction and five empty dots before the first attempt; the provisional P after it', () => {
    const noop = (): void => undefined;
    const first = render(h(WatchingScreen, { s: en, snap: snap('watching'), compare: false, onSelectModel: noop, onStartOver: noop }));
    expect(text(first)).toContain('Hold still. Then bring the phone to your nose.');
    expect(byTestId(first, 'pm-dots')[0].props['data-k']).toBe(0);
    expect(byTestId(first, 'pm-prob')).toHaveLength(0);
    const after = render(h(WatchingScreen, { s: en, snap: snap('provisional'), compare: true, onSelectModel: noop, onStartOver: noop }));
    expect(text(after)).toContain('after 1 touch · not final');
    expect(text(after)).toContain('Touch again.');
    expect(byTestId(after, 'pm-dots')[0].props['data-k']).toBe(1);
    expect(byTestId(after, 'pm-compare')).toHaveLength(1);
    const hint = render(h(WatchingScreen, { s: en, snap: snap('hint'), compare: false, onSelectModel: noop, onStartOver: noop }));
    expect(text(byTestId(hint, 'pm-hint')[0])).toBe('Turn the screen towards your face');
  });

  it('a forced answer (k_max below p_stop) is a muted leaning, a confident one keeps the verdict', () => {
    const noop = (): void => undefined;
    const base = snap('answer');
    const forced: EngineSnapshot = {
      ...base,
      decision: { ...base.decision, status: 'k_max', pRight: 0.55, percentRight: 55, nUsed: 5, hand: 'right' },
    };
    const root = render(h(AnswerScreen, { s: en, snap: forced, compare: false, onAgain: noop, onToggleCompare: noop, onSelectModel: noop }));
    const v = byTestId(root, 'pm-verdict')[0];
    expect(text(v)).toBe('Leaning RIGHT · not confident · 5 touches');
    expect(v.props['data-confident']).toBe('0');
    const sure = render(h(AnswerScreen, { s: en, snap: base, compare: false, onAgain: noop, onToggleCompare: noop, onSelectModel: noop }));
    expect(byTestId(sure, 'pm-verdict')[0].props['data-confident']).toBe('1');
  });

  it('the demo: the CNN row says n/a (sim), with no check mark', () => {
    const rows: ModelComparison[] = [
      { modelId: 'cnn', pRight: 0.01, percentRight: 1, k: 2, confident: true },
      { modelId: 'm4', pRight: 0.95, percentRight: 95, k: 2, confident: true },
    ];
    const root = render(h(CompareTable, { s: en, rows, driver: 'm4', k: 2, onSelect: () => undefined, sim: true }));
    expect(text(byTestId(root, 'pm-compare-cnn')[0])).toBe('Final CNNn/a (sim)');
    expect(text(byTestId(root, 'pm-compare-m4')[0])).toContain('✓');
    const real = render(h(CompareTable, { s: en, rows, driver: 'm4', k: 2, onSelect: () => undefined }));
    expect(text(byTestId(real, 'pm-compare-cnn')[0])).toContain('✓');
  });

  it('answer: P(right) with its meaning, the hand, Again and the compare toggle', () => {
    const noop = (): void => undefined;
    const s = snap('answer');
    const root = render(h(AnswerScreen, { s: en, snap: s, compare: false, onAgain: noop, onToggleCompare: noop, onSelectModel: noop }));
    expect(text(root)).toMatch(/P\(right\) = \d+ %probability that the RIGHT hand was used/);
    expect(text(byTestId(root, 'pm-verdict')[0])).toBe('RIGHT hand · 2 touches');
    expect(text(byTestId(root, 'pm-again')[0])).toBe('Again');
    expect(byTestId(root, 'pm-compare')).toHaveLength(0);
    expect(byTestId(root, 'pm-compare-toggle')[0].props['aria-expanded']).toBe(false);
  });

  it('abstain and grip: one message and one button', () => {
    const noop = (): void => undefined;
    expect(text(render(h(AbstainScreen, { s: en, snap: snap('abstain'), onAgain: noop })))).toBe('No touch found – bring the phone to your nose, then backAgain');
    expect(text(render(h(GripScreen, { s: en, onStartOver: noop })))).toBe('Turn the screen towards your faceStart over');
  });

  it('unsupported: one message and one action per failure', () => {
    expect(failureView('in_app_browser', null)).toEqual({ text: 'u_inapp', action: 'copy' });
    expect(failureView('insecure_context', null)).toEqual({ text: 'u_https', action: 'demo' });
    expect(failureView('not_supported', null)).toEqual({ text: 'u_desktop', action: 'demo' });
    expect(failureView('permission_denied', { engine: 'webkit', desktop: false })).toEqual({ text: 'denied_ios', action: 'copy' });
    expect(failureView('not_supported', { engine: 'webkit', desktop: false })).toEqual({ text: 'u_inapp', action: 'copy' });
    expect(failureView('engine_error', null)).toEqual({ text: 'u_error', action: 'reload' });
    expect(failureView('permission_denied', { engine: 'blink', desktop: false })).toEqual({ text: 'denied_other', action: 'reload' });
    expect(failureView('no_events', { engine: 'blink', desktop: true })).toEqual({ text: 'u_desktop', action: 'demo' });
    expect(failureView('no_events', { engine: 'blink', desktop: false })).toEqual({ text: 'u_events_android', action: 'reload' });
    expect(failureView('no_events', { engine: 'webkit', desktop: false })).toEqual({ text: 'u_events', action: 'reload' });
    expect(failureView('no_gyroscope', null)).toEqual({ text: 'u_gyro', action: 'demo' });
    expect(failureView('no_accelerometer', null)).toEqual({ text: 'u_acc', action: 'demo' });
    const onAction = vi.fn();
    const root = render(h(UnsupportedScreen, { s: en, failure: 'in_app_browser', preflight: null, copied: false, onAction }));
    expect(text(root)).toBe('Open this link in Safari or Chrome.Copy link');
    (byTestId(root, 'pm-action-copy')[0].props.onClick as () => void)();
    expect(onAction).toHaveBeenCalledWith('copy');
    expect(text(render(h(UnsupportedScreen, { s: en, failure: 'in_app_browser', preflight: null, copied: true, onAction })))).toContain('Link copied');
  });

  it('every preview state renders (dev screenshots)', () => {
    for (const name of PREVIEW_NAMES) expect(previewState(name, models, 'm4'), name).not.toBeNull();
    expect(previewState('nope', models, 'cnn')).toBeNull();
    expect(snap('answer').decision.answered).toBe(true);
  });
});

describe('chrome', () => {
  const specs: ModelSpec[] = models.map((m) => m.spec);

  it('the footer names the model, the first 8 characters of its hash and the version; the demo says so', () => {
    const noop = (): void => undefined;
    const root = render(h(Footer, { s: en, driver: 'cnn', sha256: '1a2b3c4d5e6f', version: '0.3.0', demo: true, onInfo: noop }));
    expect(text(byTestId(root, 'pm-footer')[0])).toBe('cnn · 1a2b3c4d · v0.3.0 · demo');
    const live = render(h(Footer, { s: en, driver: 'm4', sha256: binding.specs[1].sha256, version: '0.3.0', demo: false, onInfo: noop }));
    expect(text(byTestId(live, 'pm-footer')[0])).toBe(`m4 · ${binding.specs[1].sha256.slice(0, 8)} · v0.3.0`);
  });

  it('the model sheet: four radio rows (label and one line) and the compare switch', () => {
    const onSelect = vi.fn();
    const onCompare = vi.fn();
    const root = render(h(ModelSheet, { s: en, specs, driver: 'phys6', compare: true, onSelect, onCompare, onClose: () => undefined }));
    const radios = findAll(root, (e) => e.type === 'input' && e.props.type === 'radio');
    expect(radios.map((r) => [r.props.value, r.props.checked])).toEqual([
      ['cnn', false],
      ['m4', false],
      ['phys6', true],
      ['yaw', false],
    ]);
    expect(text(byTestId(root, 'pm-model-cnn')[0])).toBe('Final CNN1D CNN, 4 members');
    expect(text(byTestId(root, 'pm-model-m4')[0])).toBe('Detector modellane m4, yaw logistic');
    (radios[3].props.onChange as () => void)();
    expect(onSelect).toHaveBeenCalledWith('yaw');
    const sw = findAll(root, (e) => e.type === 'input' && e.props.role === 'switch')[0];
    expect(sw.props.checked).toBe(true);
    (sw.props.onChange as (e: unknown) => void)({ currentTarget: { checked: false } });
    expect(onCompare).toHaveBeenCalledWith(false);
  });

  it('the debug panel lists the attempts with every model and says whose thresholds these are', () => {
    const root = render(h(DebugPanel, { snap: snap('answer'), ids: ['cnn', 'm4', 'phys6', 'yaw'], engineKind: 'real', ignoredParams: ['seed'] }));
    expect(findAll(root, (e) => e.type === 'tr')).toHaveLength(3); // header + 2 attempts
    expect(text(root)).toContain("lane m4's");
    expect(text(root)).toContain('seed');
  });
});
