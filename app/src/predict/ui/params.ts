/**
 * URL parameters of predict mode (types.ts PredictParams; docs/stage3/APP_SPEC.md: &model, &view=compare,
 * &sim=1, &hand, &seed, &lang, &debug). main.tsx already decided predict mode (?predict=1 or
 * ?mode=predict). A value that is not understood falls back to the default and its name is listed in
 * `ignored` (shown in the debug panel only).
 */
import type { Hand, ModelId, PredictLang, PredictParams } from '../types';

export const MODEL_IDS: readonly ModelId[] = ['cnn', 'm4', 'phys6', 'yaw'];
export const DEFAULT_MODEL: ModelId = 'cnn';
/**
 * The desktop demo's default (&sim=1 without &model): lane m4's classifier. The final CNN reads the simulator's
 * touches with the wrong sign (a kinematic model, not a recording: docs/stage3/README.md, known limits), so the demo
 * would name the wrong hand; &model=cnn still selects it, and the compare view shows every model.
 */
export const DEMO_DEFAULT_MODEL: ModelId = 'm4';

/** true when the page should run in predict mode (the same test main.tsx makes before loading the chunk) */
export function isPredictSearch(search: string): boolean {
  const q = new URLSearchParams(search);
  return q.get('predict') === '1' || q.get('mode') === 'predict';
}

function isModelId(v: string): v is ModelId {
  return (MODEL_IDS as readonly string[]).includes(v);
}

export function parsePredictParams(search: string): PredictParams {
  const q = new URLSearchParams(search);
  const ignored: string[] = [];
  const read = <T>(name: string, fallback: T, parse: (v: string) => T | undefined): T => {
    const raw = q.get(name);
    if (raw === null) return fallback;
    const v = parse(raw.trim().toLowerCase());
    if (v === undefined) {
      ignored.push(name);
      return fallback;
    }
    return v;
  };
  const flag = (v: string): boolean | undefined => (v === '1' || v === 'true' ? true : v === '0' || v === 'false' || v === '' ? false : undefined);

  const asked = read<ModelId | null>('model', null, (v) => (isModelId(v) ? v : undefined));
  const compare = read<boolean>('view', false, (v) => (v === 'compare' ? true : v === '' || v === 'single' ? false : undefined));
  const sim = read<boolean>('sim', false, flag);
  const simHand = read<Hand | null>('hand', null, (v) => (v === 'left' || v === 'right' ? v : undefined));
  const simSeed = read<number | null>('seed', null, (v) => {
    if (!/^\d{1,10}$/.test(v)) return undefined;
    const n = Number(v);
    return n <= 0xffffffff ? n : undefined;
  });
  const lang = read<PredictLang>('lang', 'en', (v) => (v === 'he' || v === 'iw' ? 'he' : v === 'en' ? 'en' : undefined));
  const debug = read<boolean>('debug', false, flag);
  const model = asked ?? (sim ? DEMO_DEFAULT_MODEL : DEFAULT_MODEL);
  return { model, compare, sim, simHand, simSeed, lang, debug, ignored };
}

/** `search` with `name` set to `value` (or removed when null), for history.replaceState (&model=). */
export function withParam(search: string, name: string, value: string | null): string {
  const q = new URLSearchParams(search);
  if (value === null) q.delete(name);
  else q.set(name, value);
  const s = q.toString();
  return s ? `?${s}` : '';
}

/** The demo link offered on a desktop: predict mode in simulation, the language kept. */
export function demoSearch(search: string): string {
  const q = new URLSearchParams(search);
  const out = new URLSearchParams();
  out.set('predict', '1');
  out.set('sim', '1');
  for (const keep of ['lang', 'model', 'debug']) {
    const v = q.get(keep);
    if (v !== null) out.set(keep, v);
  }
  return `?${out.toString()}`;
}
