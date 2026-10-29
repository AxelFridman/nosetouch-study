/**
 * The segment classifier (approach / return / other) and the production attempt rule.
 *
 *  - segmentProba: m4_models.SegmentClassifier.proba with the 'robust' scaling (analysis/stage2/lanes/m4_models.py:
 *    162-172, 231-237) and sklearn's multinomial predict_proba (decision_function then utils.extmath.softmax):
 *      Z = (F - center) / scale;  Z = 0 where not finite;  Z = clip(Z, -clip, clip)
 *      L = coef @ Z + intercept;  P = exp(L - max L) / sum(exp(L - max L))
 *    Classes are mapped BY NAME (the bundle lists them in sklearn's sorted order: approach, other, return).
 *  - attemptRule: m4_e2e.FittedPipeline.attempts without pairing (m4_e2e.py:260-295; m4_config.PRODUCTION_PAIRING
 *    False): cls = argmax over (approach, return, other), ties to the first; an attempt is a bout whose class is
 *    'approach' with P(approach) >= detThreshold and a canonical window.
 *
 * The classifier and the threshold were fitted for lane m4's own hand classifier (lane prod, which would refit the
 * detector for the final CNN, has not run).
 */
import type { DetectorBundle, IgnoredMovement, SegmentClass } from '../types';

export type SegmentClassifierJson = DetectorBundle['segmentClassifier'];

/** m4_detect.CLASSES: the arg-max order of the attempt rule (bundle attemptRule.argmaxOrder). */
export const ARGMAX_ORDER: readonly SegmentClass[] = ['approach', 'return', 'other'];

/** P(class) of one segment from its features (name -> value; a missing name or NaN counts as NaN). */
export function segmentProba(sc: SegmentClassifierJson, features: Readonly<Record<string, number>>): Record<SegmentClass, number> {
  if (sc.scaling !== 'robust') throw new Error(`detect: segment classifier scaling '${sc.scaling}' is not supported`);
  const nf = sc.features.length;
  const clip = sc.clip ?? Infinity;
  const z = new Float64Array(nf);
  for (let j = 0; j < nf; j++) {
    const v = features[sc.features[j]];
    let zj = ((v === undefined ? NaN : v) - sc.center[j]) / sc.scale[j];
    if (!Number.isFinite(zj)) zj = 0.0; // NaN -> 0, the training median (m4_models.py:164)
    z[j] = zj < -clip ? -clip : zj > clip ? clip : zj;
  }
  const nc = sc.classes.length;
  const L = new Float64Array(nc);
  for (let c = 0; c < nc; c++) {
    const w = sc.coef[c];
    let s = 0;
    for (let j = 0; j < nf; j++) s += z[j] * w[j];
    L[c] = s + sc.intercept[c];
  }
  let mx = L[0];
  for (let c = 1; c < nc; c++) if (L[c] > mx) mx = L[c];
  const e = new Float64Array(nc);
  let sum = 0;
  for (let c = 0; c < nc; c++) {
    e[c] = Math.exp(L[c] - mx);
    sum += e[c];
  }
  const out: Record<SegmentClass, number> = { approach: 0, return: 0, other: 0 }; // a class sklearn never saw: 0
  for (let c = 0; c < nc; c++) out[sc.classes[c]] = e[c] / sum;
  return out;
}

/** np.argmax over ARGMAX_ORDER (ties and NaN: the first, as numpy). */
export function argmaxClass(p: Readonly<Record<SegmentClass, number>>): SegmentClass {
  let best: SegmentClass = ARGMAX_ORDER[0];
  let m = p[best];
  if (Number.isNaN(m)) return best;
  for (let i = 1; i < ARGMAX_ORDER.length; i++) {
    const c = ARGMAX_ORDER[i];
    const v = p[c];
    if (Number.isNaN(v)) return c;
    if (v > m) {
      m = v;
      best = c;
    }
  }
  return best;
}

/** The production attempt rule (no pairing). */
export function isAttempt(
  cls: SegmentClass,
  p: Readonly<Record<SegmentClass, number>>,
  canonicalOk: boolean,
  rule: DetectorBundle['attemptRule'],
): boolean {
  if (rule.pairReturns) throw new Error('detect: the pairing attempt rule is not the production rule (pairReturns must be false)');
  return cls === 'approach' && p.approach >= rule.detThreshold && canonicalOk;
}

/** Why a confirmed bout is not an attempt (debug view; IgnoredMovement.reason). null for an attempt. */
export function ignoredReason(
  cls: SegmentClass,
  p: Readonly<Record<SegmentClass, number>>,
  canonicalOk: boolean,
  rule: DetectorBundle['attemptRule'],
): IgnoredMovement['reason'] | null {
  if (!canonicalOk) return 'no_window';
  if (cls === 'return') return 'return';
  if (cls === 'other') return 'other';
  return p.approach >= rule.detThreshold ? null : 'below_threshold';
}
