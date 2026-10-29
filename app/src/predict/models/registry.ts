/**
 * The bundled hand models of predict mode (ModelsApi in ../types.ts), in registry order cnn, m4, phys6, yaw.
 *
 * The JSON files next to this module are imported statically, so they live in the lazily loaded predict chunk and
 * nothing is fetched at run time. They were fitted on ALL persons (P01..P20, 593 trials, CAP_PER_PERSON 40, seed
 * 20261027) with the evaluated code paths and written by analysis/stage3 (python -m stage3 export);
 * final_cnn.json is lane final's final_model.json, the same document. manifest.json records each file's sha256 (a
 * vitest test recomputes them from the files on disk) and each model's LOSO accuracy.
 *
 * Every model computes its own calibrated P(right) per attempt and its own k-attempt aggregation (aggregate.ts).
 * The detector (detector(): segmentation, segment classifier, attempt rule) and the stopping thresholds
 * (p_stop 0.9, k_max 5, p_answer 0.5) are lane m4's and were fitted for lane m4's own hand classifier; lane prod,
 * which would refit them for the final CNN (and for PHYS6 and the yaw reference), has not run.
 */
import type {
  AggregationSpec,
  AttemptInput,
  CalibratorJson,
  FinalModelJson,
  HandModel,
  ModelAttemptOutput,
  ModelId,
  ModelKind,
  ModelSpec,
  ModelsApi,
  SegmentClass,
  StoppingParams,
} from '../types';
import { aggregateOutputs } from './aggregate';
import { effectiveMethod } from './calibrators';
import { buildFinalCnn, cnnForward, type FinalCnn } from './finalCnn';
import type { ExportedCalibrator, M4DetectorJson, M4ProductionJson, ManifestJson, Phys6Json, YawRefJson } from './jsonTypes';
import {
  buildM4Hand,
  buildPhys6,
  buildYawRef,
  m4Forward,
  phys6Forward,
  yawForward,
  type M4Hand,
  type Phys6,
  type YawRef,
} from './logistic';
import cnnJson from './final_cnn.json';
import m4Json from './m4_production.json';
import manifestJson from './manifest.json';
import phys6Json from './phys6.json';
import yawJson from './yaw_ref.json';

/** registry order (the selector's and the compare view's) */
export const MODEL_ORDER: readonly ModelId[] = ['cnn', 'm4', 'phys6', 'yaw'];

/** the bundled file of each model (manifest.json > models[].file) */
export const MODEL_FILES: Readonly<Record<ModelId, string>> = {
  cnn: 'final_cnn.json',
  m4: 'm4_production.json',
  phys6: 'phys6.json',
  yaw: 'yaw_ref.json',
};

const manifest = manifestJson as unknown as ManifestJson;
const cnnDoc = cnnJson as unknown as FinalModelJson;
const m4Doc = m4Json as unknown as M4ProductionJson;
const phys6Doc = phys6Json as unknown as Phys6Json;
const yawDoc = yawJson as unknown as YawRefJson;

/** Why the stopping thresholds and the detector apply to every model (manifest.json > note). */
export const THRESHOLDS_NOTE: string = manifest.note;

const CAL_METHODS: readonly CalibratorJson['method'][] = [
  'none',
  'temperature',
  'platt',
  'beta',
  'isotonic',
  'isotonic_sym',
  'stacked',
];

function calibrationOf(doc: ExportedCalibrator): CalibratorJson['method'] {
  const m = effectiveMethod(doc);
  if (!(CAL_METHODS as readonly string[]).includes(m)) throw new RangeError(`calibration method ${JSON.stringify(m)}`);
  return m as CalibratorJson['method'];
}

function manifestEntry(id: ModelId) {
  const e = manifest.models.find((m) => m.id === id);
  if (!e) throw new RangeError(`manifest.json has no model ${JSON.stringify(id)}`);
  if (e.file !== MODEL_FILES[id]) throw new RangeError(`manifest.json: model ${id} is in ${e.file}`);
  return e;
}

function aggregationOf(id: ModelId): AggregationSpec {
  switch (id) {
    case 'cnn':
      return { kind: 'sum_over_tau', tau: { ...cnnDoc.aggregation.tau } };
    case 'm4': {
      const a = m4Doc.aggregation;
      if (a.kind !== 'scaled_mean') throw new RangeError(`m4 aggregation ${JSON.stringify(a.kind)}`);
      return { kind: 'scaled_mean', s: a.s === null ? null : { ...a.s } };
    }
    case 'phys6': {
      const a = phys6Doc.aggregation;
      if (a.kind !== 'sum_over_k_pow' || !Number.isFinite(a.alpha)) throw new RangeError('phys6 aggregation');
      return { kind: 'sum_over_k_pow', alpha: a.alpha };
    }
    case 'yaw': {
      if (yawDoc.aggregation.kind !== 'mean_p') throw new RangeError('yaw aggregation');
      return { kind: 'mean_p' };
    }
  }
}

function specOf(id: ModelId): ModelSpec {
  const e = manifestEntry(id);
  const doc = { cnn: cnnDoc, m4: m4Doc, phys6: phys6Doc, yaw: yawDoc }[id];
  const cal = calibrationOf((doc as { calibrator: ExportedCalibrator }).calibrator);
  const kind: ModelKind = id === 'cnn' ? 'cnn1d_ensemble' : 'logistic';
  return Object.freeze({
    id,
    stage2Name: e.stage2Name,
    label: e.label,
    kind,
    sha256: e.sha256,
    calibration: cal,
    aggregation: Object.freeze(aggregationOf(id)),
    trainedOn: Object.freeze({ persons: doc.training.n_persons, trials: doc.training.n_trials }),
    isDefault: manifest.default === id,
  });
}

/** every bundled model, registry order: cnn, m4, phys6, yaw */
export const specs: readonly ModelSpec[] = Object.freeze(MODEL_ORDER.map(specOf));

/** the model the page starts with ('cnn') */
export const defaultModel: ModelId = manifest.default;

export function specById(id: ModelId): ModelSpec {
  const s = specs.find((x) => x.id === id);
  if (!s) throw new RangeError(`unknown model ${JSON.stringify(id)}`);
  return s;
}

/** What the selector and the info sheet may show besides the label: one line, and the Stage 2 LOSO accuracy. */
export interface ModelInfo {
  readonly id: ModelId;
  /** e.g. '1D CNN, 4 members' */
  readonly short: string;
  /** leave-one-person-out accuracy (Stage 2) and its 95 % CI; what it measured is in `what` */
  readonly accuracy: number | null;
  readonly ci95: readonly [number, number] | null;
  readonly what: string;
  /** first 8 hex characters of the file's sha256 (the footer) */
  readonly shortHash: string;
}

export const modelInfo: Readonly<Record<ModelId, ModelInfo>> = Object.freeze(
  Object.fromEntries(
    MODEL_ORDER.map((id) => {
      const e = manifestEntry(id);
      const ci = e.loso?.ci95;
      const info: ModelInfo = {
        id,
        short: e.short,
        accuracy: e.loso ? e.loso.accuracy : null,
        ci95: ci && ci.length === 2 ? [ci[0], ci[1]] : null,
        what: e.loso?.what ?? '',
        shortHash: e.sha256.slice(0, 8),
      };
      return [id, Object.freeze(info)];
    }),
  ) as Record<ModelId, ModelInfo>,
);

// ───────────────────────────── the detector (lane m4) ─────────────────────────────

const SEGMENT_CLASSES: readonly SegmentClass[] = ['approach', 'return', 'other'];

function checkDetector(d: M4DetectorJson): M4DetectorJson {
  const sc = d.segmentClassifier;
  const nf = sc.features.length;
  const bad = (msg: string): never => {
    throw new RangeError(`m4_production.json detector: ${msg}`);
  };
  if (sc.classes.length !== 3 || !SEGMENT_CLASSES.every((c) => sc.classes.includes(c))) bad('classes');
  if (!['robust', 'z', 'none'].includes(sc.scaling)) bad(`scaling ${JSON.stringify(sc.scaling)}`);
  if (sc.coef.length !== 3 || sc.coef.some((r) => r.length !== nf) || sc.intercept.length !== 3) bad('coef shape');
  if (sc.center.length !== nf || sc.scale.length !== nf) bad('center/scale');
  const st = d.stopping;
  if (!(st.pStop > 0.5 && st.pStop <= 1 && Number.isInteger(st.kMax) && st.kMax >= 1 && st.pAnswer >= 0.5)) bad('stopping');
  if (!(d.causalDelayMs > 0) || d.activitySos.length < 1 || d.lowpassSos.length < 1) bad('filters');
  return d;
}

let detectorCache: M4DetectorJson | null = null;

/**
 * Lane m4's production detector (m4_production.json > detector): segmentation parameters (the fitted SEG_GRID
 * entry), filter coefficients exported from scipy, the segment classifier, the attempt rule, the stopping
 * thresholds and the grip gate. They were fitted for lane m4's classifier; lane prod has not run.
 */
export function detector(): M4DetectorJson {
  if (!detectorCache) detectorCache = Object.freeze(checkDetector(m4Doc.detector)) as M4DetectorJson;
  return detectorCache;
}

/** lane m4's stopping thresholds, applied to every model's own P_k (fitted for m4's classifier: see THRESHOLDS_NOTE) */
export function stoppingParams(): StoppingParams {
  const s = detector().stopping;
  return { pStop: s.pStop, kMax: s.kMax, pAnswer: s.pAnswer };
}

// ───────────────────────────── the models ─────────────────────────────

/** the unpacked parameters of every model (built once; the parity tests and the debug panel read them) */
export interface ModelBundle {
  readonly cnn: FinalCnn;
  readonly m4: M4Hand;
  readonly phys6: Phys6;
  readonly yaw: YawRef;
}

let bundleCache: ModelBundle | null = null;

export function bundle(): ModelBundle {
  if (!bundleCache) {
    if (m4Doc.sigmoidClip !== 40) throw new RangeError('m4_production.json: sigmoidClip must be m4_models.sigmoid clip 40');
    bundleCache = Object.freeze({
      cnn: buildFinalCnn(cnnDoc),
      m4: buildM4Hand(m4Doc),
      phys6: buildPhys6(phys6Doc),
      yaw: buildYawRef(yawDoc),
    });
  }
  return bundleCache;
}

function forward(id: ModelId, input: AttemptInput): ModelAttemptOutput {
  const b = bundle();
  switch (id) {
    case 'cnn':
      return cnnForward(b.cnn, input).output;
    case 'm4':
      return m4Forward(b.m4, input.window).output;
    case 'phys6':
      return phys6Forward(b.phys6, input.window).output;
    case 'yaw':
      return yawForward(b.yaw, input.window).output;
  }
}

function makeModel(spec: ModelSpec): HandModel {
  return Object.freeze({
    spec,
    predict: (input: AttemptInput): ModelAttemptOutput => forward(spec.id, input),
    aggregate: (outputs: readonly ModelAttemptOutput[]): number => aggregateOutputs(spec.aggregation, outputs),
  });
}

let modelsCache: readonly HandModel[] | null = null;

/** every bundled model, registry order (cnn, m4, phys6, yaw); pure, synchronous, no I/O */
export function loadModels(): HandModel[] {
  if (!modelsCache) {
    bundle();
    modelsCache = Object.freeze(specs.map(makeModel));
  }
  return modelsCache.slice();
}

export function modelById(id: ModelId): HandModel {
  const m = loadModels().find((x) => x.spec.id === id);
  if (!m) throw new RangeError(`unknown model ${JSON.stringify(id)}`);
  return m;
}

/** ModelsApi of ../types.ts */
export const modelsApi: ModelsApi = { specs, loadModels, detector, defaultModel };
