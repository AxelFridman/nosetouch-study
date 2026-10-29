/**
 * The shapes of the exported model files in this folder (written by analysis/stage3/export.py and runtime.py
 * to_json_*; final_cnn.json is lane final's final_model.json, the same document). Type-only: the registry
 * validates what it reads before any model uses it. These are the documents as they are, so they can say more
 * than the shared contracts in ../types.ts (ModelSpec, AggregationSpec, CalibratorJson, DetectorBundle).
 */
import type { DetectorBundle, ModelId } from '../types';

/**
 * A calibrator as exported: final_infer.calibrate's methods (none, temperature, platt, beta, isotonic, stacked)
 * plus lane m4's (runtime.calibrator_to_json): 'nested' (the inner-CV choice among none, temperature and
 * isotonic_sym) and 'isotonic_sym' with scikit-learn's IsotonicRegression thresholds. A key that is absent
 * means "not applied" (no logit clip, no p clip).
 */
export type ExportedCalibrator =
  | {
      readonly method: 'none';
      readonly logit_clip?: number;
      readonly p_clip?: number;
      /** PHYS6: m1_models._finish clips p to [lo, hi] = [0, 1] after the sigmoid */
      readonly p_clip_out?: readonly number[];
    }
  | { readonly method: 'temperature'; readonly T: number; readonly logit_clip?: number; readonly p_clip?: number }
  | { readonly method: 'platt'; readonly a: number; readonly b: number; readonly logit_clip?: number; readonly p_clip?: number }
  | {
      readonly method: 'beta';
      readonly a: number;
      readonly b: number;
      readonly c: number;
      readonly p0_clip?: number;
      readonly p_clip?: number;
    }
  | { readonly method: 'isotonic'; readonly x: readonly number[]; readonly y: readonly number[]; readonly p_clip?: number }
  | {
      readonly method: 'isotonic_sym';
      readonly x_thresholds: readonly number[];
      readonly y_thresholds: readonly number[];
      readonly x_min: number;
      readonly x_max: number;
      readonly p_lo: number;
      readonly p_hi: number;
      readonly increasing?: boolean;
      readonly out_of_bounds?: string;
    }
  | { readonly method: 'stacked'; readonly inner: ExportedCalibrator; readonly outer: ExportedCalibrator }
  | {
      readonly method: 'nested';
      readonly chosen: string;
      readonly inner: ExportedCalibrator;
      readonly cv_log_loss?: Readonly<Record<string, number>>;
    };

export interface TrainingJson {
  readonly n_trials: number;
  readonly n_persons: number;
  readonly cap_per_person?: number;
  readonly seed?: number;
  readonly persons?: readonly string[];
}

/** m4_production.json > detector: DetectorBundle plus what the export adds (PORT_SPEC 4.2, section 8). */
export interface M4DetectorJson extends DetectorBundle {
  readonly segCfg: string;
  readonly lowpassSos: readonly (readonly number[])[];
  readonly lowpassZi: readonly (readonly number[])[];
  readonly lowpassPadlen: number;
  readonly lowpassRule?: string;
  readonly note?: string;
}

/** m4_production.json ('nosetouch-m4-production-1'): lane m4's ProductionModel. */
export interface M4ProductionJson {
  readonly format: 'nosetouch-m4-production-1';
  readonly id: 'm4';
  readonly label: string;
  readonly short?: string;
  readonly stage2Name: string;
  readonly detector: M4DetectorJson;
  readonly hand: {
    readonly kind: 'yaw8';
    readonly channel: string;
    readonly index: number;
    readonly step: number;
    readonly n: number;
    readonly C: number;
    readonly coef: readonly number[];
    readonly intercept: number;
    readonly nanToNum: boolean;
  };
  readonly calibrator: ExportedCalibrator;
  /** m4_models.EPS: the calibrated logit clips p to [eps, 1 - eps] */
  readonly logitEps: number;
  /** m4_models.sigmoid clips its argument to +-40 */
  readonly sigmoidClip: number;
  readonly aggregation: {
    readonly kind: 'scaled_mean';
    readonly name: string;
    readonly form?: string;
    readonly s1?: number | null;
    readonly b?: number | null;
    /** s_k by k ('1'..'5'); null = s_k 1 (the plain mean logit) */
    readonly s: Readonly<Record<string, number>> | null;
  };
  readonly note: string;
  readonly training: TrainingJson;
}

/** phys6.json ('nosetouch-phys6-1'): lane m1's m1_lr_phys6_nested refitted on all persons. */
export interface Phys6Json {
  readonly format: 'nosetouch-phys6-1';
  readonly id: 'phys6';
  readonly label: string;
  readonly short?: string;
  readonly stage2Name: string;
  readonly features: readonly string[];
  readonly option: string;
  readonly C: number;
  readonly winsor: 'none' | 'clip' | 'asinh';
  readonly winsor_med: readonly number[];
  readonly winsor_s: readonly number[];
  readonly winsor_lo: readonly (number | null)[];
  readonly winsor_hi: readonly (number | null)[];
  readonly scale_kind: string;
  readonly sign: readonly number[];
  readonly centre: readonly number[];
  readonly scale: readonly number[];
  readonly coef: readonly number[];
  readonly intercept: number;
  readonly fit_intercept: boolean;
  readonly calibrator: ExportedCalibrator;
  readonly logitEps: number;
  readonly aggregation: { readonly kind: 'sum_over_k_pow'; readonly alpha: number };
  readonly input: { readonly body_frame: { readonly h_min: number; readonly nan_below_h_min: boolean } };
  readonly training: TrainingJson;
}

/** yaw_ref.json ('nosetouch-yaw-ref-1'): stage2/scripts/reference.py example_yaw_logreg refitted on all persons. */
export interface YawRefJson {
  readonly format: 'nosetouch-yaw-ref-1';
  readonly id: 'yaw';
  readonly label: string;
  readonly short?: string;
  readonly stage2Name: string;
  readonly channel: string;
  readonly index: number;
  readonly step: number;
  readonly n: number;
  readonly coef: readonly number[];
  readonly intercept: number;
  readonly coef_dtype?: string;
  readonly C: number;
  readonly nan_to_num: boolean;
  readonly calibrator: ExportedCalibrator;
  readonly aggregation: { readonly kind: 'mean_p'; readonly calibrated_beyond_k1?: boolean };
  readonly training: TrainingJson;
}

/** One entry of manifest.json > models. */
export interface ManifestModelJson {
  readonly id: ModelId;
  readonly file: string;
  readonly format: string;
  readonly sha256: string;
  readonly bytes: number;
  readonly label: string;
  readonly short: string;
  readonly stage2Name: string;
  readonly kind: string;
  readonly calibration: string;
  readonly isDefault: boolean;
  readonly trainedOn: { readonly persons: number; readonly trials: number };
  readonly loso?: {
    readonly accuracy: number;
    readonly ci95: readonly number[];
    readonly what?: string;
    readonly source?: string;
  };
}

/** manifest.json ('nosetouch-models-manifest-1'). */
export interface ManifestJson {
  readonly format: 'nosetouch-models-manifest-1';
  readonly default: ModelId;
  readonly detector: ModelId;
  readonly order: readonly ModelId[];
  readonly models: readonly ManifestModelJson[];
  readonly note: string;
}
