/**
 * motionlab/preprocess.py, the parts stream_from_record runs (PORT_SPEC sections 1.3-1.7):
 *   repeated_sample_mask (:575-595), _prepare_times (:40-55), uniform_grid (:58-64),
 *   resample_uniform (:67-108), finite_runs (:111-119), lowpass (:122-145), fill_gaps_linear (:148-159).
 * Every function keeps numpy's order of operations; see numeric.ts for the primitives.
 */
import type { BrowserEngine, Rows3 } from '../types';
import { LOWPASS_PADLEN, LOWPASS_SOS, LOWPASS_ZI, MAX_GAP_MS, REPEAT_MIN_DPS, STEP_MS } from './constants';
import { interpBetween, makeSos, norm3, npInterp, sosfiltfiltRun, type Sos } from './numeric';

/** The stream low-pass: Butterworth order 4 at 15 Hz, fs 60 (scipy coefficients, constants.ts). */
export const LOWPASS: Sos = makeSos(LOWPASS_SOS, LOWPASS_ZI);

/** The three values of row i equal those of row i - 1 (NaN equal to NaN). */
function tripleRepeats(d: Float64Array, i: number): boolean {
  const a = 3 * i;
  for (let c = 0; c < 3; c++) {
    const u = d[a + c];
    const v = d[a - 3 + c];
    if (!(u === v || (u !== u && v !== v))) return false;
  }
  return true;
}

function tripleAnyFinite(d: Float64Array, i: number): boolean {
  const a = 3 * i;
  return Number.isFinite(d[a]) || Number.isFinite(d[a + 1]) || Number.isFinite(d[a + 2]);
}

/** Row i of the 9 raw columns [acc, accg, rr] equals row i - 1 (NaN equal to NaN). */
function rowRepeats(acc: Float64Array, accg: Float64Array, rr: Float64Array, i: number): boolean {
  return tripleRepeats(acc, i) && tripleRepeats(accg, i) && tripleRepeats(rr, i);
}

function rowAnyFinite(acc: Float64Array, accg: Float64Array, rr: Float64Array, i: number): boolean {
  return tripleAnyFinite(acc, i) || tripleAnyFinite(accg, i) || tripleAnyFinite(rr, i);
}

/**
 * `same[i]` of repeated_sample_mask: row i repeats row i - 1 exactly (every column equal, NaN equal to NaN) and
 * holds at least one finite value; same[0] = false (preprocess.py:589-590).
 */
export function sameAsPrevious(raw: { acc: Rows3; accg: Rows3; rr: Rows3 }, i: number): boolean {
  return i > 0 && rowRepeats(raw.acc.data, raw.accg.data, raw.rr.data, i) && rowAnyFinite(raw.acc.data, raw.accg.data, raw.rr.data, i);
}

/** The drop rule for an isolated repeat: always on WebKit, else only while |rr| > 20 deg/s (preprocess.py:592-595). */
export function dropIsolated(rr: Rows3, i: number, engine: BrowserEngine | null): boolean {
  if (engine === 'webkit') return true;
  const d = rr.data;
  return norm3(d[3 * i], d[3 * i + 1], d[3 * i + 2]) > REPEAT_MIN_DPS;
}

/**
 * repeated_sample_mask(acc, accg, rr, engine): 1 for an isolated exact repeat of the previous row that is
 * dropped (raw, unsigned values; the sign does not change equality). Rows are in record order.
 */
export function repeatedSampleMask(raw: { acc: Rows3; accg: Rows3; rr: Rows3 }, n: number, engine: BrowserEngine | null): Uint8Array {
  const mask = new Uint8Array(n);
  if (n < 2) return mask;
  const same = new Uint8Array(n);
  for (let i = 1; i < n; i++) same[i] = sameAsPrevious(raw, i) ? 1 : 0;
  for (let i = 0; i < n; i++) {
    const isolated = same[i] === 1 && !(i + 1 < n && same[i + 1] === 1) && !(i >= 1 && same[i - 1] === 1);
    if (isolated && dropIsolated(raw.rr, i, engine)) mask[i] = 1;
  }
  return mask;
}

/**
 * _prepare_times on the kept rows: the indices of the rows with a finite t, stably sorted by t, keeping the first
 * of equal times. `rows` are row indices in record order.
 */
export function prepareTimes(t: ArrayLike<number>, rows: readonly number[]): number[] {
  const ok = rows.filter((i) => Number.isFinite(t[i]));
  let sorted = true;
  for (let k = 1; k < ok.length; k++) {
    if (!(t[ok[k]] >= t[ok[k - 1]])) {
      sorted = false;
      break;
    }
  }
  if (!sorted) ok.sort((x, y) => t[x] - t[y]); // Array.prototype.sort is stable (ES2019), like argsort(kind='stable')
  const out: number[] = [];
  for (const i of ok) {
    if (out.length === 0 || t[i] - t[out[out.length - 1]] > 0) out.push(i);
  }
  return out;
}

/** uniform_grid(t_start, t_end, 60 Hz): count = floor((t_end - t_start) / step + 1e-9) + 1 (preprocess.py:58-64). */
export function gridCount(tStart: number, tEnd: number): number {
  if (!(Number.isFinite(tStart) && Number.isFinite(tEnd)) || tEnd < tStart) return 0;
  return Math.floor((tEnd - tStart) / STEP_MS + 1e-9) + 1;
}

/** t[i] = t_start + i * step (numpy: t_start + arange(n) * step). */
export function gridTime(tStart: number, i: number): number {
  return tStart + i * STEP_MS;
}

/**
 * resample_uniform's value of one column at grid time `tn` (preprocess.py:96-107), from the column's finite
 * samples tv[0:len] (strictly increasing) and xv, where r = searchsorted(tv, tn, 'left'):
 *   exact (tv[r] == tn)                      -> np.interp = xv[r]
 *   inside (1 <= r < len), tv[r] - tv[r-1] <= MAX_GAP_MS -> np.interp(tn, tv, xv)
 *   otherwise (before the first, after the last sample, or inside a longer gap) -> NaN
 */
export function resampleAt(tv: Float64Array, xv: Float64Array, len: number, tn: number, r: number): number {
  if (r < len && tv[r] === tn) return xv[r];
  // inside: tv[r - 1] < tn < tv[r], so np.interp takes j = r - 1 (neither the last point nor an exact hit)
  if (r >= 1 && r < len && tv[r] - tv[r - 1] <= MAX_GAP_MS) return interpBetween(tn, tv[r - 1], tv[r], xv[r - 1], xv[r]);
  return NaN;
}

/** finite_runs of a column: [start, stop) of every run of finite values. */
export function finiteRuns(x: ArrayLike<number>, n: number): [number, number][] {
  const runs: [number, number][] = [];
  let a = -1;
  for (let i = 0; i < n; i++) {
    const ok = Number.isFinite(x[i]);
    if (ok && a < 0) a = i;
    else if (!ok && a >= 0) {
      runs.push([a, i]);
      a = -1;
    }
  }
  if (a >= 0) runs.push([a, n]);
  return runs;
}

/**
 * lowpass(x) for one column (preprocess.py:122-145): zero-phase Butterworth over every finite run separately;
 * runs shorter than 3 are left unfiltered; padlen = min(15, run - 1). Filters `col` in place.
 */
export function lowpassColumn(col: Float64Array, n: number): void {
  if (n < 3) return; // X.shape[0] < 3: copy
  for (const [a, b] of finiteRuns(col, n)) {
    if (b - a < 3) continue;
    sosfiltfiltRun(LOWPASS, col, a, b - a, col, LOWPASS_PADLEN);
  }
}

/**
 * fill_gaps_linear(t, X) on (n, 3) rows: per column, the NaN rows get np.interp(t[nan], t[ok], X[ok]) (edges hold
 * the nearest finite value); an all-finite or all-NaN column is unchanged. Returns a new array (row-major 3n).
 */
export function fillGapsLinear(t: ArrayLike<number>, x: Float64Array, n: number): Float64Array {
  const out = x.slice(0, 3 * n);
  const tOk = new Float64Array(n);
  const xOk = new Float64Array(n);
  for (let c = 0; c < 3; c++) {
    let m = 0;
    for (let i = 0; i < n; i++) {
      const v = out[3 * i + c];
      if (Number.isFinite(v)) {
        tOk[m] = t[i];
        xOk[m] = v;
        m++;
      }
    }
    if (m === n || m === 0) continue;
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(out[3 * i + c])) out[3 * i + c] = npInterp(t[i], tOk, xOk, m);
    }
  }
  return out;
}
