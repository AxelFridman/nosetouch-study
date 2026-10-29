/**
 * Presentational components of predict mode (docs/stage3/APP_SPEC.md section 5). Pure functions of their
 * props: no hooks and no state, so the vitest render tests can expand them without a DOM.
 *
 * Hand colours (LEFT #D1495B, RIGHT #00739A) only colour the bar and a small swatch; text stays in the
 * page colour and the hands are always named in words. The bar stays left-to-right in Hebrew (LEFT and
 * RIGHT are physical sides). P(right), numbers and model ids are wrapped in <bdi dir="ltr">.
 */
import type { ComponentChildren } from 'preact';
import type { Strings } from '../i18n/strings';
import { MODEL_CHIP } from '../i18n/strings';
import type { AttemptResult, EngineSnapshot, Hand, IgnoredMovement, ModelComparison, ModelId, ModelSpec } from '../types';
import { barLabel, barRightWidth, formatPercent, leftPercent, prob, seconds } from './format';

// ───────────────────────────── the probability ─────────────────────────────

export interface HandBarProps {
  readonly s: Strings;
  readonly pRight: number;
  readonly percentRight: number;
  /** 'mini': the compare rows (no words, no numbers) */
  readonly size?: 'full' | 'mini';
}

export function HandBar({ s, pRight, percentRight, size = 'full' }: HandBarProps) {
  const right = barRightWidth(pRight);
  const label = barLabel(percentRight, s.t('left'), s.t('right'));
  const bar = (
    <div class={size === 'mini' ? 'pm-bar pm-bar-mini' : 'pm-bar'} role="img" aria-label={label} dir="ltr" data-testid={size === 'full' ? 'pm-bar' : undefined}>
      <div class="pm-bar-left" style={{ width: `${100 - right}%` }} />
      <div class="pm-bar-right" style={{ width: `${right}%` }} />
      <div class="pm-bar-tick" />
    </div>
  );
  if (size === 'mini') return bar;
  return (
    <div class="pm-handbar" dir="ltr">
      <div class="pm-bar-row">
        <span class="pm-bar-word">{s.t('left')}</span>
        {bar}
        <span class="pm-bar-word">{s.t('right')}</span>
      </div>
      <div class="pm-bar-nums" aria-hidden="true">
        <span>{formatPercent(leftPercent(percentRight))}</span>
        <span>{formatPercent(percentRight)}</span>
      </div>
    </div>
  );
}

export interface ProbabilityProps {
  readonly s: Strings;
  readonly pRight: number;
  readonly percentRight: number;
  /** final: the answer; provisional: watching after k >= 1 attempts; muted: shown under an abstention */
  readonly mode: 'final' | 'provisional' | 'muted';
  /** attempts used */
  readonly k: number;
  readonly hand: Hand | null;
  /** final only: false for a forced answer below p_stop (k_max, end of stream): a muted 'leaning' line */
  readonly confident?: boolean;
}

export function ProbabilityDisplay({ s, pRight, percentRight, mode, k, hand, confident = true }: ProbabilityProps) {
  return (
    <section class={`pm-prob pm-prob-${mode}`} aria-live="polite" data-testid="pm-prob" data-p={String(pRight)}>
      <div class="pm-pline">
        <bdi dir="ltr">
          <span class="pm-plabel">P(right) =</span> <span class="pm-pnum" data-testid="pm-percent">{formatPercent(percentRight)}</span>
        </bdi>
      </div>
      <div class="pm-pmeaning">{s.t('p_meaning')}</div>
      <HandBar s={s} pRight={pRight} percentRight={percentRight} />
      {mode === 'final' && confident ? (
        <div class="pm-verdict" data-testid="pm-verdict" data-confident="1">
          {hand && <span class={`pm-swatch pm-swatch-${hand}`} aria-hidden="true" />}
          {hand ? s.t('verdict', { HAND: s.hand(hand), TOUCHES: s.touches(k) }) : s.touches(k)}
        </div>
      ) : mode === 'final' ? (
        <div class="pm-verdict pm-verdict-weak" data-testid="pm-verdict" data-confident="0">
          {hand ? s.t('leaning', { HAND: s.hand(hand), TOUCHES: s.touches(k) }) : s.touches(k)}
        </div>
      ) : (
        <div class="pm-provisional" data-testid="pm-provisional">
          {mode === 'provisional' ? s.t('provisional', { TOUCHES: s.touches(k) }) : s.t('compare_after', { TOUCHES: s.touches(k) })}
        </div>
      )}
    </section>
  );
}

// ───────────────────────────── attempts ─────────────────────────────

export function AttemptDots({ k, kMax }: { readonly k: number; readonly kMax: number }) {
  const dots = [];
  for (let i = 0; i < kMax; i++) {
    dots.push(
      <span key={i} class={i < k ? 'pm-dot pm-dot-on' : 'pm-dot'}>
        {i < k ? '✓' : ''}
      </span>,
    );
  }
  return (
    <div class="pm-dots" aria-hidden="true" data-testid="pm-dots" data-k={k}>
      {dots}
    </div>
  );
}

/** The calm or pulsing ring of the watching screen (pulses while the phone moves). */
export function Orb({ still }: { readonly still: boolean }) {
  return <div class={still ? 'pm-orb' : 'pm-orb pm-orb-moving'} aria-hidden="true" data-still={still ? '1' : '0'} />;
}

// ───────────────────────────── compare ─────────────────────────────

export interface CompareProps {
  readonly s: Strings;
  readonly rows: readonly ModelComparison[];
  readonly driver: ModelId;
  readonly k: number;
  readonly onSelect: (id: ModelId) => void;
  /** the desktop demo: the CNN does not apply to simulated motion (no number, no check mark) */
  readonly sim?: boolean;
}

/** models that do not apply to the desktop demo's simulated motion (the CNN names the wrong hand on it) */
export const SIM_NOT_APPLICABLE: readonly ModelId[] = ['cnn'];

export function CompareTable({ s, rows, driver, k, onSelect, sim = false }: CompareProps) {
  return (
    <section class="pm-compare" data-testid="pm-compare">
      <div class="pm-compare-head">
        <span>{s.t('compare_after', { TOUCHES: s.touches(k) })}</span>
        <bdi dir="ltr">P(right)</bdi>
      </div>
      {rows.map((r) => {
        const na = sim && SIM_NOT_APPLICABLE.includes(r.modelId);
        const cls = ['pm-crow', r.modelId === driver ? 'is-driver' : '', na ? 'is-na' : ''].filter(Boolean).join(' ');
        return (
          <button
            key={r.modelId}
            type="button"
            class={cls}
            aria-pressed={r.modelId === driver}
            data-testid={`pm-compare-${r.modelId}`}
            data-p={r.k > 0 ? String(r.pRight) : undefined}
            data-na={na ? '1' : undefined}
            onClick={() => onSelect(r.modelId)}
          >
            <span class="pm-crow-name">{s.modelLabel(r.modelId)}</span>
            <HandBar s={s} pRight={na ? 0.5 : r.pRight} percentRight={na ? 50 : r.percentRight} size="mini" />
            <span class="pm-crow-p">
              <bdi dir="ltr">{na ? s.t('sim_na') : r.k > 0 ? formatPercent(r.percentRight) : s.t('unavailable')}</bdi>
            </span>
            <span class="pm-crow-ok" title={r.confident && !na ? s.t('confident') : undefined}>
              {r.confident && !na ? '✓' : ''}
            </span>
          </button>
        );
      })}
    </section>
  );
}

// ───────────────────────────── chrome: top bar, footer, sheets ─────────────────────────────

export function TopBar({ s, driver, onModels }: { readonly s: Strings; readonly driver: ModelId; readonly onModels: () => void }) {
  return (
    <header class="pm-top">
      <span class="pm-brand">{s.t('brand')}</span>
      <button type="button" class="pm-chip" aria-haspopup="dialog" data-testid="pm-model-chip" onClick={onModels}>
        <bdi dir="ltr">{MODEL_CHIP[driver]}</bdi>
        <span aria-hidden="true" class="pm-chip-caret">
          ▾
        </span>
      </button>
    </header>
  );
}

export interface FooterProps {
  readonly s: Strings;
  readonly driver: ModelId;
  readonly sha256: string;
  readonly version: string;
  readonly demo: boolean;
  readonly onInfo: () => void;
  /** the build's short git commit (which deploy is loaded); omitted when unknown */
  readonly build?: string;
}

export function Footer({ s, driver, sha256, version, demo, onInfo, build }: FooterProps) {
  const parts = [driver, sha256.slice(0, 8), build ? `v${version}+${build}` : `v${version}`];
  if (demo) parts.push('demo');
  return (
    <footer class="pm-foot">
      <bdi dir="ltr" data-testid="pm-footer">
        {parts.join(' · ')}
      </bdi>
      <button type="button" class="pm-info-btn" aria-label={s.t('info_title')} data-testid="pm-info-btn" onClick={onInfo}>
        ?
      </button>
    </footer>
  );
}

function Sheet({ title, s, onClose, testid, children }: { title: string; s: Strings; onClose: () => void; testid: string; children: ComponentChildren }) {
  return (
    <div class="pm-sheet-layer">
      <div class="pm-scrim" onClick={onClose} />
      <div class="pm-sheet" role="dialog" aria-modal="true" aria-label={title} data-testid={testid}>
        <div class="pm-sheet-head">
          <h2>{title}</h2>
          <button type="button" class="pm-x" aria-label={s.t('close')} onClick={onClose}>
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

export interface ModelSheetProps {
  readonly s: Strings;
  readonly specs: readonly ModelSpec[];
  readonly driver: ModelId;
  readonly compare: boolean;
  readonly onSelect: (id: ModelId) => void;
  readonly onCompare: (on: boolean) => void;
  readonly onClose: () => void;
  /** the desktop demo: the CNN's line says it does not apply to simulated motion */
  readonly sim?: boolean;
}

export function ModelSheet({ s, specs, driver, compare, onSelect, onCompare, onClose, sim = false }: ModelSheetProps) {
  return (
    <Sheet title={s.t('model')} s={s} onClose={onClose} testid="pm-model-sheet">
      <div class="pm-radios" role="radiogroup" aria-label={s.t('model')}>
        {specs.map((spec) => (
          <label key={spec.id} class={spec.id === driver ? 'pm-radio is-on' : 'pm-radio'} data-testid={`pm-model-${spec.id}`}>
            <input type="radio" name="pm-model" value={spec.id} checked={spec.id === driver} onChange={() => onSelect(spec.id)} />
            <span class="pm-radio-text">
              <span class="pm-radio-label">{s.modelLabel(spec.id)}</span>
              <span class={sim && SIM_NOT_APPLICABLE.includes(spec.id) ? 'pm-radio-line pm-na' : 'pm-radio-line'}>
                {sim && SIM_NOT_APPLICABLE.includes(spec.id) ? s.t('sim_cnn_line') : s.modelLine(spec.id)}
              </span>
            </span>
          </label>
        ))}
      </div>
      <label class="pm-switch" data-testid="pm-compare-switch">
        <span>{s.t('compare')}</span>
        <input type="checkbox" role="switch" checked={compare} onChange={(e) => onCompare((e.currentTarget as HTMLInputElement).checked)} />
      </label>
    </Sheet>
  );
}

export function InfoSheet({ s, onClose }: { readonly s: Strings; readonly onClose: () => void }) {
  return (
    <Sheet title={s.t('info_title')} s={s} onClose={onClose} testid="pm-info-sheet">
      <p class="pm-info-line">{s.t('info1')}</p>
      <p class="pm-info-line">{s.t('info2')}</p>
      <p class="pm-info-line">{s.t('info3')}</p>
    </Sheet>
  );
}

// ───────────────────────────── icons ─────────────────────────────

export function PhoneIcon() {
  return (
    <svg class="pm-icon" viewBox="0 0 64 64" aria-hidden="true">
      <rect x="20" y="6" width="24" height="44" rx="5" fill="none" stroke="currentColor" stroke-width="3" />
      <path d="M28 44h8" stroke="currentColor" stroke-width="3" stroke-linecap="round" />
      <path d="M12 58c6-5 14-7 20-7s14 2 20 7" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" opacity="0.5" />
    </svg>
  );
}

export function TurnIcon() {
  return (
    <svg class="pm-icon" viewBox="0 0 64 64" aria-hidden="true">
      <rect x="23" y="8" width="18" height="32" rx="4" fill="none" stroke="currentColor" stroke-width="3" />
      <path d="M29 35h6" stroke="currentColor" stroke-width="3" stroke-linecap="round" />
      <path d="M10 44c4 7 12 11 22 11s18-4 22-11" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" />
      <path d="M46 42l8 2-2 8" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  );
}

export function AlertIcon() {
  return (
    <svg class="pm-icon" viewBox="0 0 64 64" aria-hidden="true">
      <circle cx="32" cy="32" r="24" fill="none" stroke="currentColor" stroke-width="3" />
      <path d="M32 19v17" stroke="currentColor" stroke-width="4" stroke-linecap="round" />
      <circle cx="32" cy="45" r="2.6" fill="currentColor" />
    </svg>
  );
}

// ───────────────────────────── debug (&debug=1) ─────────────────────────────

function attemptRow(a: AttemptResult, ids: readonly ModelId[]) {
  return (
    <tr key={`a${a.k}`}>
      <td>{a.k}</td>
      <td>
        {seconds(a.bout.onsetMs)}–{seconds(a.bout.endMs)}
      </td>
      <td>{prob(a.pApproach)}</td>
      {ids.map((id) => (
        <td key={id}>{prob(a.outputs[id]?.pRight)}</td>
      ))}
    </tr>
  );
}

function ignoredRow(m: IgnoredMovement, i: number) {
  return (
    <tr key={`i${i}`}>
      <td>{m.reason}</td>
      <td>
        {seconds(m.bout.onsetMs)}–{seconds(m.bout.endMs)}
      </td>
      <td>{prob(m.pApproach)}</td>
    </tr>
  );
}

export interface DebugProps {
  readonly snap: EngineSnapshot;
  readonly ids: readonly ModelId[];
  readonly engineKind: string;
  readonly ignoredParams: readonly string[];
  /** the session's own timing (wall clock): the longest engine call and the largest pause between two samples */
  readonly timing?: { readonly maxPushMs: number; readonly maxGapMs: number } | null;
}

export function DebugPanel({ snap, ids, engineKind, ignoredParams, timing = null }: DebugProps) {
  const d = snap.decision;
  return (
    <details class="pm-debug" open data-testid="pm-debug" dir="ltr" lang="en">
      <summary>debug</summary>
      <div class="pm-debug-grid">
        <span>engine</span>
        <span>{engineKind}</span>
        <span>phase</span>
        <span>
          {snap.phase} / {d.status}
        </span>
        <span>stream</span>
        <span>
          {seconds(snap.tMs)} s · {snap.nSamples} samples · {snap.rateHz === null ? '–' : snap.rateHz.toFixed(0)} Hz
          {snap.rateHz !== null && snap.rateHz < 50 ? ' (low)' : ''}
        </span>
        {timing && (
          <>
            <span>timing</span>
            <span data-testid="pm-debug-timing">
              longest tick {timing.maxPushMs.toFixed(1)} ms · largest gap {timing.maxGapMs.toFixed(0)} ms
              {timing.maxGapMs > 100 ? ' (gap)' : ''}
            </span>
          </>
        )}
        <span>still</span>
        <span>
          {snap.still ? 'yes' : 'no'}
          {snap.screenAwayHint ? ' · screen away' : ''}
        </span>
        <span>rest up_z</span>
        <span>{snap.restUpZ === null ? '–' : snap.restUpZ.toFixed(2)}</span>
        <span>decision</span>
        <span>
          {snap.driver} P={prob(d.pRight)} k={d.nUsed} {d.trajectory.map((p) => prob(p)).join(' → ')}
        </span>
        <span>stopping</span>
        <span>
          p_stop {snap.stopping.pStop} · k_max {snap.stopping.kMax} · p_answer {snap.stopping.pAnswer}
        </span>
        {ignoredParams.length > 0 && (
          <>
            <span>ignored URL</span>
            <span>{ignoredParams.join(', ')}</span>
          </>
        )}
      </div>
      <table class="pm-debug-table">
        <thead>
          <tr>
            <th>k</th>
            <th>s</th>
            <th>P(appr)</th>
            {ids.map((id) => (
              <th key={id}>{id}</th>
            ))}
          </tr>
        </thead>
        <tbody>{snap.attempts.map((a) => attemptRow(a, ids))}</tbody>
      </table>
      {snap.ignored.length > 0 && (
        <table class="pm-debug-table">
          <thead>
            <tr>
              <th>ignored</th>
              <th>s</th>
              <th>P(appr)</th>
            </tr>
          </thead>
          <tbody>{snap.ignored.map((m, i) => ignoredRow(m, i))}</tbody>
        </table>
      )}
      <p class="pm-debug-note">
        Detector and stopping thresholds are lane m4's (fitted for m4's classifier); lane prod, which would refit them for the other
        models, has not run. The final CNN is not yet validated end to end on detected windows.
      </p>
    </details>
  );
}
