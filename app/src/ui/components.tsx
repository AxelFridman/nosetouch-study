/**
 * Shared UI primitives. Screens compose these; they never style raw <button>/<input> themselves.
 * Every interactive primitive accepts `testId` → data-testid (used by the Playwright tests).
 */
import type { ComponentChildren, JSX } from 'preact';
import { useAppState } from '../app/store';
import { useStrings } from '../i18n';

type Tone = 'info' | 'warn' | 'error' | 'ok';

export function Screen(props: {
  title?: string;
  onBack?: () => void;
  headerRight?: ComponentChildren;
  footer?: ComponentChildren;
  children?: ComponentChildren;
  className?: string;
  testId?: string;
}) {
  const s = useStrings();
  const st = useAppState();
  return (
    <div class={`screen ${props.className ?? ''}`} data-testid={props.testId}>
      {(props.title || props.onBack || props.headerRight) && (
        <header class="screen-header">
          {props.onBack && (
            <button class="icon-btn" aria-label={s.common.back} onClick={props.onBack} data-testid="btn-back">
              <span class="flip-rtl" aria-hidden="true">
                ←
              </span>
            </button>
          )}
          <h1>{props.title}</h1>
          {st.settings.simulate && <span class="badge badge-sim">{s.common.simulatedBadge}</span>}
          {props.headerRight}
        </header>
      )}
      <main class="screen-body">{props.children}</main>
      {props.footer && <footer class="screen-footer">{props.footer}</footer>}
    </div>
  );
}

export function Button(props: {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'md' | 'lg' | 'xl';
  block?: boolean;
  disabled?: boolean;
  type?: 'button' | 'submit';
  onClick?: (e: JSX.TargetedMouseEvent<HTMLButtonElement>) => void;
  children?: ComponentChildren;
  testId?: string;
  ariaLabel?: string;
}) {
  const cls = ['btn', `btn-${props.variant ?? 'primary'}`, `btn-${props.size ?? 'md'}`, props.block ? 'btn-block' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <button
      type={props.type ?? 'button'}
      class={cls}
      disabled={props.disabled}
      onClick={props.onClick}
      data-testid={props.testId}
      aria-label={props.ariaLabel}
    >
      {props.children}
    </button>
  );
}

export function Field(props: {
  label: string;
  hint?: string;
  error?: string | null;
  htmlFor?: string;
  children?: ComponentChildren;
}) {
  return (
    <div class="field">
      <label class="field-label" for={props.htmlFor}>
        {props.label}
      </label>
      {props.children}
      {props.hint && <div class="field-hint">{props.hint}</div>}
      {props.error && (
        <div class="field-error" role="alert">
          {props.error}
        </div>
      )}
    </div>
  );
}

export function TextInput(props: {
  id?: string;
  value: string;
  onInput: (v: string) => void;
  placeholder?: string;
  maxLength?: number;
  inputMode?: 'text' | 'email' | 'url' | 'numeric' | 'decimal';
  autoComplete?: string;
  /** phone keyboards: 'words' capitalises names */
  autoCapitalize?: 'off' | 'none' | 'sentences' | 'words' | 'characters';
  testId?: string;
}) {
  return (
    <input
      id={props.id}
      class="input"
      type="text"
      value={props.value}
      placeholder={props.placeholder}
      maxLength={props.maxLength}
      inputMode={props.inputMode}
      autoComplete={props.autoComplete ?? 'off'}
      autoCapitalize={props.autoCapitalize}
      onInput={(e) => props.onInput((e.currentTarget as HTMLInputElement).value)}
      data-testid={props.testId}
    />
  );
}

/**
 * Numeric input that reports null when empty/invalid. inputMode picks the phone keyboard: 'numeric'
 * (digits only, e.g. an age) or 'decimal' (default: digits and a decimal separator).
 */
export function NumberInput(props: {
  id?: string;
  value: number | null;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  inputMode?: 'numeric' | 'decimal';
  testId?: string;
}) {
  return (
    <input
      id={props.id}
      class="input"
      type="number"
      inputMode={props.inputMode ?? 'decimal'}
      value={props.value ?? ''}
      min={props.min}
      max={props.max}
      step={props.step ?? 1}
      placeholder={props.placeholder}
      onInput={(e) => {
        const raw = (e.currentTarget as HTMLInputElement).value.trim();
        const n = raw === '' ? null : Number(raw);
        props.onChange(n === null || Number.isFinite(n) ? n : null);
      }}
      data-testid={props.testId}
    />
  );
}

/** Radio group rendered as large tappable tiles. testId of each option: `${testIdPrefix}-${value}`. */
export function ChoiceGroup<T extends string>(props: {
  name: string;
  value: T | null;
  options: { value: T; label: ComponentChildren; hint?: string }[];
  onChange: (v: T) => void;
  columns?: 1 | 2 | 3 | 4;
  testIdPrefix?: string;
  ariaLabel?: string;
}) {
  return (
    <div
      class="choice-group"
      role="radiogroup"
      aria-label={props.ariaLabel}
      style={{ '--cols': String(props.columns ?? 2) } as JSX.CSSProperties}
    >
      {props.options.map((o) => (
        <label
          key={o.value}
          class={`choice ${props.value === o.value ? 'is-checked' : ''}`}
          data-testid={props.testIdPrefix ? `${props.testIdPrefix}-${o.value}` : undefined}
        >
          <input
            type="radio"
            name={props.name}
            value={o.value}
            checked={props.value === o.value}
            onChange={() => props.onChange(o.value)}
          />
          <span>{o.label}</span>
          {o.hint && <span class="choice-hint">{o.hint}</span>}
        </label>
      ))}
    </div>
  );
}

/** Multi-select tile group. testId of each option: `${testIdPrefix}-${value}`. */
export function MultiChoice<T extends string>(props: {
  values: T[];
  options: { value: T; label: ComponentChildren; hint?: string }[];
  onChange: (v: T[]) => void;
  columns?: 1 | 2 | 3 | 4;
  testIdPrefix?: string;
}) {
  const toggle = (v: T) =>
    props.onChange(props.values.includes(v) ? props.values.filter((x) => x !== v) : [...props.values, v]);
  return (
    <div class="choice-group" style={{ '--cols': String(props.columns ?? 2) } as JSX.CSSProperties}>
      {props.options.map((o) => (
        <label
          key={o.value}
          class={`choice ${props.values.includes(o.value) ? 'is-checked' : ''}`}
          data-testid={props.testIdPrefix ? `${props.testIdPrefix}-${o.value}` : undefined}
        >
          <input type="checkbox" checked={props.values.includes(o.value)} onChange={() => toggle(o.value)} />
          <span>{o.label}</span>
          {o.hint && <span class="choice-hint">{o.hint}</span>}
        </label>
      ))}
    </div>
  );
}

export function Toggle(props: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ComponentChildren;
  hint?: string;
  testId?: string;
}) {
  return (
    <label class="toggle" data-testid={props.testId}>
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(e) => props.onChange((e.currentTarget as HTMLInputElement).checked)}
      />
      <span class="toggle-text">
        <span>{props.label}</span>
        {props.hint && <span class="field-hint">{props.hint}</span>}
      </span>
    </label>
  );
}

export function Card(props: { title?: string; tone?: Tone; children?: ComponentChildren; testId?: string }) {
  return (
    <section class={`card ${props.tone ? `card-${props.tone}` : ''}`} data-testid={props.testId}>
      {props.title && <h2>{props.title}</h2>}
      {props.children}
    </section>
  );
}

export function Notice(props: { kind: Tone; children?: ComponentChildren; testId?: string }) {
  return (
    <div class={`notice notice-${props.kind}`} role={props.kind === 'error' ? 'alert' : 'status'} data-testid={props.testId}>
      {props.children}
    </div>
  );
}

export function ProgressBar(props: { value: number; label?: ComponentChildren; right?: ComponentChildren }) {
  const pct = Math.max(0, Math.min(1, props.value)) * 100;
  return (
    <div>
      {(props.label || props.right) && (
        <div class="progress-label">
          <span>{props.label}</span>
          <span>{props.right}</span>
        </div>
      )}
      <div class="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}>
        <div style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function Stat(props: { label: ComponentChildren; value: ComponentChildren; testId?: string }) {
  return (
    <div class="stat" data-testid={props.testId}>
      <span class="stat-label">{props.label}</span>
      <span class="stat-value">{props.value}</span>
    </div>
  );
}

export function Spinner(props: { label?: string }) {
  return (
    <div class="row center" role="status">
      <div class="spinner" aria-hidden="true" />
      {props.label && <span class="muted">{props.label}</span>}
    </div>
  );
}

/** Global toast, rendered once by App. */
export function Toast() {
  const st = useAppState();
  if (!st.toast) return null;
  return (
    <div class={`toast toast-${st.toast.kind}`} role="status" data-testid="toast">
      {st.toast.text}
    </div>
  );
}
