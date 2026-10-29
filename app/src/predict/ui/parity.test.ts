/**
 * Parity of what the UI computes itself against the Python reference, on the exported fixtures:
 *  - percent_right = int(round(100 * p)) with Python's round-half-even (m4_production.predict_stream), for
 *    every model's decision and every P_k of its k-curve, offline ('reference') and live;
 *  - the display never shows "100 %" or "0 %";
 *  - each decision maps to the screen the spec gives it (answer / abstain / watching / grip).
 * The committed fixtures are synthetic (app/src/predict/__fixtures__); fixtures from real recordings stay in
 * analysis/data/stage3_fixtures/ (git-ignored) and this check is skipped where that folder is absent (CI).
 */
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import type { DecisionStatus, EngineSnapshot } from '../types';
import { formatPercent, percentOf } from './format';
import { screenOf } from './screens';

const SYNTH = new URL('../__fixtures__/', import.meta.url);
const REAL = new URL('../../../../analysis/data/stage3_fixtures/', import.meta.url);

interface FixtureDecision {
  p_right: number;
  percent_right: number;
  stopped: string;
  answered: boolean;
  final?: boolean;
  n_used: number;
  /** P_k after k attempts (the exporter writes {k, p, decided_ms, seg}) */
  k_curve?: (number | { k: number; p: number })[];
}

interface Fixture {
  id: string;
  reference?: { decisions?: Record<string, FixtureDecision> };
  live?: { decisions?: Record<string, FixtureDecision> };
}

function load(dir: URL): Fixture[] {
  const index = JSON.parse(readFileSync(new URL('index.json', dir), 'utf8')) as { cases: { file: string }[] };
  return index.cases.map((c) => JSON.parse(gunzipSync(readFileSync(new URL(c.file, dir))).toString('utf8')) as Fixture);
}

/** a minimal snapshot carrying one fixture decision (only what screenOf reads) */
function snapshotOf(d: FixtureDecision, live: boolean): EngineSnapshot {
  const status = d.stopped as DecisionStatus;
  const final = live ? d.final !== false : true;
  const phase: EngineSnapshot['phase'] = status === 'grip' ? 'grip' : final ? 'final' : 'watching';
  return { phase, decision: { status, answered: d.answered } } as unknown as EngineSnapshot;
}

function check(fixtures: Fixture[]): { decisions: number; pk: number; worstPercent: number } {
  let decisions = 0;
  let pk = 0;
  let worstPercent = 0;
  for (const f of fixtures) {
    for (const [mode, block] of [
      ['reference', f.reference],
      ['live', f.live],
    ] as const) {
      for (const [model, d] of Object.entries(block?.decisions ?? {})) {
        const where = `${f.id} ${mode} ${model}`;
        decisions += 1;
        worstPercent = Math.max(worstPercent, Math.abs(percentOf(d.p_right) - d.percent_right));
        expect(percentOf(d.p_right), where).toBe(d.percent_right);
        const shown = formatPercent(d.percent_right);
        expect(shown, where).not.toMatch(/^(100|0) %$/);
        for (const e of d.k_curve ?? []) {
          pk += 1;
          const pc = percentOf(typeof e === 'number' ? e : e.p);
          expect(pc, where).toBeGreaterThanOrEqual(0);
          expect(pc, where).toBeLessThanOrEqual(100);
        }
        const screen = screenOf({ stage: 'running', failure: null, snapshot: snapshotOf(d, mode === 'live') });
        if (d.stopped === 'grip') expect(screen, where).toBe('grip');
        else if (mode === 'reference' || d.final !== false) expect(screen, where).toBe(d.answered ? 'answer' : 'abstain');
        else expect(screen, where).toBe('watching');
      }
    }
  }
  return { decisions, pk, worstPercent };
}

describe.skipIf(!existsSync(new URL('index.json', SYNTH)))('UI parity on the committed synthetic fixtures', () => {
  it('percent_right, the display clamps and the screen of every decision match the Python reference', () => {
    const r = check(load(SYNTH));
    expect(r.decisions).toBeGreaterThan(0);
    expect(r.worstPercent).toBe(0);
    console.info(`[ui parity] synthetic: ${r.decisions} decisions, ${r.pk} P_k values, worst |d percent| = ${r.worstPercent}`);
  });
});

describe.skipIf(!existsSync(new URL('index.json', REAL)))('UI parity on the local real-recording fixtures (git-ignored)', () => {
  it('percent_right, the display clamps and the screen of every decision match the Python reference', () => {
    const r = check(load(REAL));
    expect(r.decisions).toBeGreaterThan(0);
    expect(r.worstPercent).toBe(0);
    console.info(`[ui parity] real: ${r.decisions} decisions, ${r.pk} P_k values, worst |d percent| = ${r.worstPercent}`);
  });
});
