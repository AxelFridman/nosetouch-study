import { describe, expect, it } from 'vitest';
import { EN, HE, MODEL_CHIP, strings, type StringKey } from './strings';

const keys = Object.keys(EN) as StringKey[];
const placeholders = (s: string): string[] => [...s.matchAll(/\{([A-Za-z]+)\}/g)].map((m) => m[1]).sort();

describe('predict strings', () => {
  it('Hebrew has every English key, non-empty, with the same placeholders', () => {
    expect(Object.keys(HE).sort()).toEqual([...keys].sort());
    for (const k of keys) {
      expect(HE[k].trim(), k).not.toBe('');
      expect(placeholders(HE[k]), k).toEqual(placeholders(EN[k]));
    }
  });

  it('keeps the wording of the spec (docs/stage3/APP_SPEC.md 5.3)', () => {
    const en = strings('en');
    expect(en.t('title')).toBe('Which hand?');
    expect(en.t('p_meaning')).toBe('probability that the RIGHT hand was used');
    expect(en.t('abstain')).toBe('No touch found – bring the phone to your nose, then back');
    expect(en.t('grip')).toBe('Turn the screen towards your face');
    expect(en.t('verdict', { HAND: en.hand('right'), TOUCHES: en.touches(2) })).toBe('RIGHT hand · 2 touches');
    expect(en.t('provisional', { TOUCHES: en.touches(1) })).toBe('after 1 touch · not final');
    expect(en.t('demo_badge', { HAND: en.hand('left') })).toBe('Demo: simulated LEFT hand');
    const he = strings('he');
    expect(he.t('p_meaning')).toBe('ההסתברות שהשתמשו ביד ימין');
    expect(he.t('verdict', { HAND: he.hand('right'), TOUCHES: he.touches(1) })).toBe('יד ימין · נגיעה אחת');
    expect(he.touches(3)).toBe('3 נגיעות');
  });

  it('keeps text short: one line per screen (no string over 100 characters)', () => {
    for (const k of keys) expect(EN[k].length, k).toBeLessThanOrEqual(100);
  });

  it('names every model, with a chip label', () => {
    const en = strings('en');
    for (const id of ['cnn', 'm4', 'phys6', 'yaw'] as const) {
      expect(en.modelLabel(id)).not.toMatch(/^m_/);
      expect(en.modelLine(id)).not.toMatch(/^m_/);
      expect(MODEL_CHIP[id].length).toBeLessThanOrEqual(6);
    }
    expect(en.modelLabel('cnn')).toBe('Final CNN');
  });

  it('leaves an unknown placeholder as it is', () => {
    expect(strings('en').t('verdict', { HAND: 'LEFT' })).toBe('LEFT hand · {TOUCHES}');
  });
});
