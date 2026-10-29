import { describe, expect, it } from 'vitest';
import { demoSearch, isPredictSearch, parsePredictParams, withParam } from './params';

describe('predict mode routing', () => {
  it('is on for ?predict=1 and ?mode=predict only', () => {
    expect(isPredictSearch('?predict=1')).toBe(true);
    expect(isPredictSearch('?mode=predict&lang=he')).toBe(true);
    expect(isPredictSearch('?sim=1')).toBe(false);
    expect(isPredictSearch('?predict=0')).toBe(false);
    expect(isPredictSearch('')).toBe(false);
  });
});

describe('parsePredictParams', () => {
  it('defaults: the final CNN, English, real sensors, nothing ignored', () => {
    expect(parsePredictParams('?predict=1')).toEqual({
      model: 'cnn',
      compare: false,
      sim: false,
      simHand: null,
      simSeed: null,
      lang: 'en',
      debug: false,
      ignored: [],
    });
  });

  it('reads every option', () => {
    expect(parsePredictParams('?predict=1&model=phys6&view=compare&sim=1&hand=left&seed=7&lang=he&debug=1')).toEqual({
      model: 'phys6',
      compare: true,
      sim: true,
      simHand: 'left',
      simSeed: 7,
      lang: 'he',
      debug: true,
      ignored: [],
    });
    expect(parsePredictParams('?mode=predict&model=M4&hand=RIGHT').model).toBe('m4');
    expect(parsePredictParams('?mode=predict&model=yaw').model).toBe('yaw');
    // the desktop demo defaults to lane m4's classifier (the CNN misreads simulated touches); &model still wins
    expect(parsePredictParams('?predict=1&sim=1').model).toBe('m4');
    expect(parsePredictParams('?predict=1&sim=1&model=cnn').model).toBe('cnn');
    expect(parsePredictParams('?predict=1').model).toBe('cnn');
    expect(parsePredictParams('?predict=1&seed=4294967295').simSeed).toBe(4294967295);
  });

  it('falls back to the default on a value it does not understand, and lists the parameter', () => {
    const p = parsePredictParams('?predict=1&model=gpt&hand=both&seed=-3&lang=fr&sim=yes&view=table&debug=2');
    expect(p.model).toBe('cnn');
    expect(p.simHand).toBeNull();
    expect(p.simSeed).toBeNull();
    expect(p.lang).toBe('en');
    expect(p.sim).toBe(false);
    expect(p.compare).toBe(false);
    expect(p.debug).toBe(false);
    expect(p.ignored).toEqual(['model', 'view', 'sim', 'hand', 'seed', 'lang', 'debug']);
    expect(parsePredictParams('?predict=1&seed=4294967296').ignored).toEqual(['seed']);
  });
});

describe('URL helpers', () => {
  it('withParam sets or removes one parameter and keeps the others', () => {
    expect(withParam('?predict=1&lang=he', 'model', 'm4')).toBe('?predict=1&lang=he&model=m4');
    expect(withParam('?predict=1&model=m4', 'model', 'yaw')).toBe('?predict=1&model=yaw');
    expect(withParam('?predict=1&model=m4', 'model', null)).toBe('?predict=1');
    expect(withParam('', 'model', null)).toBe('');
  });

  it('the demo link keeps the language, the model and debug, and drops everything else', () => {
    expect(demoSearch('?predict=1&lang=he&model=m4&foo=bar')).toBe('?predict=1&sim=1&lang=he&model=m4');
    expect(demoSearch('?mode=predict')).toBe('?predict=1&sim=1');
  });
});
