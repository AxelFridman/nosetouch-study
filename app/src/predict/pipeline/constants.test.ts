/**
 * The pipeline's literal constants equal what the Python exporter wrote into the model bundle
 * (models/m4_production.json detector: scipy's lowpass sos / zi / padlen and the stage2 constants) and the
 * channel order of the final model (models/final_cnn.json input.channels = standardize.CHANNELS).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  CHANNELS,
  FS_HZ,
  GRAVITY,
  LOWPASS_PADLEN,
  LOWPASS_SOS,
  LOWPASS_ZI,
  MAG_MAX,
  MAG_MIN,
  MAX_GAP_MS,
  MIN_REST_N,
  MIN_ROWS,
  N_NORM,
  REPEAT_MIN_DPS,
  REST_MS,
  STEP_MS,
  STILL_GYRO_DPS,
  YAW_CUM,
  channelIndex,
} from './constants';

const MODELS = new URL('../models/', import.meta.url);
const m4 = JSON.parse(readFileSync(new URL('m4_production.json', MODELS), 'utf8')) as {
  detector: {
    lowpassSos: number[][];
    lowpassZi: number[][];
    lowpassPadlen: number;
    constants: Record<string, number>;
  };
};

describe('pipeline constants = the exported bundle', () => {
  it('Butterworth sos, zi and pad length are the exported scipy values, bit for bit', () => {
    expect(LOWPASS_SOS.map((r) => [...r])).toEqual(m4.detector.lowpassSos);
    expect(LOWPASS_ZI.map((r) => [...r])).toEqual(m4.detector.lowpassZi);
    for (let s = 0; s < 2; s++) {
      for (let c = 0; c < 6; c++) expect(Object.is(LOWPASS_SOS[s][c], m4.detector.lowpassSos[s][c])).toBe(true);
      for (let c = 0; c < 2; c++) expect(Object.is(LOWPASS_ZI[s][c], m4.detector.lowpassZi[s][c])).toBe(true);
    }
    expect(LOWPASS_PADLEN).toBe(m4.detector.lowpassPadlen);
  });

  it('stage2 constants', () => {
    const k = m4.detector.constants;
    expect(FS_HZ).toBe(k.FS_HZ);
    expect(MAX_GAP_MS).toBe(k.MAX_GAP_MS);
    expect(STILL_GYRO_DPS).toBe(k.STILL_GYRO_DPS);
    expect(REST_MS).toBe(k.REST_MS);
    expect(MIN_REST_N).toBe(k.MIN_REST_N);
    expect(N_NORM).toBe(k.N_NORM);
    expect(GRAVITY).toBe(k.GRAVITY);
    expect(MAG_MIN).toBe(k.MAG_MIN);
    expect(MAG_MAX).toBe(k.MAG_MAX);
    expect(MIN_ROWS).toBe(k.MIN_ROWS);
    expect(REPEAT_MIN_DPS).toBe(k.REPEAT_MIN_DPS);
    expect(STEP_MS).toBe(16.666666666666668);
  });

  it('channel order is standardize.CHANNELS, as the final model reads it', () => {
    const cnn = JSON.parse(readFileSync(new URL('final_cnn.json', MODELS), 'utf8')) as { input: { channels: string[] } };
    expect([...CHANNELS]).toEqual(cnn.input.channels);
    expect(channelIndex('yaw_cum')).toBe(YAW_CUM);
    expect(CHANNELS.length).toBe(17);
  });
});
