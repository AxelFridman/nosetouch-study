import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { APP_VERSION, ORIENTATIONS, SCHEMA_VERSION } from '../src/core/constants';
import { recordsSchema } from './helpers/schema';

describe('constants', () => {
  it('APP_VERSION matches package.json', () => {
    const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
    expect(APP_VERSION).toBe(pkg.version);
  });

  it('SCHEMA_VERSION is accepted by the schema pattern', () => {
    const pattern = new RegExp(recordsSchema.$defs.SessionRecord.properties.schema_version.pattern);
    expect(pattern.test(SCHEMA_VERSION)).toBe(true);
  });

  it('orientation mirror map is an involution and matches the schema enum', () => {
    const ids = Object.keys(ORIENTATIONS);
    expect(new Set(ids)).toEqual(new Set(recordsSchema.$defs.OrientationId.enum));
    for (const id of ids) {
      const m = ORIENTATIONS[id as keyof typeof ORIENTATIONS].mirror;
      expect(ORIENTATIONS[m].mirror).toBe(id);
    }
  });
});
