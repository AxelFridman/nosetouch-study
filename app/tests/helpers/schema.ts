/**
 * Validates records against schema/records.schema.json (the single source of truth shared with the
 * Python analysis kit and the Apps Script backend). Use in tests:
 *   expectValid('TrialRecord', record)
 * The validator itself is in schemaValidator.ts (no Vitest import, so the e2e journeys reuse it).
 */
import { expect } from 'vitest';
import { validate, type SchemaDef } from './schemaValidator';

export { recordsSchema, validate, type SchemaDef } from './schemaValidator';

export function expectValid(def: SchemaDef, value: unknown): void {
  const errors = validate(def, value);
  expect(errors, `${def} schema errors:\n${errors.join('\n')}`).toEqual([]);
}

/** All columns of a stream object must have equal length. */
export function expectEqualColumnLengths(stream: Record<string, unknown> | null): void {
  if (!stream) return;
  const lengths = Object.entries(stream)
    .filter(([, v]) => Array.isArray(v))
    .map(([k, v]) => [k, (v as unknown[]).length] as const);
  const n = lengths[0]?.[1];
  for (const [k, len] of lengths) expect(len, `column ${k}`).toBe(n);
}
