/**
 * Validates records against schema/records.schema.json (the single source of truth shared with the
 * Python analysis kit and the Apps Script backend). Use in tests:
 *   expectValid('TrialRecord', record)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Ajv2020 } from 'ajv/dist/2020.js';
import { expect } from 'vitest';

const schemaPath = fileURLToPath(new URL('../../../schema/records.schema.json', import.meta.url));
export const recordsSchema = JSON.parse(readFileSync(schemaPath, 'utf8'));

const ajv = new Ajv2020({ allErrors: true, strict: false });
ajv.addSchema(recordsSchema);

export type SchemaDef = 'SessionRecord' | 'TrialRecord' | 'ParticipantProfile' | 'UploadEnvelope' | 'ExportBundle' | 'CalibrationResult' | 'PlannedBlock';

export function validate(def: SchemaDef, value: unknown): string[] {
  const fn = ajv.getSchema(`${recordsSchema.$id}#/$defs/${def}`);
  if (!fn) throw new Error(`schema def ${def} not found`);
  if (fn(value)) return [];
  return (fn.errors ?? []).map((e) => `${e.instancePath || '(root)'} ${e.message} ${JSON.stringify(e.params)}`);
}

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
