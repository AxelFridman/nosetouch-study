/**
 * Validates records against schema/records.schema.json (the single source of truth shared with the
 * Python analysis kit and the backends). No test-runner import: used by the Vitest helpers
 * (tests/helpers/schema.ts) and by the Playwright journeys (e2e/), which validate what the app exports.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Ajv2020 } from 'ajv/dist/2020.js';

const schemaPath = fileURLToPath(new URL('../../../schema/records.schema.json', import.meta.url));
export const recordsSchema = JSON.parse(readFileSync(schemaPath, 'utf8'));

const ajv = new Ajv2020({ allErrors: true, strict: false });
ajv.addSchema(recordsSchema);

export type SchemaDef = 'SessionRecord' | 'TrialRecord' | 'ParticipantProfile' | 'UploadEnvelope' | 'ExportBundle' | 'CalibrationResult' | 'PlannedBlock';

/** The schema errors of `value` as `def` (empty when valid). */
export function validate(def: SchemaDef, value: unknown): string[] {
  const fn = ajv.getSchema(`${recordsSchema.$id}#/$defs/${def}`);
  if (!fn) throw new Error(`schema def ${def} not found`);
  if (fn(value)) return [];
  return (fn.errors ?? []).map((e) => `${e.instancePath || '(root)'} ${e.message} ${JSON.stringify(e.params)}`);
}
