/**
 * Validates any $defs entry of schema/records.schema.json (the shared helper only exposes the
 * record-level defs; the sensors module produces DeviceInfo, SensorCapabilities, MeasuredRates,
 * PoseStats and AxisCheck). The value is JSON round-tripped first, exactly like an upload: a NaN
 * becomes null and an undefined field disappears, so both fail here instead of in the backend.
 */
import { Ajv2020 } from 'ajv/dist/2020.js';
import { expect } from 'vitest';
import { recordsSchema } from './schema';

const ajv = new Ajv2020({ allErrors: true, strict: false });
ajv.addSchema(recordsSchema);

export function schemaErrors(def: string, value: unknown): string[] {
  const fn = ajv.getSchema(`${recordsSchema.$id}#/$defs/${def}`);
  if (!fn) throw new Error(`schema def ${def} not found`);
  const json: unknown = JSON.parse(JSON.stringify(value));
  if (fn(json)) return [];
  return (fn.errors ?? []).map((e) => `${e.instancePath || '(root)'} ${e.message} ${JSON.stringify(e.params)}`);
}

export function expectValidDef(def: string, value: unknown): void {
  const errors = schemaErrors(def, value);
  expect(errors, `${def} schema errors:\n${errors.join('\n')}`).toEqual([]);
}
