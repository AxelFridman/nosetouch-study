import { afterEach, describe, expect, it, vi } from 'vitest';
import { deliverExport, exportFileName, REVOKE_DELAY_MS } from '../src/core/exporter';
import type { ExportBundle } from '../src/core/types';
import { makeBundle, makeProfile, makeSession } from './helpers/dataFixtures';
import { expectValid } from './helpers/schema';

interface FakeAnchor {
  href: string;
  download: string;
  rel: string;
  style: Record<string, string>;
  parentNode: { removeChild: (el: FakeAnchor) => void } | null;
  click: ReturnType<typeof vi.fn>;
}

/** Minimal DOM for the <a download> fallback. */
function stubDom() {
  const anchors: FakeAnchor[] = [];
  const body = {
    appendChild: vi.fn((el: FakeAnchor) => {
      el.parentNode = body;
      return el;
    }),
    removeChild: vi.fn((el: FakeAnchor) => {
      el.parentNode = null;
      return el;
    }),
  };
  vi.stubGlobal('document', {
    body,
    createElement: vi.fn((tag: string) => {
      expect(tag).toBe('a');
      const a: FakeAnchor = { href: '', download: '', rel: '', style: {}, parentNode: null, click: vi.fn() };
      anchors.push(a);
      return a;
    }),
  });
  const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:https://app.example/1234');
  const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
  return { anchors, body, createObjectURL, revokeObjectURL };
}

/** navigator with Web Share; canShare decides per MIME type. */
function stubShare(opts: { accept: (f: File) => boolean; share?: (d: ShareData) => Promise<void> }) {
  const canShare = vi.fn((d: ShareData) => (d.files ?? []).every(opts.accept));
  const share = vi.fn(opts.share ?? (async () => undefined));
  vi.stubGlobal('navigator', { canShare, share });
  return { canShare, share };
}

const sharedFile = (share: ReturnType<typeof vi.fn>, call = 0): File =>
  ((share.mock.calls[call][0] as ShareData).files as File[])[0];

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('exportFileName', () => {
  it('uses the single participant id and the local date/time', () => {
    const when = new Date(2026, 9, 23, 13, 5); // local time
    expect(exportFileName(makeBundle(), when)).toBe('nosetouch_P-ABCDEF_20261023-1305.json');
  });

  it("uses 'all' for several participants or an empty bundle", () => {
    const when = new Date(2026, 0, 2, 3, 4);
    const two = makeBundle({
      participants: [makeProfile(), makeProfile({ participant_id: 'P-ZZZZZZ' })],
    });
    expect(exportFileName(two, when)).toBe('nosetouch_all_20260102-0304.json');
    const mixed = makeBundle({ sessions: [makeSession({ participant_id: 'P-ZZZZZZ' })] });
    expect(exportFileName(mixed, when)).toBe('nosetouch_all_20260102-0304.json');
    const empty = makeBundle({ participants: [], sessions: [], trials: [] });
    expect(exportFileName(empty, when)).toBe('nosetouch_all_20260102-0304.json');
  });

  it('falls back to now for an invalid date', () => {
    expect(exportFileName(makeBundle(), new Date(Number.NaN))).toMatch(/^nosetouch_P-ABCDEF_\d{8}-\d{4}\.json$/);
  });
});

describe('deliverExport', () => {
  it('shares the bundle as a .json file when the platform accepts JSON (iOS)', async () => {
    const bundle = makeBundle();
    expectValid('ExportBundle', bundle);
    const { share } = stubShare({ accept: () => true });
    const dom = stubDom();

    expect(await deliverExport(bundle)).toBe('shared');
    expect(share).toHaveBeenCalledTimes(1);
    const file = sharedFile(share);
    expect(file.name).toMatch(/^nosetouch_P-ABCDEF_\d{8}-\d{4}\.json$/);
    expect(file.type).toBe('application/json');
    expect(JSON.parse(await file.text()) as ExportBundle).toEqual(bundle);
    expect((share.mock.calls[0][0] as ShareData).title).toBeTruthy();
    expect(dom.createObjectURL).not.toHaveBeenCalled();
  });

  it('retries as text/plain …json.txt when canShare rejects JSON (Chrome Android allowlist)', async () => {
    const bundle = makeBundle();
    const { canShare, share } = stubShare({ accept: (f) => f.type === 'text/plain' });
    stubDom();

    expect(await deliverExport(bundle)).toBe('shared');
    expect(canShare).toHaveBeenCalledTimes(2);
    expect(share).toHaveBeenCalledTimes(1);
    const file = sharedFile(share);
    expect(file.name).toMatch(/\.json\.txt$/);
    expect(file.type).toBe('text/plain');
    expect(JSON.parse(await file.text())).toEqual(bundle); // the SAME JSON content
  });

  it("returns 'cancelled' when the participant dismisses the share sheet", async () => {
    stubShare({
      accept: () => true,
      share: async () => {
        throw new DOMException('Share canceled', 'AbortError');
      },
    });
    const dom = stubDom();
    expect(await deliverExport(makeBundle())).toBe('cancelled');
    expect(dom.anchors).toHaveLength(0);
  });

  it('falls back to a download when share fails for another reason (e.g. expired activation)', async () => {
    const { share } = stubShare({
      accept: () => true,
      share: async () => {
        throw new DOMException('Must be handling a user gesture', 'NotAllowedError');
      },
    });
    const dom = stubDom();
    expect(await deliverExport(makeBundle())).toBe('downloaded');
    expect(share).toHaveBeenCalledTimes(1); // no second share attempt without activation
    expect(dom.anchors).toHaveLength(1);
  });

  it('downloads through an <a download> with an object URL that is revoked later', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('navigator', {}); // no Web Share at all (desktop Firefox)
    const dom = stubDom();
    const bundle = makeBundle();

    expect(await deliverExport(bundle)).toBe('downloaded');
    expect(dom.createObjectURL).toHaveBeenCalledTimes(1);
    const blob = dom.createObjectURL.mock.calls[0][0] as Blob;
    expect(blob.type).toBe('application/json');
    expect(JSON.parse(await blob.text())).toEqual(bundle);
    const [a] = dom.anchors;
    expect(a.href).toBe('blob:https://app.example/1234');
    expect(a.download).toMatch(/^nosetouch_P-ABCDEF_\d{8}-\d{4}\.json$/);
    expect(a.click).toHaveBeenCalledTimes(1);
    expect(dom.body.appendChild).toHaveBeenCalledWith(a);
    expect(dom.body.removeChild).toHaveBeenCalledWith(a);

    expect(dom.revokeObjectURL).not.toHaveBeenCalled(); // iOS still needs it while its sheet is open
    vi.advanceTimersByTime(REVOKE_DELAY_MS);
    expect(dom.revokeObjectURL).toHaveBeenCalledWith('blob:https://app.example/1234');
  });

  it('treats a throwing canShare as "cannot share"', async () => {
    vi.stubGlobal('navigator', {
      canShare: () => {
        throw new TypeError('not supported');
      },
      share: vi.fn(),
    });
    const dom = stubDom();
    expect(await deliverExport(makeBundle())).toBe('downloaded');
    expect(dom.anchors).toHaveLength(1);
  });

  it('throws only when neither sharing nor downloading is possible', async () => {
    vi.stubGlobal('navigator', {});
    vi.stubGlobal('document', undefined);
    await expect(deliverExport(makeBundle())).rejects.toThrow(/neither share nor download/);
  });
});
