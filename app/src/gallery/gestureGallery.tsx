/**
 * Dev-only gallery of GestureAnimation: every useful combination with its id, for eyeballing and for
 * screenshots (scripts/render_gesture_images.ts). Served by the dev server at /gesture-gallery.html;
 * it is not a production build input.
 *
 * Query parameters
 *   ?only=<id>          show one item. id = hand.orientation.speed.contact.variant.view[.compact]
 *                       e.g. right.upright_toward.natural.straight.standard.both (hand: right|left|any,
 *                       contact: straight|angled|any); any combination is accepted, listed or not
 *   &all=1              every hand × grip (upright, top left, top right) × speed × contact × variant
 *   &size=<px>          width of each item (default 440, compact items 300)
 *   &t=<seconds>        start every animation at time t (negative animation-delay)
 *   &paused=1           freeze every animation (animation-play-state: paused), e.g. at &t
 *   &theme=dark|light   force the colour scheme (the prefers-color-scheme rules are rewritten)
 *   &dir=rtl            right-to-left page (as in Hebrew): the drawings must NOT mirror
 *   &motion=reduce      force prefers-reduced-motion (the three still frames)
 *   &still=1            render every item with playing={false}
 *   &bare=1             no titles or help text (screenshots)
 * Automation hooks: window.gxSetTime(t) freezes every animation at t seconds without a reload;
 * window.gxInfo() lists the rendered gestures (key, loop length, number of takes, halfway and touch
 * instants of the first take, touch instant of every take); <body data-gx-ready="1"> once rendered.
 */
import { render } from 'preact';
import { useEffect } from 'preact/hooks';
import { ALL_ORIENTATIONS } from '../core/constants';
import type { OrientationId } from '../core/types';
import {
  GestureAnimation,
  type GestureContact,
  type GestureHand,
  type GestureSpeed,
  type GestureVariant,
  type GestureView,
} from '../ui/GestureAnimation';
import '../ui/styles.css';
import '../ui/session.css';

interface Item {
  hand: GestureHand;
  orientation: OrientationId;
  speed: GestureSpeed;
  contact: GestureContact;
  variant: GestureVariant;
  view: GestureView;
  compact: boolean;
}

const HANDS: GestureHand[] = ['right', 'left', null];
const GRIPS: OrientationId[] = ['upright_toward', 'top_left_toward', 'top_right_toward'];
const SPEEDS: GestureSpeed[] = ['fast', 'natural', 'slow'];
const CONTACTS: GestureContact[] = ['straight', 'angled'];
/** 'any' alternates a straight and an angled loop (accepted in ids, shown in its own sections) */
const ALL_CONTACTS: GestureContact[] = [...CONTACTS, 'any'];
const VARIANTS: GestureVariant[] = ['standard', 'curved', 'pause'];
const VIEWS: GestureView[] = ['side', 'pov', 'both'];

const base: Item = {
  hand: 'right',
  orientation: 'upright_toward',
  speed: 'natural',
  contact: 'straight',
  variant: 'standard',
  view: 'both',
  compact: false,
};
const item = (patch: Partial<Item>): Item => ({ ...base, ...patch });

export function itemId(it: Item): string {
  return [it.hand ?? 'any', it.orientation, it.speed, it.contact, it.variant, it.view, it.compact ? 'compact' : '']
    .filter(Boolean)
    .join('.');
}

export function parseItemId(id: string): Item | null {
  const p = id.split('.');
  if (p.length < 6 || p.length > 7) return null;
  const [hand, orientation, speed, contact, variant, view, compact] = p;
  const h: GestureHand | undefined = hand === 'right' || hand === 'left' ? hand : hand === 'any' ? null : undefined;
  if (h === undefined) return null;
  if (!(ALL_ORIENTATIONS as string[]).includes(orientation)) return null;
  if (!(SPEEDS as string[]).includes(speed) || !(ALL_CONTACTS as string[]).includes(contact)) return null;
  if (!(VARIANTS as string[]).includes(variant) || !(VIEWS as string[]).includes(view)) return null;
  if (compact !== undefined && compact !== 'compact') return null;
  return {
    hand: h,
    orientation: orientation as OrientationId,
    speed: speed as GestureSpeed,
    contact: contact as GestureContact,
    variant: variant as GestureVariant,
    view: view as GestureView,
    compact: compact === 'compact',
  };
}

const SECTIONS: { title: string; note?: string; items: Item[] }[] = [
  {
    title: 'Hands × grips',
    note: 'natural speed, straight contact. Right hand: the person faces right and the forearm enters bottom-right; left hand mirrored; no hand = participant\'s choice ("?").',
    items: HANDS.flatMap((hand) => GRIPS.map((orientation) => item({ hand, orientation }))),
  },
  { title: 'Speeds', note: 'loops of about 1.6 s, 2.8 s and 5 s', items: SPEEDS.map((speed) => item({ speed })) },
  {
    title: 'Contact',
    note: 'straight = arrives upright; angled = arrives tilted ~32°, the upper edge/corner touches; any = one loop of each',
    items: (['right', 'left'] as GestureHand[]).flatMap((hand) => ALL_CONTACTS.map((contact) => item({ hand, contact }))),
  },
  {
    title: "Participant's choice (no hand)",
    note: 'the loops alternate the right and the left hand, in a neutral colour, with a "?" on each panel',
    items: [
      item({ hand: null }),
      item({ hand: null, contact: 'any' }),
      item({ hand: null, contact: 'any', view: 'side', compact: true }),
      item({ hand: null, contact: 'angled', view: 'side', compact: true }),
    ],
  },
  {
    title: 'Variants',
    note: 'curved = swings out to the side first (visible in the "what you see" view); pause = stops halfway',
    items: (['right', 'left'] as GestureHand[]).flatMap((hand) => VARIANTS.map((variant) => item({ hand, variant }))),
  },
  {
    title: 'Views and compact',
    items: [
      item({ view: 'side' }),
      item({ view: 'pov' }),
      item({ view: 'both', compact: true }),
      item({ hand: 'left', orientation: 'top_left_toward', speed: 'fast', view: 'pov', compact: true }),
      item({ hand: null, view: 'both', compact: true }),
    ],
  },
  {
    title: 'Other grips',
    items: ALL_ORIENTATIONS.filter((o) => !GRIPS.includes(o)).map((orientation) => item({ orientation })),
  },
];

function allItems(): Item[] {
  const out: Item[] = [];
  for (const hand of HANDS)
    for (const orientation of GRIPS)
      for (const speed of SPEEDS)
        for (const contact of CONTACTS) for (const variant of VARIANTS) out.push(item({ hand, orientation, speed, contact, variant }));
  return out;
}

// ───────────────────────────── page environment ─────────────────────────────

const params = new URLSearchParams(location.search);
const num = (v: string | null) => (v !== null && v.trim() !== '' && Number.isFinite(Number(v)) ? Number(v) : null);

/** Freeze (or offset) every gesture animation: negative animation-delay + paused play state. */
function setFreeze(t: number | null, paused: boolean): void {
  let el = document.getElementById('gx-freeze') as HTMLStyleElement | null;
  if (!el) {
    el = document.createElement('style');
    el.id = 'gx-freeze';
    document.head.appendChild(el);
  }
  const decl: string[] = [];
  if (t !== null) decl.push(`animation-delay:${-t}s!important`);
  if (paused) decl.push('animation-play-state:paused!important');
  el.textContent = decl.length ? `.gx-anim{${decl.join(';')}}` : '';
}

/**
 * Forces a media feature by rewriting the @media rules that test it (same-origin sheets: in dev every
 * imported stylesheet is a <style>): matching rules → "all", the others → "not all".
 */
const originalMedia = new WeakMap<CSSMediaRule, string>();
function forceMedia(feature: string, value: string): void {
  const visit = (rules: CSSRuleList) => {
    for (const r of Array.from(rules)) {
      if (r instanceof CSSMediaRule) {
        const text = originalMedia.get(r) ?? r.media.mediaText;
        const m = new RegExp(`^\\(?\\s*${feature}\\s*:\\s*([a-z-]+)\\s*\\)?$`).exec(text.trim());
        if (m) {
          originalMedia.set(r, text);
          r.media.mediaText = m[1] === value ? 'all' : 'not all';
        }
      }
      if ('cssRules' in r) visit((r as CSSGroupingRule).cssRules);
    }
  };
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      visit(sheet.cssRules);
    } catch {
      /* cross-origin sheet */
    }
  }
}

function applyEnvironment(): void {
  const theme = params.get('theme');
  if (theme === 'dark' || theme === 'light') {
    forceMedia('prefers-color-scheme', theme);
    document.documentElement.style.colorScheme = theme;
  }
  if (params.get('motion') === 'reduce') forceMedia('prefers-reduced-motion', 'reduce');
}

interface GxInfo {
  key: string;
  cycle: number;
  takes: number;
  mid: number;
  touch: number;
  touches: number[];
}

declare global {
  interface Window {
    gxSetTime: (t: number) => void;
    gxInfo: () => GxInfo[];
  }
}

// ───────────────────────────── page ─────────────────────────────

function Card(props: { it: Item; size: number | null; still: boolean; bare: boolean }) {
  const { it } = props;
  const width = props.size ?? (it.compact ? 300 : 440);
  return (
    <figure class="gal-card" data-item={itemId(it)} style={{ width: `${width}px` }}>
      <div class="gal-shot">
        <GestureAnimation
          hand={it.hand}
          orientation={it.orientation}
          speed={it.speed}
          contact={it.contact}
          variant={it.variant}
          view={it.view}
          compact={it.compact}
          playing={!props.still}
        />
      </div>
      {!props.bare && <figcaption class="gal-id">{itemId(it)}</figcaption>}
    </figure>
  );
}

function Gallery() {
  const only = params.get('only');
  const size = num(params.get('size'));
  const still = params.get('still') === '1';
  const bare = params.get('bare') === '1';
  useEffect(() => {
    document.body.dataset.gxReady = '1';
  }, []);

  if (only) {
    const it = parseItemId(only);
    if (!it) return <p class="gal-error">Unknown item id: {only}</p>;
    return (
      <main class="gal gal-only">
        <Card it={it} size={size} still={still} bare={bare} />
      </main>
    );
  }
  const sections = params.get('all') === '1' ? [{ title: 'All combinations', items: allItems() }] : SECTIONS;
  return (
    <main class="gal">
      {!bare && (
        <header class="gal-head">
          <h1>Gesture gallery</h1>
          <p class="muted small">
            Dev only. Params: only, all, size, t, paused, theme=dark|light, dir=rtl, motion=reduce, still, bare (see
            src/gallery/gestureGallery.tsx).
          </p>
        </header>
      )}
      {sections.map((sec) => (
        <section key={sec.title} class="gal-section">
          {!bare && <h2>{sec.title}</h2>}
          {!bare && 'note' in sec && sec.note && <p class="muted small">{sec.note}</p>}
          <div class="gal-grid">
            {sec.items.map((it) => (
              <Card key={itemId(it)} it={it} size={size} still={still} bare={bare} />
            ))}
          </div>
        </section>
      ))}
    </main>
  );
}

const GALLERY_CSS = `
.gal { padding: 16px; display: flex; flex-direction: column; gap: 20px; }
.gal-only { padding: 0; display: block; }
.gal-only .gx { max-width: none; } /* one item: it takes exactly &size= */
.gal-head h1 { margin: 0 0 4px; }
.gal-section h2 { margin: 0 0 4px; font-size: 1.05rem; }
.gal-grid { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.gal-card { margin: 0; display: flex; flex-direction: column; gap: 6px; }
.gal-shot { background: var(--bg); padding: 10px; }
.gal-id { font: 12px var(--mono); color: var(--muted); overflow-wrap: anywhere; }
.gal-error { padding: 16px; color: var(--danger); }
`;

function boot(): void {
  if (params.get('dir') === 'rtl') {
    document.documentElement.dir = 'rtl';
    document.documentElement.lang = 'he';
  }
  const style = document.createElement('style');
  style.textContent = GALLERY_CSS;
  document.head.appendChild(style);
  const t = num(params.get('t'));
  const paused = params.get('paused') === '1';
  setFreeze(t, paused);
  window.gxSetTime = (time: number) => setFreeze(time, true);
  window.gxInfo = () =>
    Array.from(document.querySelectorAll<HTMLElement>('[data-gesture]')).map((el) => ({
      key: el.dataset.gesture ?? '',
      cycle: Number(el.dataset.cycleS),
      takes: Number(el.dataset.takes ?? 1),
      mid: Number(el.dataset.tMid),
      touch: Number(el.dataset.tTouch),
      touches: (el.dataset.tTouches ?? el.dataset.tTouch ?? '')
        .split(',')
        .filter((s) => s !== '')
        .map(Number),
    }));
  render(<Gallery />, document.getElementById('gallery')!);
  applyEnvironment();
  // the dev server re-injects stylesheets on edits: apply the forced media again
  new MutationObserver(() => applyEnvironment()).observe(document.head, { childList: true, subtree: true, characterData: true });
}

boot();
