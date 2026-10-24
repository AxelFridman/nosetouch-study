import { useEffect, useState } from 'preact/hooks';
import { initialState, persistSettings, type AppState, type LocalSettings, type ScreenId } from './state';

type Listener = (s: AppState) => void;

let state: AppState = initialState();
const listeners = new Set<Listener>();

export function getState(): AppState {
  return state;
}

export function setState(patch: Partial<AppState>): void {
  state = { ...state, ...patch };
  for (const l of listeners) l(state);
}

export function subscribe(l: Listener): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

/** Navigate to a screen. Secondary screens remember where to return. Keeps the history guard in step. */
export function navigate(screen: ScreenId, opts: { returnTo?: ScreenId } = {}): void {
  setState({ screen, returnTo: opts.returnTo ?? state.returnTo, toast: null });
  syncHistory();
  if (typeof window !== 'undefined') window.scrollTo?.(0, 0);
}

export function goBack(): void {
  navigate(state.returnTo);
}

export function updateSettings(patch: Partial<LocalSettings>): void {
  const settings = { ...state.settings, ...patch };
  persistSettings(settings);
  setState({ settings });
}

let toastTimer: ReturnType<typeof setTimeout> | undefined;
export function showToast(text: string, kind: 'info' | 'ok' | 'warn' | 'error' = 'info', ms = 3500): void {
  setState({ toast: { text, kind } });
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => setState({ toast: null }), ms);
}

/** Re-renders the component whenever the app state changes. */
export function useAppState(): AppState {
  const [s, set] = useState(state);
  useEffect(() => {
    const unsubscribe = subscribe(set);
    set(state); // catch updates that happened between first render and subscription
    return unsubscribe;
  }, []);
  return s;
}

// ───────────────────────────── system back button ─────────────────────────────
//
// The app is a single page. Without history entries the Android back button or gesture, or an iOS
// edge swipe, would leave it from any screen, also in the middle of a trial. So while the app is
// past its root screen (welcome) it keeps ONE guard entry on top of the page's own entry:
//  - navigate() pushes the guard on a move away from the root (a forward move), replaces it on the
//    moves after that and drops it (history.back()) on the way back to the root, where back leaves;
//  - a back press pops the guard ('popstate'). The screen's back handler runs first (the session
//    screen stops a running trial and opens its Pause dialog, then the guard is pushed again, so the
//    next press is caught too); without one the app goes where the screen's ← arrow goes
//    (backTargetOf), with a fresh guard unless that is the root.
// History is never deeper than two entries, whatever path the participant took.

type HistoryLike = Pick<History, 'pushState' | 'replaceState' | 'back' | 'state'>;
type PopTarget = Pick<EventTarget, 'addEventListener' | 'removeEventListener'>;

const GUARD = 'nosetouch-guard';
/** The popstate of our own history.back() arrives within milliseconds; after this it is not waited for. */
const OWN_POP_MS = 1000;

let historyOn = false;
/** the current history entry is our guard */
let guarded = false;
/** popstate events still to come from our own history.back(), expected until ownPopsUntil */
let ownPops = 0;
let ownPopsUntil = 0;
let backHandler: (() => boolean) | null = null;

function browserHistory(): HistoryLike | null {
  if (!historyOn) return null;
  try {
    return typeof history === 'undefined' ? null : history;
  } catch {
    return null;
  }
}

function isGuard(entryState: unknown): boolean {
  return typeof entryState === 'object' && entryState !== null && (entryState as { nosetouch?: unknown }).nosetouch === GUARD;
}

/** Every screen but the root keeps a guard entry (the loading screen never has one). */
function needsGuard(screen: ScreenId): boolean {
  return screen !== 'welcome' && screen !== 'loading';
}

function pushGuard(h: HistoryLike): void {
  try {
    h.pushState({ nosetouch: GUARD, screen: state.screen }, '');
    guarded = true;
  } catch (e) {
    console.warn('history.pushState failed', e); // e.g. Safari's rate limit: back may then leave the app
  }
}

function dropGuard(h: HistoryLike): void {
  ownPops++;
  ownPopsUntil = Date.now() + OWN_POP_MS;
  guarded = false;
  try {
    h.back();
  } catch (e) {
    ownPops--;
    console.warn('history.back failed', e);
  }
}

function syncHistory(): void {
  const h = browserHistory();
  if (!h) return;
  if (!needsGuard(state.screen)) {
    if (guarded) dropGuard(h);
  } else if (!guarded) {
    pushGuard(h);
  } else {
    try {
      h.replaceState({ nosetouch: GUARD, screen: state.screen }, '');
    } catch (e) {
      console.warn('history.replaceState failed', e);
    }
  }
}

/**
 * Where the system back button goes from a screen: the same place as the screen's ← arrow (done:
 * the Home button). null at the root, where back leaves the app. The participant path (spec v2 §1):
 * welcome ← profile ← session; the repair screen (sensors) and the researchers' setup go back to the
 * profile. The session screen normally takes the press itself (setBackHandler: its pause dialog);
 * 'profile' is its fallback when no session is running.
 */
export function backTargetOf(st: Pick<AppState, 'screen' | 'returnTo'>): ScreenId | null {
  switch (st.screen) {
    case 'profile':
    case 'done':
      return 'welcome';
    case 'sensors':
    case 'setup':
    case 'session':
      return 'profile';
    case 'data':
    case 'settings':
    case 'sandbox':
    case 'about':
      return st.returnTo === st.screen ? 'welcome' : st.returnTo;
    default:
      return null;
  }
}

/**
 * Lets the current screen take the system back button. The handler returns true when it handled
 * the press: the app then stays and keeps a guard entry, so the next press is caught as well.
 * Returns a function that removes the handler again (only if it is still the registered one).
 */
export function setBackHandler(handler: () => boolean): () => void {
  backHandler = handler;
  return () => {
    if (backHandler === handler) backHandler = null;
  };
}

/** The 'popstate' handler (exported for tests): the guard entry was popped by back or forward. */
export function handlePopState(entryState: unknown): void {
  const h = browserHistory();
  if (!h) return;
  if (ownPops > 0 && Date.now() > ownPopsUntil) ownPops = 0; // it never came: this one is the participant's
  if (ownPops > 0) {
    ownPops--;
    guarded = isGuard(entryState);
    return;
  }
  if (isGuard(entryState)) {
    // forward onto an old guard entry: forward means nothing here, only re-sync
    guarded = true;
    if (!needsGuard(state.screen)) dropGuard(h);
    return;
  }
  guarded = false;
  let handled = false;
  try {
    handled = backHandler?.() ?? false;
  } catch (e) {
    console.warn('back handler failed', e);
  }
  if (handled) {
    pushGuard(h);
    return;
  }
  const target = backTargetOf(state);
  if (!target) return;
  if (state.screen === 'done') setState({ session: null }); // what the done screen's Home button does
  navigate(target);
}

/** Connects the system back button to the app (App calls it once at start). Returns the uninstaller. */
export function installHistory(target: PopTarget): () => void {
  historyOn = true;
  guarded = false;
  ownPops = 0;
  const h = browserHistory();
  // A reload on a guarded screen restores our guard as the page's own entry: make it a plain one.
  if (h && isGuard(h.state)) {
    try {
      h.replaceState(null, '');
    } catch {
      // harmless: the entry below then reloads the app instead of leaving it
    }
  }
  const onPop = (e: Event) => handlePopState((e as PopStateEvent).state);
  target.addEventListener('popstate', onPop);
  syncHistory();
  return () => {
    target.removeEventListener('popstate', onPop);
    historyOn = false;
    guarded = false;
    ownPops = 0;
  };
}
