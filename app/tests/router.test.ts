/**
 * The router (app/store.ts) for the v2 participant path (spec §1): welcome ← profile ← session; the
 * repair screen (sensors) and the researchers' setup go back to the profile; done goes home. The
 * consent and calibration screens no longer exist. The system back button walks the same targets as
 * the screens' ← arrows and leaves the app only from the root.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { initialState, type ScreenId } from '../src/app/state';
import { backTargetOf, getState, installHistory, navigate, setBackHandler, setState } from '../src/app/store';

/** Every screen id, checked exhaustively at compile time (a new screen must be added here). */
const SCREENS: Record<ScreenId, true> = {
  loading: true,
  welcome: true,
  profile: true,
  sensors: true,
  setup: true,
  session: true,
  done: true,
  data: true,
  settings: true,
  sandbox: true,
  about: true,
};
const ALL = Object.keys(SCREENS) as ScreenId[];

describe('backTargetOf (the ← arrow and the system back button)', () => {
  it('participant path: welcome ← profile ← session; sensors and setup ← profile; done → welcome', () => {
    const expected: Partial<Record<ScreenId, ScreenId | null>> = {
      loading: null,
      welcome: null,
      profile: 'welcome',
      sensors: 'profile',
      setup: 'profile',
      session: 'profile',
      done: 'welcome',
    };
    for (const [screen, target] of Object.entries(expected)) {
      expect(backTargetOf({ screen: screen as ScreenId, returnTo: 'welcome' }), screen).toBe(target);
    }
  });

  it('secondary screens go back where they were opened from (welcome when that is themselves)', () => {
    for (const screen of ['data', 'settings', 'sandbox', 'about'] as ScreenId[]) {
      expect(backTargetOf({ screen, returnTo: 'welcome' })).toBe('welcome');
      expect(backTargetOf({ screen, returnTo: 'profile' })).toBe('profile');
      expect(backTargetOf({ screen, returnTo: screen })).toBe('welcome');
    }
  });

  it('every screen has a target except the root and the loading screen, and no target is a removed screen', () => {
    for (const screen of ALL) {
      const target = backTargetOf({ screen, returnTo: 'welcome' });
      if (screen === 'welcome' || screen === 'loading') expect(target).toBeNull();
      else expect(ALL).toContain(target);
    }
    expect(ALL).not.toContain('consent');
    expect(ALL).not.toContain('calibration');
  });
});

/** A browser history with popstate delivered asynchronously, like a real one. */
class FakeHistory {
  entries: unknown[] = [null];
  index = 0;
  left = false;
  constructor(private readonly target: EventTarget) {}
  get state(): unknown {
    return this.entries[this.index];
  }
  pushState(s: unknown): void {
    this.entries = [...this.entries.slice(0, this.index + 1), structuredClone(s)];
    this.index++;
  }
  replaceState(s: unknown): void {
    this.entries[this.index] = structuredClone(s);
  }
  back(): void {
    const i = this.index - 1;
    if (i < 0) {
      this.left = true;
      return;
    }
    this.index = i;
    const ev = new Event('popstate');
    Object.defineProperty(ev, 'state', { value: this.state });
    setTimeout(() => this.target.dispatchEvent(ev), 0);
  }
}

describe('system back button on the v2 path', () => {
  const settle = () => new Promise((r) => setTimeout(r, 5));
  const g = globalThis as { history?: unknown };
  let fake: FakeHistory;
  let uninstall: () => void;

  beforeEach(() => {
    const target = new EventTarget();
    fake = new FakeHistory(target);
    g.history = fake;
    setState({ ...initialState(), screen: 'loading', returnTo: 'welcome' });
    uninstall = installHistory(target);
    navigate('welcome'); // what boot does
  });

  afterEach(() => {
    uninstall();
    delete g.history;
  });

  const press = async () => {
    fake.back();
    await settle();
  };

  it('profile → repair screen: back walks sensors → profile → welcome, then leaves', async () => {
    navigate('profile');
    navigate('sensors');
    expect(fake.entries).toHaveLength(2); // one guard entry, whatever the depth
    for (const expected of ['profile', 'welcome'] as ScreenId[]) {
      await press();
      expect(getState().screen).toBe(expected);
      expect(fake.left).toBe(false);
    }
    await press();
    expect(fake.left).toBe(true);
  });

  it('the researchers setup screen goes back to the profile', async () => {
    navigate('profile');
    navigate('setup');
    await press();
    expect(getState().screen).toBe('profile');
  });

  it('the session screen takes the press (pause dialog); without a running session back goes to the profile', async () => {
    const handler = vi.fn(() => true);
    const remove = setBackHandler(handler);
    navigate('profile');
    navigate('session');
    await press();
    expect(handler).toHaveBeenCalledOnce();
    expect(getState().screen).toBe('session');
    remove();
    const declines = setBackHandler(() => false);
    await press();
    expect(getState().screen).toBe('profile');
    declines();
  });

  it('back from done goes to the welcome screen and closes the finished session', async () => {
    setState({ session: { session_id: 's1' } as never });
    navigate('done');
    await press();
    expect(getState().screen).toBe('welcome');
    expect(getState().session).toBeNull();
  });
});
