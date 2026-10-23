import type { Language } from '../core/types';
import { useAppState } from '../app/store';
import { en } from './en';
import { he } from './he';

export type Strings = typeof en;

const TABLE: Record<Language, Strings> = { en, he };

export function stringsFor(lang: Language): Strings {
  return TABLE[lang] ?? en;
}

/** Strings for the current UI language; re-renders when the language changes. */
export function useStrings(): Strings {
  return stringsFor(useAppState().settings.language);
}

/** Sets <html lang/dir>; Hebrew is right-to-left. */
export function applyDocumentLanguage(lang: Language): void {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'he' ? 'rtl' : 'ltr';
}

export const LANGUAGES: { value: Language; label: string }[] = [
  { value: 'en', label: 'English' },
  { value: 'he', label: 'עברית' },
];
