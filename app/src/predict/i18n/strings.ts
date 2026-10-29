/**
 * Strings of predict mode (docs/stage3/APP_SPEC.md section 5.3). Minimal on purpose: one short line per
 * screen. English is the reference; Hebrew (&lang=he) has the same keys (a test checks it). LEFT and
 * RIGHT name physical sides, so the hand bar stays left-to-right in Hebrew too.
 */
import type { Hand, ModelId, PredictLang } from '../types';

export const EN = {
  brand: 'NoseTouch',
  title: 'Which hand?',
  instruct: 'Hold the phone out, screen towards you. Bring it to your nose, then back.',
  start: 'Start',
  permission: 'Allow motion access',
  denied_ios: 'Motion access was refused. Close Safari completely, open the link again and tap Allow.',
  denied_other: "Motion access was refused. Allow it in the browser's site settings and reload.",
  watch: 'Hold still. Then bring the phone to your nose.',
  watch_sim: 'A simulated person will touch their nose. Watch.',
  not_counted: 'Not counted – try again',
  again_hint: 'Touch again.',
  p_meaning: 'probability that the RIGHT hand was used',
  left: 'LEFT',
  right: 'RIGHT',
  verdict: '{HAND} hand · {TOUCHES}',
  leaning: 'Leaning {HAND} · not confident · {TOUCHES}',
  provisional: 'after {TOUCHES} · not final',
  touch_one: '1 touch',
  touch_many: '{k} touches',
  abstain: 'No touch found – bring the phone to your nose, then back',
  grip: 'Turn the screen towards your face',
  again: 'Again',
  start_over: 'Start over',
  compare: 'Compare models',
  compare_after: 'after {TOUCHES}',
  confident: 'confident',
  unavailable: 'n/a',
  model: 'Model',
  close: 'Close',
  demo: 'Demo',
  demo_badge: 'Demo: simulated {HAND} hand',
  sim_cnn: 'Simulated motion: the CNN only reads real motion',
  sim_na: 'n/a (sim)',
  sim_cnn_line: 'not valid on simulated motion',
  u_desktop: 'This page needs a phone with motion sensors.',
  u_inapp: 'Open this link in Safari or Chrome.',
  u_https: 'This page needs a secure (https) link.',
  u_gyro: 'This phone reports no gyroscope.',
  u_acc: 'This phone reports no accelerometer.',
  u_events: 'No sensor data. Reload the page.',
  u_events_android: "No sensor data. Allow motion sensors in Chrome's site settings, then reload.",
  u_error: 'Something went wrong – reload',
  try_demo: 'Try the demo',
  copy_link: 'Copy link',
  copied: 'Link copied',
  reload: 'Reload',
  restarted: 'Restarted: the page was hidden.',
  info_title: 'About',
  info1: "P(right) is the model's calibrated probability that the right hand was used.",
  info2: 'Hold the phone with the screen towards you.',
  info3: 'Nothing leaves this phone.',
  m_cnn: 'Final CNN',
  m_cnn_line: '1D CNN, 4 members',
  m_m4: 'Detector model',
  m_m4_line: 'lane m4, yaw logistic',
  m_phys6: 'Physics logistic',
  m_phys6_line: 'PHYS6: yaw, roll, lateral',
  m_yaw: 'Yaw reference',
  m_yaw_line: 'logistic on yaw, 8 instants',
} as const;

export type StringKey = keyof typeof EN;

export const HE: Record<StringKey, string> = {
  brand: 'NoseTouch',
  title: 'באיזו יד?',
  instruct: 'החזיקו את הטלפון ביד מושטת, המסך מולכם. קרבו אותו לאף וחזרו.',
  start: 'התחלה',
  permission: 'אשרו גישה לחיישני התנועה',
  denied_ios: 'הגישה לחיישני התנועה נדחתה. סגרו את Safari לגמרי, פתחו שוב את הקישור ואשרו.',
  denied_other: 'הגישה לחיישני התנועה נדחתה. אשרו אותה בהגדרות האתר בדפדפן וטענו מחדש.',
  watch: 'החזיקו בלי לזוז. ואז קרבו את הטלפון לאף.',
  watch_sim: 'אדם מדומה יקרב את הטלפון לאף. צפו.',
  not_counted: 'לא נספר – נסו שוב',
  again_hint: 'געו שוב באף.',
  p_meaning: 'ההסתברות שהשתמשו ביד ימין',
  left: 'שמאל',
  right: 'ימין',
  verdict: 'יד {HAND} · {TOUCHES}',
  leaning: 'נוטה ליד {HAND} · לא בטוח · {TOUCHES}',
  provisional: 'אחרי {TOUCHES} · לא סופי',
  touch_one: 'נגיעה אחת',
  touch_many: '{k} נגיעות',
  abstain: 'לא נמצאה נגיעה – קרבו את הטלפון לאף וחזרו',
  grip: 'סובבו את המסך אל הפנים',
  again: 'שוב',
  start_over: 'להתחיל מחדש',
  compare: 'השוואת מודלים',
  compare_after: 'אחרי {TOUCHES}',
  confident: 'בטוח',
  unavailable: 'לא זמין',
  model: 'מודל',
  close: 'סגירה',
  demo: 'הדגמה',
  demo_badge: 'הדגמה: יד {HAND} מדומה',
  sim_cnn: 'תנועה מדומה: ה-CNN קורא רק תנועה אמיתית',
  sim_na: 'לא חל (הדמיה)',
  sim_cnn_line: 'לא תקף על תנועה מדומה',
  u_desktop: 'הדף צריך טלפון עם חיישני תנועה.',
  u_inapp: 'פתחו את הקישור ב-Safari או ב-Chrome.',
  u_https: 'הדף צריך קישור מאובטח (https).',
  u_gyro: "הטלפון הזה לא מדווח על ג'ירוסקופ.",
  u_acc: 'הטלפון הזה לא מדווח על מד תאוצה.',
  u_events: 'לא מתקבלים נתוני חיישנים. טענו את הדף מחדש.',
  u_events_android: 'לא מתקבלים נתוני חיישנים. אשרו חיישני תנועה בהגדרות האתר ב-Chrome וטענו מחדש.',
  u_error: 'משהו השתבש – טענו מחדש',
  try_demo: 'להדגמה',
  copy_link: 'העתקת הקישור',
  copied: 'הקישור הועתק',
  reload: 'טעינה מחדש',
  restarted: 'התחלנו מחדש: הדף הוסתר.',
  info_title: 'מידע',
  info1: 'P(right) היא ההסתברות המכוילת של המודל שהשתמשו ביד ימין.',
  info2: 'החזיקו את הטלפון כשהמסך מולכם.',
  info3: 'שום מידע לא יוצא מהטלפון.',
  m_cnn: 'Final CNN',
  m_cnn_line: 'CNN חד-ממדי, 4 רשתות',
  m_m4: 'Detector model',
  m_m4_line: 'מסלול m4, רגרסיה לוגיסטית על yaw',
  m_phys6: 'Physics logistic',
  m_phys6_line: 'PHYS6: yaw, roll, תנועה צידית',
  m_yaw: 'Yaw reference',
  m_yaw_line: 'רגרסיה לוגיסטית על yaw, 8 נקודות',
};

export type Translate = (key: StringKey, vars?: Readonly<Record<string, string | number>>) => string;

export interface Strings {
  readonly lang: PredictLang;
  readonly t: Translate;
  /** "1 touch" / "3 touches" */
  touches(k: number): string;
  /** LEFT / RIGHT in the page language */
  hand(h: Hand): string;
  /** the selector label of a model and its one-line description */
  modelLabel(id: ModelId): string;
  modelLine(id: ModelId): string;
}

function fill(template: string, vars?: Readonly<Record<string, string | number>>): string {
  if (!vars) return template;
  return template.replace(/\{([A-Za-z]+)\}/g, (m, name: string) => (name in vars ? String(vars[name]) : m));
}

export function strings(lang: PredictLang): Strings {
  const table: Record<StringKey, string> = lang === 'he' ? HE : EN;
  const t: Translate = (key, vars) => fill(table[key], vars);
  const touches = (k: number): string => (k === 1 ? t('touch_one') : t('touch_many', { k }));
  return {
    lang,
    t,
    touches,
    hand: (h) => t(h === 'left' ? 'left' : 'right'),
    modelLabel: (id) => t(`m_${id}` as StringKey),
    modelLine: (id) => t(`m_${id}_line` as StringKey),
  };
}

/** Short name on the model chip of the top bar. */
export const MODEL_CHIP: Readonly<Record<ModelId, string>> = {
  cnn: 'CNN',
  m4: 'm4',
  phys6: 'PHYS6',
  yaw: 'Yaw',
};
