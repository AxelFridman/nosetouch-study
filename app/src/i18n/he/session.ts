import type { session as en } from '../en/session';

// Hebrew mirror of ../en/session.ts. Gender-neutral: plural imperatives, infinitives and impersonal
// (passive) forms. A trial is "תנועה" for participants, a block is "סבב", a session is "מפגש".
// '\u200E' (LRM) before a number followed by a Latin unit ("60 Hz", "°/s") keeps number and unit in
// left-to-right order; without it the bidi algorithm shows "°/s" as "s/°" inside Hebrew text.
export const session: typeof en = {
  trialOf: (i, n) => `תנועה ${i} מתוך ${n}`,
  blockOf: (i, n) => `סבב ${i} מתוך ${n}`,
  trialInBlock: (i, n) => `תנועה ${i} מתוך ${n} בסבב הזה`,
  // "label: count" form: "1 מתוך 40 הושלמו" would not agree in number
  trialsDone: (done, total) => `הושלמו: ${done} מתוך ${total}`,
  attempt: (n) => `ניסיון מס׳ ${n}`,
  saving: 'שמירה…',
  saveFailed: 'לא ניתן היה לשמור את הניסיון הזה בטלפון. נסו שוב.',
  noSession: 'אין כרגע מפגש פעיל.',
  goHome: 'למסך הפתיחה',
  simMismatch: 'המפגש הזה התחיל במצב חיישנים אחר (סימולציה או חיישנים אמיתיים). לכל ניסיון נרשם באיזה מצב הוא הוקלט.',

  exit: {
    open: 'הפסקה',
    title: 'לעשות הפסקה?',
    body: 'אפשר לעצור עכשיו ולהמשיך מאוחר יותר ממסך הפתיחה, או לסיים את המפגש כאן. כל מה שהוקלט עד עכשיו נשמר.',
    pause: 'עצירה והמשך מאוחר יותר',
    end: 'סיום המפגש עכשיו',
    cancel: 'חזרה למפגש',
    paused: 'המפגש הושהה. אפשר להמשיך מאוחר יותר ממסך הפתיחה.',
  },

  intro: {
    kicker: 'הסבב הבא',
    newHand: 'החליפו יד!',
    odd: 'בסבב הזה כל תנועה קצת שונה ולא רגילה. קראו כל הוראה לפני ההקשה על „מוכנים”.',
    free: 'בסבב הזה החזיקו את הטלפון בכל דרך שמרגישה לכם טבעית.',
    start: 'התחלה',
  },

  ready: {
    phone: 'טלפון',
    speed: 'מהירות',
    startPosition: 'נקודת התחלה',
    movement: 'תנועה',
    hint: 'הקישו על „מוכנים”, מתחו את היד קדימה והחזיקו בלי לזוז. זזים רק כשנשמע הצפצוף.',
    ready: 'מוכנים',
    preparing: 'מתכוננים…',
    // Closing the tab is not enough on iOS: WebKit caches the answer until the app showing the page
    // is closed. No Private tab mid-session: new storage = a new participant code.
    permissionDenied: {
      safari:
        'הטלפון לא איפשר גישה לחיישני התנועה. סגרו את Safari לגמרי (החליקו אותו למעלה במחליף היישומים), פתחו שוב את הקישור, הקישו על „המשך המפגש” ואשרו גישה לתנועה כשתופיע הבקשה.',
      homeScreen:
        'הטלפון לא איפשר גישה לחיישני התנועה. סגרו את האפליקציה לגמרי (החליקו אותה למעלה במחליף היישומים), פתחו אותה שוב ממסך הבית, הקישו על „המשך המפגש” ואשרו גישה לתנועה כשתופיע הבקשה. אם זה לא עוזר, פתחו את הקישור ב-Safari והתחילו שם מחדש.',
      browser: (browser) =>
        `הטלפון לא איפשר גישה לחיישני התנועה. סגרו את ${browser} לגמרי (החליקו אותו למעלה במחליף היישומים), פתחו שוב את הקישור, הקישו על „המשך המפגש” ואשרו גישה לתנועה כשתופיע הבקשה.`,
      inApp:
        'הטלפון לא איפשר גישה לחיישני התנועה. הדפדפן שבתוך האפליקציה הזו לא יכול לבקש שוב: נא לפתוח את הקישור ב-Safari ולהתחיל שם מחדש.',
      // "next to" rather than "left of": a Hebrew phone UI mirrors Chrome's address bar.
      android:
        'הטלפון לא איפשר גישה לחיישני התנועה. הקישו על הסמל שליד שורת הכתובת › הרשאות (או הגדרות אתר) › חיישני תנועה, ובחרו לאפשר. אחר כך הקישו שוב על „מוכנים”.',
      other:
        'הטלפון לא איפשר גישה לחיישני התנועה. אפשרו בהגדרות הדפדפן גישה לחיישני תנועה עבור האתר הזה, ואז הקישו שוב על „מוכנים”.',
    },
  },

  live: {
    starting: 'מפעילים את החיישנים…',
    messages: {
      hold_still: 'לא לזוז',
      too_early: 'מוקדם מדי! חכו לצפצוף',
      go: 'עכשיו!',
      moving: 'געו באף',
      hold_at_nose: 'השאירו אותו שם…',
      done: 'סיימנו!',
      no_movement: 'לא זוהתה תנועה',
      too_long: 'זה לקח יותר מדי זמן',
      sensor_stalled: 'חיישני התנועה הפסיקו לעבוד',
    },
    waitBeep: 'יד ישרה, בלי לזוז, וחכו לצפצוף',
    stop: 'לחיצה ארוכה לעצירה',
    stopAria: 'לחצו והחזיקו כשנייה כדי לעצור את הניסיון',
    sim: 'סימולציה',
  },

  review: {
    heading: 'הוקלט!',
    duration: (seconds) => `זמן התנועה: ${seconds} שניות`,
    noDuration: 'לא ניתן היה למדוד את זמן התנועה.',
    plotAria: 'מהירות הסיבוב והתאוצה במהלך הניסיון, עם סימון הצפצוף, תחילת התנועה וסופה.',
    rotation: (peak) => `סיבוב (שיא \u200E${peak} °/s)`,
    acceleration: (peak) => `תאוצה (שיא \u200E${peak} m/s²)`,
    markers: { go: 'צפצוף', onset: 'התחלה', end: 'סוף', touch: 'נגיעה' },
    orientationMismatch: (expected, detected) =>
      `נראה שהטלפון הוחזק ${detected} במקום ${expected}. אם הטלפון הוחזק אחרת מהציור, הקישו על „בעיה” ובחרו „החזקתי את הטלפון בצורה לא נכונה”.`,
    question: 'האם התנועה בוצעה לפי ההוראות?',
    next: 'כן, הבא',
    redo: 'עוד פעם',
    problem: 'בעיה',
    flagTitle: 'מה השתבש?',
    flags: {
      wrong_hand: 'השתמשתי ביד הלא נכונה',
      wrong_orientation: 'החזקתי את הטלפון בצורה לא נכונה',
      other_problem: 'משהו אחר השתבש',
    },
    flagCancel: 'לא משנה',
    details: (n, hz) => `${n === 1 ? 'דגימה אחת' : `${n} דגימות`} · \u200E${hz} Hz`,
  },

  rotations: {
    upright: 'זקוף',
    upside_down: 'הפוך',
    top_left: 'לרוחב עם הצד העליון שמאלה',
    top_right: 'לרוחב עם הצד העליון ימינה',
    flat: 'שטוח',
  },

  failed: {
    heading: 'בואו ננסה את זה שוב',
    reasons: {
      no_movement: 'לא זוהתה תנועה אחרי הצפצוף. התחילו לזוז מיד כשהוא נשמע.',
      too_long: 'התנועה לא הסתיימה בזמן. הביאו את הטלפון אל האף והשאירו אותו שם עד הצפצוף הכפול.',
      aborted: 'הניסיון נעצר.',
      interrupted: 'הניסיון נקטע: המסך כבה או שהאפליקציה עברה לרקע.',
      sensor_stalled: 'חיישני התנועה הפסיקו לשלוח נתונים. אם זה חוזר על עצמו, סגרו את האפליקציה ופתחו שוב את הקישור.',
      error: 'משהו השתבש במהלך ההקלטה.',
    },
    kept: 'הניסיון הזה נשמר לבקרת איכות, אבל לא נספר.',
    retry: 'נסו שוב',
  },

  breakScreen: {
    heading: (i, n) => `סבב ${i} מתוך ${n} הושלם!`,
    cheers: ['כל הכבוד!', 'עבודה מצוינת!', 'הולך מעולה!', 'מצוין, ממשיכים כך!', 'יפה מאוד!'],
    rest: 'הורידו את היד, נערו אותה קצת ונוחו לרגע.',
    next: (hand, orientation) => `בסבב הבא: ${hand}, ${orientation}`,
    continue: 'המשך',
    pauseExit: 'הפסקה ויציאה',
  },

  done: {
    title: 'תודה רבה!',
    completed: 'המפגש הושלם. תודה על העזרה למחקר שלנו!',
    abandoned: 'המפגש הסתיים מוקדם. תודה! כל מה שהוקלט עדיין מועיל.',
    noSession: 'תודה על ההשתתפות!',
    summary: 'המפגש שלכם',
    accepted: 'תנועות שהוקלטו',
    attempts: 'ניסיונות (כולל חזרות)',
    duration: 'משך',
    upload: 'שליחת הנתונים',
    allSent: 'כל הנתונים נשלחו לצוות המחקר.',
    pending: (n) => (n === 1 ? 'פריט אחד ממתין לשליחה.' : `${n} פריטים ממתינים לשליחה.`),
    failed: (n) =>
      n === 1
        ? 'פריט אחד עוד לא נשלח. האפליקציה מנסה שוב כל עוד היא פתוחה ומחוברת לאינטרנט; אם זה ממשיך להיכשל, אפשר לראות למה דרך „הנתונים שלי” במסך הפתיחה.'
        : `${n} פריטים עוד לא נשלחו. האפליקציה מנסה שוב כל עוד היא פתוחה ומחוברת לאינטרנט; אם זה ממשיך להיכשל, אפשר לראות למה דרך „הנתונים שלי” במסך הפתיחה.`,
    keepOpen: 'השאירו את האפליקציה פתוחה ומחוברת לאינטרנט עד שהכול יישלח.',
    notConfigured: 'במחקר הזה הנתונים נאספים כקובץ: הקישו על „שמירה / שיתוף של הנתונים” ושלחו את הקובץ לצוות המחקר.',
    uploadNow: 'שליחה עכשיו',
    uploading: 'שליחה…',
    uploadResult: (sent, failed) =>
      failed > 0 ? `נשלחו: ${sent}. עוד לא נשלחו: ${failed}. פרטים דרך „הנתונים שלי” במסך הפתיחה.` : `נשלחו: ${sent}.`,
    nothingToUpload: 'אין עוד מה לשלוח.',
    exportData: 'שמירה / שיתוף של הנתונים',
    exporting: 'מכינים את הקובץ…',
    exported: {
      shared: 'הקובץ שותף.',
      downloaded: 'הקובץ נשמר בתיקיית ההורדות.',
      cancelled: 'השיתוף בוטל.',
    },
    exportFailed: 'לא ניתן היה ליצור את קובץ הנתונים.',
    // not "מסך הבית": in Hebrew that is the phone's home screen
    home: 'למסך הפתיחה',
  },

  sandbox: {
    title: 'ארגז חול לחיישנים',
    intro: 'כלי לצוות המחקר: אותות החיישנים בזמן אמת וגלאי התנועה. שום דבר לא נשמר.',
    start: 'התחלה',
    stop: 'עצירה',
    rate: 'קצב דגימה',
    samples: 'דגימות',
    accSource: 'תאוצה קווית',
    detector: 'גלאי',
    accSources: {
      linear: 'מדווחת על ידי הטלפון',
      highpass: 'מוערכת (מסנן מעביר גבוהים)',
      none: 'לא זמינה',
    },
    states: {
      idle: 'מושבת',
      waiting_still: 'ממתין ליציבות',
      still: 'יציב, הצפצוף בדרך',
      armed: 'ניתן אות, זזים עכשיו',
      moving: 'זוהתה תנועה',
      ended: 'התנועה הסתיימה',
      failed: 'אין תנועה תקפה',
    },
    gyroPlot: 'מהירות סיבוב |ω| (\u200E°/s, מוחלקת)',
    accPlot: 'תאוצה (m/s²)',
    series: { gyro: '|ω|', acc: '|a| ללא כבידה', gx: 'כבידה x', gy: 'כבידה y', gz: 'כבידה z' },
    thresholds: { still: 'יציב', onset: 'התחלה' },
    markers: { go: 'אות', onset: 'התחלה', end: 'סוף' },
    movements: 'תנועות שזוהו',
    noMovements: 'עדיין אין. החזיקו את הטלפון בלי לזוז, חכו לצפצוף ואז הביאו אותו אל האף.',
    movementRow: (i, ms, peak) => `#${i}: \u200E${ms} ms, שיא \u200E${peak} °/s`,
    noMovement: 'אין תנועה אחרי הצפצוף',
    tooLong: 'תנועה ארוכה מדי',
    permissionDenied: 'הגישה לחיישני התנועה נדחתה.',
    errors: {
      not_supported: 'הדפדפן הזה לא מספק חיישני תנועה.',
      permission_denied: 'הגישה לחיישני התנועה נדחתה.',
      insecure_context: 'חיישני תנועה דורשים דף מאובטח (https).',
      no_events: 'לא מגיעים אירועי תנועה. במחשב, פתחו את האפליקציה עם \u200E?sim=1 כדי להשתמש בסימולטור.',
      sensor_error: 'חיישני התנועה דיווחו על שגיאה.',
    },
  },

  figure: {
    top: 'עליון',
    screenUp: 'מסך למעלה',
  },
};
