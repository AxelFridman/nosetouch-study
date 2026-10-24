// Strings of the screens outside the session: welcome (with the consent) → profile → (sensor repair)
// → (researchers' session setup), and the done screen. Functions are allowed for interpolation.
// Hebrew mirror: ../he/onboarding.ts (identical keys and array lengths; tests/i18n.test.ts checks this).
// NOTE for the PI: the consent wording (welcome.summary*, welcome.agree and consent.*) is the version
// in public/config.json consent_version (2026-10-v2); bump it whenever this wording changes.
import type { HandednessSelf, PhoneHand, ProtocolPreset, Sex } from '../../core/types';

export const onboarding = {
  welcome: {
    title: 'Nose-touch study',
    lead: 'Help us teach a phone to tell which hand moved it.',
    gestureLabel:
      'Animation: a person holds the phone with the arm stretched out, brings it in to touch the tip of the nose, then stretches the arm out again.',
    // the four key facts of the consent, shown before everything else
    summaryDuration:
      "You'll hold your phone at arm's length and touch your nose with it, about 40 times — around 5 minutes. You can stop at any time; everything you've done is kept.",
    summarySensors: "We record only the phone's motion sensors (movement and rotation) — no camera, microphone or location.",
    /** uploads: data goes to the research server; names: the name goes too (config.upload_nickname) */
    summaryStored: (uploads: boolean, names: boolean): string =>
      uploads
        ? names
          ? 'We store your name, age, sex, dominant hand and height with the data on a private research server.'
          : 'We store your age, sex, dominant hand and height with the data on a private research server. Your name stays on this phone.'
        : 'Your name, age, sex, dominant hand and height are stored with the data on this phone until it is sent to the researcher.',
    summaryDelete: 'To have your data deleted, contact the researcher (details below).',
    fullInfo: 'Full information',
    /** region: StudyConfig.region in this language ('' → no location clause) */
    agree: (minAge: number, region: string) =>
      `I agree, I'm ${minAge} or older, ${region ? `I'm taking part from ${region}, ` : ''}and I have no arm or hand injury that makes this uncomfortable.`,
    permissionNote: 'Your phone may ask to use its motion sensors: tap Allow.',
    start: 'Start',
    continueAs: (name: string) => `Continue as ${name}`,
    continueHint: 'Tap Start to check your answers and go on.',
    newPerson: 'New person',
    resumeTitle: 'Unfinished session',
    resumeBody: (done: number, total: number) => `${done} of ${total} ${total === 1 ? 'movement' : 'movements'} recorded.`,
    resumeStarted: (when: string) => `Started ${when}.`,
    resume: 'Resume session',
    resuming: 'Opening…',
    resumeFailed: 'That session can no longer be resumed.',
    resumeNeedsSensors: 'Motion access is needed before you can continue. Please check the sensors first.',
    language: 'Language',
    menuTitle: 'More',
    menuData: 'My data',
    menuSettings: 'Settings',
    menuSandbox: 'Sensor playground',
    menuAbout: 'About the study',
    warnInsecure:
      'This page was not opened through a secure https:// link, so the phone will not share its motion sensors. Please open the exact link you received.',
    warnNoMotion:
      'This browser does not provide motion sensors. Please open the link on your phone, in Safari (iPhone) or Chrome (Android).',
    warnInApp: (app: string) =>
      `You opened this link inside the ${app} app. Motion sensors often do not work there: tap the ⋯ or ⋮ menu and choose "Open in browser", or copy the link into Safari (iPhone) or Chrome (Android).`,
    warnInAppGeneric:
      'You opened this link inside another app. Motion sensors often do not work there: tap the ⋯ or ⋮ menu and choose "Open in browser", or copy the link into Safari (iPhone) or Chrome (Android).',
    warnSim:
      'Simulation mode is on: a built-in simulator replaces the real sensors. For testing only — it switches off when the app is reloaded.',
    warnSetup:
      'Researcher mode: after the profile you choose the session on the setup screen. It switches off when the app is reloaded.',
    pendingUploads: (n: number) =>
      `${n} ${n === 1 ? 'item is' : 'items are'} waiting to be uploaded. ${n === 1 ? 'It is' : 'They are'} sent automatically when you are online.`,
    version: (v: string) => `Version ${v}`,
  },

  // "Full information" on the welcome screen (a <details> below the four key facts).
  consent: {
    purposeTitle: 'What is this study about?',
    /** institution: StudyConfig.institution in this language ('' → neutral wording) */
    purpose: (institution: string) =>
      `${institution ? `Researchers at ${institution}` : 'Researchers'} are studying whether the motion sensors of a phone can tell if it is held in the left or in the right hand. Your recordings will be used to develop and test such a method.`,
    procedureTitle: 'What will I do?',
    // street preset: 29 movements, plus 10 in the optional bonus round
    procedure:
      'You will hold the phone with your arm stretched straight out in front of you and bring it in to touch the tip of your nose, about 30 times (about 40 if you also do the optional bonus round at the end). You will use both hands (one after the other), hold the phone upright and sideways with the screen facing you, move at different speeds and do a few slightly different movements. Any part of the phone may touch your nose. The app shows what to do and beeps to start each movement. No table or other equipment is needed.',
    durationTitle: 'How long does it take?',
    // street preset: estimateMinutes 5 min (the required part), plus the bonus round of about 1 minute
    duration:
      'About 5 minutes, and at the end an optional bonus round of about 1 minute. You can pause or stop at any time; everything recorded until then is kept.',
    risksTitle: 'Risks and safety',
    risks: [
      'You might drop the phone: hold it firmly and, if you can, stand or sit where it would land on a bed or a sofa.',
      'You might bump your nose: always touch it gently, also in the "fast" movements.',
      'Stop right away if anything hurts or feels uncomfortable.',
    ],
    voluntaryTitle: 'Taking part is voluntary',
    voluntary:
      'You can stop at any moment without giving a reason. You can delete the data stored on this phone at any time on the "My data" screen. To have data that was already uploaded deleted, contact the researcher and quote your participant code (it is shown at the end).',
    collectedTitle: 'What is recorded',
    collected: [
      'The motion sensors (accelerometer, gyroscope, orientation and, where the phone has one, the compass direction) and screen touches, only while a movement is being recorded.',
      'Your answers to the short questions: your name, age, sex, dominant hand, the hand you usually hold your phone in, and your height (optional).',
      'Technical details of the phone and the browser, for example the phone model and how fast the sensors report.',
    ],
    notCollectedTitle: 'What is never recorded',
    notCollected: 'No camera, no microphone, no location, no contacts and nothing from other apps.',
    nameUploaded:
      'Your name is stored with your data on the private research server, so that the researcher can find your recordings (for example to delete them). It is never published.',
    nameLocal: 'Your name stays on this phone and is not uploaded.',
    storageTitle: 'Where the data is stored',
    storageUpload:
      'On this phone and on a private research server managed by the research team. Each movement is sent there as soon as it is recorded, whenever the phone is online.',
    storageLocal: 'On this phone. You can export it as a file and send it to the research team.',
    retentionTitle: 'How long it is kept',
    retention: (institution: string) =>
      `The data is kept only as long as it is needed for this research and its publications, according to the rules of ${institution || 'the research institution'}. Publications never contain names.`,
    contactIntro: 'For questions or to have your data deleted, contact:',
    contactNone: 'For questions or to have your data deleted, contact the researcher who sent you this link.',
    version: (v: string) => `Consent version ${v}`,
  },

  profile: {
    title: 'About you',
    name: 'Full name',
    namePlaceholder: 'First and last name',
    nameHintUploaded: 'Optional. Stored with your data on the private research server.',
    nameHintLocal: 'Optional. Stays on this phone.',
    age: 'Age',
    sex: 'Sex',
    sexes: {
      female: 'Female',
      male: 'Male',
      other: 'Other',
      prefer_not_to_say: 'Prefer not to say',
    } as Record<Sex, string>,
    handedness: 'Dominant hand',
    handednessOptions: {
      right: 'Right',
      left: 'Left',
      ambidextrous: 'Both',
    } as Record<HandednessSelf, string>,
    phoneHand: 'Which hand do you usually hold your phone in?',
    phoneHandOptions: {
      right: 'Right',
      left: 'Left',
      both: 'Both',
    } as Record<PhoneHand, string>,
    height: 'Height (cm, optional)',
    returning: 'Returning? Enter your code',
    notYou: 'Not you? Enter your own code',
    codeYours: 'Participant code of this visit:',
    codeInput: 'Code from your earlier visit',
    codeHint: 'If you took part before, enter the code you got then. Leave it empty if this is your first time.',
    codeFormat: 'A code looks like P-ABC234: "P-" and 6 letters or digits.',
    codeChecking: 'Checking the code…',
    codeFound: 'Welcome back! Your answers were loaded from this phone. Please check them.',
    codeNew: 'Code accepted. Please fill in your details below.',
    save: 'Continue',
    saving: 'One moment…',
    errors: {
      ageRequired: 'Please enter your age.',
      ageRange: 'Please enter your age in whole years (up to 110).',
      ageTooYoung: (minAge: number) => `This study is only for people aged ${minAge} or older.`,
      required: 'Please choose an answer.',
      heightRange: 'Height must be between 50 and 250 cm.',
      fixBelow: 'Please check the highlighted answers.',
    },
  },

  // The repair screen: shown only when the background check after Start found that the sensors cannot
  // record (permission denied, no events, too slow), or when resuming needs the permission again.
  sensors: {
    title: 'Sensor check',
    intro:
      'The study needs the motion sensors of your phone, and they are not working for this page yet. Follow the advice below, then measure again.',
    stepsTitle: 'What happens',
    steps: [
      'Tap the button below.',
      'If your phone asks, allow access to motion and orientation.',
      'Hold the phone normally for about 3 seconds while we measure.',
    ],
    privacyNote: 'Only motion is read: never the camera, the microphone or your location.',
    allow: 'Allow motion access and test',
    measureAgain: 'Measure again',
    requesting: 'Waiting for permission…',
    measuring: 'Measuring… hold the phone normally',
    resultsTitle: 'Results',
    motionRate: 'Motion sensor rate',
    orientationRate: 'Orientation rate',
    gyroscope: 'Gyroscope (rotation)',
    linearAcc: 'Linear acceleration',
    quaternion: 'Orientation sensor (extra)',
    browser: 'Browser',
    available: 'Available',
    missing: 'Not available',
    hz: (n: number) => `${n} Hz`,
    none: '—',
    ok: 'Your sensors work. You are ready to continue.',
    simOk: 'Simulation mode: the built-in simulator provides the sensor data.',
    // WebKit keeps the answer in memory per app process and data store: quitting the app that shows
    // the page resets it; deleting website data does not. Picked per context by deniedText()
    // (SensorCheckScreen.tsx). Never a Private tab: new storage = a new participant code.
    deniedIos:
      'Motion access is blocked. Safari remembers this until it is fully closed: swipe Safari away in the app switcher, open this link again and tap Allow.',
    deniedIosHomeScreen:
      'Motion access is blocked. The app remembers this until it is fully closed: swipe it away in the app switcher, open it again from your Home Screen and tap Allow. If that does not help, open the link in Safari.',
    deniedIosBrowser: (browser: string) =>
      `Motion access is blocked. ${browser} remembers this until it is fully closed: swipe ${browser} away in the app switcher, open this link again and tap Allow.`,
    deniedIosInApp:
      'Motion access is blocked in the browser inside this app. Please open the link in Safari: tap the ⋯ or share menu and choose "Open in Safari", or copy the link into Safari.',
    deniedAndroid:
      'Motion access is blocked. Tap the icon to the left of the address bar › Permissions (or Site settings) › Motion sensors › Allow, then tap "Measure again".',
    deniedOther: 'Motion access is blocked. Allow motion sensors for this site in your browser settings and try again.',
    noEvents:
      'Your browser is not sending motion data. Some browsers (for example Brave and other privacy browsers) and the browsers inside other apps block the sensors: please open this link in Safari (iPhone) or Chrome (Android). Also make sure Low Power Mode / Battery Saver is off.',
    noEventsDesktop: 'This device does not seem to have motion sensors. Please open the link on your phone.',
    insecure:
      'This page was not loaded through a secure https:// link, so the sensors are blocked. Please open the exact link you received.',
    tooSlow: (hz: number) =>
      `The sensors report only ${hz === 1 ? 'once' : `${hz} times`} per second, which is too slow to record the movement. Turn off Low Power Mode / Battery Saver, close other apps and measure again.`,
    slow: (hz: number) =>
      `The sensors report ${hz} times per second (about 60 is typical). Turning off Low Power Mode / Battery Saver may help. You can still continue.`,
    noGyro: 'This phone does not report rotation (gyroscope). You can continue, but the recordings will be less complete.',
    failed: 'The measurement did not work. Please try again.',
    tipsTitle: 'Before you continue',
    tips: [
      'Turn off Low Power Mode (iPhone) or Battery Saver (Android).',
      'Turn the volume up so you can hear the beeps, and make sure the phone is not in Silent mode.',
      'Keep this page open and the phone unlocked during the whole session.',
    ],
    testBeep: 'Play a test beep',
  },

  // Researchers only (?setup=1 or Settings): choose the session instead of the study's default.
  setup: {
    title: 'Session setup',
    intro: 'For researchers: choose the session. Participants never see this screen; they get the study session.',
    presetLabel: 'Session',
    presets: {
      street: {
        label: 'Street',
        desc: 'The 5-minute visit: both hands, upright and sideways, then a few different movements. No table needed; a short bonus round is optional.',
      },
      quick: { label: 'Quick', desc: '3 ways of holding the phone, 3 speeds.' },
      standard: { label: 'Standard', desc: '4 ways of holding the phone, speeds, start positions and a few unusual movements.' },
      full: { label: 'Full', desc: 'All 5 ways of holding the phone, every speed and start position, unusual and free movements.' },
      custom: { label: 'Custom', desc: 'Build your own session (for researchers).' },
    } as Record<ProtocolPreset, { label: string; desc: string }>,
    /** bonus: movements of the optional bonus round (0 = none) */
    presetMeta: (trials: number, minutes: number, bonus: number) =>
      `${trials} ${trials === 1 ? 'movement' : 'movements'}${bonus > 0 ? ` (+${bonus} bonus)` : ''} · about ${minutes} min`,
    posture: 'Will you stand or sit?',
    postureHint: 'Standing is preferred. Please keep the same posture for the whole session.',
    customTitle: 'Custom session',
    orientations: 'Ways of holding the phone',
    speeds: 'Speeds',
    starts: 'Start positions',
    odd: 'Unusual movements (one of each per hand)',
    reps: 'Repetitions of each combination',
    repsHint: '1 to 5',
    free: 'Add a free-style block for each hand',
    freeHint: 'The phone is held in any way the participant likes (3 movements per hand).',
    errOrientation: 'Choose at least one way of holding the phone.',
    errSpeed: 'Choose at least one speed.',
    errStart: 'Choose at least one start position.',
    errReps: 'Repetitions must be a whole number from 1 to 5.',
    summaryTitle: 'Summary',
    /** bonus: movements of the optional bonus round (0 = none) */
    summary: (trials: number, minutes: number, bonus: number) =>
      `${trials} ${trials === 1 ? 'movement' : 'movements'}, about ${minutes} ${minutes === 1 ? 'minute' : 'minutes'} including short breaks${bonus > 0 ? `, plus an optional bonus round of ${bonus} ${bonus === 1 ? 'movement' : 'movements'}` : ''}.`,
    participant: 'Participant',
    bothHands: 'You will switch between your right and your left hand several times. The app always tells you which hand to use.',
    checklistTitle: 'Before you start',
    checklist: [
      'Make some room: you will stretch your arm out in front of you.',
      'Turn the volume up so you can hear the beeps.',
      'If you can, stand or sit where the phone would land on a bed or a sofa if it slips.',
    ],
    safetyTitle: 'Safety',
    start: 'Start recording',
    starting: 'Preparing…',
    needProfile: 'Please fill in your profile first.',
    needConsent: 'Please agree to take part on the start screen first.',
  },

  done: {
    /** firstName: first word of the participant's name ('' → no name) */
    thanks: (firstName: string) => (firstName ? `Thank you, ${firstName}!` : 'Thank you!'),
    completed: 'You finished the session. Thank you for helping our research!',
    endedEarly: 'You stopped before the end. Thank you: everything you did is kept and useful.',
    abandoned: 'The session ended early. Thank you: everything you recorded is still useful.',
    noSession: 'Thank you for taking part!',
    recorded: (n: number) => `${n} ${n === 1 ? 'movement' : 'movements'} recorded`,
    uploadTitle: 'Sending your data',
    allSent: 'All data uploaded ✓',
    sending: (left: number) => `Uploading… please keep this page open a moment (${left} left)`,
    offline: (left: number) =>
      `This phone is offline: ${left} ${left === 1 ? 'item is' : 'items are'} waiting. ${left === 1 ? 'It is' : 'They are'} sent automatically the next time this page is open with an internet connection.`,
    failed: (n: number) =>
      `${n === 1 ? '1 item' : `${n} items`} could not be sent yet. The app keeps trying while this page is open; if it keeps failing, "My data" on the start screen shows why.`,
    notConfigured: 'This study collects data as a file: tap "Save / share my data" and send the file to the researcher.',
    uploadNow: 'Upload now',
    uploading: 'Uploading…',
    uploadResult: (sent: number, failed: number) =>
      failed > 0 ? `Sent ${sent}, ${failed} not sent yet. See "My data" on the start screen for details.` : `Sent ${sent}.`,
    nothingToUpload: 'Nothing left to upload.',
    codeLabel: 'Your code:',
    codeHint: '(to delete your data later)',
    detailsTitle: 'Details',
    attempts: 'Attempts (including repeats)',
    duration: 'Time',
    nextPerson: 'Next person',
    home: 'Home',
    exportData: 'Save / share my data',
    exporting: 'Preparing your file…',
    exported: {
      shared: 'Shared.',
      downloaded: 'The file was saved to your downloads.',
      cancelled: 'Sharing was cancelled.',
    },
    exportFailed: 'The data file could not be created.',
  },
};
