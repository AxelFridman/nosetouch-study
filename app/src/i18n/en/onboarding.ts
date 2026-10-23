// Strings of the onboarding flow: welcome → consent → profile → sensors → calibration → setup.
// Functions are allowed for interpolation. Hebrew mirror: ../he/onboarding.ts (identical keys and
// array lengths; tests/i18n.test.ts checks this).
import type { EhiAnswer, EhiShortForm, HandednessSelf, PhoneHand, ProtocolPreset, Sex } from '../../core/types';

export const onboarding = {
  welcome: {
    tagline: 'Can a phone feel which hand is holding it? Help us find out.',
    aboutTitle: 'What you will do',
    aboutBody:
      'Hold your phone with your arm stretched out in front of you, then bring it in to gently touch the tip of your nose. You will repeat this with each hand, holding the phone in different ways and at different speeds.',
    needsTitle: 'You need',
    needs: ['About 10–30 minutes', 'This phone', 'A little space around you'],
    privacy:
      'Pseudonymous: you get a random participant code instead of your name. Only the motion sensors are recorded, and only during the movements.',
    start: 'Start',
    continueAs: 'Continue as',
    newParticipant: 'New participant (someone else)',
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
    pendingUploads: (n: number) =>
      `${n} ${n === 1 ? 'item is' : 'items are'} waiting to be uploaded. ${n === 1 ? 'It is' : 'They are'} sent automatically when you are online.`,
    version: (v: string) => `Version ${v}`,
  },

  consent: {
    title: 'Information and consent',
    intro: 'Please read this before you decide whether to take part.',
    purposeTitle: 'What is this study about?',
    /** institution: StudyConfig.institution in this language ('' → neutral wording) */
    purpose: (institution: string) =>
      `${institution ? `Researchers at ${institution}` : 'Researchers'} are studying whether the motion sensors of a phone can tell if it is held in the left or in the right hand. Your recordings will be used to develop and test such a method.`,
    procedureTitle: 'What will I do?',
    procedure:
      'You will hold the phone with your arm stretched straight out in front of you and bring it in to gently touch the tip of your nose, many times. You will use both hands (one after the other), hold the phone in several ways (upright, sideways, upside down, screen toward you or away from you), move at different speeds, and do a few deliberately unusual movements. The app explains every part and beeps to start each movement.',
    durationTitle: 'How long does it take?',
    // estimateMinutes: quick 7.5, standard 14, full 29.5 min of movements, plus the setup steps
    duration:
      'About 10–30 minutes, depending on the session you choose (the standard session takes about 15–20 minutes, including the setup). You can pause or stop at any time.',
    risksTitle: 'Risks and safety',
    risks: [
      'You might drop the phone: hold it firmly and, if you can, stand or sit where it would land on a bed or a sofa.',
      'You might bump your nose: always touch it gently, also in the "fast" movements.',
      'Stop right away if anything hurts or feels uncomfortable.',
    ],
    voluntaryTitle: 'Taking part is voluntary',
    voluntary:
      'You can stop at any moment without giving a reason. You can delete the data stored on this phone at any time on the "My data" screen. To have data that was already uploaded deleted, contact the researcher and quote your participant code.',
    collectedTitle: 'What is recorded',
    collected: [
      'The motion sensors (accelerometer, gyroscope, orientation) and screen touches, only while a movement is being recorded.',
      'Your answers to the short profile questions (age, sex, handedness, height, phone details).',
      'Technical details of the phone and the browser, for example how fast the sensors report.',
    ],
    notCollectedTitle: 'What is never recorded',
    notCollected: 'No camera, no microphone, no location, no contacts and nothing from other apps.',
    nicknameLocal:
      'Your name is not needed. An optional nickname stays on this phone and is not uploaded.',
    nicknameUploaded:
      'Your name is not needed. An optional nickname is stored with your data, so please do not use your full name.',
    storageTitle: 'Where the data is stored',
    storageUpload:
      'On this phone and in the study storage managed by the research team. It is sent there automatically when you are online.',
    storageLocal:
      'On this phone. You can export it as a file and send it to the research team.',
    retentionTitle: 'How long it is kept',
    retention: (institution: string) =>
      `The pseudonymous data is kept only as long as it is needed for this research and its publications, according to the rules of ${institution || 'the research institution'} and of the ethics committee that approved the study.`,
    contactTitle: 'Questions?',
    contactIntro: 'For questions or requests about your data, contact:',
    contactNone: 'Contact the researcher who sent you this link.',
    version: (v: string) => `Consent version ${v}`,
    /** region: StudyConfig.region in this language ('' → no location clause) */
    agree: (minAge: number, region: string) =>
      `I am at least ${minAge} years old, ${region ? `I am taking part from ${region}, ` : ''}I have read the information above, and I agree to take part.`,
    accept: 'I agree, continue',
    decline: 'Not now',
  },

  profile: {
    title: 'About you',
    intro: 'A few quick questions. They help us understand for whom the method works well and for whom it does not.',
    codeTitle: 'Your participant code',
    codeHint:
      'Write it down or take a screenshot. You need it to continue on another day or phone, or to ask for your data to be deleted.',
    haveCodeTitle: 'Already have a participant code?',
    otherCodeTitle: 'Not you? Enter your own participant code',
    haveCodeHint: 'Enter it to keep using the same code. Leave it empty if this is your first time.',
    codeFormat: 'A code looks like P-ABC234: "P-" and 6 letters or digits.',
    codeChecking: 'Checking the code…',
    codeFound: 'Welcome back! Your answers were loaded from this phone. Please check them.',
    codeNew: 'Code accepted. Please fill in your details below.',
    nickname: 'Nickname',
    nicknameHintLocal: 'Optional. Stays on this phone, so you can recognise your profile.',
    nicknameHintUploaded: 'Optional. It is stored with your data, so please do not use your full name.',
    age: 'Age',
    ageHint: 'In years',
    sex: 'Sex',
    sexes: {
      female: 'Female',
      male: 'Male',
      other: 'Other',
      prefer_not_to_say: 'Prefer not to say',
    } as Record<Sex, string>,
    handedness: 'Are you right-handed or left-handed?',
    handednessOptions: {
      right: 'Right-handed',
      left: 'Left-handed',
      ambidextrous: 'Both equally (ambidextrous)',
    } as Record<HandednessSelf, string>,
    ehiTitle: 'Which hand do you use for…',
    ehiHint: 'Four short questions from a standard handedness questionnaire.',
    ehiItems: {
      writing: 'Writing',
      throwing: 'Throwing',
      toothbrush: 'Holding a toothbrush',
      spoon: 'Holding a spoon',
    } as Record<keyof EhiShortForm, string>,
    ehiAnswers: {
      always_right: 'Always right',
      usually_right: 'Usually right',
      both: 'Both equally',
      usually_left: 'Usually left',
      always_left: 'Always left',
    } as Record<EhiAnswer, string>,
    phoneHand: 'Which hand do you usually hold your phone in?',
    phoneHandOptions: {
      right: 'Right',
      left: 'Left',
      both: 'Both / it varies',
    } as Record<PhoneHand, string>,
    height: 'Height (cm)',
    armIssue: 'Do you have an injury or a condition that affects how you move your arms or hands?',
    phoneModel: 'Phone model',
    phoneModelHint: 'For example iPhone 13 or Galaxy S23.',
    phoneCase: 'Is your phone in a case or a cover?',
    optional: 'Optional',
    save: 'Save and continue',
    saving: 'Saving…',
    errors: {
      ageRequired: 'Please enter your age.',
      ageRange: 'Please enter your age in whole years (up to 110).',
      ageTooYoung: (minAge: number) => `This study is only for people aged ${minAge} or older.`,
      required: 'Please choose an answer.',
      heightRange: 'Height must be between 50 and 250 cm.',
      fixBelow: 'Please check the highlighted answers.',
    },
  },

  sensors: {
    title: 'Sensor check',
    intro:
      'The study records the motion sensors of your phone. Your browser may ask whether this page may use "motion and orientation": if it does, please tap Allow.',
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

  calibration: {
    title: 'Calibration',
    intro:
      'Three short steps (about 30 seconds) let us read the sensors of your phone correctly. You may skip them, but they make the data better.',
    stepLabel: (n: number) => `Step ${n} of 3`,
    flatTitle: 'Phone flat on a table',
    flatHow:
      'Put the phone on a table with the screen facing up. Tap "Measure", take your hand away and do not touch the phone until you hear the double beep.',
    uprightTitle: 'Phone upright in front of you',
    uprightHow:
      'Hold the phone upright in front of you with the screen facing you, as if reading a message. Keep it still until the double beep.',
    tiltTitle: 'Tilt test',
    tiltHow:
      'Hold the phone flat in front of you, screen facing up. At the beep, slowly tilt it to the RIGHT until it stands on its side, then bring it back flat (about 3 seconds).',
    tiltSim: 'Skipped in simulation mode: the simulator cannot tilt the phone.',
    measure: 'Measure',
    measureAgain: 'Measure again',
    getReady: (sec: number) => `Starting in ${sec}…`,
    holdStill: 'Measuring… keep still',
    tiltNow: 'Tilt slowly to the right… and back',
    flatOk: 'Done ✓',
    flatOkInverted: 'Done ✓ This browser reports gravity with the opposite sign (normal on an iPhone). Nothing to do.',
    flatFaceDown:
      'The phone seems to be lying screen-down. Put it on the table with the screen facing UP, take your hand away and tap "Measure again". If the screen was already facing up, you can continue.',
    flatMoving: 'The phone moved. Put it on the table, take your hand away and try again.',
    flatNotFlat: 'The phone does not seem to lie flat. Put it screen-up on a level table and try again.',
    flatOdd: 'The readings look unusual. Try again; if it keeps happening, skip this step.',
    noData: 'No sensor data arrived. Go back to the sensor check and try again.',
    uprightOk: 'Done ✓',
    uprightInverted:
      'The readings do not match step 1. Please repeat step 1 (phone flat, screen facing UP), then this step (top edge up, screen facing you).',
    uprightMoving: 'Please keep the phone still while it measures, then try again.',
    uprightUnclear: 'Hold the phone upright (top edge up) with the screen facing you and try again.',
    axisOk: 'Done ✓ The rotation readings look standard.',
    axisNoted: 'Done ✓ This phone reports rotation in an unusual way. That is useful to know and has been noted.',
    axisRadians: 'Done ✓ This phone reports rotation in radians. That has been noted.',
    axisUnclear:
      'We could not see a clear tilt. Tilt slowly, about a quarter turn to the right, and back. You can try again or simply continue.',
    doneEarlier: 'Your earlier calibration is kept. Continue, or measure again.',
    continue: 'Continue',
    skip: 'Skip calibration',
  },

  setup: {
    title: 'Your session',
    intro: 'Choose how long you want to record. Longer sessions cover more ways of holding the phone.',
    presetLabel: 'Session',
    presets: {
      quick: { label: 'Quick', desc: '4 ways of holding the phone, 3 speeds.' },
      standard: { label: 'Standard', desc: '5 ways of holding the phone, speeds, start positions and a few unusual movements.' },
      full: { label: 'Full', desc: 'All 9 ways of holding the phone, every speed and start position, unusual and free movements.' },
      custom: { label: 'Custom', desc: 'Build your own session (for researchers).' },
    } as Record<ProtocolPreset, { label: string; desc: string }>,
    presetMeta: (trials: number, minutes: number) =>
      `${trials} ${trials === 1 ? 'movement' : 'movements'} · about ${minutes} min`,
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
    summary: (trials: number, minutes: number) =>
      `${trials} ${trials === 1 ? 'movement' : 'movements'}, about ${minutes} ${minutes === 1 ? 'minute' : 'minutes'} including short breaks.`,
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
    needConsent: 'Please read and accept the consent first.',
  },
};
