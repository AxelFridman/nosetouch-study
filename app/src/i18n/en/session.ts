// Strings for the session runner, the done screen and the researcher sandbox. OWNED BY the session UI
// agent. Hebrew mirror with the same keys: ../he/session.ts. Protocol wording (hands, orientations,
// speeds, start positions, variants, safety) lives in common.ts and is not repeated here.
import type { Hand, OrientationId, ParticipantFlag, PhoneRotation, SensorError, Speed, TrialMessage } from '../../core/types';

const HAND_CAPS = (hand: Hand) => (hand === 'left' ? 'LEFT' : 'RIGHT');
const TIERS = ['Try it once', 'Part 1 of 3 · Basics', 'Part 2 of 3 · Sideways', 'Part 3 of 3 · Different movements', 'Bonus'];

export const session = {
  trialOf: (i: number, n: number) => `Trial ${i} of ${n}`,
  blockOf: (i: number, n: number) => `Block ${i} of ${n}`,
  trialInBlock: (i: number, n: number) => `Movement ${i} of ${n} in this block`,
  trialsDone: (done: number, total: number) => `${done} of ${total} done`,
  attempt: (n: number) => `Attempt ${n}`,
  saving: 'Saving…',
  saveFailed: 'This trial could not be saved on the phone. Please try again.',
  noSession: 'There is no session in progress.',
  goHome: 'Go to the start screen',
  simMismatch:
    'This session was started in a different sensor mode (simulation or real sensors). Every trial records which mode it used.',

  // The How-to card: the intro of the first (choice) block. Its Start button runs the free trial.
  howTo: {
    title: 'How to do it',
    anyPart: 'Any part of the phone may touch your nose.',
    steps: [
      'Hold the phone in front of you, arm straight, screen facing you.',
      'At the beep, bring it to touch your nose.',
      'Hold still until the double beep, then stretch your arm out again — the next one starts by itself.',
    ],
    safety: 'Touch gently and hold the phone firmly.',
    sitting: 'I’m sitting',
    try: 'Try it once — your way',
    gestureAria: 'Animation: the arm is stretched out, the phone comes in and touches the nose, then the arm stretches out again.',
    angledAria: 'Animation: the phone arrives tilted and touches the nose with its top edge.',
  },

  // Asked right after the free trial of the choice block.
  hand: {
    question: 'Which hand did you use?',
    left: 'Left hand',
    right: 'Right hand',
    unsure: 'Not sure / something went wrong',
  },

  intro: {
    kicker: 'Next block',
    newHand: 'Switch hands!',
    odd: 'In this block every movement is a little different and unusual. Read each instruction before you tap Ready.',
    free: 'In this block, hold the phone in whatever way feels natural to you.',
    choice: 'Try it once, your way: use whichever hand and grip feel natural. Afterwards we ask which hand you used.',
    start: 'Start',
    /** the block-start button names the hand (tap it with the free hand) */
    startIn: (hand: Hand) => `Start — phone in ${HAND_CAPS(hand)} hand`,
    phoneIn: (hand: Hand) => `Phone in your ${HAND_CAPS(hand)} hand`,
    keepIn: (hand: Hand) => `Keep the phone in your ${HAND_CAPS(hand)} hand`,
    /** tier label of a street block: 0 choice, 1 basics, 2 sideways, 3 variety, 4 bonus */
    tier: (tier: number) => TIERS[tier] ?? `Part ${tier}`,
    basicsDone: 'Basics done — great! The rest adds variety. You can stop any time.',
    speedsVary: 'Speeds change — the screen tells you each time.',
    variesList: (items: string) => `Each movement is a little different: ${items}. The screen tells you which one.`,
    redo: 'Once more: the last movement is done again.',
    gestureAria: (hand: string, grip: string) => `Animation: ${hand}, ${grip}, the phone comes to the nose and goes back.`,
  },

  // Researcher review mode (settings: review every trial): the Ready screen before each trial.
  ready: {
    phone: 'Phone',
    speed: 'Speed',
    startPosition: 'Start',
    movement: 'Movement',
    hint: 'Tap Ready, stretch your arm out and hold still. Move only when you hear the beep.',
    ready: 'Ready',
    preparing: 'Preparing…',
    // Closing the tab is not enough on iOS: WebKit caches the answer until the app showing the page
    // is closed. Picked per browser context (deniedContext in SensorCheckScreen.tsx). Never a Private
    // tab mid-session: new storage = a new participant code, and the session stays behind.
    permissionDenied: {
      safari:
        'The phone did not allow access to its motion sensors. Fully close Safari (swipe it away in the app switcher), open the link again, tap “Resume session” and allow motion access when asked.',
      homeScreen:
        'The phone did not allow access to its motion sensors. Fully close this app (swipe it away in the app switcher), open it again from your Home Screen, tap “Resume session” and allow motion access when asked. If that does not help, open the link in Safari and start again there.',
      browser: (browser: string) =>
        `The phone did not allow access to its motion sensors. Fully close ${browser} (swipe it away in the app switcher), open the link again, tap “Resume session” and allow motion access when asked.`,
      inApp:
        'The phone did not allow access to its motion sensors. The browser inside this app cannot ask again: please open the link in Safari and start again there.',
      android:
        'The phone did not allow access to its motion sensors. Tap the icon to the left of the address bar › Permissions (or Site settings) › Motion sensors › Allow, then try again.',
      other:
        'The phone did not allow access to its motion sensors. Allow motion sensors for this site in the browser settings, then try again.',
    },
  },

  live: {
    starting: 'Starting the sensors…',
    messages: {
      hold_still: 'Arm straight — hold still',
      too_early: 'Too early! Wait for the beep',
      go: 'GO!',
      moving: 'Touch your nose',
      hold_at_nose: 'Hold…',
      done: 'Done!',
      no_movement: 'No movement detected',
      too_long: 'That took too long',
      sensor_stalled: 'The motion sensors stopped',
      return_arm: 'Arm out again',
      wrong_grip: 'Hold the phone like the picture',
    } as Record<TrialMessage, string>,
    /** the status line of the live panel, by phase (see SessionScreen.statusKey) */
    status: {
      starting: 'Starting the sensors…',
      waiting: 'Arm straight — hold still',
      foreperiod: 'Get ready…',
      go: 'GO!',
      moving: 'Touch your nose',
      at_nose: 'Hold…',
      returning: 'Arm out again',
      wrong_grip: 'Hold the phone like the picture',
      too_early: 'Too early! Wait for the beep',
      sensor_stalled: 'The motion sensors stopped',
      no_movement: 'No movement detected',
      too_long: 'That took too long',
      done: 'Done!',
    },
    /** the huge speed word of the live panel */
    speedWords: { natural: 'NORMAL', fast: 'FAST', slow: 'SLOW' } as Record<Speed, string>,
    /** the hand word of the choice trial */
    anyHand: 'YOUR CHOICE',
    minutesLeft: (n: number) => (n <= 0 ? 'almost done' : n === 1 ? 'about 1 min left' : `about ${n} min left`),
    waitBeep: 'Arm straight, keep still and wait for the beep',
    pause: 'Hold to pause',
    pauseAria: 'Press and hold for about one second to pause',
    sim: 'SIM',
  },

  // The pause sheet: a long press on the live panel's pause control, the system back button, the app
  // going to the background, stalled sensors, a long wait without GO, two failures in a row.
  pause: {
    open: 'Pause',
    title: 'Paused',
    reasons: {
      paused: 'Take your time. Everything you did so far is saved.',
      hidden: 'The app went to the background, so the movement in progress was stopped.',
      stalled: 'The motion sensors stopped sending data. Hold the phone still for a moment, then tap Continue.',
      idle: 'Nothing happened for a while. Stretch your arm out, hold still and wait for the beep.',
      failed_twice: 'That did not work twice in a row. Read the instruction on the screen again, then tap Continue.',
      save_failed: 'The last movement could not be saved on the phone. Tap Continue to try again.',
      error: 'Something went wrong while recording. Tap Continue to try again.',
    },
    continue: 'Continue',
    redo: 'Redo last movement',
    wrongHand: 'Last one was the wrong hand',
    finish: 'Finish now',
    later: 'Continue later',
    laterToast: 'Session paused. You can continue later from the start screen.',
    kept: 'Everything you recorded is kept, also if you finish now.',
  },

  // After the main part (tier 3), before the optional bonus blocks.
  checkpoint: {
    title: 'Main part done — thank you!',
    body: (n: number) => (n === 1 ? 'You recorded 1 movement.' : `You recorded ${n} movements.`),
    bonusHint: 'If you have a little more time, the bonus adds a few more movements. Both are fine.',
    finish: 'Finish',
    bonus: (minutes: number) => (minutes <= 1 ? 'Bonus: about 1 more minute' : `Bonus: about ${minutes} more minutes`),
  },

  // Speech (only where the platform allows it: never outside a tap on iOS; silent on computers).
  speech: {
    speeds: { natural: 'Normal', fast: 'Fast', slow: 'Slow' } as Record<Speed, string>,
    grips: {
      upright_toward: 'Upright, screen facing you',
      upright_away: 'Upright, screen facing away',
      top_left_toward: 'Sideways, top of the phone to your left',
      top_right_toward: 'Sideways, top of the phone to your right',
      upside_down_toward: 'Upside down, screen facing you',
      upside_down_away: 'Upside down, screen facing away',
      top_left_away: 'Sideways, top to your left, screen facing away',
      top_right_away: 'Sideways, top to your right, screen facing away',
      flat_up: 'Flat, screen up',
      free: 'Any way you like',
    } as Record<OrientationId, string>,
  },

  review: {
    heading: 'Recorded!',
    duration: (seconds: string) => `Movement time: ${seconds} s`,
    noDuration: 'The movement time could not be measured.',
    plotAria: 'Rotation speed and acceleration during the trial, with the beep and the start and end of the movement marked.',
    rotation: (peak: string) => `rotation (peak ${peak} °/s)`,
    acceleration: (peak: string) => `acceleration (peak ${peak} m/s²)`,
    markers: { go: 'beep', onset: 'start', end: 'end', touch: 'touch' },
    orientationMismatch: (expected: string, detected: string) =>
      `The phone seemed to be held ${detected} instead of ${expected}. If you held it differently from the picture, tap “Problem” and choose “I held the phone the wrong way”.`,
    question: 'Did you do it as instructed?',
    next: 'Yes, next',
    redo: 'Redo',
    problem: 'Problem',
    flagTitle: 'What went wrong?',
    flags: {
      wrong_hand: 'I used the wrong hand',
      wrong_orientation: 'I held the phone the wrong way',
      other_problem: 'Something else went wrong',
    } as Record<ParticipantFlag, string>,
    flagCancel: 'Never mind',
    details: (n: number, hz: string) => `${n} ${n === 1 ? 'sample' : 'samples'} · ${hz} Hz`,
  },

  rotations: {
    upright: 'upright',
    upside_down: 'upside down',
    top_left: 'sideways with the top to the left',
    top_right: 'sideways with the top to the right',
    flat: 'flat',
  } as Record<PhoneRotation, string>,

  failed: {
    heading: 'Let’s try that one again',
    reasons: {
      no_movement: 'No movement was detected after the beep. Start moving as soon as you hear it.',
      too_long: 'The movement did not finish in time. Bring the phone to your nose and keep it there until the double beep.',
      aborted: 'You stopped the trial.',
      interrupted: 'The trial was interrupted: the screen turned off or the app went to the background.',
      sensor_stalled: 'The motion sensors stopped sending data. If this keeps happening, close the app and open the link again.',
      error: 'Something went wrong while recording.',
    },
    kept: 'This attempt is kept for quality control but does not count.',
    retry: 'Try again',
  },

  breakScreen: {
    heading: (i: number, n: number) => `Block ${i} of ${n} done!`,
    cheers: ['Great job!', 'Nicely done!', 'You are doing great!', 'Excellent, keep it up!', 'Well done!'],
    rest: 'Lower your arm, shake it out and rest for a moment.',
    next: (hand: string, orientation: string) => `Next: ${hand}, ${orientation}`,
    continue: 'Continue',
    pauseExit: 'Pause & exit',
  },

  done: {
    title: 'Thank you!',
    completed: 'You finished the session. Thank you for helping our research!',
    abandoned: 'The session ended early. Thank you: everything you recorded is still useful.',
    endedEarly: 'You finished early. Thank you: everything you recorded is kept and useful.',
    noSession: 'Thank you for taking part!',
    thanksName: (name: string) => `Thank you, ${name}!`,
    movements: (n: number) => (n === 1 ? '1 movement recorded' : `${n} movements recorded`),
    summary: 'Your session',
    accepted: 'Movements recorded',
    attempts: 'Attempts (including redos)',
    duration: 'Time',
    upload: 'Sending your data',
    allSent: 'All your data has been sent to the researchers.',
    uploadedAll: 'All data uploaded ✓',
    uploadingLeft: (k: number) => `Uploading… please keep this page open a moment (${k} left)`,
    offline: 'You are offline. Your data is saved on this phone and is sent the next time the app is open and online.',
    pending: (n: number) => (n === 1 ? '1 item is waiting to be sent.' : `${n} items are waiting to be sent.`),
    failed: (n: number) =>
      n === 1
        ? '1 item could not be sent yet. The app retries while it is open and online; if it keeps failing, “My data” on the start screen shows why.'
        : `${n} items could not be sent yet. The app retries while it is open and online; if they keep failing, “My data” on the start screen shows why.`,
    keepOpen: 'Please keep the app open and stay online until everything is sent.',
    notConfigured: 'This study collects data as a file: tap “Save / share my data” and send the file to the researcher.',
    uploadNow: 'Upload now',
    uploading: 'Uploading…',
    uploadResult: (sent: number, failed: number) =>
      failed > 0 ? `Sent ${sent}, ${failed} not sent yet. See “My data” on the start screen for details.` : `Sent ${sent}.`,
    nothingToUpload: 'Nothing left to upload.',
    exportData: 'Save / share my data',
    exporting: 'Preparing your file…',
    exported: {
      shared: 'Shared.',
      downloaded: 'The file was saved to your downloads.',
      cancelled: 'Sharing was cancelled.',
    },
    exportFailed: 'The data file could not be created.',
    code: (id: string) => `Your code: ${id} (to delete your data later)`,
    nextPerson: 'Next person',
    home: 'Home',
  },

  sandbox: {
    title: 'Sensor playground',
    intro: 'Researcher tool: live sensor signals and the online movement detector. Nothing is saved.',
    start: 'Start',
    stop: 'Stop',
    rate: 'Sample rate',
    samples: 'Samples',
    accSource: 'Linear acceleration',
    detector: 'Detector',
    accSources: {
      linear: 'reported by the phone',
      highpass: 'estimated (high-pass)',
      none: 'not available',
    },
    states: {
      idle: 'stopped',
      waiting_still: 'waiting for stillness',
      still: 'still, beep coming',
      armed: 'GO given, move now',
      moving: 'movement detected',
      ended: 'movement ended',
      failed: 'no valid movement',
    },
    gyroPlot: 'Rotation speed |ω| (°/s, smoothed)',
    accPlot: 'Acceleration (m/s²)',
    series: { gyro: '|ω|', acc: '|a| without gravity', gx: 'gravity x', gy: 'gravity y', gz: 'gravity z' },
    thresholds: { still: 'still', onset: 'onset' },
    markers: { go: 'GO', onset: 'on', end: 'end' },
    movements: 'Detected movements',
    noMovements: 'None yet. Hold the phone still, wait for the beep, then bring it to your nose.',
    movementRow: (i: number, ms: number, peak: string) => `#${i}: ${ms} ms, peak ${peak} °/s`,
    noMovement: 'no movement after the beep',
    tooLong: 'movement too long',
    permissionDenied: 'Motion sensor access was denied.',
    errors: {
      not_supported: 'This browser does not provide motion sensors.',
      permission_denied: 'Motion sensor access was denied.',
      insecure_context: 'Motion sensors need a secure (https) page.',
      no_events: 'No motion events arrive. On a computer, open the app with ?sim=1 to use the simulator.',
      sensor_error: 'The motion sensors reported an error.',
    } as Record<SensorError['code'], string>,
  },

  figure: {
    top: 'top',
    screenUp: 'screen up',
  },
};
