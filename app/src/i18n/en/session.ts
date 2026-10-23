// Strings for the session runner, the done screen and the researcher sandbox. OWNED BY the session UI
// agent. Hebrew mirror with the same keys: ../he/session.ts. Protocol wording (hands, orientations,
// speeds, start positions, variants, safety) lives in common.ts and is not repeated here.
import type { ParticipantFlag, PhoneRotation, SensorError, TrialMessage } from '../../core/types';

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

  exit: {
    open: 'Pause',
    title: 'Take a break?',
    body: 'You can pause now and continue later from the start screen, or end the session here. Everything recorded so far is kept.',
    pause: 'Pause, continue later',
    end: 'End the session now',
    cancel: 'Keep going',
    paused: 'Session paused. You can continue later from the start screen.',
  },

  intro: {
    kicker: 'Next block',
    newHand: 'Switch hands!',
    odd: 'In this block every movement is a little different and unusual. Read each instruction before you tap Ready.',
    free: 'In this block, hold the phone in whatever way feels natural to you.',
    start: 'Start',
  },

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
        'The phone did not allow access to its motion sensors. Tap the icon to the left of the address bar › Permissions (or Site settings) › Motion sensors › Allow, then tap Ready again.',
      other:
        'The phone did not allow access to its motion sensors. Allow motion sensors for this site in the browser settings, then tap Ready again.',
    },
  },

  live: {
    starting: 'Starting the sensors…',
    messages: {
      hold_still: 'Hold still',
      too_early: 'Too early! Wait for the beep',
      go: 'GO!',
      moving: 'Touch your nose',
      hold_at_nose: 'Hold it there…',
      done: 'Done!',
      no_movement: 'No movement detected',
      too_long: 'That took too long',
      sensor_stalled: 'The motion sensors stopped',
    } as Record<TrialMessage, string>,
    waitBeep: 'Arm straight, keep still and wait for the beep',
    stop: 'Hold to stop',
    stopAria: 'Press and hold for about one second to stop this trial',
    sim: 'SIM',
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
    noSession: 'Thank you for taking part!',
    summary: 'Your session',
    accepted: 'Movements recorded',
    attempts: 'Attempts (including redos)',
    duration: 'Time',
    upload: 'Sending your data',
    allSent: 'All your data has been sent to the researchers.',
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
