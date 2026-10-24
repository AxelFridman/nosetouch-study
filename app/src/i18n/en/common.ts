import type { Hand, OrientationId, Posture, Speed, StartPosition, Variant } from '../../core/types';

/** Shared vocabulary + the protocol wording (the exact instructions participants follow). */
export const common = {
  appName: 'NoseTouch Study',
  continue: 'Continue',
  back: 'Back',
  cancel: 'Cancel',
  save: 'Save',
  start: 'Start',
  retry: 'Try again',
  skip: 'Skip',
  yes: 'Yes',
  no: 'No',
  done: 'Done',
  next: 'Next',
  close: 'Close',
  loading: 'Loading…',
  optional: 'optional',
  minutes: (n: number) => `${n} min`,
  trials: (n: number) => (n === 1 ? '1 trial' : `${n} trials`),
  errorGeneric: 'Something went wrong. Please try again.',
  /** The app could not start (storage blocked); `detail` is the browser's own message, shown left-to-right. */
  bootFailed: (detail: string) =>
    `The app could not start on this browser: ${detail}. Please open the link in Safari (iPhone) or Chrome (Android), not inside another app, and not in a private window.`,
  simulatedBadge: 'SIMULATION',

  hands: {
    left: 'Left hand',
    right: 'Right hand',
  } as Record<Hand, string>,
  handShort: {
    left: 'LEFT',
    right: 'RIGHT',
  } as Record<Hand, string>,
  holdWith: (hand: Hand) => `Hold the phone in your ${hand === 'left' ? 'LEFT' : 'RIGHT'} hand`,

  orientations: {
    upright_toward: {
      label: 'Upright, screen facing you',
      how: 'Hold the phone upright (top edge up) with the screen facing you, as if reading it.',
    },
    upright_away: {
      label: 'Upright, screen facing away',
      how: 'Hold the phone upright with the screen facing away from you (the main camera looks at you).',
    },
    top_left_toward: {
      label: 'Sideways, top to the left, screen facing you',
      how: 'Turn the phone sideways so its top edge points to your LEFT, screen facing you.',
    },
    top_right_toward: {
      label: 'Sideways, top to the right, screen facing you',
      how: 'Turn the phone sideways so its top edge points to your RIGHT, screen facing you.',
    },
    upside_down_toward: {
      label: 'Upside down, screen facing you',
      how: 'Hold the phone upside down (top edge pointing down), screen facing you.',
    },
    upside_down_away: {
      label: 'Upside down, screen facing away',
      how: 'Hold the phone upside down with the screen facing away from you.',
    },
    top_left_away: {
      label: 'Sideways, top to the left, screen facing away',
      how: 'Turn the phone sideways with its top edge to your LEFT and the screen facing away from you.',
    },
    top_right_away: {
      label: 'Sideways, top to the right, screen facing away',
      how: 'Turn the phone sideways with its top edge to your RIGHT and the screen facing away from you.',
    },
    flat_up: {
      label: 'Flat, screen up',
      how: 'Hold the phone flat like a tray, screen facing the ceiling.',
    },
    free: {
      label: 'Your choice',
      how: 'Hold the phone however feels most natural to you.',
    },
  } as Record<OrientationId, { label: string; how: string }>,

  speeds: {
    natural: { label: 'Natural speed', how: 'Move at your normal, comfortable pace.' },
    fast: { label: 'Fast', how: 'Move quickly, as fast as you comfortably and safely can. Touch your nose gently!' },
    slow: { label: 'Slow', how: 'Move slowly and smoothly, taking about 3 seconds.' },
  } as Record<Speed, { label: string; how: string }>,

  // label: 1-3 words, shown big on the live panel and said aloud; how: one short line, read at arm's length
  startPositions: {
    natural: { label: 'Arm straight in front', how: 'Stretch your arm straight out in front of you, at shoulder height.' },
    shoulder: { label: 'Side start', how: 'Arm straight, phone in front of that shoulder' },
    center: { label: 'Middle start', how: 'Arm straight, phone right in front of your nose' },
  } as Record<StartPosition, { label: string; how: string }>,

  variants: {
    standard: { label: 'Normal movement', how: 'Bring the phone straight to your nose.' },
    curved: { label: 'Curve', how: 'Swing out to the side, then in' },
    pause: { label: 'Stop halfway', how: 'Stop 1 second halfway, then go on' },
    twist: { label: 'Twist', how: 'Twist the phone on the way in' },
    elbow_high: { label: 'Elbow up', how: 'Lift your elbow out to the side' },
    bent_start: { label: 'Elbow bent', how: 'Start with your elbow a little bent' },
    lean_in: { label: 'Lean in', how: 'Move your head to meet the phone' },
    chin: { label: 'Touch your chin', how: 'This time touch your CHIN instead of your nose.' },
    forehead: { label: 'Touch your forehead', how: 'This time touch your FOREHEAD instead of your nose.' },
    jerky: { label: 'Stop-and-go', how: 'Stop and go a few times on the way' },
    tilted_touch: { label: 'Top edge', how: 'Tilt it so the top edge touches your nose' },
  } as Record<Variant, { label: string; how: string }>,

  postures: {
    standing: 'Standing',
    sitting: 'Sitting',
  } as Record<Posture, string>,

  // The double beep (end cue) means "arm out again": in the hands-free flow the next trial starts by itself.
  movementSummary:
    'Stretch your arm out and hold still. At the beep, bring the phone to gently touch your nose (any part of the phone may touch it) and hold it there until the double beep, then stretch your arm out again.',
  safety: 'Hold the phone firmly and always touch your nose gently.',
};
