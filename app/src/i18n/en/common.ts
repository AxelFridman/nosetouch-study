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

  startPositions: {
    natural: { label: 'Arm straight in front', how: 'Stretch your arm straight out in front of you, at shoulder height.' },
    shoulder: { label: 'In line with your shoulder', how: 'Arm fully straight, hand directly in front of the same shoulder.' },
    center: { label: 'In front of your face', how: 'Arm fully straight, phone centered in front of your nose.' },
  } as Record<StartPosition, { label: string; how: string }>,

  variants: {
    standard: { label: 'Normal movement', how: 'Bring the phone straight to your nose.' },
    curved: { label: 'Curved path', how: 'Take a curved path: swing out to the side first, then to your nose.' },
    pause: { label: 'Pause halfway', how: 'Stop halfway for about one second, then continue to your nose.' },
    twist: { label: 'Twist while moving', how: 'Rotate (twist) the phone in your hand while bringing it to your nose.' },
    elbow_high: { label: 'Elbow up', how: 'Raise your elbow high out to the side as you bring the phone in.' },
    bent_start: { label: 'Bent elbow start', how: 'Start with your elbow slightly bent instead of fully straight.' },
    lean_in: { label: 'Lean in', how: 'Lean your head forward to meet the phone halfway.' },
    chin: { label: 'Touch your chin', how: 'This time touch your CHIN instead of your nose.' },
    forehead: { label: 'Touch your forehead', how: 'This time touch your FOREHEAD instead of your nose.' },
    jerky: { label: 'Jerky movement', how: 'Move to your nose in a jerky, stop-and-go way.' },
    tilted_touch: {
      label: 'Tilted touch',
      how: 'Touch your nose with the top edge of the phone: tilt it as you arrive.',
    },
  } as Record<Variant, { label: string; how: string }>,

  postures: {
    standing: 'Standing',
    sitting: 'Sitting',
  } as Record<Posture, string>,

  // The double beep (end cue) means "arm out again": in the hands-free flow the next trial starts by itself.
  movementSummary:
    'Stretch your arm out and hold still. At the beep, bring the phone to gently touch your nose (any part of the phone may touch it) and hold it there until the double beep, then stretch your arm out again.',
  safety: 'Hold the phone firmly (doing this over a bed or sofa is a good idea) and always touch your nose gently.',
};
