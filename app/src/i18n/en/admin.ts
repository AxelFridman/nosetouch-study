// Strings of the admin screens: My data, Settings, About. Functions are allowed for interpolation.
// Hebrew mirror: ../he/admin.ts (identical keys and array lengths; tests/i18n.test.ts checks this).
import type { SessionRecord, UploadMode } from '../../core/types';

export const admin = {
  /** BCP 47 locale used to format dates and numbers on these screens. */
  locale: 'en-GB',

  data: {
    title: 'My data',
    intro: 'Everything recorded on this phone. You can export it, upload it or delete it at any time.',
    sessionsTitle: 'Sessions',
    noSessions: 'No sessions recorded on this phone yet.',
    movements: (accepted: number, planned: number) =>
      `${accepted} of ${planned} ${planned === 1 ? 'movement' : 'movements'}`,
    status: {
      in_progress: 'Unfinished',
      completed: 'Completed',
      abandoned: 'Stopped early',
    } as Record<SessionRecord['status'], string>,
    simulated: 'Simulated',
    uploadState: {
      off: 'Upload off',
      none: 'Not queued for upload',
      done: 'Uploaded ✓',
      partial: (sent: number, total: number) => `Uploading: ${sent} of ${total} sent`,
      failed: (n: number) => `${n} not sent yet (will retry)`,
      refused: (n: number) => `${n} refused by the server`,
    },
    exportSession: 'Export',
    deleteSession: 'Delete',
    confirmDeleteSession: (when: string) => `Delete the session of ${when} from this phone? This cannot be undone.`,
    deletedSession: 'The session was deleted from this phone.',
    exportAll: 'Export all data',
    uploadNow: 'Upload now',
    busy: 'Working…',
    exportShared: 'File shared.',
    exportDownloaded: 'The file was saved to your downloads.',
    exportCancelled: 'Export cancelled.',
    exportFailed: 'The export did not work. Please try again.',
    uploadOff:
      'Uploading is not set up for this study. Use "Export all data" and send the file to the research team.',
    uploadNothing: 'Nothing to upload: everything has been sent.',
    uploadDone: (n: number) => `${n} ${n === 1 ? 'item' : 'items'} uploaded.`,
    uploadSomeFailed: (sent: number, failed: number) =>
      `${sent} uploaded, ${failed} not sent. It will retry automatically.`,
    uploadFailed: 'Upload failed. Check the internet connection; it will retry automatically.',
    // Items the server refused (e.g. the study key was changed since this page was loaded). The app
    // does not retry them on its own.
    refused: (n: number) =>
      `The server refused ${n === 1 ? '1 item' : `${n} items`}. Reload this page so that the latest settings are used, then tap "Upload now". If ${n === 1 ? 'it is' : 'they are'} refused again, use "Export all data" and send the file to the research team.`,
    uploadRefused: (n: number) =>
      `${n === 1 ? '1 item was' : `${n} items were`} refused by the server. Reload this page, then tap "Upload now" again.`,
    queueTitle: 'Upload queue',
    queue: (pending: number, failed: number, sent: number, refused: number) =>
      `${sent} sent · ${pending} waiting · ${failed} to retry${refused > 0 ? ` · ${refused} refused` : ''}`,
    storageTitle: 'Storage on this phone',
    storageUsed: (used: string, quota: string) => `${used} used (about ${quota} available)`,
    participantsTitle: 'Participants on this phone',
    noParticipants: 'No participant profiles on this phone.',
    since: (when: string) => `Since ${when}`,
    sessionsCount: (n: number) => `${n} ${n === 1 ? 'session' : 'sessions'}`,
    deleteParticipant: 'Delete this participant and their data',
    // n is 0 for a profile that never recorded a session ("all 0 of their sessions" otherwise).
    confirmDeleteParticipant: (code: string, n: number) =>
      `Delete participant ${code}${n === 1 ? ' and their session' : n > 1 ? ` and all ${n} of their sessions` : ''} from this phone? This cannot be undone.`,
    deletedParticipant: 'The participant was deleted from this phone.',
    uploadedNote:
      'Copies that were already uploaded are not deleted from here. To have them deleted, contact the research team and quote your participant code.',
    contact: 'Contact:',
    storageTip:
      'Tip: Safari can delete the data of websites that were not visited for 7 days. Upload or export your data soon, or add this app to your home screen.',
  },

  settings: {
    title: 'Settings',
    language: 'Language',
    feedbackTitle: 'Sounds and feedback',
    audio: 'Beeps',
    audioHint: 'A beep starts each movement and a double beep ends it.',
    speech: 'Spoken instructions',
    speechHint: 'Reads the instructions aloud when a new block starts.',
    vibrate: 'Vibrate when a new block starts',
    vibrateHint: 'Android only. Never during a movement: the vibration would disturb the sensors.',
    testingTitle: 'Testing',
    simulate: 'Simulation mode (testing only)',
    simulateHint:
      'A built-in simulated participant replaces the real sensors. Never use it for real data. It switches off when the app is reloaded.',
    simulateLocked: 'Cannot be changed while a session is in progress.',
    researcherTitle: 'For researchers: upload',
    researcherHint: 'Overrides the upload settings of the study (config.json) on this phone only.',
    uploadMode: 'Upload mode',
    uploadModes: {
      apps_script: { label: 'Google Apps Script', hint: 'The free backend: the /exec URL of the web app deployment.' },
      json_post: { label: 'JSON POST', hint: 'Any HTTPS endpoint that accepts JSON and answers CORS requests.' },
      none: { label: 'No upload', hint: 'Data stays on the phone; use Export.' },
    } as Record<UploadMode, { label: string; hint: string }>,
    endpoint: 'Endpoint URL',
    endpointHint: 'For example https://script.google.com/macros/s/…/exec',
    studyKey: 'Study key',
    studyKeyHint: 'A shared key checked by the backend (a spam filter, not a password).',
    save: 'Save upload settings',
    saved: 'Upload settings saved.',
    useDefaults: 'Use the study defaults',
    defaultsRestored: 'Using the upload settings of the study.',
    testUpload: 'Test connection',
    testing: 'Testing…',
    errEndpoint: 'The address must start with https:// (or be a path on this site, such as /api/ingest).',
    errEndpointRequired: 'Enter the endpoint address.',
    warnAppsScript: 'This does not look like an Apps Script web app address (…/macros/s/…/exec).',
    test: {
      ok: 'Connection OK: the backend answered.',
      okInfo: (info: string) => `Connection OK: ${info}`,
      notOk: (message: string) => `The backend answered with an error: ${message}`,
      notJson: (status: number) =>
        `A server answered (HTTP ${status}), but not like the study backend. Check the address.`,
      http: (status: number) => `The server answered with HTTP ${status}.`,
      network:
        'Could not reach the backend. For Apps Script this usually means that the deployment is not shared with "Anyone", was deleted or must be authorised again — or that you are offline.',
      timeout: 'The backend did not answer in time.',
      noEndpoint: 'No endpoint is set.',
    },
    effectiveTitle: 'Active configuration',
    studyCode: 'Study code',
    mode: 'Upload mode',
    endpointShort: 'Endpoint',
    key: 'Study key',
    keySet: (n: number) => `set (${n} ${n === 1 ? 'character' : 'characters'})`,
    override: 'Override on this phone',
    consentVersion: 'Consent version',
    appVersion: 'App version',
    schemaVersion: 'Data format version',
    protocolVersion: 'Protocol version',
    notSet: '—',
    resetTitle: 'Reset this device',
    resetHint:
      'Deletes every participant, session and recording stored on this phone. Copies that were already uploaded are not affected.',
    reset: 'Delete all study data on this phone',
    resetConfirm1: 'Delete ALL study data on this phone (every participant and every session)?',
    resetConfirm2: 'Are you sure? This cannot be undone.',
    resetDone: 'All study data on this phone was deleted.',
  },

  about: {
    title: 'About the study',
    whatTitle: 'What is this?',
    /** institution: StudyConfig.institution in this language ('' → not named) */
    what: (institution: string) =>
      `NoseTouch is a research app${institution ? ` of ${institution}` : ''}. It records how a phone moves while people bring it from an outstretched arm to their nose, with the right and with the left hand. The goal is to find out whether the phone itself can tell which hand is holding it.`,
    stagesTitle: 'How the research works',
    stages: [
      'Collect recordings from many people, phones and ways of holding the phone.',
      'Build and test a model that tells the left hand from the right one.',
      'Run that model live in this app.',
    ],
    dataTitle: 'Your data',
    data: [
      'Pseudonymous: you are identified only by a random participant code.',
      'Only the motion sensors and screen touches during the movements: never the camera, the microphone, your location or your contacts.',
      'Stored on this phone and, when uploading is set up, in the study storage of the research team.',
      'You can export or delete the data on this phone at any time on the "My data" screen.',
    ],
    contactTitle: 'Contact',
    contactNone: 'Contact the researcher who sent you the link.',
    installTitle: 'Install as an app (optional)',
    installIos: 'iPhone: tap Share › Add to Home Screen.',
    installAndroid: 'Android: tap ⋮ › Add to Home screen (or Install app).',
    installWhy: 'An app on the home screen keeps its data even when the browser clears old website data.',
    sandboxTitle: 'Sensor playground',
    sandbox: 'See the live sensor signals and practise the movement without recording anything.',
    openSandbox: 'Open the playground',
    versionsTitle: 'Versions',
    studyCode: 'Study code',
  },
};
