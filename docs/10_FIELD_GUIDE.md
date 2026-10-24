# 10 · Field guide: running sessions

One page for the researcher who runs NoseTouch visits with friends, on the street or through a WhatsApp link. Each visit takes about 5 minutes on the friend's own phone, and every movement uploads as soon as it is recorded. Background: [01 Protocol](01_PROTOCOL.md) (what the app does), [06 Considerations](06_CONSIDERATIONS.md) (consent and privacy), [backend/heroku](../backend/heroku/README.md) (the upload service).

## The link

- **<https://axelfridman.github.io/nosetouch-study/>**, or `…/nosetouch-study/?lang=he` to open in Hebrew.
- Never share a link with `?sim=1` (simulated data, no sensors) or `?setup=1` (the researchers' setup screen), and never a tunnel address.
- **QR code.** Make one of the link and keep it as a photo on your phone or on a printed card. A friend scans it with the phone's camera, which opens the link in the phone's own browser, not inside a chat app. Chrome on a computer makes one (⋮ menu › Cast, save, and share › Create QR Code), or offline: `analysis\.venv\Scripts\python.exe -m pip install segno`, then `analysis\.venv\Scripts\python.exe -c "import segno; segno.make('https://axelfridman.github.io/nosetouch-study/', error='m').save('nosetouch_qr.png', scale=12, border=4)"`.

## What to say

> Do you have 5 minutes to help with my study for the lab I'm applying to? You hold your phone at arm's length and touch your nose with it about 30 times, with both hands; the app records only the phone's motion sensors (no camera, microphone or location), and you can stop at any time.

> יש לך 5 דקות לעזור לי במחקר שלי לקראת הקבלה למעבדה? מחזיקים את הטלפון ביד מתוחה ונוגעים איתו באף בערך 30 פעמים, בשתי הידיים; האפליקציה מקליטה רק את חיישני התנועה של הטלפון (בלי מצלמה, מיקרופון או מיקום), ואפשר להפסיק בכל רגע.

In WhatsApp, add: "Open it in Safari (iPhone) or Chrome (Android): https://axelfridman.github.io/nosetouch-study/". Participants must be 18 or older; the app's first screen holds the consent and the checkbox.

## Browser

- **iPhone: Safari. Android: Chrome** (Samsung Internet works too). Not Brave, which blocks the motion sensors.
- **Leave WhatsApp's in-app browser.** If the page opens inside WhatsApp (or Instagram, Facebook, …), with the app's own bar at the top instead of the browser's address bar, tap its menu (⋮ or ⋯) and choose "Open in Chrome", "Open in browser" or "Open in Safari". Or long-press the link in the chat, Copy, and paste it into Safari or Chrome. The welcome screen recognises these in-app browsers (WhatsApp, Instagram, Facebook, TikTok and others) and asks the friend to switch, with a "Copy the link" button; on an iPhone, Start then reads "Continue here anyway": do not use it on a visit.
- Volume up (the How-to card has "Play a test beep"); Do Not Disturb on (a call pauses the visit); Low Power Mode or Battery Saver off; a free arm's length around them, away from traffic. Standing is the default; a seated friend ticks "I’m sitting" on the How-to card.

## During the visit

1. Welcome: tick the box, tap **Start**. An iPhone asks for motion access: **Allow**.
2. About you (name optional, age, sex, dominant hand, phone hand, height optional) › Continue.
3. How-to card ("Play a test beep" checks the volume) › "Try it once" with any hand › "Which hand did you use?".
4. Each part starts with one tap on "Start — phone in LEFT hand" (tapped with the free hand). Then the movements run by themselves: arm out and still, beep, touch the nose, hold until the double beep, arm out again.
5. Let the app instruct; do not demonstrate a path or a speed.
6. **Wrong hand or a slip:** press "Hold to pause" for about a second, then "Last one was the wrong hand" or "Redo last movement". **To stop early:** "Hold to pause" › "Finish now" › "Tap again to finish". Nothing recorded is lost.
7. "Main part done": **Finish**, or the bonus (about 1 more minute).
8. Done screen: wait for "All data uploaded ✓ — you can close this page." "Uploading… please keep this page open a moment" means a few seconds more; "This phone is offline" means the data wait on the phone until the page is open again online. The screen shows the friend's code (P-XXXXXX), needed for deletion later: suggest a screenshot. "Send the link to a friend" (where the browser can share) opens the phone's share sheet; "Someone else on this phone" starts the next person without the first one's name or profile.

A call, switching apps or a locked screen pauses the session: tap **Continue**. "Exit for now" in the pause sheet leaves the visit unfinished: opening the link again within 24 hours shows "Resume session" ("Start over" begins a new visit and closes the old one), and a visit left longer is closed as `abandoned` the next time the app opens on that phone. Its recorded movements are uploaded either way.

## If an iPhone denied motion access

After "About you" a "Sensor check" screen says "Motion access is blocked". Safari remembers the refusal until it is fully closed:
1. Open the app switcher (swipe up from the bottom and hold, or double-press the Home button) and swipe Safari away.
2. Open the link again (WhatsApp or the QR code).
3. The welcome screen says "Continue as {name}": tick the box, tap **Start**, tap **Allow** this time, then Continue on the filled-in profile.

Chrome, Firefox or Edge on an iPhone: close that browser the same way. A Home Screen icon: close it and reopen it from the Home Screen. Inside WhatsApp or Instagram: open the link in Safari. Never switch to a Private tab (new storage, so a new code and a lost session), and do not delete website data (no reliable reset, and it deletes unsent data). Android: tap the icon left of the address bar › Permissions (or Site settings) › Motion sensors › Allow, then "Measure again".

## Check uploads live

1. Open **<https://nosetouch-study-14d4b1cd9830.herokuapp.com/admin/login>** on your laptop or your own phone.
2. Open `%USERPROFILE%\.nosetouch\heroku.env` (`notepad "$env:USERPROFILE\.nosetouch\heroku.env"`), copy the value after `ADMIN_TOKEN=` and paste it into the form. The browser then stays signed in for 7 days with a cookie that does not hold the token.
3. The page refreshes every 15 s. **Recent trials**: each movement within seconds, `Accepted` yes. **Recent sessions**: `in_progress`, then `completed` (about 29/39 accepted/planned without the bonus) or `ended_early`; `abandoned` marks an unfinished visit that was replaced or left. **Participants**: name, platform, left/right counts. **Refused or ignored uploads**: a `rejected` row needs attention (`bad_study_key` means a phone with an old page: reload it and tap "Upload now"); a `kept_newer` row is an older copy that arrived late and is harmless.

**Never share the token.** No chat, e-mail, screenshot or shared URL (`/admin/login?token=…` lands in Heroku's router logs); never sign in on a friend's phone; carry it to your phone with a password manager. If it may have leaked: `heroku config:set ADMIN_TOKEN=<new> -a <app>` (signs every browser out), then update `heroku.env`.

## Pull the data

From the repository root (PowerShell):

```powershell
analysis\.venv\Scripts\python.exe analysis\scripts\pull_live.py --real-only
analysis\.venv\Scripts\python.exe analysis\scripts\qa_report.py analysis\data\live --simulated exclude
```

`pull_live.py` reads `APP_URL` and `ADMIN_TOKEN` from `heroku.env`, replaces `analysis\data\live\` (git-ignored) with the server's current state and prints the counts per participant; `--participant P-XXXXXX --out analysis\data\live_P-XXXXXX` pulls one person into a folder of its own. The QA report goes to `analysis\reports\qa_live\`. Copy a pull to a dated folder (`analysis\data\real\<date>\`) before analysing it. The raw records contain names: never commit or share them.

## Delete a participant on request

1. Find them: the code from their Done screen, or their name in the admin page's **Participants** table.
2. Admin page › **Delete a participant** › the code › Delete › confirm. This removes the participant, their sessions, their trials and their upload-log lines. (The API call is in [backend/heroku](../backend/heroku/README.md#deletion-on-request).)
3. Ask them to delete it on their phone too: welcome screen › **My data** › "Delete this participant and their data". Records still queued there would otherwise upload again.
4. Pull again (`pull_live.py` replaces `analysis\data\live\`) and delete their folder from every snapshot and derived copy.
5. Note the date and the code, nothing else. Database backups keep a copy until they expire (on Essential plans 7 days for the daily ones, longer for the weekly one).

**On collection days:** no pushes to `main` that touch `app/` (they redeploy the app), phones charged, and the admin page open.
