# 03 Sensors and platforms

What a phone browser actually delivers on iOS and Android, how the app measures each device's conventions (calibration), the exact normalisation rules the analysis must apply, the online quality checks, and when a native or high-rate tool is worth it.

Most platform facts below come from reading the current WebKit, Chromium and Gecko source code, the W3C specifications and vendor documentation (October 2026). Source-code readings have not all been confirmed on real phones yet; [section 8](#8-pilot-verification-checklist) lists what the pilot must verify.

Related documents: [Data schema](02_DATA_SCHEMA.md) (field definitions), [Segmentation](04_SEGMENTATION.md), [Modeling plan](05_MODELING_PLAN.md), [Stage 3 predictor](08_STAGE3_PREDICTOR.md).

## 1. Summary

1. **About 60 Hz everywhere.** iOS polls Core Motion with a 1/60 s timer, Chromium pumps `devicemotion` at 60 Hz and caps the Generic Sensor API at 60 Hz, and Firefox registers its sensors at 60 Hz. A web page cannot get more. For this task 60 Hz is adequate ([section 7.1](#71-is-60-hz-enough)); higher rates need a native app.
2. **iOS inverts the accelerometer.** Every iOS browser is WebKit (outside the EU, where Apple allows other engines since iOS 17.4; the app detects the engine instead of assuming it), and WebKit reports `acceleration` and `accelerationIncludingGravity` with all three axes negated relative to W3C and Android (face-up z ≈ −9.81 instead of +9.81). `rotationRate` is **not** inverted. A sign flip of the accelerometer alone reverses every chirality feature, so an un-normalised iPhone right-hand trial looks like an Android left-hand trial. The sign is therefore measured per device ([section 4](#4-calibration-procedure)) and normalised offline ([section 5](#5-normalisation-layer-for-analysis)).
3. **`rotationRate.alpha/beta/gamma` are the rates about device x/y/z.** This is the W3C definition since 2019, followed by WebKit, Chromium and Gecko. MDN's `DeviceMotionEventRotationRate.alpha` page ("around the Z axis") is outdated, and `deviceorientation` angles use a different convention.
4. **`event.interval` is not comparable across platforms**: about 0.0167 (seconds) on iOS, a constant 16 (ms) on Chrome, a constant 100 on Firefox. It is stored as metadata and never used for timing.
5. **Chrome quantises, iOS does not.** Chromium rounds readings to 0.1 m/s², 0.1 deg/s and 0.1°; WebKit passes full doubles. A model can learn the platform from this, so the analysis can quantise iOS data to the same grid; an ablation decides whether that becomes the default ([section 5.4](#54-platform-harmonisation)).
6. **Permission.** iOS shows the motion prompt only during a tap, and a Cancel sticks until Safari is force-quit. Chrome 151+ also ships `requestPermission()`, so the app calls it on every browser where it exists and never uses its presence to detect iOS.
7. **Foreground only.** All browsers stop motion events while the page is hidden (screen lock, app or tab switch). Trials whose movement spans a hide are marked (`quality.hidden_during`); a hide during the return of the arm only adds the warning `hidden_after_end`.
8. **Euler angles are unusable at the start pose.** The upright phone sits at beta ≈ 90°, the gimbal-lock point of the `deviceorientation` angles. They are recorded as metadata only.
9. **Phone storage is a buffer.** Safari can delete a site's data after 7 days without interaction, and iOS can drop IndexedDB connections. Every attempt is uploaded right away.

## 2. What the browser delivers

### 2.1 Engines

| Phone | Browsers | Engine (`device.engine`) |
|---|---|---|
| iPhone, iPad | Safari, Chrome, Firefox, Edge for iOS, in-app browsers (Instagram, Facebook, ...), Home Screen web apps | `webkit` |
| Android | Chrome, Samsung Internet, Edge, Opera, Android WebView (in-app browsers) | `blink` (Chromium) |
| Android | Firefox | `gecko` |
| Android | Brave | Chromium, but motion sensors are disabled ([Brave staff, 2021](https://community.brave.app/t/device-motion-data-accelerometer-gyro-not-available-on-android/292620)) |

The app sets `engine = 'webkit'` if and only if `'webkitCompassHeading' in DeviceOrientationEvent.prototype`, which exists only in iOS WebKit's IDL. It must never infer iOS from `typeof DeviceMotionEvent.requestPermission === 'function'`: Chrome 151+ has that function too ([Chrome 151 release notes](https://developer.chrome.com/release-notes/151)). Source code also suggests "check `requestPermission`" in places; that advice is outdated.

### 2.2 Platform comparison

| Property | iOS: WebKit (all iOS browsers) | Android: Chromium (Chrome, Samsung Internet, WebView) | Android: Firefox (Gecko) | Sources |
|---|---|---|---|---|
| **Motion data** | | | | |
| `devicemotion` rate | ~60 Hz. A 1/60 s timer reads the latest Core Motion sample, so occasional duplicated or skipped samples are expected | 60 Hz pump; each sensor runs at min(60 Hz, its maximum); first event only after all three sensors have reported | Sensors registered at 60 Hz; an event fires when all three have reported, or after 100 ms with the missing parts `null` | [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.h), [Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_sensor_event_pump.h), [Gecko](https://github.com/mozilla-firefox/firefox/blob/main/dom/system/nsDeviceSensors.cpp) |
| `event.timeStamp` | Event construction in the web process after an IPC hop, floored to 1 ms | Pump tick (dispatch time), not sensor time | Derived from the Android sensor timestamps | [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/dom/Event.cpp), [Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/core/dom/events/event.cc), [Gecko](https://github.com/mozilla-firefox/firefox/blob/main/dom/events/DeviceMotionEvent.cpp) |
| `event.interval` | ≈ 0.016667, i.e. **seconds** (the spec requires ms) | 16 (ms), constant | 100, constant | [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/DeviceMotionClientIOS.mm), [Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_motion_event_pump.cc), [Gecko](https://github.com/mozilla-firefox/firefox/blob/main/dom/system/nsDeviceSensors.cpp) |
| Value precision | Unrounded doubles | Rounded (anti-fingerprinting): acceleration 0.1 m/s², rotation 0.1 deg/s, Euler 0.1°, quaternion ≈ 0.1° | Not established by our sources | [Chromium](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h), [W3C](https://w3c.github.io/deviceorientation/) |
| Accelerometer sign (`acceleration`, `accelerationIncludingGravity`) | **All axes inverted**: face-up z ≈ −9.81. Re-measured on iPadOS 26.6.2 in Oct 2026 | W3C: face-up z ≈ +9.81 | W3C: +9.81 | [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm), [W3C list](https://lists.w3.org/Archives/Public/public-geolocation/2014Jul/0024.html), [measurement](https://github.com/soroushdty/gaitscope/issues/96), [Android](https://developer.android.com/develop/sensors-and-location/sensors/sensors_motion) |
| `acceleration` (gravity removed) | Core Motion `userAcceleration` (Apple's sensor fusion) | `TYPE_LINEAR_ACCELERATION`; without that sensor, a Chromium low-pass gravity filter that is wrong during rotations | Linear-acceleration sensor | [Chromium](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/linear_acceleration_fusion_algorithm_using_accelerometer.cc), [Gecko](https://github.com/mozilla-firefox/firefox/blob/main/mobile/android/geckoview/src/main/java/org/mozilla/gecko/GeckoSensor.java) |
| `rotationRate` | deg/s; alpha = x, beta = y, gamma = z; **not** inverted | deg/s (since Chrome 66; rad/s before); same mapping | deg/s; same mapping | [W3C PR #43](https://github.com/w3c/deviceorientation/pull/43), [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm), [Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_motion_event_pump.cc), [Gecko](https://github.com/mozilla-firefox/firefox/blob/main/mobile/android/geckoview/src/main/java/org/mozilla/gecko/GeckoSensor.java) |
| Missing sensor | Not relevant: all current iPhones have a gyroscope | Object present with `null` fields, e.g. `rotationRate` on gyro-less budget phones | `null` fields after the 100 ms timeout; rate drops to ~10 Hz (inferred) | [Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_motion_event_pump.cc), [Gecko](https://github.com/mozilla-firefox/firefox/blob/main/dom/system/nsDeviceSensors.cpp) |
| **Orientation** | | | | |
| `deviceorientation` | Z-X'-Y'' angles computed from the Core Motion attitude. Alpha has an arbitrary zero (`xArbitraryZVertical`), probably reset whenever motion updates restart. `webkitCompassHeading` gives the magnetic heading; no `absolute` flag, no `deviceorientationabsolute` | Relative (game rotation vector) since Chrome 50; fires only when an angle changes by ≥ 0.1°; `deviceorientationabsolute` exists | `deviceorientationabsolute` since Firefox 110 | [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm), [Apple](https://developer.apple.com/documentation/coremotion/cmattitudereferenceframe/xarbitraryzvertical), [Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_orientation_event_pump.cc), [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/DeviceOrientationEvent.json) |
| Gimbal lock | At beta = ±90° (upright portrait, the start pose) alpha and gamma are undefined; upright landscape sits at the gamma = ±90° wrap | same (spec convention) | same | [W3C](https://w3c.github.io/deviceorientation/) |
| Generic Sensor API (`RelativeOrientationSensor`, quaternion) | None | Chrome 67+ (`GravitySensor` 91+). Max 60 Hz. `sensor.timestamp` is the sample time but may be offset from `performance.now()` by accumulated deep-sleep time. Suspended when the page is hidden **or loses focus** | None | [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/RelativeOrientationSensor.json), [Chromium caps](https://github.com/chromium/chromium/blob/main/services/device/public/cpp/generic_sensor/sensor_traits.h), [Chromium focus](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/sensor/sensor_proxy.cc) |
| **Permission and lifecycle** | | | | |
| Motion permission | `DeviceMotionEvent.requestPermission()` and `DeviceOrientationEvent.requestPermission()` share one permission per origin. The prompt appears only while a user gesture is being processed; otherwise `NotAllowedError`. The decision is kept in memory for the app process: it survives reloads, and a Cancel is just as sticky until Safari is force-quit (deleting the site's data is not a reliable reset). Insecure context or third-party iframe: silent `denied` | Chrome 151+ (stable 28 Jul 2026) ships `requestPermission()`. It currently resolves `granted` (the Sensors setting defaults to Allow); Google plans Ask-by-default, after which pages get no events until they call it. Samsung Internet may not have it yet | No `requestPermission()`; sensors disabled under `resistFingerprinting` | [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/dom/DeviceMotionEvent.cpp), [WebKit cache](https://github.com/WebKit/WebKit/blob/main/Source/WebKit/UIProcess/WebsiteData/WebDeviceOrientationAndMotionAccessController.cpp), [Apple forum](https://developer.apple.com/forums/thread/128376), [Chrome 151](https://developer.chrome.com/release-notes/151), [chromestatus](https://chromestatus.com/feature/5915984063889408), [Gecko](https://github.com/mozilla-firefox/firefox/blob/main/mobile/android/geckoview/src/main/java/org/mozilla/gecko/GeckoSensor.java) |
| Gesture tracking | WebKit checks for a gesture *currently being processed*; any `await` before the call (fetch, IndexedDB, AudioContext, a worker round trip) can lose it | Not an issue at present (Allow default) | n/a | [WebKit bug 269706](https://bugs.webkit.org/show_bug.cgi?id=269706) |
| Hidden page, screen lock, app switch | Motion updates suspended; Core Motion restarts afterwards | Updates stop; Generic Sensors also stop on focus loss | Events stop (spec: only while visible) | [W3C](https://w3c.github.io/deviceorientation/), [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/page/Page.cpp), [Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/core/frame/platform_event_controller.cc) |
| Screen Wake Lock | Safari 16.4+; in Home Screen web apps only from iOS 18.4. Released when the page is hidden; older WebKit needed a fresh user activation to re-acquire it | Chrome 84+; released when hidden | Firefox 126+ | [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/WakeLock.json), [WebKit bug 254545](https://bugs.webkit.org/show_bug.cgi?id=254545), [WebKit bug 255363](https://bugs.webkit.org/show_bug.cgi?id=255363) |
| Screen-orientation lock | `screen.orientation.lock()` not supported | Supported | Not checked | [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/ScreenOrientation.json) |
| Timer resolution (`performance.now()`, `timeStamp`) | 1 ms | 100 µs (5 µs when cross-origin isolated) | 1 ms | [WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/page/Performance.cpp), [Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/core/timing/time_clamper.h) |
| **Cues and export** | | | | |
| Web Audio | Needs a user gesture. The ringer switch mutes it unless `navigator.audioSession.type = 'playback'` (Safari 16.4+, effective since iOS 17) | `AudioContext.resume()` after a gesture; no `audioSession` API | Not checked | [WebKit bug 237322](https://bugs.webkit.org/show_bug.cgi?id=237322), [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/AudioSession.json), [Chrome autoplay](https://developer.chrome.com/blog/autoplay) |
| Vibration API | Not available in any iOS browser | Chrome 32+; needs a user gesture since Chrome 60 | No-op that returns `true` | [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/Navigator.json), [WebKit bug 288846](https://bugs.webkit.org/show_bug.cgi?id=288846) |
| `speechSynthesis` | Available, but speaks only when called during a user gesture; reportedly silenced by the ringer switch | Chrome 33+; not in Android WebView | Not checked | [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/SpeechSynthesis.json), [WebKit bug 223473](https://bugs.webkit.org/show_bug.cgi?id=223473) |
| Web Share with files | Safari 14+ | Chrome 76+. The allowlist rejects `.json`, `application/json` and `.zip`; at most 10 files and 50 MiB. Android WebView has no Web Share | Share since 79, no files | [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/Navigator.json), [Chromium allowlist](https://github.com/chromium/chromium/blob/main/components/browser_ui/webshare/android/java/src/org/chromium/components/browser_ui/webshare/ShareServiceImpl.java) |
| **Storage and metadata** | | | | |
| Quota and eviction | Up to 60% of disk per origin; LRU eviction. ITP deletes all script-writable storage after 7 days of Safari use without interaction with the site (Home Screen web apps exempt). `persist()` since 15.2 (heuristic), `estimate()` since 17 | Up to 60% per origin; LRU when disk runs low; `persist()` decided automatically (engagement, installed) | `persist()` prompts the user | [WebKit storage](https://webkit.org/blog/14403/updates-to-storage-policy/), [WebKit ITP](https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/), [web.dev](https://web.dev/articles/storage-for-the-web) |
| IndexedDB reliability | "UnknownError: Connection to Indexed Database server lost" when WebKit's networking process dies; transactions are rolled back when a backgrounded Safari is suspended | No known issue in our sources | Not checked | [WebKit bug 273827](https://bugs.webkit.org/show_bug.cgi?id=273827), [WebKit bug 197050](https://bugs.webkit.org/show_bug.cgi?id=197050) |
| User agent | OS version frozen in the UA since iOS 26; iPad Safari presents as a Mac; no device model | Reduced UA since Chrome 110 (`Linux; Android 10; K`); model and OS version only via `navigator.userAgentData.getHighEntropyValues(['model', 'platformVersion'])` | Not checked | [Chromium UA reduction](https://www.chromium.org/updates/ua-reduction/), [Safari 26 notes](https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes) |

iOS 27 / Safari 27 (14 Sep 2026) lists no changes to motion, orientation, Wake Lock, Web Share or storage policy, and WebKit's iOS motion code is unchanged through September 2026 ([Safari 27 notes](https://developer.apple.com/documentation/safari-release-notes/safari-27-release-notes)). The iOS column is therefore treated as current.

### 2.3 In-app browsers, WebViews and Home Screen apps

- **iOS in-app browsers** (Instagram, Facebook, Gmail, ...) are WKWebViews: same WebKit conventions, 60 Hz and sign inversion. The host app may grant, deny or show WebKit's prompt, and WKWebView does not remember the decision ([WebKit bug 213468](https://bugs.webkit.org/show_bug.cgi?id=213468), [Apple WKUIDelegate](https://developer.apple.com/documentation/webkit/wkuidelegate/webview(_:requestdeviceorientationandmotionpermissionfor:initiatedbyframe:decisionhandler:))). Chrome for iOS does not implement the delegate, so the default WebKit prompt appears ([Chromium iOS](https://github.com/chromium/chromium/blob/main/ios/web/web_state/ui/crw_wk_ui_handler.mm)).
- **Android WebView**: sensor access is always allowed and `requestPermission()` returns `granted`, but there is no Web Share and no `speechSynthesis`, and storage belongs to the host app ([chromestatus](https://chromestatus.com/feature/5915984063889408), [BCD](https://github.com/mdn/browser-compat-data/blob/main/api/Navigator.json)).
- **iOS Home Screen web apps** run the same WebKit code but have their own permission cache and storage, separate from Safari tabs (inferred from WebKit's per-data-store cache). Since iOS 26 a site added to the Home Screen opens as a web app by default ([Safari 26 notes](https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes)). A participant who switches from Safari to the Home Screen copy starts with an empty local profile.

## 3. How the app handles each platform issue

| Platform fact | What the app does | Where |
|---|---|---|
| iOS prompts only during a tap; `await` before the call loses the gesture | The tap handler calls `DeviceMotionEvent.requestPermission?.()` and `DeviceOrientationEvent.requestPermission?.()` as its first statements, starting both before awaiting either, on every browser where they exist. The grant is re-requested on every page load (instant once decided) | [`core/sensors.ts`](../app/src/core/sensors.ts) `requestMotionPermission` |
| `denied` without a prompt means an insecure context or iframe; a Cancel is sticky | Result stored in `capabilities.permission_state` (`NotAllowedError` gives `unknown`); the sensor check shows recovery steps that match the context (item 3 below) | `core/sensors.ts`, `SensorCheckScreen.tsx` |
| Sensors can be blocked without an error (Brave, privacy settings) | If no `devicemotion` event arrives within 1500 ms of starting, the source reports `no_events`. The sensor check ignores that early error: it measures for 2.5 s and reports `no_events` only when no data arrived in that window ([section 6](#6-online-quality-checks)) | `core/sensors.ts`, `SensorCheckScreen.tsx` |
| Engines differ in sign and units | `device.engine` detection plus the implicit calibration from the trials ([section 4](#4-calibration-procedure)); raw values stored unchanged | `core/sensors.ts`, [`core/calibration.ts`](../app/src/core/calibration.ts) |
| In-app browsers behave unpredictably | Instagram, FBAN/FBAV (Facebook), TikTok/musical_ly/BytedanceWebview, Line, Snapchat, GSA (Google app), LinkedIn, WeChat, Pinterest, WhatsApp, the Android WebView `wv` token and an iOS user agent without a `Safari/` token outside Home Screen apps (`ios_webview`) are detected into `device.in_app_browser` | `core/sensors.ts` |
| `event.interval` units differ | Stored only as `interval_raw`; sample times come from `performance.now()` in the handler and `event.timeStamp` | `core/sensors.ts`, [`core/recorder.ts`](../app/src/core/recorder.ts) |
| Hidden page stops the sensors | Screen Wake Lock acquired and re-acquired on `visibilitychange` → visible and in the Start, block-intro and Continue taps; a real-phone attempt that reached GO without the lock held gets the warning `wake_lock_not_held`. The page's visibility is polled at every sample and at the stop: `quality.hidden_during` is true when the page was hidden before `end_detected_ms + post_roll_ms`, and a later hide (during the return of the arm) adds `hidden_after_end` instead. The UI stops a trial when the page is hidden or samples stop for 2 s: `interrupted`, or still `complete` with `return_cut_short` when only the return was cut | [`core/cues.ts`](../app/src/core/cues.ts) `WakeLockGuard`, [`core/trialController.ts`](../app/src/core/trialController.ts), [`app/sessionRunner.ts`](../app/src/app/sessionRunner.ts) |
| The vibration motor contaminates the accelerometer; iOS has no Vibration API | Vibration only for `block_start`, never for `ready` / `go` / `end` / `error`, which happen while recording | `core/cues.ts` |
| The ringer switch mutes Web Audio on iOS | `navigator.audioSession.type = 'playback'` set in `unlock()`; large colour cues in addition to beeps, because the screen may face away | `core/cues.ts` |
| iOS Web Audio can stay silent after a call, an alarm or an app switch: the context reads `interrupted`, ignores `resume()`, or reads `running` while silent (WebKit bugs 276016, 273511 and 283419, as cited in `core/cues.ts`) | The taps call `unlock()` (Start, block intro, Continue, the test beep). On iPhones and iPads it runs `suspend()` then `resume()` inside every such tap, on other devices only for a context that is neither running nor suspended (e.g. `interrupted`); a repaired context that is still not running 300 ms later is closed and replaced at the next tap. The How-to card has a "Play a test beep" button | `core/cues.ts` |
| iOS speech works only inside a tap, and keeps other audio ducked for a moment after it | Speech only for instructions spoken from a tap (on other platforms, chained trials also say their speed word); GO and end are synthesised Web Audio beeps. GO waits until the words are over (estimated as 250 ms + 75 ms per character), plus 1 s on iPhones (WebKit bug 278598, as cited in `app/sessionRunner.ts`) | `core/cues.ts`, `app/sessionRunner.ts` |
| Chrome Android will not share `.json` | Export retries the same content as `text/plain` `…json.txt`, else downloads | [`core/exporter.ts`](../app/src/core/exporter.ts) |
| iOS drops IndexedDB connections | One short transaction per operation; on a connection error, reopen once and retry once | [`core/storage.ts`](../app/src/core/storage.ts) |
| Safari's 7-day deletion and eviction | Upload after every attempt; `navigator.storage.persist()` requested; "export my data" fallback | [`core/uploader.ts`](../app/src/core/uploader.ts), `core/storage.ts` |
| Euler angles hit gimbal lock at the start pose | Onset and end use \|ω\| and \|a\| only; angles stored as metadata | [`core/segmentation.ts`](../app/src/core/segmentation.ts) |
| Reduced UA, frozen iOS version, no iOS model | UA Client Hints `model` and `platformVersion` on Android. On iOS `device.model_hint` is only the device class (`iPhone` / `iPad` / `iPod`), or the hardware id in Instagram/Facebook webviews; a "Macintosh" UA with touch counts as an iPad (`platform` `ios`). Protocol 2.0.0 no longer asks for the phone model; the analysis groups iPhones by screen size (`motionlab.io.device_model`) | `core/sensors.ts` |

Participant-facing handling that the platform research called for, and its status:

1. **In-app browsers and blocked sensors.** Done. The welcome screen asks the participant to open the link in Safari or Chrome when `device.in_app_browser` is set (with "Copy the link"; on an iPhone, Start becomes the secondary "Continue here anyway"), and the `no_events` message names privacy browsers and in-app browsers as likely causes (`WelcomeScreen.tsx`, `SensorCheckScreen.tsx`).
2. **Slow or incomplete sensors.** Done. The background rate check after the Start tap opens the repair screen only below 20 Hz or without data. A rate from 20 to under 45 Hz, or a missing gyroscope, does not stop the visit: the How-to card shows a note instead ("Your phone’s motion sensors report N times per second (about 60 is usual). Turning off Low Power Mode / Battery Saver may help.", or "This phone does not report rotation (gyroscope): you can take part, but some data will be missing.") ([section 6](#6-online-quality-checks), `sensorNote` in `SessionScreen.tsx`).
3. **Recovery after `denied`.** Done, on the sensor repair screen and on the session's block intro (`deniedContext` in `SensorCheckScreen.tsx`). The advice depends on where the app runs:
   - iOS, Safari tab: fully close Safari (swipe it away in the app switcher), open the link again and tap Allow.
   - iOS, Home Screen app: close the app in the app switcher and open it again from the Home Screen. Its permission cache is separate from Safari's ([section 2.3](#23-in-app-browsers-webviews-and-home-screen-apps)).
   - Chrome, Firefox or Edge on iOS: close that browser completely, not Safari.
   - iOS in-app browser: open the link in Safari.
   - Android: allow the site's "Motion sensors" setting ([web sensor findings](https://developer.apple.com/forums/thread/128376)).

   The decision is cached in memory per app process, and deleting the site's website data is not a reliable reset. The app never suggests a Private tab. Its storage is separate and temporary, so the participant would get a new participant code, could not resume the session in progress, and would lose unsent data when the tab closes.
4. **Wake lock.** Done. The Start, block-intro and Continue taps re-request it, in addition to `visibilitychange` (older WebKit needed a fresh activation, [bug 255363](https://bugs.webkit.org/show_bug.cgi?id=255363)). When the lock is refused (Home Screen apps before iOS 18.4, some in-app browsers), the How-to card asks the participant to set Auto-Lock / Screen timeout to 5 minutes, and real-phone attempts that reached GO without the lock carry the warning `wake_lock_not_held`. Not done: a looping-video fallback for browsers without the Wake Lock API, and a capabilities field that records the refusal.
5. **Low Power Mode, Do Not Disturb and screen rotation.** Partly done. The How-to card tells every participant "For these 5 minutes: Do Not Disturb on, Low Power Mode / Battery Saver off." (a call pauses the session; on iOS Low Power Mode caps `requestAnimationFrame` at 30 fps, and its effect on motion events is undocumented, [WebKit bug 215745](https://bugs.webkit.org/show_bug.cgi?id=215745)), and the slow-sensor note and the repair screen repeat the Low Power advice. iOS cannot lock the screen orientation, so the page may rotate in sideways grips: the grip gate names the grip in words and draws the grip picture only when the page turned with the phone (with Rotation Lock on, a picture of a sideways phone on a portrait page would look turned), and a rotation between GO and the end of the movement adds the warning `screen_rotated_during_movement`. Still to check on real phones: whether such rotations cause sample gaps ([section 8](#8-pilot-verification-checklist), check 15).

## 4. Calibration procedure

**App 0.2.0 (protocol 2.0.0) has no calibration step.** Participants on the street could not lay the phone on a table, so the session stores an implicit calibration (`method: implicit_v1`) filled from the still pre-GO hold of the first complete upright trial: its `PoseStats` as `upright`, the sign of `accg_y` as `gravity_sign` when \|`accg_y`\| > 7 m/s², and `upright_check` against the engine prior (note `upright_sign_contradicts_engine` on a contradiction); no flat window, so no gyro bias, and no tilt step, so no `axis_check` ([02, section 4.5](02_DATA_SCHEMA.md#45-calibration-calibrationresult)). The analysis confirms the sign with the engine and the trial votes and infers the rotationRate mapping from the movements themselves ([section 5](#5-normalisation-layer-for-analysis)). The guided procedure below (`method: guided_v1`, or absent) is what app 0.1.0 recorded, the pilot included.

Calibration measures the conventions of the participant's phone and browser instead of assuming them. Three reasons: the accelerometer sign differs by engine, and a future WebKit change could flip it silently; the `rotationRate` field-to-axis mapping changed in the spec in 2019, and other wrappers still disagree (Expo's iOS DeviceMotion maps alpha to z, [section 7.3](#73-pitfalls-if-going-native)); and Chrome reported rad/s until Chrome 66. Results go to `session.calibration` ([02, section 4.5](02_DATA_SCHEMA.md#45-calibration-calibrationresult)). The math is in `core/calibration.ts`; all inputs are raw samples.

| Step | Participant does | Output | Expected |
|---|---|---|---|
| (a) Flat | Lays the phone flat on a table, screen up, hands off | `flat` (`PoseStats`), `gravity_sign`, `gravity_magnitude`, `gyro_bias_dps` | iOS −1, Android +1; magnitude ≈ 9.8 |
| (b) Upright | Holds the phone upright in front of the face, screen toward them | `upright` (`PoseStats`), `upright_check` | `ok` |
| (c) Tilt | From flat, tilts the phone sideways by at least 30°, then back | `axis_check` | `w3c`, `deg_per_s`, ratio ≈ 1 |

**(a) Flat: gravity sign.** `poseStats` computes the mean and population SD of accg, acc and rr over the window. The window counts as still when every accg axis SD < 0.25 m/s², every rr SD < 3 deg/s (or rr is missing) and n ≥ 20. `gravityFromFlat` accepts the window when 8.6 ≤ \|mean accg\| ≤ 11.0 m/s² and \|z\| ≥ 0.9·\|mean accg\| (tilt below about 26°). Then `gravity_sign = sign(z)` and `gravity_magnitude = |mean accg|`; otherwise both are `null` and `notes` says why ("not flat", "moving", "implausible magnitude"). A phone lying screen-down passes the same checks with the opposite sign. The calibration screen therefore compares the sign with the engine expectation (`webkit` −1, `blink` / `gecko` +1, [section 2.1](#21-engines); an unknown engine, or the simulator, accepts either sign). A contradiction is shown as a warning, not as success, and asks the participant to lay the phone flat with the screen up and measure again; if they continue anyway, `notes` records `flat_sign_contradicts_engine`. `gyro_bias_dps` is the mean rotationRate of the flat window, or `null` when that window gave no gravity sign (a hand on the phone measures motion, not bias). The analysis uses it only when the window was still and the bias norm is below 3 deg/s ([section 5.1](#51-rules)).

**(b) Upright: consistency check.** With the phone upright, W3C accg points along +y. `uprightCheck` computes `y · (gravity_sign ?? 1)`: above 7 gives `ok`, below −7 gives `inverted`, anything else `unclear`. `inverted` means the two steps disagree, typically because one was done face-down or upside down, so the screen asks the participant to repeat steps 1 and 2 (flat and upright). A session that still carries `inverted` is flagged for review. The normalisation keeps its calibrated `gravity_sign` unless the trial votes clearly contradict it (≥ 5 votes, ≥ 80 % against), in which case the vote sign replaces it ([section 5.2](#52-gravity-sign-resolution)). Check such sessions by hand against the trial votes and the engine before using accelerometer-based features.

**(c) Tilt: rotationRate axis, sign and unit (`axisCheck`).**

1. Smooth accg with an EMA (τ = 100 ms). `g0` is the mean over the first 300 ms; `i*` is the sample with the largest angle to `g0`; that angle is `tilt_deg`. Below 30° the verdict is `unclear`.
2. The device rotation axis seen from gravity is `n = −normalize(g0 × g_i*)`: a world-fixed vector seen from the device rotates opposite to the device. Negating both gravity vectors (the iOS sign) leaves the cross product unchanged, so this step does not depend on the accelerometer sign.
3. Integrate each raw rotationRate field over time (trapezoidal, from the first sample to `i*`): `I_alpha`, `I_beta`, `I_gamma`.
4. Candidates in device x/y/z: `w3c` = [I_alpha, I_beta, I_gamma]; `legacy` = [I_beta, I_gamma, I_alpha]; plus their negations. The expected vector is `tilt_deg · n`. The best candidate has the highest cosine similarity; `magnitude_ratio` = \|best\| / `tilt_deg`.
5. Verdict = the best candidate's name if similarity ≥ 0.85 and the ratio lies in [0.6, 1.6] (`deg_per_s`) or in [0.6, 1.6] / 57.3 (`rad_per_s`); otherwise `unclear` (and `units` null when neither range fits). Missing rotationRate gives `unclear`.

Only the two mappings that browsers have actually used are tested. A tilt excites one horizontal axis, which is enough to separate them.

**Engine detection** (`device.engine`, [section 2.1](#21-engines)) is recorded alongside as an independent expectation: `webkit` should give `gravity_sign` −1 and every other engine +1. The calibration screen checks the flat sign against it (step a), and the analysis falls back on it when neither the calibration nor the trial votes settle the sign ([section 5.2](#52-gravity-sign-resolution)).

**What calibration does not establish**: heading (irrelevant for this task), screen toward vs away (gravity cannot tell), and accelerometer scale beyond the magnitude plausibility check.

## 5. Normalisation layer for analysis

Raw records are never modified. Every analysis applies one versioned normalisation step (`norm_v2`; `norm_v1` before protocol 2.0.0) before computing any feature, and every derived dataset records that version (`qc_norm_version`). [`analysis/motionlab/conventions.py`](../analysis/motionlab/conventions.py) implements it (`session_normalization`, `resolve_gravity_sign`, `trial_gravity_vote`, `infer_rate_mapping`, `normalize_acceleration`, `device_signals`); [`analysis/README.md`](../analysis/README.md#normalisation-rules-norm_v2) documents it in full. This section is the specification that code must match.

`norm_v2` differs from `norm_v1` in three rules, all needed because protocol 2.0.0 sessions carry only an implicit calibration: (1) the engine expectation confirmed by the trial votes is a sign source of its own (`engine_vote`, not flagged; [5.2](#52-gravity-sign-resolution)); (2) a session without a definite `axis_check` gets the rotationRate mapping its trials' gravity closure supports (below); (3) no gyro bias is subtracted without a trusted calibration bias (on the pilot, subtracting the in-hand pre-GO mean worsened the median gravity closure from 1.6° to 2.7°).

### 5.1 Rules

| Quantity | Raw fields | Rule | Result |
|---|---|---|---|
| Accelerometer sign | `acc_*`, `accg_*`, `segmentation.baseline.gravity`, calibration vectors | Multiply every component by the session's resolved sign `s` ([5.2](#52-gravity-sign-resolution)) | W3C convention: face-up `f_z` = +9.81, accg at rest points up |
| Gyroscope axes | `rr_alpha`, `rr_beta`, `rr_gamma` | Map by `calibration.axis_check.verdict` (table below) | `w_x`, `w_y`, `w_z` about device x, y, z |
| Gyroscope units | | `units == 'rad_per_s'`: multiply by 180/π. All current browsers report deg/s | deg/s |
| Gyroscope bias (optional) | `calibration.gyro_bias_dps` | Map like the rates, then subtract before integrating (1 deg/s of bias integrates to 1° per second). Use it only when the flat window was still (`calibration.flat.still`) and the bias norm is below 3 deg/s; otherwise subtract nothing (`norm_v2`: a hand-held pre-GO mean is tremor, not bias; `norm_v1` subtracted the mean rate of the 400 ms before GO when its norm was below 3 deg/s) | |
| Time | `t`, `ts`, `interval_raw` | `t` is the sample clock; `ts` (plus `ts_first_raw`) only for checks; `interval_raw` never | ms since t0 |
| Orientation angles, compass, `screen.angle_*` | | Not used for features or axis remapping | metadata |

| `axis_check.verdict` | `w_x` | `w_y` | `w_z` |
|---|---|---|---|
| `w3c` | `rr_alpha` | `rr_beta` | `rr_gamma` |
| `w3c_inverted` | −`rr_alpha` | −`rr_beta` | −`rr_gamma` |
| `legacy` | `rr_beta` | `rr_gamma` | `rr_alpha` |
| `legacy_inverted` | −`rr_beta` | −`rr_gamma` | −`rr_alpha` |
| `unclear`, `null` or absent (schema 1.0, implicit calibration) | the closure check below; failing that, as `w3c`, flagged `axis_unverified` | | |

*Closure check* (`infer_rate_mapping`, run on up to 16 complete trials of every session without a definite verdict). For each candidate mapping (the four verdicts, in deg/s and rad/s) the rest gravity before the movement is propagated through the movement with the gyroscope and compared with the rest gravity at the nose. The candidate with the smallest session median is applied (source `closure`) when it closes below 3°, below half the runner-up and over at least 8 trials; otherwise W3C deg/s is assumed and flagged. The accelerometer sign does not matter here, because both rest windows flip together. On the pilot the check picked W3C (1.8° against 4.9° for the runner-up), in agreement with its tilt check.

The mirror augmentation of [05](05_MODELING_PLAN.md) operates on these normalised x/y/z signals: a ↦ (−a_x, a_y, a_z), w ↦ (w_x, −w_y, −w_z).

### 5.2 Gravity-sign resolution

| Step | Source | Used when | Marked |
|---|---|---|---|
| 1 | `calibration.gravity_sign` of a guided calibration (`method` `guided_v1` or absent) | it is +1 or −1, unless the trial votes clearly contradict it (≥ 5 votes, ≥ 80% against): then the vote sign replaces it | `calibration`. Flag for review when `upright_check == 'inverted'` or when it differs from the engine expectation. A sign replaced by the votes gets source `inferred`, a warning and the detail "calibration sign overridden by the trial votes" |
| 2 | Engine expectation (`webkit` −1, `blink` / `gecko` +1) confirmed by a confident trial vote | the engine is known and the vote agrees; the upright trials vote alone when they are confident | `engine_vote`, not flagged: the normal case of protocol 2.0.0. An implicit calibration's sign is one of these votes, never a source of its own; when it contradicts the result the session is flagged |
| 3 | Trial vote | no calibrated sign; ≥ 3 votes with ≥ 80% agreement | `inferred` (warning; a disagreement with the engine is named) |
| 4 | Engine expectation alone | no calibration and no confident vote | `engine` (warning). The order of steps 3 and 4 is configurable in `motionlab.conventions.resolve_gravity_sign` |
| 5 | The implicit calibration's sign | nothing above | `implicit` (warning) |
| 6 | +1 | nothing else | `default` (warning); exclude the session from accelerometer-based chirality features (rule S3, [02 section 12](02_DATA_SCHEMA.md#12-data-quality-and-exclusion-rules)) |

Trial vote: for each trial with a fixed instructed orientation, compare the raw `segmentation.baseline.gravity` with the expected "up" direction of that orientation ([02, section 3](02_DATA_SCHEMA.md#3-coordinate-frames)). The vote is the sign of the cosine, counted only when \|cos\| ≥ cos 45° and \|g\| ≥ 0.3 g. `free` trials do not vote. The upright trials are preferred when they are confident on their own: a flipped sign turns `top_left` into `top_right`, so sideways votes confound the sign with a grip mix-up, while upright votes cannot.

Why this matters: a global sign flip of the accelerometer is an improper transformation, while the gyroscope stays as it is. Every chirality quantity that combines both (net yaw about gravity ∫ω·ĝ, (v × a)·ĝ, ω·â, det[a(t), a(t+τ), ĝ]) changes sign, which is exactly what the other hand looks like ([05](05_MODELING_PLAN.md)).

### 5.3 Reference implementation of the mapping

Tested on synthetic iOS-like records. `sign`, `verdict` and `units` come from [5.2](#52-gravity-sign-resolution) and `calibration.axis_check`.

```python
import numpy as np

RR_FIELDS = {  # axis_check.verdict -> (sign, rotationRate fields giving device x, y, z)
    "w3c": (1, ("rr_alpha", "rr_beta", "rr_gamma")),
    "w3c_inverted": (-1, ("rr_alpha", "rr_beta", "rr_gamma")),
    "legacy": (1, ("rr_beta", "rr_gamma", "rr_alpha")),
    "legacy_inverted": (-1, ("rr_beta", "rr_gamma", "rr_alpha")),
}

def col(stream, key):
    return np.array([np.nan if v is None else v for v in stream[key]], dtype=float)

def normalise_motion(motion, sign, verdict=None, units=None):
    """Raw MotionStream -> W3C-sign f (with gravity) and a (without) in m/s^2, w in deg/s about device x, y, z.
    sign: resolved gravity sign of the session (+1 / -1). verdict, units: from calibration.axis_check."""
    rr_sign, fields = RR_FIELDS.get(verdict, RR_FIELDS["w3c"])  # 'unclear' or absent: W3C, flag it
    k = rr_sign * (180 / np.pi if units == "rad_per_s" else 1.0)
    out = {"t_ms": col(motion, "t")}
    for i, axis in enumerate("xyz"):
        out["f" + axis] = sign * col(motion, "accg_" + axis)
        out["a" + axis] = sign * col(motion, "acc_" + axis)
        out["w" + axis] = k * col(motion, fields[i])
    return out
```

### 5.4 Platform harmonisation

- **Quantisation.** Chrome rounds to 0.1 m/s² and 0.1 deg/s; iOS does not ([Chromium](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h)). Quantise iOS data (and any high-rate reference data) to the same grid, at least as an ablation, and always report results per platform, including leave-one-platform-out.
- **Rate.** Web data is about 60 Hz everywhere. Resample with `motionlab.preprocess.resample_uniform`, which interpolates linearly and never bridges gaps longer than `max_gap_ms` (default 100 ms). Models that will run in the app ([08](08_STAGE3_PREDICTOR.md)) are trained on 60 Hz data.
- **Rest-window precision.** Chrome's 0.1 m/s² step is comparable to physiological tremor at arm's length (about 0.14 m/s², [Elble 2005](https://smithengineering.queensu.ca/mme/faculty/deluzio/jam/files/Elble.pdf)), so baseline SDs can collapse on Android. This is why the online thresholds have floors ([04](04_SEGMENTATION.md)).

### 5.5 Derived quantities

- **Gravity during the movement**: average the still windows before GO (`baseline`) and at the nose, propagate them with the gyroscope (dg/dt = −ω × g), and blend forward and backward estimates ([05](05_MODELING_PLAN.md)). Accelerometer-corrected filters mis-estimate tilt while the hand accelerates ([segmentation findings](https://arxiv.org/abs/2502.03681)). `accg − acc` (the platform's own fusion, `motionlab.preprocess.estimate_gravity`) is a quality-control comparison.
- **Linear acceleration**: recompute it from `f` and the gravity estimate. Browser `acceleration` on Android phones without a linear-acceleration sensor comes from a crude low-pass filter.
- **Never use as features**: Euler angles (gimbal lock), `compass` and absolute heading (noisy indoors, and they encode the room), `interval_raw`.

### 5.6 Tests the normalisation must pass

1. A face-up still window gives f ≈ (0, 0, +9.81) for an iOS-like record (`gravity_sign` −1) and an Android-like record (+1).
2. iOS-like and Android-like simulations of the same movement (`npm run synth` alternates platforms) give the same normalised signals, up to noise and quantisation.
3. A chirality feature such as net yaw has the same sign on both platform versions of one movement, and the opposite sign after mirroring.
4. A `legacy` verdict maps `rr_beta` to `w_x`, `rr_gamma` to `w_y`, `rr_alpha` to `w_z`.
5. Mirroring twice returns the original signals.

## 6. Online quality checks

What the app checks while collecting, and where the result is stored. The online checks drive cues and participant feedback; the offline analysis re-checks everything from the raw streams.

| Stage | Check | Stored in | Effect |
|---|---|---|---|
| Opening the link | Secure context, `DeviceMotionEvent` / `DeviceOrientationEvent` present, Generic Sensor, Wake Lock, Vibration, Web Share files | `capabilities.*` | Feature availability |
| | Engine and in-app browser | `device.engine`, `device.in_app_browser` | |
| Start tap | `requestPermission()` result | `capabilities.permission_state` | `denied` sends the participant to the repair screen after the profile |
| Background rate check (after the Start tap) | Motion data arrive during the 2.5 s rate measurement, at ≥ 20 Hz (warning below 45 Hz) | `capabilities.measured.*` (no data → issue `no_events`) | The repair screen opens only without data, below 20 Hz, on a denial or an insecure page; otherwise the session starts, and a rate under 45 Hz shows a note on the How-to card |
| | Gyroscope present | `capabilities.measured.rotation_available` | A note on the How-to card (and `no_gyro` on the repair screen when that opens); the visit goes on |
| How-to card, each trial | Screen wake lock held | `quality.warnings` `wake_lock_not_held` (real phones, attempts that reached GO) | The Auto-Lock note on the How-to card while the lock is not held |
| Implicit calibration | The first still upright hold: magnitude plausibility, the sign against the engine | `calibration.*`, `calibration.notes` | Nothing on screen; a contradiction is noted (`upright_sign_contradicts_engine`) and the grip check keeps the engine's sign |
| Each trial: before GO | Hold-still gate (`still_gyro_dps`, `still_acc_ms2`, `still_hold_ms`); the grip gate, the wait for the return of the arm and the reading time of a new instruction ([04 §4.1](04_SEGMENTATION.md#41-around-the-segmenter-gates-false-starts-and-the-return-of-the-arm-protocol-200)); larger movement before GO counts as a false start | `timing.still_ms`, `timing.false_starts` | Messages `hold_still`, `wrong_grip`, `return_arm`, `read`, `too_early`; error beep for a false start |
| Each trial: after GO | No onset within `max_wait_ms`; movement longer than `max_movement_ms` | `outcome.status` `no_movement` / `too_long` | Error beep; the trial runs again by itself (twice in a row: the pause sheet) |
| Each trial: always | Page hidden; samples stop arriving for 2 s; no GO within 25 s; a screen rotation between GO and the end | `quality.hidden_during` (the movement window only; a later hide adds `hidden_after_end`), `outcome.status` `interrupted`, `quality.warnings` `screen_rotated_during_movement` | The trial stops and the session pauses (no GO within 25 s while the grip gate held GO back: the pause sheet names the grip); a rotation is only noted |
| Each trial: after it | Rate, gaps, orientation from gravity vs the instruction | `quality.*`, `detected.*` | Hands-free: accepted, or run again when the grip was detected as another one. Tap mode (researchers): Next, Redo or flag a problem |
| Uploads | Every attempt queued and confirmed | `outbox` (local) | Pending / sent / failed counts shown. A permanent rejection (`retry: false`) is not retried automatically; "Upload now" sends it again |

Thresholds for excluding data afterwards are in [02, section 12](02_DATA_SCHEMA.md#12-data-quality-and-exclusion-rules).

## 7. Sampling rate and native alternatives

### 7.1 Is 60 Hz enough?

For left/right classification, yes (recommendation, to be confirmed by the validation in 7.4):

- For minimum-jerk movements ([Flash & Hogan profile](https://www.socsci.ru.nl/wilberth/physics/smooth.html)), 99% of the acceleration energy lies below 2.53/D Hz and 99.9% below 5.56/D Hz, where D is the duration (our computation). For D = 0.25 s that is 10.1 Hz and 22 Hz, both under the 30 Hz Nyquist limit of 60 Hz sampling.
- Natural one-way finger-to-nose movements take about 0.6 s ([Chen 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8072355/)); fast 50 cm pointing takes 276–651 ms ([Terrier 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3045308/)).
- Sampling at 60 Hz quantises onset times with an SD of T/√12 = 4.8 ms, far below the tens to hundreds of ms by which onset methods differ ([04](04_SEGMENTATION.md)).
- Activity-recognition studies find 10 Hz sufficient for most activities ([Yamane 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12196717/)) and accuracy peaking between 12 and 63 Hz ([Khan 2016](https://doi.org/10.1016/j.patrec.2016.01.001)).

Where 60 Hz may cost something: gyro-integrated orientation in fast and odd trials (work cited by Yamane 2025 needed 100–400 Hz for fast cyclic movements), jerk and the impact at the nose, and onset precision for fast trials.

### 7.2 Options compared

| Option | iPhone rate | Android rate | Timestamps | Participant effort | Protocol control | Cost, data path | Sources |
|---|---|---|---|---|---|---|---|
| **This PWA** | ~60 Hz | ~60 Hz (0.1-unit rounding) | Handler time and dispatch-time `timeStamp` | Open a link or QR, one permission tap | Full: profile, consent, protocol, segmentation, upload | $0; own backend | [section 2](#2-what-the-browser-delivers) |
| **phyphox 1.2.1** (RWTH Aachen, GPL-3.0) | ~100 Hz | Device maximum, up to ~500 Hz (declares `HIGH_SAMPLING_RATE_SENSORS`) | Experiment time from a monotonic clock, one buffer per sensor | Store install plus QR scan (1–2 min) | Partial: XML experiment with numeric label buffers and timed runs; no profile, consent or segmentation | Free; CSV/xlsx export or `http/post` without retry | [release](https://phyphox.org/?p=5037), [rates](https://phyphox.org/forums/printthread.php?tid=2356), [format](https://phyphox.org/docs/file-format/input/) |
| **Sensor Logger** (closed source, UK) | ~100 Hz | ~70–400 Hz, device-dependent | UTC epoch ns from the sensors | Store install | Annotations; "Studies" need paid tiers | Free tier stores 10 recordings; vendor cloud or export | [help](https://www.tszheichoi.com/sensorloggerhelp), [rates](https://github.com/tszheichoi/awesome-sensor-logger/blob/main/SAMPLING.md) |
| **Native app** (Expo development build with a custom module) | 100 Hz (Core Motion cap) | 200 Hz without, 400–500 Hz or more with `HIGH_SAMPLING_RATE_SENSORS` | Hardware sensor timestamps | TestFlight install, or APK sideload | Full | $99/year Apple account, App Review, 90-day builds; development time | [WWDC23](https://developer.apple.com/videos/play/wwdc2023/10179), [Android](https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview), [TestFlight](https://developer.apple.com/help/app-store-connect/test-a-beta-version/testflight-overview/) |
| **Expo Go** | — | — | — | Not usable by arbitrary participants since May 2026 | — | — | [Expo changelog](https://expo.dev/changelog/expo-go-loading-changes-may-2026) |

### 7.3 Pitfalls if going native

- **iOS ceiling.** Core Motion's maximum is 100 Hz ([WWDC23 session 10179](https://developer.apple.com/videos/play/wwdc2023/10179)); in January 2026 Apple DTS confirmed that no API or entitlement allows more ([forum](https://developer.apple.com/forums/thread/813136)). Use one `CMMotionManager` per app. `CMBatchedSensorManager` (800 Hz) works only on Apple Watch during workouts.
- **Android caps.** Apps targeting API 31+ are capped at 200 Hz unless they declare `HIGH_SAMPLING_RATE_SENSORS`, and are rate-limited regardless when the user turns off microphone access ([Android docs](https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview)). The Android 17 CDD requires at least 100 Hz for handheld accelerometers and gyroscopes ([CDD](https://source.android.com/docs/compatibility/17/android-17-cdd)). Log the achieved rate per session.
- **Stock `expo-sensors` is not a research logger.** On Android 12+ without `HIGH_SAMPLING_RATE_SENSORS` it falls back to `SENSOR_DELAY_NORMAL` (nominally 5 Hz) and thins events by arrival time (default 100 ms) ([source](https://github.com/expo/expo/blob/main/packages/expo-sensors/android/src/main/java/expo/modules/sensors/SensorSubscription.kt)). `DeviceMotion` emits one snapshot per display frame and flips Android's `accelerationIncludingGravity` to the iOS sign ([source](https://github.com/expo/expo/blob/main/packages/expo-sensors/android/src/main/java/expo/modules/sensors/modules/DeviceMotionModule.kt)). On iOS its `rotationRate` alpha/gamma map to z/x, but to x/z on Android, and `interval` is seconds on iOS but ms on Android ([source](https://github.com/expo/expo/blob/main/packages/expo-sensors/ios/DeviceMotionModule.swift)). A native path needs a custom module that registers once at the maximum rate and keeps every hardware-timestamped sample.
- **Expo Go.** Since 12 May 2026 it loads only projects owned by your account or organisation, and since 3 Sep 2026 iOS Expo Go requires the same account in the CLI and the app ([May](https://expo.dev/changelog/expo-go-loading-changes-may-2026), [September](https://expo.dev/changelog/expo-go-57-login)).
- **Distribution.** iOS needs the $99/year Apple Developer Program; TestFlight external testing needs App Review for the first build, and builds expire after 90 days ([Apple](https://developer.apple.com/programs/), [TestFlight](https://developer.apple.com/help/app-store-connect/test-a-beta-version/testflight-overview/)). Android APKs still sideload, but Google's developer verification applies to all certified devices worldwide in 2027 ([Android](https://developer.android.com/developer-verification/guides)).

### 7.4 Recommendation

Recommended, not yet built:

1. **PWA for every participant.** It is the main instrument.
2. **phyphox as the high-rate reference** for a lab-run validation cohort of 10–20 participants on both operating systems:
   - Use an experiment file `nose-touch.phyphox` with accelerometer, gyroscope, linear_acceleration, gravity and attitude, each with `rate="0"` or `rateStrategy="request"`. The default `auto` strategy duplicates or drops samples to hit a target rate.
   - Add numeric label buffers for hand, orientation, speed/variant and a participant code, and use a timed run.
   - Declare file format 1.19: phyphox refuses newer formats, and 1.20 would lock out phyphox 1.2.0 installs. Host it next to the PWA and distribute it with an online QR code.
   - Alternate PWA and phyphox blocks on the same phone.
   - Train and test on phyphox data at its native rate, and again after emulating the PWA (sample-and-hold to 60 Hz, 0.1-unit rounding). Compare accuracy and onset/end timing.
   - phyphox's remote interface is plain HTTP without authentication: use it only on a trusted lab network, never from the HTTPS PWA ([remote interface](https://phyphox.org/docs/remote-interface/)).
3. **Sensor Logger only for the student's own pilot recordings**, e.g. to tune segmentation on 100–400 Hz data. It is closed source, its free tier is capped, and its cloud is a third party.
4. **A native app only if the validation shows a material loss** of accuracy or timing at 60 Hz.

## 8. Pilot verification checklist

Run on at least two iPhones (iOS 26 or 27, one with ProMotion) and two Android phones (one mid-range, Chrome and Samsung Internet) before recruiting.

| # | Check | How | Expected (from source reading) | Where it shows up |
|---|---|---|---|---|
| 1 | Accelerometer sign | A street visit (the implicit calibration fills from the first still upright trial) | iOS `gravity_sign` −1, Android +1, consistent with `device.engine`; `upright_check` `ok`; the analysis resolves the sign as `engine_vote` | `calibration.gravity_sign`, `qa_report.py` |
| 2 | rotationRate mapping, sign, unit | `qa_report.py` on a street visit: the gravity-closure check of the session's trials compares the candidate mappings | `w3c` wins decisively (< 3°, under half the runner-up) on both platforms | QA report |
| 3 | `event.interval` | Inspect | iOS ≈ 0.0167, Chrome 16, Firefox 100 | `capabilities.measured.interval_raw` |
| 4 | Achieved rate and jitter | Sensor check plus full sessions; include Low Power Mode on/off and one in-app browser | ≈ 60 Hz, median dt ≈ 16.7 ms, no gaps during trials | `capabilities.measured.*`, `quality.*` |
| 5 | Value precision | Histogram of values at rest | Chrome: multiples of 0.1; iOS: unrounded | streams |
| 6 | Repeated samples on iOS | Count identical consecutive samples | Occasional only | streams |
| 7 | Permission flows | First visit; Cancel then retry; reload; force-quit; Home Screen copy; Chrome 151+; Samsung Internet | iOS Cancel sticks until Safari is force-quit; Home Screen copy asks again; Chrome resolves `granted`, check whether it needs a gesture | `capabilities.permission_state` |
| 8 | Privacy browsers | Brave, DuckDuckGo, Firefox with `resistFingerprinting` | No events, so `no_events` (Brave's 2021 policy may have changed) | sensor check |
| 9 | Hidden page | Lock the screen and switch apps mid-trial; once during the movement, once during the return of the arm (about a second after the double beep) | During the movement (or within `post_roll_ms` of the double beep): `interrupted`, `hidden_during`, gaps. During the return: `complete` with `return_cut_short` and `hidden_after_end`, `hidden_during` false | `outcome`, `quality` |
| 10 | Wake lock | Wait on an instruction screen; Home Screen app; Low Power Mode on | Screen stays on (Home Screen apps only from iOS 18.4); where the lock is refused, the How-to card's Auto-Lock note and `wake_lock_not_held` on the trials | `quality.warnings` |
| 11 | Audio with the ringer switch on silent, and after a call | Run a trial; take a call (or open another app) mid-block, come back and tap Continue | Beeps audible (`audioSession` playback); after the interruption the beeps come back with the Continue tap (the audio repair, [section 3](#3-how-the-app-handles-each-platform-issue)) | — |
| 12 | GO-beep latency | Film the screen and record the sound with a second device | Output latency estimate (not stored in records) | `cues` |
| 13 | Nose touches | Screen-toward trials | `touchstart` close to the online end | `timing.first_touch_ms` vs `end_ms` |
| 14 | Clipping | Fast trials, firm nose contact | Peaks below the sensor range | streams |
| 15 | Sideways grips and screen rotation | `top_left_*`, `top_right_*`, once with Rotation Lock on and once off; hold a sideways block in the wrong grip | UI stays usable when the screen rotates; device frame unaffected; no gaps over 50 ms around `screen_events`; a rotation during a movement adds `screen_rotated_during_movement`; the wrong grip is named in words, and its picture appears only when the page turned with the phone | `screen.*`, `streams.screen_events`, `quality.*`, `detected.*` |
| 16 | Quaternion convention (Android) | Static poses: compare `RelativeOrientationSensor` with the Euler-derived matrix | Same rotation, v_reference = R·v_device, as the simulator and mirror code assume | `streams.quaternion` |
| 17 | Storage robustness and export | Kill Safari mid-session; export on iOS and Chrome Android | Data intact; `.json` or `.json.txt` arrives | — |
| 18 | Upload path | The Heroku service from Safari and Chrome, with and without a content blocker | `ok: true`; each trial on the admin page within seconds; failures retried | outbox counts, admin page |
| 19 | Mirror sanity | A few people do both hands | Mirrored right-hand trials match left-hand trials best with acc (−, +, +), gyro (+, −, −) among the 64 sign patterns ([05](05_MODELING_PLAN.md)) | analysis |
| 20 | Generic Sensor timestamp offset | Only if `sensor.timestamp` is added to the schema | Possibly a constant offset to `performance.now()` (deep sleep) | — |
| 21 | The hands-free chain | A whole street visit on each platform, once with Low Power Mode on | The motion prompt on the Start tap; beeps after the Start and block-intro taps; GO only after the new instruction could be read ("Read this first…"); chained trials start by themselves; "Hold to pause" works and a nose touch never triggers it; the screen stays on through a block | records, admin page |
| 22 | In-app browsers | Open the link from WhatsApp (also a WhatsApp Business chat) and Instagram, on an iPhone and on an Android phone | The welcome screen names the app and offers "Copy the link"; on an iPhone, Start reads "Continue here anyway"; `device.in_app_browser` is set (`whatsapp`, `instagram`, `ios_webview`, `android_webview`); note whether the motion prompt, speech and the wake lock work there | `device.in_app_browser`, `capabilities.*` |

## 9. Open questions for the PI

1. Accept every browser with quality flags, or require Safari or Chrome (blocking in-app browsers and Firefox)?
2. Minimum device requirements: is a gyroscope mandatory, and is about 50 Hz the minimum acceptable rate?
3. Approve the phyphox validation cohort (10–20 participants, both platforms, lab time)?
4. Is a native app in scope at all if the validation shows a gap? Does the lab or Bar-Ilan have an Apple Developer organisation account?
5. Which phone and browser will the PI use for the evaluation, and may it start with the 2.5-s upright check ([08 §1](08_STAGE3_PREDICTOR.md#1-what-the-pi-uses))? The check is what makes the gravity sign of an unknown browser safe.
