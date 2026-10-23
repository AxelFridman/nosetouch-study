/**
 * The phone drawn FROM THE PARTICIPANT'S POINT OF VIEW for an OrientationId:
 *  - screen side (screen facing the participant): light screen, notch at the top, home bar at the bottom;
 *  - back side (screen facing away): darker body, camera module at the TOP-LEFT corner as seen from the back;
 *  - rotation: upright 0°, upside_down 180°, top_left = top edge points LEFT (−90°), top_right = top
 *    edge points RIGHT (+90°) — rotating the drawing is exactly the in-plane rotation the participant
 *    sees, so the camera of top_left_away ends up bottom-left, as on a real phone;
 *  - flat_up: a tilted parallelogram (seen from above and in front) with the top edge pointing away;
 *  - free: a phone with a question mark.
 * An arrow (and a small "top" label) marks the phone's top edge. Optional hand badge (L/R) sits on the
 * physical side of that hand (also on RTL pages). role="img", aria-label = the orientation label.
 * Geometry lives in a 200×200 viewBox; clear at 160–260 px.
 */
import type { Hand, OrientationId } from '../core/types';
import { ORIENTATIONS } from '../core/constants';
import { useStrings } from '../i18n';

export type FigureSide = 'screen' | 'back' | 'flat' | 'free';
export type FigureRotation = 0 | 90 | 180 | -90;

export interface PhoneFigureSpec {
  side: FigureSide;
  /** clockwise rotation of the upright drawing, as seen by the participant */
  rotationDeg: FigureRotation;
}

/** What to draw for an orientation (pure; derived from the ORIENTATIONS table). */
export function phoneFigureSpec(o: OrientationId): PhoneFigureSpec {
  const info = ORIENTATIONS[o];
  if (!info || info.rotation === null) return { side: 'free', rotationDeg: 0 };
  if (info.rotation === 'flat') return { side: 'flat', rotationDeg: 0 };
  const rotationDeg: FigureRotation =
    info.rotation === 'upside_down' ? 180 : info.rotation === 'top_left' ? -90 : info.rotation === 'top_right' ? 90 : 0;
  return { side: info.screen === 'away' ? 'back' : 'screen', rotationDeg };
}

// Upright phone body in the 200×200 box, centred on (100, 100).
const BODY = { x: 62, y: 30, w: 76, h: 140, r: 13 };

/** Where the (unrotated) "top" label goes for each rotation: next to the arrow, never upside down. */
const TOP_LABEL: Record<FigureRotation, { x: number; y: number }> = {
  0: { x: 128, y: 18 },
  180: { x: 72, y: 192 },
  90: { x: 184, y: 82 },
  [-90]: { x: 16, y: 82 },
};

function TopArrow() {
  // drawn for the upright phone; the caller's rotation turns it with the phone
  return (
    <g class="pf-arrow" aria-hidden="true">
      <line x1="100" y1="27" x2="100" y2="14" stroke-width="4" stroke-linecap="round" />
      <polygon points="100,2 91,15 109,15" />
    </g>
  );
}

function ScreenSide(props: { question?: boolean }) {
  return (
    <g>
      <rect class="pf-body" x={BODY.x} y={BODY.y} width={BODY.w} height={BODY.h} rx={BODY.r} />
      <rect class="pf-screen" x={BODY.x + 5} y={BODY.y + 5} width={BODY.w - 10} height={BODY.h - 10} rx="9" />
      <rect class="pf-notch" x="88" y="39" width="24" height="7" rx="3.5" />
      <rect class="pf-home" x="86" y="160" width="28" height="3" rx="1.5" />
      {props.question ? (
        <text class="pf-question" x="100" y="124" text-anchor="middle">
          ?
        </text>
      ) : (
        // left-aligned "content" bars make the rotation readable even without the arrow
        <g class="pf-content">
          <rect x="74" y="62" width="40" height="7" rx="3.5" />
          <rect x="74" y="78" width="52" height="5" rx="2.5" />
          <rect x="74" y="90" width="46" height="5" rx="2.5" />
          <rect x="74" y="102" width="50" height="5" rx="2.5" />
          <rect x="74" y="114" width="30" height="5" rx="2.5" />
        </g>
      )}
    </g>
  );
}

function BackSide() {
  return (
    <g>
      <rect class="pf-back" x={BODY.x} y={BODY.y} width={BODY.w} height={BODY.h} rx={BODY.r} />
      <rect class="pf-camera" x="68" y="36" width="31" height="31" rx="9" />
      <circle class="pf-lens" cx="76.5" cy="44.5" r="5.2" />
      <circle class="pf-lens" cx="76.5" cy="58.5" r="5.2" />
      <circle class="pf-lens" cx="90.5" cy="51.5" r="5.2" />
      <circle class="pf-flash" cx="91" cy="41" r="2.3" />
      <circle class="pf-logo" cx="100" cy="104" r="9" />
    </g>
  );
}

// Flat phone: point(u, v) = (52 + 76u + 22v, 150 − 70v); u across the width, v towards the (far) top edge.
const flatPt = (u: number, v: number) => `${(52 + 76 * u + 22 * v).toFixed(1)},${(150 - 70 * v).toFixed(1)}`;

function FlatPhone(props: { screenUp: string }) {
  return (
    <g>
      <polygon class="pf-body" points={[flatPt(0, 0), flatPt(1, 0), flatPt(1, 1), flatPt(0, 1)].join(' ')} stroke-linejoin="round" />
      <polygon class="pf-screen" points={[flatPt(0.07, 0.06), flatPt(0.93, 0.06), flatPt(0.93, 0.94), flatPt(0.07, 0.94)].join(' ')} />
      <line class="pf-notch-line" x1="101.8" y1="88.4" x2="117" y2="88.4" stroke-width="3.5" stroke-linecap="round" />
      {/* top edge points away from the participant, along the phone's long axis */}
      <g class="pf-arrow" aria-hidden="true">
        <line x1="113.2" y1="76.2" x2="118.2" y2="60.4" stroke-width="4" stroke-linecap="round" />
        <polygon points="121.6,49.6 112.3,59.6 123.9,63.2" />
      </g>
      <text class="pf-label" x="100" y="22" text-anchor="middle">
        {props.screenUp}
      </text>
    </g>
  );
}

export function PhoneFigure(props: { orientation: OrientationId; hand?: Hand; size?: number }) {
  const s = useStrings();
  const size = props.size ?? 200;
  const spec = phoneFigureSpec(props.orientation);
  const label = s.common.orientations[props.orientation]?.label ?? props.orientation;
  const rot = spec.rotationDeg;
  const topLabel = TOP_LABEL[rot];

  return (
    <div class="phone-figure" style={{ width: `${size}px`, height: `${size}px` }} data-orientation={props.orientation}>
      <svg viewBox="0 0 200 200" width={size} height={size} role="img" aria-label={label} focusable="false">
        {spec.side === 'flat' ? (
          <FlatPhone screenUp={s.session.figure.screenUp} />
        ) : spec.side === 'free' ? (
          <ScreenSide question />
        ) : (
          <>
            <g transform={rot === 0 ? undefined : `rotate(${rot} 100 100)`}>
              {spec.side === 'screen' ? <ScreenSide /> : <BackSide />}
              <TopArrow />
            </g>
            <text class="pf-label" x={topLabel.x} y={topLabel.y} text-anchor="middle">
              {s.session.figure.top}
            </text>
          </>
        )}
      </svg>
      {props.hand && (
        <span class={`pf-hand pf-hand-${props.hand}`} aria-hidden="true">
          {s.common.handShort[props.hand]}
        </span>
      )}
    </div>
  );
}
