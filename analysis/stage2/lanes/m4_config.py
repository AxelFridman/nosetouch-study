"""Lane m4: the one place for every evaluation and production choice that more than one module reads.

``m4_pipeline`` re-exports these names; ``m4_aggregate``, ``m4_production`` and ``m4_online`` import them from
here, so the sessions the aggregate scale is fitted on, the sessions it is evaluated on and the runtime stopping
rule can never drift apart.
"""
from __future__ import annotations

from stage2 import config as C

from .m4_detect import INSTRUCTED_STILLNESS_S

# ───────────────────────────── stress test of single recordings ─────────────────────────────
#: stillness kept before the approach (s); 'recorded' (nothing cut) is added by the compute step
KEEP_S_STRESS = (0.0, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0)
#: stillness settings at which every held-out recording is ALSO scored as its screen-away twin (the phone turned
#: 180 deg about its long axis: standardize.screen_flip_channels on the raw device vectors)
STRESS_FLIP_KEEPS = ('recorded', INSTRUCTED_STILLNESS_S)
#: Grip check of the inference page (model-free; m4_detect.rest_up_z): while the phone rests at arm's length before
#: the first movement, the screen-normal component of the up vector must be at least this value (the screen faces
#: the user, tilted upwards); below it the page does not answer and asks to turn the screen towards the face. The
#: production detector reads the screen-normal displacement and pose, which change sign with a screen-away grip.
#: (The first remedy, a check of the chosen segment's own start pose, missed the screen-away recordings whose return
#: was chosen: the stress test keeps it as 'gate_seg' for the record.)
GRIP_GATE_UPZ = 0.0
#: the rest pose is read from the still samples before the first movement, or from the first GRIP_REST_MIN_S seconds
#: of the stream when there are fewer still samples than that
GRIP_REST_MIN_S = 0.25

# ───────────────────────────── synthetic multi-attempt sessions ─────────────────────────────
KEEP_S_SESSIONS = (0.0, 0.5, 1.0, 2.0)            # stillness at each junction (s)
#: recordings per synthetic session: enough that K_MAX accepted attempts can be reached although about a third of
#: the true approaches are not accepted (the previous 5-recording sessions stopped short of K_MAX in 27 %)
SESSION_RECORDINGS = 8
#: a new session starts every SESSION_STRIDE recordings of a (person, hand) cell's shuffled list (sessions overlap
#: when SESSION_STRIDE < SESSION_RECORDINGS; the list wraps around), so the number of sessions stays as before
SESSION_STRIDE = 5
K_MAX = 5                                         # maximum accepted attempts of the stopping rule
#: attempt rules compared on the sessions: the pairing rule (the bout after an accepted approach is its return and
#: never an attempt) and the plain rule (every bout classified as an approach is an attempt)
SESSION_RULES = {'pairing': True, 'no_pairing': False}
#: the production attempt rule: plain. The pairing rule discarded true approaches that followed an accepted
#: arm-settling movement (recall 67 % against 77 %, precision unchanged; previous sessions, lane audit), so it is
#: dropped; both rules are still evaluated (exploratory, counted in N_CONFIGS_SESSIONS)
PRODUCTION_PAIRING = False

# ───────────────────────────── stopping rule ─────────────────────────────
P_STOP_GRID = (0.75, 0.8, 0.85, 0.9, 0.95)        # stopping thresholds evaluated
P_STOP = 0.9                                      # product choice: stop when the aggregate says 90 % one hand
P_ANSWER_GRID = (0.5, 0.6, 0.7, 0.8)              # after K_MAX attempts: answer only when max(P, 1 - P) >= this
P_ANSWER = 0.5                                    # production: always answer after K_MAX attempts
ABSTAIN_TAU = 0.7                                 # single-touch confidence threshold reported as an operating point
#: aggregations of the attempts the stopping rule is evaluated with (m4_online.AGGREGATIONS), and how many
#: parameters each one fits on the training persons (the order of simplicity of the declared choice rule below)
STOP_AGGREGATIONS = ('scale_recordings', 'scale_sessions', 'mean')
AGGREGATION_N_PARAMS = {'mean': 0, 'scale_sessions': 2, 'scale_recordings': K_MAX}
#: Declared rule for the production aggregation (applied to the saved session results by
#: m4_pipeline.production_aggregation, never typed): among the aggregations whose stopping-rule answer is
#: consistent with calibration (ECE inside its 95 % perfect-calibration null band AND calibration-slope 95 % CI
#: containing 1), the one with the fewest fitted parameters; if none qualifies, the one with the lowest log loss.
#: An exploratory choice on the outer sessions (counted in N_CONFIGS_SESSIONS).
AGGREGATION_RULE = 'calibrated_then_simplest'

# ───────────────────────────── statistics ─────────────────────────────
N_BOOT = C.N_BOOT                                 # person-cluster bootstrap resamples everywhere in the lane
MIN_PERSONS_CI = 5                                # strata with fewer persons get no bootstrap CI
ALPHA = 0.05                                      # family-wise level of the Holm-adjusted paired tests
