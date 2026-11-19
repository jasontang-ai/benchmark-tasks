"""Lock a steering wheel with a Club: hook it across the wheel and pull the sliding half out until it bites the rim.

On the station table, straight ahead of the robot: a 14 in steering wheel on a plywood column stand with a dash,
instrument hood and windshield, its face tilted 25 degrees like a car's. In front of it the Club (Winner International
The Club 1000) stands on its two hooks.

- The Club is two telescoping steel pieces with one hook each; the hooks open toward the bar's ends, so they fit
  inside the rim and bite outward. The outer piece carries the lock, the long handle and the upper hook; the inner bar,
  with the lower hook, slides out of it on a self-locking ratchet (it extends, it never slides back).
- The hooks open toward the bar's ends, so the lower hook can carry the Club on the bottom of the rim but the upper
  one can't. Either hook may go on first (the maker: one hook first, then pull the other until it fits snugly); the
  demo seats the upper hook under the rim's 12 o'clock, holds the whole 1.6 kg Club up there with one hand and pulls
  the inner bar down along the bar's own line with the other. Stop where the second hook first touches and the
  ratchet holds the bar a tooth short: on release it rattles loose. The score reads only the end state.
- Around the wheel, as in a car: the dash, an instrument hood 130 mm behind the rim's top and a windshield above it,
  all part of the stand, so the 558 mm bar is swung in under them from the driver's side.

On the kit: the wheel stand is one heavy rigid body; the Club is a free body (the outer piece) with the inner bar on a
slider (dry drag 5 N, static = sliding). The ratchet is one rule: the slider's retract limit follows the last 4.8 mm
tooth the bar has passed (a pawl). The hooks hold the rim by contact alone.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ourkit as k  # noqa: E402
from ourkit import demo as D  # noqa: E402
from ourkit.rules import Rule, lock_limits  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.0  # the full score already needs 1 s still with the hands clear
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Lock the steering wheel with the Club across it, its handle pointing up, and let go of it."
MM = 0.001
R = math.radians

# == the wheel (x forward from the arm bases' midpoint, y left, z up from the table top) ================================
TILT = R(25.0)  # the wheel's face leans back 25 degrees from vertical, as a car's
C = np.array([0.300, 0.0, 0.200])  # wheel centre
U = np.array([math.sin(TILT), 0.0, math.cos(TILT)])  # 12 o'clock, along the face
ND = np.array([-math.cos(TILT), 0.0, math.sin(TILT)])  # the face's normal toward the robot (the driver's side)
Y = np.array([0.0, 1.0, 0.0])  # 9 o'clock seen from the driver (the robot's left)
RIM_OD, GRIP_T = 0.356, 0.026  # 14 in wheel, its grip 26 mm thick
R_C = RIM_OD / 2 - GRIP_T / 2  # 0.165 m: the grip's centre circle
R_IN = RIM_OD / 2 - GRIP_T  # 0.152 m: the grip's inner edge (inner diameter 304 mm)
N_SEG = 40  # the round grip as 40 straight pieces of square section, flat faces at 12 and 6 o'clock
DISH = 0.095  # 3.75 in from the grip's centre plane back to the hub face
SPOKES_AT = (45.0, 135.0, 225.0, 315.0)  # four spokes, clockwise from 12 o'clock seen from the driver
STAND_KG = 30.0  # the stand with its dash and windshield, held as its two C-clamps hold it to the table
STAND_MOVE = 0.010  # the stand moved further than this: score 0
VINYL = 0.6  # friction of the grip's vinyl and the Club's vinyl coat (static = dynamic)

# == the Club (its own frame: x across, y along the bar toward the handle, z toward the driver when fitted) ==========
TUBE, INNER = 0.025, 0.019  # outer tube and inner bar, square sections
HOOK_T, THROAT, LIP, LIP_T = 0.015, 0.038, 0.022, 0.012  # hook rod, U opening (bar to lip), lip length, lip thickness
S0 = 0.229  # m between the two hook stems' centres with the inner bar pushed fully in
EXT_MAX = 0.171  # the inner bar's travel: 244 mm inside the hooks, closed; 415 mm fully out (the maker's maximum)
TUBE_END = -0.185  # the outer tube's mouth, 185 mm from the upper stem
HANDLE_END = 0.299  # the handle's end past the upper stem: 558 mm overall, closed
LOCK = (0.038, 0.070)  # lock body: across, long
PITCH = 0.0048  # ratchet: one tooth every 3/16 in of travel
DRAG = 5.0  # N: the inner bar's sliding drag (spring scale)
OUTER_KG, INNER_KG = 1.2, 0.4  # 1.6 kg in all
LIP_Z = -(TUBE / 2 + THROAT)  # the lips' front faces in the Club's frame (z), -50.5 mm
THROAT_MID = -TUBE / 2 - THROAT / 2  # -31.5 mm: the grip's centre in the middle of a hook's throat
STEM_Z0 = LIP_Z - LIP_T  # the hooks' lowest point, -62.5 mm: what the Club stands on at the start
BAR_OFF = TUBE / 2 + (THROAT - GRIP_T) / 2 + GRIP_T / 2  # 31.5 mm: bar axis in front of the grip's centre plane when fitted
RED, STEEL, BLACK, BIRCH, ALU = (0.78, 0.08, 0.07), (0.55, 0.56, 0.58), (0.08, 0.08, 0.09), (0.84, 0.74, 0.58), \
    (0.72, 0.73, 0.75)
DASH_GREY, GLASS = (0.24, 0.25, 0.27), (0.62, 0.74, 0.80)
OPEN_IN = S0 + HOOK_T  # 244 mm inside the hooks with the inner bar fully in
SNUG_EXT = 2 * R_IN - OPEN_IN  # 60 mm out: both hooks touch the grip
SNUG_PLAY = 0.0048  # snug: play at the upper hook one tooth or less
CLEAR_GAP, STILL_S, STILL_MM = 30.0, 1.0, 0.001  # released: grippers 30 mm clear; still: under 1 mm over 1 s
HANDLE_UP_DEG = 30.0  # handle within 30 degrees of 12 o'clock


def play(ext: float) -> float:
    """Play at the upper hook, m, with the Club resting on its lower hook: the grip's inner diameter less the hooks'
    inside spacing. A pawl tooth every 4.8 mm, so pulled to the grip the bar rests on the tooth 2.4 mm short."""
    return 2 * R_IN - (OPEN_IN + ext)


def rot_rpy(m) -> tuple[float, float, float]:
    """(roll, pitch, yaw) for a rotation matrix (the kit's turn = Rz Ry Rx)."""
    m = np.asarray(m, dtype=float)
    pitch = math.asin(float(np.clip(-m[2, 0], -1.0, 1.0)))
    return (math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0]))


def face(phi_deg: float, r: float, d: float = 0.0) -> np.ndarray:
    """A point on the wheel: ``phi_deg`` clockwise from 12 o'clock seen from the driver, ``r`` from the centre along
    the face, ``d`` toward the driver off the grip's centre plane."""
    a = R(phi_deg)
    return C + r * (math.cos(a) * U - math.sin(a) * Y) + d * ND


# == the car around the wheel (part of the stand body) ================================================================
# A simplified stand-in. The instrument hood's face 130 mm behind the grip at 12 o'clock (Oregon's equipment rule asks
# at least 2 in between the wheel and the instrument panel; set back so the demo's swing clears it by 5 cm); the
# windshield's lower edge 445 mm (17.5 in, a tape measurement in a car) ahead of the wheel's top, the glass rising
# toward the driver at 37 degrees from horizontal.
WHEEL_TOP = face(0.0, RIM_OD / 2)  # top of the grip, x 0.375, z 0.361
GRIP_BACK_X = float(face(0.0, R_C + GRIP_T / 2, -GRIP_T / 2)[0])  # the grip's rearmost edge at 12 o'clock, x 0.387
HOOD_GAP = 0.130  # instrument hood's face behind the grip
HOOD_X = GRIP_BACK_X + HOOD_GAP
DASH_UP, HOOD_UP = 0.040, 0.080  # the dash top and the hood's top above the top of the wheel
DASH_TOP, HOOD_TOP = float(WHEEL_TOP[2]) + DASH_UP, float(WHEEL_TOP[2]) + HOOD_UP
HOOD_W, HOOD_D = 0.360, 0.150  # instrument hood: across, front to back
DASH_W, BOARD_W, DASH_END = 0.600, 0.600, 0.850  # dash and base board across; both run back to x 85
UPPER_DASH_Z, LOWER_DASH_X = 0.250, 0.560  # the upper dash's underside; the lower dash's face (knee room)
WS_DIST = 0.445  # the windshield's lower edge from the top of the wheel
WS_RAKE = math.radians(37.0)  # the glass from horizontal
WS_X0 = float(WHEEL_TOP[0]) + math.sqrt(WS_DIST ** 2 - (DASH_TOP - float(WHEEL_TOP[2])) ** 2)
WS_LEN, WS_W, WS_T = 0.500, 0.800, 0.006  # the glass: up its slope, across, thick


def _oriented(centre, x_axis, y_axis, size, colour, label=""):
    x = np.asarray(x_axis, float) / np.linalg.norm(x_axis)
    y = np.asarray(y_axis, float) - np.dot(y_axis, x) * x
    y /= np.linalg.norm(y)
    return k.box(tuple(centre), size, colour, turn=rot_rpy(np.column_stack([x, y, np.cross(x, y)])), label=label)


def _wheel() -> list:
    out = []
    seg = 2 * R_C * math.tan(math.pi / N_SEG) * 1.12
    for i in range(N_SEG):
        phi = 360.0 * i / N_SEG
        a = R(phi)
        radial = math.cos(a) * U - math.sin(a) * Y
        tangent = np.cross(ND, radial)
        out.append(_oriented(face(phi, R_C), tangent, radial, (seg, GRIP_T, GRIP_T), BLACK,
                             label="grip_top" if i == 0 else ("grip_bottom" if i == N_SEG // 2 else "")))
    hub = C - DISH * ND
    for phi in SPOKES_AT:
        a = R(phi)
        radial = math.cos(a) * U - math.sin(a) * Y
        p0, p1 = face(phi, R_IN - 0.004, -0.004), hub + 0.040 * radial
        out.append(_oriented((p0 + p1) / 2, p1 - p0, np.cross(ND, radial), (float(np.linalg.norm(p1 - p0)), 0.030, 0.006),
                             ALU))
    out.append(_oriented(hub - 0.020 * ND, U, Y, (0.090, 0.090, 0.040), ALU))  # hub and the 3-bolt adapter
    out.append(_oriented(hub + 0.012 * ND, U, Y, (0.064, 0.064, 0.024), BLACK))  # horn button
    col0, col1 = 0.115, 0.330  # column: 100 mm square, from behind the hub back to the post
    out.append(_oriented(C - ND * (col0 + col1) / 2, U, Y, (0.100, 0.100, col1 - col0), BIRCH))
    out.append(k.span((0.450, -BOARD_W / 2, 0.0), (DASH_END + 0.010, BOARD_W / 2, 0.018), BIRCH))  # base board, clamped
    out.append(k.span((0.470, -0.050, 0.018), (0.610, 0.050, 0.090), BIRCH))  # post under the column
    # the car around the wheel: lower dash (knee panel), upper dash, instrument hood, windshield on the dash's far edge
    out.append(k.span((LOWER_DASH_X, -DASH_W / 2, 0.018), (DASH_END, DASH_W / 2, UPPER_DASH_Z), DASH_GREY))
    out.append(k.span((HOOD_X, -DASH_W / 2, UPPER_DASH_Z), (DASH_END, DASH_W / 2, DASH_TOP), DASH_GREY,
                      label="dash"))
    out.append(k.span((HOOD_X, -HOOD_W / 2, DASH_TOP - 0.001), (HOOD_X + HOOD_D, HOOD_W / 2, HOOD_TOP), BLACK,
                      label="instrument_hood"))
    along = np.array([-math.cos(WS_RAKE), 0.0, math.sin(WS_RAKE)])  # up the glass, toward the driver
    out_n = np.array([math.sin(WS_RAKE), 0.0, math.cos(WS_RAKE)])  # the glass's outer face
    ws0 = np.array([WS_X0, 0.0, DASH_TOP])
    out.append(_oriented(ws0 + along * WS_LEN / 2 + out_n * WS_T / 2, along, Y, (WS_LEN, WS_W, WS_T), GLASS,
                         label="windshield"))
    return out


WHEEL_BOXES = _wheel()
WHEEL = k.mech("wheel", (0.0, 0.0, TOP + 0.0005), WHEEL_BOXES, STAND_KG, color=BIRCH, surface=VINYL)

# -- the Club ---------------------------------------------------------------------------------------------------------


def _hook(sign: float) -> list:
    """A hook's stem (down from the bar, toward the wheel) and its lip (outward along the bar, behind the grip), in its
    own part's frame round the stem's centre. ``sign`` +1 for the upper hook (opens toward +y), -1 the lower."""
    h = HOOK_T / 2
    return [k.span((-h, -h, STEM_Z0), (h, h, 0.0), RED),
            k.span((-h, sign * h if sign > 0 else -h - LIP, STEM_Z0), (h, h + LIP if sign > 0 else -h, LIP_Z), RED)]


OUTER_BOXES = (
    k.box((0, (TUBE_END + 0.010) / 2, 0), (TUBE, 0.010 - TUBE_END, TUBE), STEEL, label="outer_tube"),
    k.span((-LOCK[0] / 2, 0.010, -TUBE / 2), (LOCK[0] / 2, 0.010 + LOCK[1], 0.025), STEEL, label="lock_body"),
    k.span((-TUBE / 2, 0.080, -TUBE / 2), (TUBE / 2, HANDLE_END, TUBE / 2), RED, label="handle"),
    *_hook(+1.0),
)
INNER_BOXES = (
    k.span((-INNER / 2, -HOOK_T / 2 - 0.030, -INNER / 2), (INNER / 2, S0 - 0.010, INNER / 2), STEEL, label="inner_bar"),
    *_hook(-1.0),
)
INNER = k.slider("inner", (0.0, -S0, 0.0), "Y", -EXT_MAX, 0.0, INNER_BOXES, friction=DRAG, damping=0.5,
                 mass=INNER_KG)
CLUB_YAW = R(-30.0)  # standing on its hooks, hooks down, the handle to the left and turned 30 degrees forward
GRIP_L = np.array([0.0, -0.016, 0.0])  # demo: left hand on the outer tube, 16 mm below the upper hook's stem
GRIP_R = np.array([0.0, -S0 + 0.022, 0.0])  # demo: right hand on the inner bar, 22 mm above the lower hook's stem
_MID = (GRIP_L + GRIP_R) / 2
CLUB_START = np.array([0.150 - (math.cos(CLUB_YAW) * _MID[0] - math.sin(CLUB_YAW) * _MID[1]),
                       0.080 - (math.sin(CLUB_YAW) * _MID[0] + math.cos(CLUB_YAW) * _MID[1]), -STEM_Z0])
CLUB = k.mech("club", (CLUB_START[0], CLUB_START[1], TOP + CLUB_START[2] + 0.0005), OUTER_BOXES, OUTER_KG,
              joints=[INNER], yaw=CLUB_YAW, color=STEEL, surface=VINYL)
OBJECTS = [WHEEL, CLUB]


def ext(st) -> float:
    """How far the inner bar is out, m (the slider runs to negative q)."""
    return -float(st["club"].q["inner"])


# -- the ratchet: the retract limit follows the last tooth passed ---------------------------------------------------


class Ratchet(Rule):
    """A self-locking ratchet on the inner bar: once the bar has passed a tooth (every ``PITCH`` of travel out), it
    can't slide back past it. It slides out freely against its drag."""

    def __init__(self):
        super().__init__("inner", "ratchet")
        self.tooth = 0

    def reset(self, ctx) -> None:
        super().reset(ctx)
        self.tooth = 0
        lock_limits(self.mech, self.joint, 0.0, "up")

    def step(self, ctx) -> None:
        n = int(math.floor((-ctx.q(self.joint) + 1e-4) / PITCH))
        if n > self.tooth:
            self.tooth = n
            lock_limits(self.mech, self.joint, -n * PITCH, "up")
            self.log(ctx, f"tooth {n}", -n * PITCH)


LOCKING = k.Rules(CLUB, [Ratchet()])

# == the scorer ========================================================================================================


def _wheel_frame(st):
    w = st["wheel"]
    rot = w.rotation()
    return w.to_world(C), rot @ U, rot @ ND


def hook_seat(st, which: str) -> tuple[float, float]:
    """(gap, depth) of a hook on the grip, m: ``gap`` from the hook stem's outer face to the grip's inner edge
    (0 touching, negative impossible), ``depth`` of the grip's centre in the hook's throat from the bar axis (-28.5 mm
    fitted). Read from the nearest point of the grip's centre circle to the hook's throat."""
    c = st["club"]
    part = "" if which == "upper" else "inner"
    sign = 1.0 if which == "upper" else -1.0
    throat = np.array([0.0, sign * (HOOK_T / 2 + LIP / 2), LIP_Z / 2 - TUBE / 4])
    t = c.to_world(throat, part)
    centre, _, n = _wheel_frame(st)
    v = t - centre
    v -= np.dot(v, n) * n
    q = centre + R_C * v / max(np.linalg.norm(v), 1e-9)
    r, tr = c.spec.frames(c.q)[part]
    local = r.T @ (c.rotation().T @ (q - c.pos) - tr)
    gap = sign * local[1] - (HOOK_T / 2 + GRIP_T / 2)
    return float(gap), float(local[2])


def hook_on(st, which: str) -> bool:
    """The grip passes through the hook's U: its inner edge within 10 mm of the stem and the grip inside the throat
    (the lip behind it, the bar in front)."""
    gap, depth = hook_seat(st, which)
    return -0.002 <= gap <= 0.010 and LIP_Z + GRIP_T / 2 - 0.004 <= depth <= -TUBE / 2 - GRIP_T / 2 + 0.004


def handle_angle(st) -> float:
    """Degrees between the handle's direction, in the wheel's face, and 12 o'clock."""
    a = st["club"].rotation() @ Y
    _, u, n = _wheel_frame(st)
    a = a - np.dot(a, n) * n
    return float(np.degrees(np.arccos(np.clip(np.dot(a, u) / max(np.linalg.norm(a), 1e-9), -1.0, 1.0))))


STAND, END = k.Sticky(), k.Settle(STILL_S, STILL_MM)
RUN = {"stand0": None}
REC = k.run_record.RunRecord(__file__, "club")


def grade(states) -> float:
    if STAND:
        return 0.0
    up, low = hook_on(states, "upper"), hook_on(states, "lower")
    return k.ladder((0.25, END.ok and (up or low)), (0.5, up and low), (0.75, play(ext(states)) <= SNUG_PLAY + 1e-9),
                    (1.0, handle_angle(states) <= HANDLE_UP_DEG))


def score(states) -> float:
    return grade(states)


def success(states) -> bool:
    return grade(states) >= 1.0


def distance(states) -> float:
    return max(0.0, play(ext(states)))


def _club_points(c):
    return np.concatenate([k.score.surface_points(c, "", OUTER_BOXES, 0.03),
                           k.score.surface_points(c, "inner", INNER_BOXES, 0.03)])


@k.on_reset
def _reset(states) -> None:
    STAND.reset(), END.reset()
    RUN["stand0"] = np.asarray(states["wheel"].pos, dtype=float)
    REC.reset()


@k.on_step
def _watch(states) -> None:
    c = states["club"]
    if np.linalg.norm(np.asarray(states["wheel"].pos) - RUN["stand0"]) > STAND_MOVE:
        STAND.trip("wheel stand moved")
    clear = k.hands_clear(_club_points(c), CLEAR_GAP)
    END.push(np.r_[c.pos, c.to_world((0, HANDLE_END, 0)), ext(states)], clear)
    REC.tick(states, lambda: (f"out {ext(states) * 1000:.1f} mm, play {play(ext(states)) * 1000:+.1f} mm, "
                              f"upper {hook_seat(states, 'upper')[0] * 1000:+.1f} mm, "
                              f"lower {hook_seat(states, 'lower')[0] * 1000:+.1f} mm, score {grade(states):g}"))


# == the demo ==========================================================================================================
# Both hands lift the standing Club, turn it to the wheel's face and put both hooks through the wheel inside the grip.
# The pair raise it until the upper hook's stem is under the grip at 12 o'clock; the left hand then keeps that hook
# there (a servo on the hook's live pose: it carries the whole Club and leads its sag) while the right pulls the inner
# bar down along the bar's line until its hook is on the grip at 6 o'clock, past the last tooth. Right lets go and
# backs off, then left; both go home and hold.


def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def _ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1.0, 0], [-math.sin(a), 0, math.cos(a)]])


R_FIT = _ry(-(math.pi / 2 - TILT)) @ _rz(-math.pi / 2)  # Club frame fitted: y along 12 o'clock, z toward the driver
G_FIT = D.Grip.deg(0, 90 - 25, 0)  # both hands at the end: pointing into the wheel along its axis, closing across
G_CLUB = R_FIT.T @ G_FIT.rotation  # the grip in the Club's frame (straight down at the start)
O0 = np.array(CLUB_START)
LIFT = 0.120
S_IN, D_PRE = R_IN - HOOK_T / 2 - LIP - 0.0085, 0.090  # upper stem 114 mm up while the hooks go through; 80 mm out
SEAT_PUSH = 0.003  # the upper stem sent 3 mm past the grip's inner edge
PULL_PAST = 0.006  # the hand sent on 6 mm once the inner bar is out to the grip
PULL_V = 0.015  # m/s down the bar's line
PULL_LAG = 0.100  # the hand may run this far ahead of the bar (the arm gives about 0.45 N/mm)
KEEP_MAX = 0.080  # the left hand's lead over its seated command, at most (it carries the whole Club)
CARRY_LEAD = 0.012  # m up while the arms carry the Club between them (about 8 N each at 0.45 N/mm)
OPEN_GAP = 0.050
N_TURN = 12
TURN_BOW = np.array([0.0, 0.080, 0.0])  # the turn bowed 80 mm to the left at its middle: keeps the left wrist off its limit
TURN_FROM = np.array([0.120, 0.0])  # the turn's line is laid from the Club's middle lifted over (12, 0) cm; from halfway
TURN_HALF = 0.5  # on it runs on that line, and before then the start's offset (3 cm forward, 8 cm left) fades out
TURN_OUT = 0.060  # along it, the Club carried up to 60 mm further left (the upper hook's lip kept off the grip at 9
HAND_X_MIN = 0.100  # o'clock) and pushed forward only where the right hand would come nearer the robot than this


def world(p) -> np.ndarray:
    return np.asarray(p, dtype=float) + np.array([0.0, 0.0, TOP])


def fitted_origin(s_stem: float, d: float) -> np.ndarray:
    """The Club's origin (the upper stem's centre on the bar axis) fitted with that stem ``s_stem`` up the face and the
    bar axis ``d`` in front of the grip's centre plane."""
    return C + s_stem * U + d * ND


def hand_at(rot, origin, local, q=0.0) -> np.ndarray:
    """A grasp point's world position for a Club pose; ``q`` the inner bar's slider (for the right hand's point)."""
    return world(origin + rot @ (np.asarray(local, dtype=float) + np.array([0.0, q, 0.0])))


def grip_of(rot) -> D.Grip:
    return D.Grip.from_matrix(rot @ G_CLUB)


def live_hand(st, local, part="") -> np.ndarray:
    return st["club"].to_world(local - (np.array([0.0, -S0, 0.0]) if part else 0.0), part)


class Hand:
    """One arm's plan whose every move stores its command (``cmd``), so a correction can add to it."""

    def __init__(self, arm: str):
        self.p = D.Plan(arm)
        self.cmd = None

    def go(self, seconds, target, grip=None, finger=None, label="move", track=False):
        if callable(target):
            fb = self.cmd if self.cmd is not None else D.P.home(self.p.arm)[0]

            def fn(st, f=target):
                self.cmd = np.asarray(f(st), dtype=float)
                return self.cmd
            self.p.move(seconds, fn, grip=grip, finger=finger, label=label, track=track, fallback=tuple(fb))
            self.cmd = np.asarray(fb, dtype=float)
        else:
            t = np.asarray(target, dtype=float)

            def fn(st, t=t):
                self.cmd = t
                return t
            self.p.move(seconds, fn, grip=grip, finger=finger, label=label, fallback=tuple(t))
            self.cmd = t
        return self


def demo_plan(how: str = "demo") -> dict:
    L, Rt = Hand("left"), Hand("right")
    r0 = _rz(CLUB_YAW)
    g0 = grip_of(r0)
    open_f, shut = D.gap_finger(OPEN_GAP), D.FINGER_CLOSED
    # 1. over the standing Club, down round its tube and bar, close
    for h, loc in ((L, GRIP_L), (Rt, GRIP_R)):
        p = hand_at(r0, O0, loc)
        h.go(2.5, p + [0, 0, 0.080], grip=g0, finger=open_f, label="over the Club")
        h.go(1.5, p, label="down round the bar")
        h.p.fingers(0.0, 1.0, "grip")
    # 2. lift together, then turn it to the wheel's face on the way over (one rigid motion for both hands)
    pre = fitted_origin(S_IN, D_PRE)
    mid0 = O0 + np.array([0.0, 0.0, LIFT]) + r0 @ ((GRIP_L + GRIP_R) / 2)
    mid1 = pre + R_FIT @ ((GRIP_L + GRIP_R) / 2)
    lead = np.array([0.0, 0.0, CARRY_LEAD])
    for h, loc in ((L, GRIP_L), (Rt, GRIP_R)):
        h.go(2.0, hand_at(r0, O0 + [0, 0, LIFT], loc) + lead, label="lift")
    from yam_arena import ik as _ik
    line0 = np.array([TURN_FROM[0], TURN_FROM[1], mid0[2]])
    for i in range(1, N_TURN + 1):
        s = i / N_TURN
        rot = _ik.slerp_matrix(r0, R_FIT, s)
        w = max(0.0, 1.0 - s / TURN_HALF)  # the start's offset, faded out by halfway
        mid = line0 + s * (mid1 - line0) + math.sin(math.pi * s) * TURN_BOW + w * (mid0 - line0) * Y \
            + (math.sin(math.pi * s / TURN_HALF) if s < TURN_HALF else 0.0) * TURN_OUT * Y
        org = mid - rot @ ((GRIP_L + GRIP_R) / 2)
        org[0] += max(w * (mid0[0] - line0[0]), HAND_X_MIN - float(hand_at(rot, org, GRIP_R)[0]))
        for h, loc in ((L, GRIP_L), (Rt, GRIP_R)):
            h.go(0.6, hand_at(rot, org, loc) + lead, grip=grip_of(rot), label="turn it to the wheel")
    g = grip_of(R_FIT)

    # 3. closed loop: each hand puts its grasp point where the plan has it (lead for the sag the pose shows)
    def fix(h, loc, part, origin, seconds, label):
        def fn(st, h=h, loc=loc, part=part):
            want = hand_at(R_FIT, origin, loc, float(st["club"].q["inner"]) if part else 0.0)
            have = live_hand(st, loc, part)
            return h.cmd + np.clip(want - have, -0.040, 0.040)
        h.go(seconds, fn, label=label)

    for _ in range(2):
        fix(L, GRIP_L, "", pre, 1.2, "line the hooks up in front of the grip")
        fix(Rt, GRIP_R, "inner", pre, 1.2, "line the hooks up in front of the grip")
    # 4. in along the wheel's axis, hooks through inside the grip; then up until the upper stem is under 12 o'clock
    seated = fitted_origin(R_IN - HOOK_T / 2 + SEAT_PUSH, BAR_OFF)
    inside = fitted_origin(S_IN, BAR_OFF)
    for h in (L, Rt):
        h.go(2.5, lambda st, h=h: h.cmd + (inside - pre), label="hooks in through the wheel")
    fix(L, GRIP_L, "", inside, 1.0, "hooks behind the grip")
    fix(Rt, GRIP_R, "inner", inside, 1.0, "hooks behind the grip")
    for h in (L, Rt):
        h.go(2.0, lambda st, h=h: h.cmd + (seated - inside), label="upper hook up under the grip")
    fix(L, GRIP_L, "", seated, 1.0, "upper hook seated")
    fix(Rt, GRIP_R, "inner", seated, 1.0, "upper hook seated")
    L.p.hold(0.5, "steady")
    Rt.p.hold(0.5, "steady")

    # 5. left keeps the upper hook seated (a servo on its live stem) while right pulls the inner bar down its line
    base = {}

    def keep(st):  # integral servo on the upper hook: its stem 0.3 mm under the grip, the grip mid-throat
        if base.get("trial") != k.LATEST.get("resets"):
            base.update(trial=k.LATEST.get("resets"), cmd=np.asarray(L.cmd, dtype=float))
        gap, depth = hook_seat(st, "upper")
        nxt = L.cmd + float(np.clip(0.35 * (gap - 0.0003), -0.003, 0.003)) * U \
            + float(np.clip(0.2 * (depth - THROAT_MID), -0.001, 0.001)) * ND
        off = np.clip(nxt - base["cmd"], -KEEP_MAX, KEEP_MAX)
        return base["cmd"] + off

    goal = SNUG_EXT - 0.001 if how != "slam" else SNUG_EXT - 0.012  # slam: stopped 12 mm short of the grip
    pull_s, after = 9.0, 0.5 + 1.0 + 1.5  # the pull; then hold, let go and back off
    sv = {}

    def pull(st):  # servo: down the bar's line at PULL_V, the inner bar kept in the wheel's face, until it is out
        if sv.get("trial") != k.LATEST.get("resets"):  # far enough; then on by PULL_PAST to bite
            sv.update(trial=k.LATEST.get("resets"), base=np.asarray(Rt.cmd, dtype=float), moved=0.0, fix=np.zeros(3),
                      done=None)
        e = ext(st)
        if sv["done"] is None and e >= goal:
            sv["done"] = sv["moved"]
            sv["here"] = live_hand(st, GRIP_R, "inner")  # where the hand has the bar now
        if how == "slam" and sv["done"] is not None:  # slam: stop pulling, the hand held where the bar is
            return sv["here"]
        limit = sv["done"] + (PULL_PAST if how != "slam" else 0.0) if sv["done"] is not None else SNUG_EXT + PULL_LAG
        sv["moved"] = min(limit, sv["moved"] + PULL_V / 20.0)
        centre, u, n = _wheel_frame(st)
        p = live_hand(st, GRIP_R, "inner") - centre
        depth = hook_seat(st, "lower")[1]  # the grip's centre in the lower hook's throat: keep it mid-throat
        err = (depth - THROAT_MID) * n + (0.0 - p[1]) * Y
        sv["fix"] = sv["fix"] + np.clip(0.3 * err, -0.002, 0.002)
        return sv["base"] - sv["moved"] * U + sv["fix"]

    if how == "wrong_order":  # left lets go of the Club before the pull
        L.p.fingers(OPEN_GAP, 1.0, "let go too early")
        L.go(1.5, lambda st: L.cmd - 0.060 * G_FIT.approach, label="back off")
    else:
        L.go(pull_s + after, keep, label="keep the upper hook seated", track=True)
    Rt.go(pull_s, pull, label="pull the inner bar down its line", track=True)
    Rt.p.hold(0.5, "hold")
    # 6. right lets go and backs off out of the wheel, then left
    Rt.p.fingers(OPEN_GAP, 1.0, "let go of the inner bar")
    Rt.go(1.5, lambda st: Rt.cmd - 0.070 * G_FIT.approach, label="back off")
    if how != "wrong_order":
        L.p.fingers(OPEN_GAP, 1.0, "let go of the Club")
        L.go(1.5, lambda st: L.cmd - 0.070 * G_FIT.approach, label="back off")
    L.p.home()
    Rt.p.home()
    return D.together(L.p, Rt.p)


def demo():
    return D.choose(demo=lambda: demo_plan("demo"), slam=lambda: demo_plan("slam"),
                    wrong_order=lambda: demo_plan("wrong_order"))


# -- what the template tests read ---------------------------------------------------------------------------------------
TEXT_NUMBERS = [("25°", 25.0), ("356 mm", RIM_OD), ("26 mm", GRIP_T), ("304 mm", 2 * R_IN),
                ("4.8 mm", PITCH), ("5 N", DRAG), ("1.6 kg", OUTER_KG + INNER_KG), ("558 mm", HANDLE_END + S0 + 0.030),
                ("244 mm", OPEN_IN), ("415 mm", OPEN_IN + EXT_MAX), ("2.4 mm", play(12 * PITCH)),
                ("30°", HANDLE_UP_DEG), ("30 mm", CLEAR_GAP / 1000), ("10 mm", STAND_MOVE),
                ("1 mm", STILL_MM), ("1 s", STILL_S), ("130 mm behind the grip", HOOD_GAP),
                ("4 cm above the top of the wheel", DASH_UP), ("8 cm above the top of the wheel", HOOD_UP),
                ("36 cm across", HOOD_W), ("60 cm across", DASH_W), ("445 mm", WS_DIST), ("37°", math.degrees(WS_RAKE)),
                ("6 mm clear acrylic", WS_T)]
CHEATS = {"none": 0.0, "slam": 0.5, "wrong_order": 0.5}  # mode: the most it may score
HOLD = {"club.inner": 0.5}  # joint: the most it may move with no command (mm)
REQUIRED_FORCES = {"pull the inner bar": 5.0}  # the left hand carries (a servo on the hook), it presses nothing
