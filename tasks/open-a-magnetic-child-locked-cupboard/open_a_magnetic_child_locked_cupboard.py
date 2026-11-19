"""Open a magnetic child-locked cupboard, on ourkit.

A low melamine cabinet stands on the table with its one door shut. A Safety 1st magnetic child lock is screwed inside
the door's top left corner, its catch under the cabinet top: nothing of it shows outside except a small sticker where
the parents marked it. The magnetic key, a 40 mm puck, lies on a marked circle beside the cabinet.

- The latch is sprung. While the key is held flat over it the latch is pulled free; the moment the key leaves, it
  drops back. Its hook needs 6 mm of door travel at the lock to clear the catch, so the key has to stay on the door,
  riding with it, while the other hand pulls the knob that far.
- Past the catch the door swings freely on its hinge (friction holds it where it is left).

On the kit: the door is a hinge; the lock is a ``Latch`` on it (catches every 0.5 mm of travel up to the 6 mm that
clears the catch, as the hook that drops back on the catch's lip), freed only while the key's pose is over the lock.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ourkit as k  # noqa: E402
from ourkit import demo as D  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.0
TOP = TABLE_TOP[SCENE]
R = math.radians
INSTRUCTION = "Open the cupboard door wide and put the key back on its spot."

# -- sizes (x forward from the arm bases' midpoint, y left, z up from the table top) ---------------------------------
CAB_W, CAB_D, CAB_H, PANEL = 0.330, 0.300, 0.340, 0.016
CAB_FRONT, CAB_Y0 = 0.375, -0.270  # carcass front edge, its right side
DOOR_W, DOOR_H, DOOR_T, DOOR_Z0 = 0.320, 0.320, 0.018, 0.010
DOOR_KG, CAB_KG = 1.4, 30.0  # MDF door; 30 kg stands in for the screws to the table
HINGE_Y = CAB_Y0 + 0.005  # the door's right edge, hinge axis at its inner face
HINGE_Z = DOOR_Z0 + DOOR_H / 2
HINGE_FRIC = 0.05  # N m: the door stays where it is left
LOCK_IN, LOCK_Z = 0.030, 0.300  # lock centre: in from the door's free edge, height
KNOB_IN, KNOB_Z = 0.035, 0.150
KNOB_D, KNOB_PROUD, KNOB_HEAD = 0.030, 0.025, 0.012
KEY_D, KEY_T, KEY_KG = 0.040, 0.014, 0.025
SPOT_D, SPOT = 0.060, (0.300, 0.200)  # the key's marked circle on the table
FRICTION = 0.5

# -- the lock (fidelity) --------------------------------------------------------------------------------------------
PLAY, CLEAR, NOTCH = 0.001, 0.006, 0.0005  # m of door travel at the lock: play, hook clears the catch, catch spacing
REL_R, REL_GAP, REL_TILT = 0.012, 0.005, R(15)  # the key frees the latch: centre within, face within, tilt under

# -- scoring --------------------------------------------------------------------------------------------------------
PAST_Q, OPEN_Q = R(3), R(45)
FLAT = R(10)
CLEAR_MM, STILL_S, STILL_MM, MOVE_MM = 40.0, 1.0, 1.0, 5.0

# door-frame points (the hinge axis at the door's inner face, joint at 0)
FACE_X = -DOOR_T - 0.001  # outer face
LOCK_L = (FACE_X, DOOR_W - LOCK_IN, LOCK_Z - HINGE_Z)
KNOB_L = (FACE_X - KNOB_PROUD + KNOB_HEAD / 2, DOOR_W - KNOB_IN, KNOB_Z - HINGE_Z)
R_LOCK = math.hypot(LOCK_L[0], LOCK_L[1])
CATCHES = [(PLAY + i * NOTCH) / R_LOCK for i in range(int(round((CLEAR - PLAY) / NOTCH)) + 1)]

WHITE, MDF, GREY, ZINC, KEYC, MARK = (0.93, 0.93, 0.91), (0.95, 0.95, 0.94), (0.70, 0.71, 0.73), (0.62, 0.63, 0.66), \
    (0.30, 0.62, 0.36), (0.82, 0.82, 0.84)
SPAWN_GAP = 0.0005


def _disc(prefix, centre, d, t, axis, color):
    """A round disc of 8 bars, 22.5 deg apart, about ``axis`` ("x" or "z")."""
    out = []
    for i in range(8):
        a = math.radians(22.5 * i)
        if axis == "z":
            out.append(k.box(centre, (d, 0.4 * d, t), color, turn=(0.0, 0.0, a), label=f"{prefix}_{i}"))
        else:
            out.append(k.box(centre, (t, d, 0.4 * d), color, turn=(a, 0.0, 0.0), label=f"{prefix}_{i}"))
    return out


def _carcass() -> list:
    s, x0, x1, y0, y1 = k.span, CAB_FRONT, CAB_FRONT + CAB_D, CAB_Y0, CAB_Y0 + CAB_W
    return [s((x0, y0, 0), (x1, y0 + PANEL, CAB_H), WHITE, label="side_right"),
            s((x0, y1 - PANEL, 0), (x1, y1, CAB_H), WHITE, label="side_left"),
            s((x0, y0 + PANEL, CAB_H - PANEL), (x1, y1 - PANEL, CAB_H), WHITE, label="top"),
            s((x0, y0 + PANEL, 0), (x1, y1 - PANEL, PANEL), WHITE, label="bottom"),
            s((x1 - PANEL, y0 + PANEL, PANEL), (x1, y1 - PANEL, CAB_H - PANEL), WHITE, label="back"),
            k.box((x0 + 0.012, HINGE_Y + DOOR_W - LOCK_IN, CAB_H - PANEL - 0.006), (0.020, 0.030, 0.012), GREY,
                  ghost=True, label="catch")]


KNOB_C = (FACE_X - KNOB_PROUD + KNOB_HEAD / 2, DOOR_W - KNOB_IN, KNOB_Z - HINGE_Z)
DOOR_BOXES = [
    k.box((-DOOR_T / 2 - 0.001, DOOR_W / 2, 0.0), (DOOR_T, DOOR_W, DOOR_H), MDF, label="door_slab"),
    k.box((FACE_X - (KNOB_PROUD - KNOB_HEAD) / 2, KNOB_C[1], KNOB_C[2]), (KNOB_PROUD - KNOB_HEAD, 0.012, 0.012), ZINC,
          label="knob_stem"),
    *_disc("knob", KNOB_C, KNOB_D, KNOB_HEAD, "x", ZINC),
    k.box((FACE_X - 0.0002, LOCK_L[1], LOCK_L[2]), (0.0004, 0.020, 0.020), (0.95, 0.80, 0.30), ghost=True,
          label="sticker"),
    k.box((0.008, LOCK_L[1], LOCK_L[2]), (0.016, 0.030, 0.050), GREY, ghost=True, label="lock_body"),
]
DOOR = k.hinge("door", (CAB_FRONT, HINGE_Y, HINGE_Z), "Z", 0.0, R(110), DOOR_BOXES, friction=HINGE_FRIC,
               damping=0.02, mass=DOOR_KG)
CUPBOARD = k.mech("cupboard", (0.0, 0.0, TOP + SPAWN_GAP), _carcass(), CAB_KG, joints=[DOOR], color=WHITE,
                  surface=FRICTION)

KEY_BOXES = _disc("key", (0, 0, 0), KEY_D, KEY_T, "z", KEYC)
KEY_REST_Z = TOP + SPAWN_GAP + 0.0005 + KEY_T / 2
KEY = k.mech("key", (SPOT[0], SPOT[1], KEY_REST_Z + SPAWN_GAP), KEY_BOXES, KEY_KG, color=KEYC, surface=FRICTION)
SPOT_MARK = Cylinder("key_spot", pos=(SPOT[0], SPOT[1], TOP + 0.00025), radius=SPOT_D / 2, height=0.0005, color=MARK)
OBJECTS = [CUPBOARD, KEY, SPOT_MARK]


# -- the key over the lock ------------------------------------------------------------------------------------------
def lock_point(states):
    """The lock centre on the door's outer face, and the face's outward normal, in the world."""
    s = states["cupboard"]
    p = s.to_world(LOCK_L, "door")
    n = s.to_world((LOCK_L[0] - 1.0, LOCK_L[1], LOCK_L[2]), "door") - p
    return p, n / np.linalg.norm(n)


def key_pose(states):
    s = states["key"]
    return np.asarray(s.pos, dtype=float), s.rotation()[:, 2]


def key_over(states) -> bool:
    p, n = lock_point(states)
    c, a = key_pose(states)
    d = c - p
    along = float(d @ n)
    lateral = float(np.linalg.norm(d - along * n))
    gap = along - KEY_T / 2
    tilt = math.acos(min(1.0, abs(float(a @ n))))
    return lateral <= REL_R and -0.003 <= gap <= REL_GAP and tilt <= REL_TILT


def key_home(states) -> bool:
    c, a = key_pose(states)
    off = math.hypot(c[0] - SPOT[0], c[1] - SPOT[1])
    tilt = math.acos(min(1.0, abs(float(a[2]))))
    return off <= (SPOT_D - KEY_D) / 2 and tilt <= FLAT and c[2] - TOP < KEY_T


LOCK = k.Rules(CUPBOARD, [k.Latch("door", CATCHES, side="up", start=CATCHES[0],
                                  engage=lambda c, at: not key_over(c.states),
                                  release=lambda c, at: key_over(c.states))])

# -- the scorer -----------------------------------------------------------------------------------------------------
MOVED, END = k.Sticky(), k.Settle(STILL_S, STILL_MM / 1000.0)
RUN = {"cab0": None, "n": 0, "door": "", "key": ""}  # traces every 0.5 s, for the run report
REC = k.run_record.RunRecord(__file__, "cupboard")


def grade(states) -> float:
    if MOVED:
        return 0.0
    q = float(states["cupboard"].q["door"])
    if q < PAST_Q - 1e-9:
        return 0.25 if key_over(states) else 0.0
    if q < OPEN_Q - 1e-9:
        return 0.5
    return 1.0 if key_home(states) and END.ok else 0.75


def score(states) -> float:
    return grade(states)


def success(states) -> bool:
    return grade(states) >= 1.0


def distance(states) -> float:
    return max(0.0, OPEN_Q - float(states["cupboard"].q["door"]))


def _watched(states):
    s = states["cupboard"]
    pts = [k.score.surface_points(s, "door", DOOR_BOXES[:2] + DOOR_BOXES[2:10], 0.03),
           k.score.surface_points(states["key"], "", KEY_BOXES[:1], 0.01)]
    return np.concatenate(pts)


@k.on_reset
def _reset(states) -> None:
    MOVED.reset(), END.reset()
    RUN.update(cab0=[float(v) for v in states["cupboard"].pos])
    REC.reset()
    RUN["n"] = 0
    RUN["door"], RUN["key"] = "", ""


@k.on_step
def _watch(states) -> None:
    s = states["cupboard"]
    if np.linalg.norm(np.asarray(s.pos) - RUN["cab0"]) > MOVE_MM / 1000.0:
        MOVED.trip("cabinet moved")
    edge = s.to_world((FACE_X, DOOR_W, 0.0), "door")
    c, _ = key_pose(states)
    clear = k.hands_clear(_watched(states), CLEAR_MM)
    END.push([edge[0], edge[1], *c], clear)
    RUN["clear"], RUN["spread_mm"] = bool(clear), round(float(END.spread) * 1000, 2) if END.rows else -1.0
    RUN["n"] += 1
    if RUN["n"] % 10 == 0:
        RUN["door"] += f" {math.degrees(float(s.q['door'])):.0f}"
        RUN["key"] += str(int(key_over(states)))
        kc, ka = key_pose(states)
        RUN["key_end"] = (f"{(kc[0] - SPOT[0]) * 1000:+.1f},{(kc[1] - SPOT[1]) * 1000:+.1f},{(kc[2] - TOP) * 1000:.1f} mm "
                          f"tilt {math.degrees(math.acos(min(1.0, abs(float(ka[2]))))):.1f} home {key_home(states)} "
                          f"settled {END.ok}")
    REC.tick(states, lambda: f"door {math.degrees(float(s.q['door'])):.1f} deg, key over {key_over(states)}, "
                             f"score {grade(states):g}")


# -- the demo -------------------------------------------------------------------------------------------------------
G_DOWN, G_FWD = D.Grip.deg(), D.Grip.deg(pitch=90)
KEY_GRASP_Z = 0.018  # TCP height over the table for the pinch: pad tips 3 mm above the table
KEY_OPEN, KEY_SQUEEZE = 0.056, 0.003
KNOB_OPEN, KNOB_SQUEEZE = 0.046, 0.003
PRESS = 0.003  # key commanded this far into the door face
UP = 0.10


def door_q(st) -> float:
    return float(st["cupboard"].q["door"])


def door_pt(q, local):
    """A door-frame point in the world at door angle q (the cabinet as built: it is heavy and does not move)."""
    c, s_ = math.cos(q), math.sin(q)
    x, y, z = local
    return np.array([CAB_FRONT + c * x - s_ * y, HINGE_Y + s_ * x + c * y, TOP + SPAWN_GAP + HINGE_Z + z])


def key_hand(how: str) -> D.Plan:
    p = D.Plan("left")
    sx, sy = SPOT
    k_fb = (sx, sy, TOP + KEY_GRASP_Z)

    def over_key(dz):
        return lambda st: (float(st["key"].pos[0]), float(st["key"].pos[1]), TOP + dz)

    p.move(2.5, over_key(UP), finger=D.gap_finger(KEY_OPEN), grip=G_DOWN, label="over the key",
           fallback=(sx, sy, TOP + UP))
    p.move(1.5, over_key(KEY_GRASP_Z), label="down to the key", fallback=k_fb)
    p.fingers(KEY_D - 2 * KEY_SQUEEZE, 0.8, "pinch the key")
    if how in ("pull_only",):
        return p.fingers(KEY_OPEN, 0.8, "let go").home()
    p.move(1.5, (sx, sy, TOP + UP), label="lift the key")

    def at_lock(stand_off):
        def fn(st):
            L, n = lock_point(st)
            want = L + n * (KEY_T / 2 + stand_off)
            c, _ = key_pose(st)
            off = p.memo.setdefault("off", None)
            return want + (p.memo["off"] if off is not None else n * 0.011)
        return fn

    p0 = door_pt(0.0, LOCK_L) + np.array([-(KEY_T / 2 + 0.020) - 0.011, 0, 0])
    p.move(3.0, tuple(p0), grip=G_FWD, label="turn the key to face the door, 20 mm off the lock")
    p.hold(0.5, "steady")

    def measure(st):  # the hand-to-key offset, read once the key faces the door
        c, _ = key_pose(st)
        p.memo["off"] = np.asarray(p0) - c
        return p0
    p.move(0.5, measure, label="read the key in the hand", fallback=tuple(p0))
    p_in = door_pt(0.0, LOCK_L) + np.array([-(KEY_T / 2 - PRESS) - 0.011, 0, 0])
    p.move(1.5, at_lock(-PRESS), track=True, label="press the key flat over the lock", fallback=tuple(p_in))
    if how == "tap":
        p.hold(1.0, "hear the click")
        p.move(1.0, tuple(p0), label="take the key away")
        p.move(2.0, (0.30, 0.14, TOP + LOCK_Z), label="out of the door's way")
    else:
        hold = 32.0 if how == "held" else 5.5
        D.follow(p, at_lock(-PRESS), hold, fallback=tuple(p_in), label="keep the key on the lock as the door moves")
        if how == "held":
            return p
        D.follow(p, at_lock(0.025), 1.0, fallback=tuple(p0), label="lift the key off the door")
        p.move(2.0, (0.30, 0.14, TOP + LOCK_Z), label="out of the door's way")
    drop = (0.22, 0.20) if how == "no_return" else SPOT
    p.move(3.0, (drop[0], drop[1], TOP + UP), grip=G_DOWN, label="key flat over its spot")
    p.move(1.5, (drop[0], drop[1], TOP + KEY_GRASP_Z + 0.002), label="set the key down")
    p.fingers(KEY_OPEN, 0.8, "let go of the key")
    p.move(1.2, (drop[0], drop[1], TOP + UP), label="rise clear")
    return p.home()


LEVEL_TO, PULL_TO, END_PITCH = R(30), R(55), 35.0
PARK_R = (0.14, -0.30, TOP + 0.18)  # right hand parks here, about 16 cm from the open door  # pulled level to 30 deg, then on to 55 deg tilting the hand down


def knob_hand(how: str) -> D.Plan:
    """Pinch the knob head level, the pinch point at its centre, and pull the door round its arc: past the catch,
    then on to 30 deg; from there keep pulling while the hand pitches down about the pinch line (the knob stays put
    between the pads), so the wrist rises clear of the door as the knob comes toward the robot."""
    p = D.Plan("right")

    def at(q, back=0.0):
        c, s_ = math.cos(q), math.sin(q)
        return tuple(door_pt(q, KNOB_C) - np.array([back * c, back * s_, 0]))

    def grip(q, pitch=90.0):
        return D.Grip.deg(yaw=math.degrees(q), pitch=pitch)

    p.move(3.0, at(0.0, 0.07), finger=D.gap_finger(KNOB_OPEN), grip=grip(0.0), label="in line with the knob")
    p.move(2.0, at(0.0), label="slide over the knob")
    p.fingers(KNOB_D - 2 * KNOB_SQUEEZE, 0.8, "grip the knob")
    if how == "held":
        return p.hold(30.0).fingers(KNOB_OPEN, 0.8, "let go").home()
    wait = 12.0 - p.duration if how != "tap" else 15.0 - p.duration
    p.hold(max(0.5, wait), "wait for the key")
    p.move(3.0, at(R(5)), grip=grip(R(5)), label="pull the door past the catch")
    p.hold(5.5, "hold while the key hand clears")
    for deg in range(10, int(round(math.degrees(PULL_TO))) + 1, 5):
        q = R(deg)
        f = max(0.0, (q - LEVEL_TO) / (PULL_TO - LEVEL_TO))
        p.move(1.2, at(q), grip=grip(q, 90.0 - f * (90.0 - END_PITCH)), label=f"pull to {deg} deg")
    p.hold(0.8, "steady")
    p.fingers(KNOB_OPEN, 0.8, "let go of the knob")
    p.back(0.05, 1.2, "back off the knob")
    p.move(2.5, PARK_R, grip=G_DOWN, label="park clear of the open door")
    return p.hold(3.0, "hold still")


def demo():
    def two(how):
        return D.together(key_hand(how), knob_hand(how))

    return D.choose(demo=lambda: two("demo"), pull_only=lambda: two("pull_only"), tap=lambda: two("tap"),
                    held=lambda: two("held"), no_return=lambda: two("no_return"))


# -- what the template tests read -------------------------------------------------------------------------------------
TEXT_NUMBERS = [("40 mm", KEY_D), ("14 mm", KEY_T), ("25 g", KEY_KG), ("320 mm", DOOR_W), ("18 mm", DOOR_T),
                ("1.4 kg", DOOR_KG), ("340 mm", CAB_H), ("30 mm round", KNOB_D), ("25 mm proud", KNOB_PROUD),
                ("30 cm up", LOCK_Z), ("15 cm up", KNOB_Z), ("1 mm of play", PLAY), ("6 mm of door travel", CLEAR),
                ("12 mm", REL_R), ("5 mm of the door", REL_GAP), ("15 deg", math.degrees(REL_TILT)),
                ("3 deg", math.degrees(PAST_Q)), ("45 deg", math.degrees(OPEN_Q)), ("60 mm", SPOT_D),
                ("40 mm clear", CLEAR_MM / 1000), ("1 mm over", STILL_MM / 1000), ("1 s", STILL_S),
                ("5 mm off", MOVE_MM / 1000), ("0.05 N m", HINGE_FRIC)]
CHEATS = {"none": 0.0, "pull_only": 0.0, "tap": 0.0, "held": 0.25, "no_return": 0.75}
HOLD = {"cupboard.door": 0.5}
REQUIRED_FORCES = {"press the key": 0.3}
