"""On the station table, straight ahead of the robot: a small window cased in pine trim, set in a painted plywood wall
section that stands on the table. A cordless spring roller shade (a Draper Cordless FlexShade, 61 cm wide) hangs in
front of the window from its roller at the top, caught part way down. Its bottom edge is a round aluminium hem bar.
A strip of blue painter's tape runs level across both side casings, below the hem bar; a steel rule stands on each
side casing.

- The shade's spring pulls the hem bar up. Pulled down, the roller turns; it locks only at each half turn, and only
  when it turns slowly. Let go while it is moving or fast, and the shade runs up past every catch to the top.
- Pulling the bar down a little frees the lock. Nothing else holds the bar.

Run it:  inspect-robots run --embodiment yam_arena -E task=lower_the_shade_to_the_tape_line.py ...
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections import deque
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import run_record  # noqa: E402
from benchkit import _ENV, _np, Part, Slider, on_reset, on_step, set_targets, tool  # noqa: E402
from yam_arena import analysis, station  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.0  # the full-score test already includes its 3 s at rest; the run stops as soon as it holds
TOP = TABLE_TOP[SCENE]
CONTROL_HZ = 20.0  # the run commands set -E control_hz=20; times below are counted in its steps
DT = 1.0 / CONTROL_HZ

INSTRUCTION = "Lower the shade so its bottom bar stays at the blue tape line."

# -- the shade (positions from the midpoint between the arm bases: x forward, y left, z up from the table top) --------
BAR_X, BAR_Z0, BAR_D = 0.340, 0.320, 0.019  # hem bar: centre line, centre height at the start catch, diameter
BAR_Y = (-0.030, 0.580)  # its ends (the shade's width, 610 mm)
SPACING = 0.062  # m of hem-bar travel per half turn of the roller: a catch every 62 mm
LINE_K = 2  # the tape line is at the second catch below the start
C_LINE = -LINE_K * SPACING  # the hem bar's travel (m, up positive) at the line's catch
START_BOTTOM = BAR_Z0 - BAR_D / 2  # 0.3105 m: the bar's bottom at the start catch
TAPE_TOP_Z = START_BOTTOM + C_LINE  # 0.1865 m: the tape's upper edge, level with the bar's bottom at the line's catch
TAPE_W = 0.024  # 24 mm painter's tape

# -- the scoring thresholds -------------------------------------------------------------------------------------------
PULL_DEPTH = 0.050  # m: the bar pulled at least this far below its start at some time
BELOW_START = 0.030  # m: at the end the bar's bottom is at least this far below its start
LINE_BAND = 0.030  # m: caught with its bottom within this of the tape's upper edge (half a catch spacing)
STILL_S = 3.0  # s at the end with the bar still and both grippers clear
STILL_STEPS = int(round(STILL_S * CONTROL_HZ)) + 1
STILL_MM = 0.002  # m: the bar's height may vary less than this over those 3 s
CLEAR_GAP = 0.030  # m: no gripper this close to the hem bar or the shade's fabric
WALL_MOVE = 0.005  # m: the wall section moved further than this from its marks: score 0

# -- the spring and the lock (README: fidelity) -------------------------------------------------------------------------
F0 = 5.0  # N: the spring's net pull up on the bar at the start catch (bar and fabric weight taken off)
RATE = 8.0  # N/m: more pull per metre the bar is drawn down (6.0 N at the line's catch)
DAMP = 0.5  # N s/m on the bar's travel: the roller's bearings and the fabric's air drag
BAR_MASS = 0.30  # kg: the hem bar and the hanging fabric
ROLLER_MASS = 0.20  # kg: the roller tube and wound fabric, as felt at the bar (their inertia over the radius squared)
REST_DROP = 0.001  # m: the bar's slider rests this far below its start catch, so the wall section settling onto the
# table never knocks it; the spring lifts it onto the catch when the trial starts. Travel q counts from that rest
PRELOAD = F0 / RATE + REST_DROP  # m: the spring's target, far above the roller: its pull at the start catch is F0
V_CATCH = 0.020  # m/s: the lock catches only when the bar rises through a catch slower than this
FREE_BELOW = 0.005  # m: pulling the bar this far below its engaged catch frees the lock
Q_LOW = 0.005 + BAR_D / 2 - BAR_Z0 + REST_DROP  # the bar's lowest travel: its bottom 5 mm above the table
Q_TOP = 0.400  # rolled fully up: the bar's top 11 mm under the roller
CATCH_K = range(-4, 7)  # every half turn the bar can reach: travel k x SPACING
CATCHES = {k: k * SPACING + REST_DROP for k in CATCH_K}  # slider travel at each catch
ENGAGE_WITHIN = 0.003  # m: a slowly rising bar this close under a catch engages it (with the step's travel added) ...
ENGAGE_LEAD_S = 0.10  # ... unless it would reach the catch sooner than this, before the stop is in
ENGAGE_MIN = 0.0003  # m: and never once it is within this of the catch


def _fresh() -> SimpleNamespace:
    return SimpleNamespace(step=0, q=REST_DROP, v=0.0, deepest=REST_DROP, locked=REST_DROP, wall0=None, wall_moved=False,
                           recent=deque(maxlen=STILL_STEPS), events=[("caught", 0.0, 0.0, 0.0)], max_rise=0.0,
                           clear_mm=None)


LOG = _fresh()
LATEST: dict = {"states": None, "steps": 0}  # the last object states, which the scripted demo reads to plan


def bottom(q: float) -> float:
    """The hem bar's bottom (m above the table) at slider travel ``q``."""
    return START_BOTTOM + q - REST_DROP


def settled() -> bool:
    """The last 3 s: the bar moved less than 2 mm and no gripper came within 30 mm of the bar or fabric."""
    if len(LOG.recent) < STILL_STEPS:
        return False
    qs = [q for q, _ in LOG.recent]
    return all(c for _, c in LOG.recent) and max(qs) - min(qs) < STILL_MM


def grade(objects) -> float:
    """The score as the outcome table states it, for the trial ending now."""
    if LOG.wall_moved:
        return 0.0
    q = float(objects["shade"].sliders["hembar"])
    pulled = START_BOTTOM - bottom(min(LOG.deepest, q)) >= PULL_DEPTH - 1e-9
    below = START_BOTTOM - bottom(q) >= BELOW_START - 1e-9
    if not (pulled and below):
        return 0.0  # never drawn down, or back at (or above) its start catch: rolled up
    if not settled():
        return 0.25  # still held, touched or moving
    if abs(bottom(q) - TAPE_TOP_Z) <= LINE_BAND + 1e-9:
        return 1.0
    return 0.5  # caught at another catch below its start


def score(objects) -> float:
    return grade(objects)


def success(objects) -> bool:
    return grade(objects) >= 1.0


def distance(objects) -> float:
    """Metres from the bar's bottom to the tape's upper edge."""
    return float(abs(bottom(float(objects["shade"].sliders["hembar"])) - TAPE_TOP_Z))


# -- the lock: a stop at each half turn, moved in only for a slow rise --------------------------------------------------


def catch_name(k: int) -> str:
    return f"catch_{k}" if k >= 0 else f"catch_n{-k}"


def stop_target(k: int, engaged: bool) -> float:
    """The spring target that holds catch ``k``'s stop in over the bar (or out), pressed on that end of its travel.
    The start catch's stop rests in (its travel runs out, +y); every other stop rests out (its travel runs in, -y)."""
    inside, outside = (0.0, STOP_TRAVEL) if k == 0 else (-STOP_TRAVEL, 0.0)
    return inside - STOP_PRESS if engaged else outside + STOP_PRESS


def _stop(k: int, engaged: bool) -> None:
    set_targets("shade", {catch_name(k): stop_target(k, engaged)})


def lock_step(q: float, v: float) -> None:
    """One control step of the lock. An engaged catch frees once the bar is pulled 5 mm below it. With none engaged, a
    bar rising slower than V_CATCH within 3 mm under a catch engages that catch, and the bar stops there; one rising
    faster runs on past it."""
    t = round(LOG.step * DT, 2)
    if LOG.locked is not None and q < LOG.locked - FREE_BELOW:
        k = round((LOG.locked - REST_DROP) / SPACING)
        LOG.events.append(("freed", t, round(LOG.locked * 1000, 1), round(v * 1000, 1)))
        LOG.locked = None
        _stop(k, False)
    if LOG.locked is not None or v <= 0.0:
        return
    LOG.max_rise = max(LOG.max_rise, v)
    above = [(c, k) for k, c in CATCHES.items() if c > q]
    if not above:
        return
    c, k = min(above)
    gap = c - q
    if v < V_CATCH and max(ENGAGE_MIN, v * ENGAGE_LEAD_S) <= gap <= ENGAGE_WITHIN + v * DT:
        LOG.locked = c
        LOG.events.append(("caught", t, round(c * 1000, 1), round(v * 1000, 1)))
        _stop(k, True)
    elif v >= V_CATCH:
        passed = [kk for kk, cc in CATCHES.items() if LOG.q < cc <= q]
        for kk in passed:
            LOG.events.append(("ran past", t, round(CATCHES[kk] * 1000, 1), round(v * 1000, 1)))


@on_reset
def _reset(states) -> None:
    global LOG
    LOG = _fresh()
    LATEST["states"], LATEST["steps"] = states or None, 0
    if states:
        LOG.wall0 = np.asarray(states["shade"].pos, dtype=float)
    # the spring's preload, the start catch engaged and every other catch out
    set_targets("shade", {"hembar": PRELOAD, **{catch_name(k): stop_target(k, k == 0) for k in CATCH_K}})
    run_record.reset(globals())


@on_step
def _watch(states, grips=None) -> None:
    """Follow the trial: the lock, the deepest pull, the wall section and the grippers' clearance (``grips`` their
    hulls; read from the arms if not given)."""
    LATEST["states"], LATEST["steps"] = states, LATEST["steps"] + 1
    shade = states["shade"]
    q = float(shade.sliders["hembar"])
    LOG.step += 1
    LOG.v = (q - LOG.q) / DT
    lock_step(q, LOG.v)
    LOG.q = q
    LOG.deepest = min(LOG.deepest, q)
    if LOG.wall0 is None:
        LOG.wall0 = np.asarray(shade.pos, dtype=float)
    if float(np.linalg.norm(np.asarray(shade.pos) - LOG.wall0)) > WALL_MOVE:
        LOG.wall_moved = True
    if grips is None:
        grips = grippers()
    gap = points_to_grips(shade_points(q), grips, beyond=CLEAR_GAP) if grips else math.inf
    LOG.clear_mm = None if math.isinf(gap) else round(gap * 1000, 1)
    LOG.recent.append((q, gap >= CLEAR_GAP))
    run_record.tick(globals(), states)


# -- what the scorer reads: the grippers -------------------------------------------------------------------------------

GRIPPER_LINKS = ["gripper", "tip_left", "tip_right"]


def gripper(q6, finger: float, base) -> tuple:
    """One gripper's hulls in the world and its position, for arm joints ``q6`` and finger joint ``finger`` (m)."""
    q = {n: float(v) for n, v in zip(ARM_JOINTS, q6, strict=True)} | {j: float(finger) for j in FINGER_JOINTS}
    poses = analysis.fk_link_poses_world(q, base)
    return analysis.world_hulls(poses, GRIPPER_LINKS), poses["gripper"][:3, 3]


def grippers() -> list:
    """Both grippers' hulls, from the arms' joints (the object states don't include the arms)."""
    env = _ENV["env"]
    if env is None:
        return []
    out = []
    for arm in ("left", "right"):
        robot = env.unwrapped.scene[f"{arm}_robot"]
        names = list(robot.joint_names)
        joints = _np(robot.data.joint_pos)[0]
        finger = float(np.mean([joints[names.index(j)] for j in FINGER_JOINTS]))
        out.append(gripper(joints[[names.index(j) for j in ARM_JOINTS]], finger, station.BASES[arm])[0])
    return out


def points_to_grips(points, grips, beyond: float = math.inf) -> float:
    """Metres from the nearest point to the nearest gripper hull (negative inside). Hulls whose bounding box is
    ``beyond`` m or more from the points' are skipped: they can't be nearer than that."""
    best = math.inf
    lo, hi = points.min(axis=0), points.max(axis=0)
    for hulls in grips:
        for _, h in hulls:
            v = getattr(h, "vertices", None)
            if v is not None and np.maximum(v.min(axis=0) - hi, lo - v.max(axis=0)).max() >= beyond:
                best = min(best, beyond)
                continue
            best = min(best, float((points @ h.planes[:, :3].T - h.planes[:, 3]).max(axis=1).min()))
    return best


def shade_points(q: float) -> np.ndarray:
    """Points on the hem bar (its surface every 3 cm) and on the hanging fabric (every 5 cm), in the world, with the
    bar at travel ``q``."""
    zc = TOP + SPAWN_GAP + BAR_Z0 - REST_DROP + q
    ys = np.linspace(BAR_Y[0], BAR_Y[1], 21)
    r = BAR_D / 2
    bar = [(BAR_X + dx, y, zc + dz) for y in ys for dx, dz in ((r, 0), (-r, 0), (0, r), (0, -r))]
    top = min(TOP + SPAWN_GAP + ROLLER_Z, zc + r + FABRIC_H)
    zs = np.arange(zc + r, top, 0.05)
    cloth = [(BAR_X, y, z) for y in ys[::2] for z in zs]
    return np.array(bar + cloth)


# -- the props, at the real items' sizes --------------------------------------------------------------------------------

CATALOGUE = {"shade": "Draper Cordless FlexShade, 24 x 36 in, small round exposed hem bar",
             "tape": "ScotchBlue 2090 painter's tape, 0.94 in", "rule": "300 mm steel rule"}

PLY, PINE, ALU, FABRIC, ACRYL = (0.80, 0.78, 0.74), (0.88, 0.80, 0.62), (0.75, 0.76, 0.78), (0.86, 0.84, 0.78), \
    (0.92, 0.94, 0.96)
STEEL, TAPE_BLUE, CAP = (0.62, 0.64, 0.67), (0.25, 0.55, 0.88), (0.95, 0.95, 0.95)
SPAWN_GAP = 0.0005  # m: the wall section starts this far above the table and settles onto it
WALL_X = 0.42  # the plywood's room face
PLY_T, TRIM_T, TRIM_W = 0.019, 0.019, 0.089  # 3/4 in plywood; 1x4 pine trim (19 x 89 mm actual)
OPEN_Y, OPEN_Z = (0.000, 0.560), (0.080, 0.720)  # the window opening
PANEL_Y = (-0.170, 0.730)  # the 900 x 900 mm panel
ROLLER_Z, ROLLER_R = 0.765, 0.024  # the roller's axis above the table; 38 mm tube with its wound fabric
ROLLER_X = BAR_X - 0.024  # regular roll: the fabric comes off the back of the roller, straight down to the bar
SPACER_X = (0.363, 0.401)  # 38 mm (1-1/2 in) pine spacer blocks on the head casing's face
BRACE = (0.150, 0.025, 0.003)  # each steel corner brace's foot on the table: long (x), wide, thick
FABRIC_H = 0.420  # m of fabric plate riding on the bar (the rest of the drop is a fixed plate up to the roller)
FABRIC_STOP_Y = 0.555  # the riding fabric plate's collider ends here (the bar's last 25 mm is drawn only): the stops
# sit over the bar's left end
FRICTION = 0.35  # aluminium bar on the pads (PhysX averages it with the pads' 0.5)
WALL_SIM_KG = 30.0  # the wall section's sim mass: it stands in for the braces screwed to the table (README, fidelity)
TRIM_X = WALL_X - TRIM_T / 2  # casing centre, 19 mm proud of the plywood


def _box(lo, hi, color, extra=None):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    spec = (tuple(((lo + hi) / 2).tolist()), tuple((hi - lo).tolist()), color)
    return spec + (extra,) if extra is not None else spec


def _wall_boxes() -> list:
    px = (WALL_X, WALL_X + PLY_T)
    tx = (WALL_X - TRIM_T, WALL_X)
    (y0, y1), (z0, z1) = OPEN_Y, OPEN_Z
    out = [
        # the plywood panel round the opening, and the acrylic "glass" in it
        _box((px[0], PANEL_Y[0], 0.0), (px[1], y0, 0.900), PLY),
        _box((px[0], y1, 0.0), (px[1], PANEL_Y[1], 0.900), PLY),
        _box((px[0], y0, 0.0), (px[1], y1, z0), PLY),
        _box((px[0], y0, z1), (px[1], y1, 0.900), PLY),
        _box((WALL_X + PLY_T / 2 - 0.0015, y0, z0), (WALL_X + PLY_T / 2 + 0.0015, y1, z1), ACRYL),
        # 1x4 casing on the room face: right side, left side, head, sill
        _box((tx[0], y0 - TRIM_W, 0.0), (tx[1], y0, z1 + TRIM_W), PINE),
        _box((tx[0], y1, 0.0), (tx[1], y1 + TRIM_W, z1 + TRIM_W), PINE),
        _box((tx[0], y0, z1), (tx[1], y1, z1 + TRIM_W), PINE),
        _box((tx[0], y0, 0.0), (tx[1], y1, z0), PINE),
        # spacer blocks, the shade's brackets and its roller (tube plus wound fabric)
        _box((SPACER_X[0], -0.060, 0.735), (SPACER_X[1], -0.035, 0.800), PINE),
        _box((SPACER_X[0], 0.585, 0.735), (SPACER_X[1], 0.610, 0.800), PINE),
        _box((ROLLER_X - 0.030, -0.035, 0.740), (SPACER_X[0], -0.032, 0.790), ALU),
        _box((ROLLER_X - 0.030, 0.582, 0.740), (SPACER_X[0], 0.585, 0.790), ALU),
        _box((ROLLER_X - ROLLER_R, -0.032, ROLLER_Z - ROLLER_R), (ROLLER_X + ROLLER_R, 0.582, ROLLER_Z + ROLLER_R),
             FABRIC),
        # the fabric coming off the roller, just behind the part riding on the bar
        _box((BAR_X + 0.0015, BAR_Y[0], 0.420), (BAR_X + 0.0035, BAR_Y[1], ROLLER_Z), FABRIC),
    ]
    # two steel corner braces each side: feet on the table, in front of and behind the panel, outside the casing
    for y in (PANEL_Y[0] + 0.040, PANEL_Y[1] - 0.040):
        out.append(_box((px[0] - BRACE[0], y - BRACE[1] / 2, 0.0), (px[0], y + BRACE[1] / 2, BRACE[2]), STEEL))
        out.append(_box((px[1], y - BRACE[1] / 2, 0.0), (px[1] + BRACE[0], y + BRACE[1] / 2, BRACE[2]), STEEL))
        out.append(_box((px[0] - 0.003, y - BRACE[1] / 2, 0.0), (px[0], y + BRACE[1] / 2, BRACE[0]), STEEL))
    return out


MID_Y = (BAR_Y[0] + BAR_Y[1]) / 2
LEN = BAR_Y[1] - BAR_Y[0]
SIDE = BAR_D / math.sqrt(2.0)  # the round bar's diamond section: two line contacts on the pads, as a round bar gives
DIAMOND = Part(turn=(0.0, math.pi / 4, 0.0))
_CLOTH_Z = BAR_D / 2 + 0.001 + FABRIC_H / 2
BAR_BOXES = (
    ((0.0, 0.0, 0.0), (SIDE, LEN - 0.006, SIDE), ALU, Part(turn=DIAMOND.turn, label="hembar_tube")),
    ((0.0, -LEN / 2 + 0.0015, 0.0), (SIDE, 0.003, SIDE), CAP, DIAMOND),  # plastic end caps
    ((0.0, LEN / 2 - 0.0015, 0.0), (SIDE, 0.003, SIDE), CAP, DIAMOND),
    ((0.0, (BAR_Y[0] + FABRIC_STOP_Y) / 2 - MID_Y, _CLOTH_Z), (0.002, FABRIC_STOP_Y - BAR_Y[0], FABRIC_H), FABRIC),
    ((0.0, (FABRIC_STOP_Y + BAR_Y[1]) / 2 - MID_Y, _CLOTH_Z), (0.002, BAR_Y[1] - FABRIC_STOP_Y, FABRIC_H), FABRIC,
     Part(ghost=True)),
)
HEMBAR = Slider("hembar", rest=(BAR_X, MID_Y, BAR_Z0 - REST_DROP), size=(0.005, 0.005, 0.005), axis="Z", lower=Q_LOW,
                upper=Q_TOP, rate=RATE, damping=DAMP, mass=BAR_MASS + ROLLER_MASS, color=ALU, boxes=BAR_BOXES,
                weightless=True)
# Hidden stops over the bar's left end, one per half turn: each slides in along -y over the bar's top to catch it.
STOP = (0.012, 0.020, 0.008)  # m: x, y, z of each stop
STOP_Y = BAR_Y[1] - STOP[1] / 2 + 0.002  # engaged: over the bar's last 18 mm
STOP_TRAVEL = 0.024  # m it slides out (+y) to clear the bar's end by 4 mm
STOP_PRESS = 0.005  # m its target goes past either end of its travel, so it sits pressed on that limit
STOP_RATE, STOP_DAMP, STOP_MASS = 2000.0, 63.0, 0.5  # N/m, N s/m, kg: in or out within about 0.08 s


def _stop_slider(k: int) -> Slider:
    z = BAR_Z0 + k * SPACING + BAR_D / 2 + STOP[2] / 2  # its underside on the bar's top at that catch
    y, lo, hi = (STOP_Y, 0.0, STOP_TRAVEL) if k == 0 else (STOP_Y + STOP_TRAVEL, -STOP_TRAVEL, 0.0)
    return Slider(catch_name(k), rest=(BAR_X, y, z), size=STOP, axis="Y", lower=lo, upper=hi,
                  rate=STOP_RATE, damping=STOP_DAMP, mass=STOP_MASS, color=ALU, weightless=True, hidden=True)


SHADE = tool("shade", pos=(0.0, 0.0, TOP + SPAWN_GAP), boxes=_wall_boxes(), mass=WALL_SIM_KG, friction=FRICTION,
             sliders=(HEMBAR, *(_stop_slider(k) for k in CATCH_K)), color=PLY, self_collide=True)

# Fixed on the wall section, in front of its casing: the tape line and the two steel rules (zero 100 mm above the table).
CASING_FACE = WALL_X - TRIM_T
RULE_Z = (0.100, 0.400)
RULE_W, RULE_T = 0.025, 0.001
RULE_Y = {"right": BAR_Y[0] - 0.007 - RULE_W / 2, "left": BAR_Y[1] + 0.007 + RULE_W / 2}
_FIX = TOP + SPAWN_GAP


def _fixed() -> list:
    out = []
    for side, (ya, yb) in (("right", (OPEN_Y[0] - TRIM_W, OPEN_Y[0])), ("left", (OPEN_Y[1], OPEN_Y[1] + TRIM_W))):
        out.append(Box(f"tape_{side}", pos=(CASING_FACE - 0.00025, (ya + yb) / 2, _FIX + TAPE_TOP_Z - TAPE_W / 2),
                       size=(0.0005, yb - ya, TAPE_W), color=TAPE_BLUE))
        out.append(Box(f"rule_{side}", pos=(CASING_FACE - 0.0005 - RULE_T / 2, RULE_Y[side], _FIX + sum(RULE_Z) / 2),
                       size=(RULE_T, RULE_W, RULE_Z[1] - RULE_Z[0]), color=STEEL))
    return out


OBJECTS = [SHADE, *_fixed()]

# -- the scripted demo (shade_demo.py) ----------------------------------------------------------------------------------
# No REQUIRED_FORCES: the hand never presses into anything. The shade's 5-6 N pull is carried by the 50 N grip on the
# bar (the arm lags about 31 mm under it, which the demo reads and corrects).

from shade_demo import plan  # noqa: E402,F401


def demo():
    """Left arm parks clear of the shade; the right hand slides onto the hem bar from beyond its end, pulls it below the
    line's catch, eases it up slowly until the lock takes it, lets go, slides off and goes home."""
    return plan(SimpleNamespace(**globals()))
