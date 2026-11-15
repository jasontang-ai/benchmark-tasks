"""On the station table, in front of the left arm: a six-well plate on a running magnetic stirrer, and a transfer pipette
standing tip down in a grey holder to their left.

- The plate's left-hand row of three wells, running away from the arms, is wells 1, 2 and 3; each holds one pipetteful
  of clear water and a stir bar. The right-hand well nearest the arms holds the dark blue stock. The other two wells
  are empty. The stirrer under the plate mixes each well as liquid lands in it.
- The pipette is a squeeze-bulb transfer pipette that takes up to one pipetteful. Its bulb's squeeze faces point
  to the robot's left and right. Grip the bulb lightly to hold it; squeeze it hard to push its contents out; ease
  back to a light grip to draw liquid up, which works only while the tip is under the liquid, and takes only what
  lies above the tip. So: squeeze before dipping, ease off low in the liquid to draw, lift, then squeeze over or
  inside the next well to dispense.

Run it:  inspect-robots run --embodiment yam_arena -E task=pipette_dye_into_cups.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from benchkit import _ENV, Slider, ToolState, _np, on_reset, on_step, recolor, tool  # noqa: E402
from ledger import Ledger, Vessel  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = (
    "Use the transfer pipette to dilute the dark stock by half in each well. The stock is in the right-hand well "
    "nearest you, and wells 1, 2 and 3 run away from you along the left-hand row, each holding water. Move one "
    "pipetteful from the stock into well 1, then one from well 1 into well 2, then one from well 2 into well 3. "
    "Then stand the empty pipette back in its holder and let go."
)

# -- the stirrer: a 2mag MIXdrive 6 MTP, a stainless steel magnetic stirrer made for six-well plates, 140 x 102 x 26 mm,
# with six stirring points 39 mm apart (three along its 140 mm side, two along its 102 mm side), one under each well.
# It runs throughout the trial; its control unit sits off the table. --
STIRRER_SIZE = (0.140, 0.102, 0.026)
PITCH = 0.039  # the stirrer's points, and the plate's wells (39.1 mm on the standard plate), centre to centre
STIR_H = STIRRER_SIZE[2]
STEEL = (0.70, 0.71, 0.72)

# -- the plate: a Corning 3516 polystyrene six-well plate, 127.8 x 85.5 mm, about 20 mm tall without its lid, wells
# 34.8 mm across and 17.4 mm deep, in two rows of three. It sits on the stirrer, long side along x, a well over each
# point. The left-hand row (y 18) is wells 1, 2 and 3, nearest the arms first; the right-hand row (y 14.1) holds the
# stock in its nearest well. The row sits where a straight-down grip can carry the 15 cm pipette 1 cm over the rims
# with reach to spare (the reach test checks every waypoint). --
PLATE_SIZE = (0.1278, 0.0855, 0.020)
WELL_R = 0.0174  # radius
WELL_DEPTH = 0.0174
PANEL_T = 0.0012  # each of a well's eight wall panels
PLATE_CLEAR = (0.88, 0.91, 0.94)  # clear polystyrene, shown as a pale opaque grey-blue
ROW_Y = 0.18
X0 = 0.205
WELL_XY = {"cup1": (X0, ROW_Y), "cup2": (X0 + PITCH, ROW_Y), "cup3": (X0 + 2 * PITCH, ROW_Y),
           "stock": (X0, ROW_Y - PITCH)}
EMPTY_XY = [(X0 + PITCH, ROW_Y - PITCH), (X0 + 2 * PITCH, ROW_Y - PITCH)]
EMPTY = ["empty0", "empty1"]
WELL_XY.update(dict(zip(EMPTY, EMPTY_XY)))
PLATE_XY = (X0 + PITCH, ROW_Y - PITCH / 2)
STIRRER_XY = PLATE_XY
PLATE_BASE = TOP + STIR_H  # the plate's underside, on the stirrer
FLOOR_Z = PLATE_BASE + PLATE_SIZE[2] - WELL_DEPTH  # each well's floor
RIM_Z = PLATE_BASE + PLATE_SIZE[2]
FLOOR = {n: FLOOR_Z for n in [*WELL_XY, "empty0", "empty1"]}
CUP_X = {n: xy[0] for n, xy in WELL_XY.items()}  # kept for the scorer's distance and the tests
PIPETTEFUL_ML = 3.4  # the real pipette's bulb draw: one full squeeze-release, in the sim and on a bench
LIQUID_PER_ALIQUOT = PIPETTEFUL_ML * 1e-6 / (math.pi * WELL_R**2)  # m of liquid one pipetteful adds to a well (3.6 mm)
FILM = 0.0006  # the coloured layer drawn on each well's floor: thin, so it stays well below the draw height
ORDER = ["stock", "cup1", "cup2", "cup3"]
START = {"stock": {"dye": 2.0}, "cup1": {"water": 1.0}, "cup2": {"water": 1.0}, "cup3": {"water": 1.0}}

CLEAR = (0.86, 0.89, 0.92)
STOCK_BLUE = (0.10, 0.16, 0.62)
WATER = (0.78, 0.87, 0.96)
GREY = (0.55, 0.56, 0.58)
WALL = 0.003  # the holder's walls

# -- the holder ----------------------------------------------------------------------------------------------------
HOLDER_XY = (0.24, 0.24)
HOLDER_INNER = 0.022
HOLDER_HEIGHT = 0.050

# -- the pipette (tool frame: origin at the bulb's centre, z up) ----------------------------------------------------
BULB = (0.020, 0.008, 0.030)  # the core, thin enough that the fingers reach full squeeze before they meet it
STEM = 0.008
STEM_LEN = 0.120  # bulb 3 cm + stem 12 cm = 15 cm overall: Globe Scientific 137010, a 5 mL large-bulb LDPE transfer pipette
PAD = (0.020, 0.010, 0.026)
SQUEEZE_TRAVEL = 0.009  # each side of the bulb moves in this far (a real bulb gives about a centimetre)
TIP_LOCAL = (0.0, 0.0, -BULB[2] / 2 - STEM_LEN)
REST_WIDTH = BULB[1] + 2 * PAD[1]  # across the bulb's sides at rest
PIPETTE_MASS = 0.03  # the body; each side adds 4 g
PIPETTE_Z = TOP + 0.003 - TIP_LOCAL[2]  # standing in the holder, tip just above the table

SIDE_RATE = 350.0  # N/m per side: a light grip (1.5 mm in) holds about 1 N by friction, three times the weight
WHITE = (0.95, 0.95, 0.95)
STEM_WHITE = (0.90, 0.91, 0.93)

PIPETTE = tool(
    "pipette",
    pos=(*HOLDER_XY, PIPETTE_Z),
    boxes=[
        ((0.0, 0.0, 0.0), BULB, WHITE),
        ((0.0, 0.0, -BULB[2] / 2 - STEM_LEN / 2), (STEM, STEM, STEM_LEN), STEM_WHITE),
    ],
    sliders=[
        Slider("side_l", rest=(0.0, BULB[1] / 2 + PAD[1] / 2, 0.0), size=PAD, axis="Y",
               lower=-SQUEEZE_TRAVEL, upper=0.0, color=WHITE, rate=SIDE_RATE),
        Slider("side_r", rest=(0.0, -BULB[1] / 2 - PAD[1] / 2, 0.0), size=PAD, axis="Y",
               lower=0.0, upper=SQUEEZE_TRAVEL, color=WHITE, rate=SIDE_RATE),
    ],
    mass=PIPETTE_MASS,
    points=[("tip", TIP_LOCAL)],
)


def _walls(name: str, xy, inner: float, height: float, color, base: float = TOP) -> list[Box]:
    """Four fixed walls around a square opening, standing on ``base``: the pipette holder."""
    x, y = xy
    off = inner / 2 + WALL / 2
    z = base + height / 2
    long = inner + 2 * WALL
    return [
        Box(f"{name}_wall_f", pos=(x + off, y, z), size=(WALL, long, height), color=color),
        Box(f"{name}_wall_b", pos=(x - off, y, z), size=(WALL, long, height), color=color),
        Box(f"{name}_wall_l", pos=(x, y + off, z), size=(inner, WALL, height), color=color),
        Box(f"{name}_wall_r", pos=(x, y - off, z), size=(inner, WALL, height), color=color),
    ]


def _well(name: str, xy) -> list[Box]:
    """A round well: eight wall panels around a 34.8 mm circle (an octagon, round at camera resolution)."""
    x, y = xy
    width = 2 * (WELL_R + PANEL_T) * math.tan(math.pi / 8) + 0.0004
    z = FLOOR_Z + WELL_DEPTH / 2
    out = []
    for k in range(8):
        a = k * math.pi / 4
        r = WELL_R + PANEL_T / 2
        out.append(Box(f"{name}_wall{k}", pos=(x + r * math.cos(a), y + r * math.sin(a), z),
                       size=(PANEL_T, width, WELL_DEPTH), color=PLATE_CLEAR, yaw=a))
    return out


def _plate() -> list[Box]:
    """The plate: its base under the wells, its four skirt walls, and six round wells."""
    x, y = PLATE_XY
    lx, ly, h = PLATE_SIZE
    base_t = FLOOR_Z - PLATE_BASE
    skirt = 0.0015
    out = [Box("plate_base", pos=(x, y, PLATE_BASE + base_t / 2), size=(lx, ly, base_t), color=PLATE_CLEAR)]
    for side, (dx, dy, sx, sy) in {"f": (lx / 2 - skirt / 2, 0, skirt, ly), "b": (-lx / 2 + skirt / 2, 0, skirt, ly),
                                   "l": (0, ly / 2 - skirt / 2, lx, skirt), "r": (0, -ly / 2 + skirt / 2, lx, skirt)}.items():
        out.append(Box(f"plate_skirt_{side}", pos=(x + dx, y + dy, PLATE_BASE + h / 2), size=(sx, sy, h),
                       color=PLATE_CLEAR))
    for name, xy in WELL_XY.items():
        out += _well(name, xy)
    return out


def _film(name: str, color) -> list[Box]:
    """The liquid on a well's floor: two squares, one turned 45 degrees, so it reads round."""
    x, y = WELL_XY[name]
    side = (WELL_R - 0.0005) * math.sqrt(2)  # its corners stop just inside the wall
    return [Box(f"{name}_liquid{k}", pos=(x, y, FLOOR_Z + FILM / 2), size=(side, side, FILM), color=color,
                yaw=k * math.pi / 4) for k in (0, 1)]


def _stirrer() -> list[Box]:
    x, y = STIRRER_XY
    return [Box("stirrer", pos=(x, y, TOP + STIR_H / 2), size=STIRRER_SIZE, color=STEEL)]


OBJECTS = [
    *_stirrer(),
    *_plate(),
    *(b for n in ORDER for b in _film(n, STOCK_BLUE if n == "stock" else WATER)),
    *(b for n in EMPTY for b in _film(n, CLEAR)),
    *_walls("holder", HOLDER_XY, HOLDER_INNER, HOLDER_HEIGHT, GREY),
    PIPETTE,
]

# -- the liquid ledger -----------------------------------------------------------------------------------------------


def _vessels() -> list[Vessel]:
    """Every well of the plate, the two empty ones too: liquid squeezed into a spare well (a discard) isn't a spill.
    They have no place in the protocol's order, so liquid from them can't make a right well."""
    return [Vessel(n, WELL_XY[n], (2 * WELL_R, 2 * WELL_R), floor_z=FLOOR_Z, rim_z=RIM_Z,
                   height_per_aliquot=LIQUID_PER_ALIQUOT, contents=dict(START.get(n, {})), catch_above=CATCH_ABOVE,
                   radius=WELL_R)
            for n in [*ORDER, *EMPTY]]


CATCH_ABOVE = 0.03  # squeezed out up to this far above a well's rim, over its opening, the liquid lands in the well
LEDGER = Ledger(_vessels(), travel=SQUEEZE_TRAVEL, order=ORDER)


def squeeze(state: ToolState) -> float:
    """How far the bulb's sides are pushed in, on average (m)."""
    return float(np.mean([abs(v) for v in state.sliders.values()])) if state.sliders else 0.0


def colour_of(fraction: float) -> tuple[float, float, float]:
    """Dye absorbs light exponentially with strength (Beer-Lambert), so each halving is a visible step and even an
    eighth is plainly tinted."""
    return tuple(w * (b / w) ** fraction for w, b in zip(WATER, STOCK_BLUE))


def liquid_colour(well: Vessel) -> tuple[float, float, float]:
    """The colour a well's liquid shows when mixed, clear if empty."""
    return CLEAR if well.volume < 1e-6 else colour_of(well.fraction("dye"))


_SHOWN: dict = {}
SETTLING: dict = {}  # well -> (dye fraction it showed before another well's liquid landed, step it landed)


def shown_fraction(name: str, well: Vessel) -> float:
    """What the well shows: as the stirrer works, its colour moves from what it was to the mixed colour over MIX_S, so a
    policy can see the stirrer still working, as on a real bench."""
    now = well.fraction("dye")
    if name not in SETTLING:
        return now
    before, start = SETTLING[name]
    k = (LEDGER.step - start) / (MIX_S * HZ)
    if k >= 1.0:
        del SETTLING[name]
        return now
    return before + (now - before) * max(k, 0.0)


def _show_liquids() -> None:
    for name, well in LEDGER.vessels.items():
        colour = CLEAR if well.volume < 1e-6 else colour_of(shown_fraction(name, well))
        colour = tuple(round(c, 3) for c in colour)
        if _SHOWN.get(name) != colour:
            _SHOWN[name] = colour
            recolor(f"{name}_liquid0", colour)
            recolor(f"{name}_liquid1", colour)


# -- let go: the pipette counts as let go once it has been still for 1 s with no gripper anywhere on it. The count
# starts again whenever it moves or a gripper's bounds overlap any part of it, stem or bulb --
HZ = 20
REST_STEPS = HZ  # 1 s
STILL = 0.020  # m/s
RUN = {"settled": 0}

# The whole gripper, from the collision meshes in the `gripper` link's frame: fingers open along its x, approach along z.
GRIPPER_BODY = 0.051  # half-width across the fingers of everything but the tips
GRIPPER_TIP = 0.0238  # a shut finger tip's reach past the middle; it opens by the finger joint's travel
GRIPPER_DEPTH = 0.034  # half-depth along y
GRIPPER_Z = (-0.0105, 0.1468)  # wrist end to pad tips


def gripper_bounds(q, finger: float, base) -> tuple[np.ndarray, np.ndarray]:
    """The world-aligned bounds of a whole gripper, tips and body, for arm joints ``q`` and finger joint ``finger``."""
    tcp, rot = ik.tcp_pose(np.asarray(q), base)
    origin = tcp - rot @ np.array(TCP_OFFSET)
    half = max(GRIPPER_BODY, GRIPPER_TIP + abs(finger))
    corners = np.array([origin + rot @ np.array((a * half, b * GRIPPER_DEPTH, z))
                        for a in (-1, 1) for b in (-1, 1) for z in GRIPPER_Z])
    return corners.min(0), corners.max(0)


def grippers():
    """Each gripper's bounds, from the arms' joints, or None when the arms can't be read: then nothing counts as let go."""
    env = _ENV["env"]
    if env is None:
        return None
    out = []
    for arm in ("left", "right"):
        robot = env.unwrapped.scene[f"{arm}_robot"]
        names = list(robot.joint_names)
        joints = _np(robot.data.joint_pos)[0]
        q = joints[[names.index(j) for j in ARM_JOINTS]]
        finger = float(np.abs(joints[[names.index(j) for j in FINGER_JOINTS]]).mean())
        out.append(gripper_bounds(q, finger, station.BASES[arm]))
    return out


def touched(pipette: ToolState, grips) -> bool:
    """A gripper's bounds overlap the pipette's, stem and bulb. Unknown grippers count as touching."""
    if grips is None:
        return True
    return any(bool(np.all(lo <= pipette.high) and np.all(pipette.low <= hi)) for lo, hi in grips)


@on_reset
def _reset(states) -> None:
    LEDGER.reset()
    RUN.update(settled=0)
    _SHOWN.clear()
    SETTLING.clear()
    _show_liquids()


def _note_landings(shown_before: dict) -> None:
    """After a step with a squeeze-out: liquid from another well starts that well settling from the colour it showed; a
    real squeeze-back (MIX_BACK or more) mixes the well at once, as pipetting up and down does."""
    for t, dst, src in transfers(LEDGER):
        if t != LEDGER.step - 1 or not src:
            continue
        if all(d.vessel == dst for d in src):
            if sum(d.volume for d in src) >= MIX_BACK:
                SETTLING.pop(dst, None)
        elif shown_before.get(dst) is not None:  # liquid landing in an empty well has nothing to mix with
            SETTLING[dst] = (shown_before[dst], t)


@on_step
def _track(states, grips="arms") -> None:
    pipette = states["pipette"]
    shown_before = {n: shown_fraction(n, v) if v.volume > 1e-9 else None for n, v in LEDGER.vessels.items()}
    LEDGER.update(pipette.point("tip"), squeeze(pipette))
    if LEDGER.events and LEDGER.events[-1].t == LEDGER.step - 1 and LEDGER.events[-1].kind == "dispense":
        _note_landings(shown_before)
    grips = grippers() if grips == "arms" else grips
    free = float(np.linalg.norm(pipette.lin_vel)) <= STILL and not touched(pipette, grips)
    RUN["settled"] = RUN["settled"] + 1 if free else 0
    _show_liquids()


def settled() -> bool:
    return RUN["settled"] >= REST_STEPS


# -- scoring -------------------------------------------------------------------------------------------------------
TARGET = {"cup1": 0.5, "cup2": 0.25, "cup3": 0.125}  # fraction of stock strength
VOLUME_RANGE = (0.75, 2.25)  # pipettefuls: one or two (a well keeps two until its pipetteful moves on)
# A well's strength counts as its target when it is closer (by ratio) to it than to half or double it: the same
# call as matching it against reference wells made two-fold apart.
CONC_BAND = (2**-0.5, 2**0.5)
PER_CUP = 0.3
SPILL_CAP = 0.5
HOLDER_MAX_TILT = 20.0  # degrees
HOLDER_TIP_BELOW = TOP + 0.02


def cup_right(ledger: Ledger, name: str) -> bool:
    well, conc = ledger.vessels[name], TARGET[name]
    low, high = VOLUME_RANGE
    return low <= well.volume <= high and CONC_BAND[0] * conc <= well.fraction("dye") < CONC_BAND[1] * conc


STOCK_LIMIT = 1.25  # pipettefuls out of the stock, net: one, with the same quarter-pipetteful margin the wells get


def stock_taken(ledger: Ledger) -> float:
    """Pipettefuls taken out of the stock, net of stock squeezed back into it: the protocol takes exactly one. A draw
    put back and drawn again is a correction, not a second pipetteful. Only the stock's own liquid counts as put back."""
    out = sum(e.volume for e in ledger.events if e.kind == "aspirate" and e.vessel == "stock")
    back = sum(sum(d.volume for d in src) for _, dst, src in transfers(ledger)
               if dst == "stock" and src and all(d.vessel == "stock" for d in src))
    return float(out - back)


def stock_spoiled(ledger: Ledger) -> bool:
    """Liquid from any other well squeezed into the stock: it waters the stock down, whatever comes after."""
    return any(dst == "stock" and any(d.vessel != "stock" for d in src) for _, dst, src in transfers(ledger))


# -- the protocol, transfer by transfer ------------------------------------------------------------------------------
# Each well must be made from the one before it: well 1 only from the stock, well 2 only from well 1, well 3 only from well 2.
# A transfer is everything drawn since the last squeeze-out, and the well it lands in.
SOURCE = {"cup1": "stock", "cup2": "cup1", "cup3": "cup2"}

# The sim mixes a well the moment liquid lands in it; a real stirrer takes time (not measured). So liquid drawn
# from a well less than MIX_S after another well's liquid last landed in it is unmixed, and the well it is squeezed into
# can't be right, as its colour would show on a real bench. Squeezing a draw back into its own well is a correction, or
# mixing by pipetting: it taints nothing, and it counts as mixing that well. The demo waits about 18 s.
MIX_S = 15.0  # conservative: a 5 x 2 mm bar at about 300 rpm in a 34.8 mm well is slow; not measured
MIX_BACK = 0.75  # pipettefuls: only a real squeeze-back mixes a well; a token one (a drop) doesn't


def transfers(ledger: Ledger) -> list[tuple[int, str, list]]:
    """(step, destination, the aspirate events drawn since the last squeeze-out) for every dispense, in order."""
    out, drawn = [], []
    for e in ledger.events:
        if e.kind == "aspirate":
            drawn.append(e)
        elif e.kind == "dispense":
            out.append((e.t, e.vessel, drawn))
            drawn = []
        elif e.kind == "spill":
            drawn = []
    return out


def unmixed_draws(ledger: Ledger) -> list:
    """Draws taken from a well before the stirrer, or pipetting, had mixed what last landed in it from another well."""
    landed: dict[str, int] = {}  # step another well's liquid last landed in a well
    mixed: dict[str, int] = {}  # step a well was last mixed by squeezing a draw back into it
    early = []
    moves = {t: (dst, src) for t, dst, src in transfers(ledger)}
    for e in ledger.events:
        if e.kind == "aspirate" and e.vessel in landed and e.vessel != "stock":
            if mixed.get(e.vessel, -1) < landed[e.vessel] and e.t - landed[e.vessel] < MIX_S * HZ:
                early.append(e)
        elif e.kind == "dispense" and e.t in moves:
            dst, src = moves[e.t]
            if src and all(d.vessel == dst for d in src):
                if sum(d.volume for d in src) >= MIX_BACK:
                    mixed[dst] = e.t
            else:
                landed[dst] = e.t
    return early


def spoiled(ledger: Ledger) -> set[str]:
    """Wells that received liquid from the wrong well, or unmixed liquid from another well."""
    early = {id(e) for e in unmixed_draws(ledger)}
    out = set()
    for _, dst, src in transfers(ledger):
        if dst not in SOURCE or not src or all(d.vessel == dst for d in src):
            continue
        if {d.vessel for d in src} != {SOURCE[dst]} or any(id(d) in early for d in src):
            out.add(dst)
    return out


def went_back(ledger: Ledger) -> bool:
    """The contamination rule, once the work has started: after the first draw from the stock, the tip went back into an
    earlier well's liquid than one it had already been in, or liquid was squeezed into an earlier well than one already
    drawn from. Before any stock is drawn there is no dye to carry back, so dips (or water drawn and returned) are fine."""
    rank = {n: i for i, n in enumerate(ORDER)}
    started, dipped, drawn = False, -1, -1
    for e in ledger.events:
        if e.vessel not in rank:
            continue
        r = rank[e.vessel]
        if e.kind == "aspirate":
            started = started or e.vessel == "stock"
            if started:
                drawn, dipped = max(drawn, r), max(dipped, r)
        elif e.kind == "dip" and started:
            if r < dipped:
                return True
            dipped = r
        elif e.kind == "dispense" and started and r < drawn:
            return True
    return False


def cups_in_order(ledger: Ledger) -> int:
    """How many of wells 1, 2, 3 are right, counting from well 1 and stopping at the first that isn't. A well that took
    liquid from the wrong well, or unmixed liquid, isn't right."""
    bad = spoiled(ledger)
    n = 0
    for name in ("cup1", "cup2", "cup3"):
        if not cup_right(ledger, name) or name in bad:
            break
        n += 1
    return n


def in_holder(state: ToolState) -> bool:
    tip = state.point("tip")
    inside = abs(tip[0] - HOLDER_XY[0]) <= HOLDER_INNER / 2 and abs(tip[1] - HOLDER_XY[1]) <= HOLDER_INNER / 2
    return bool(inside and tip[2] < HOLDER_TIP_BELOW and state.tilt_deg() <= HOLDER_MAX_TILT)


def put_away(ledger: Ledger, pipette: ToolState, let_go: bool) -> bool:
    """Standing in its holder, empty, and let go: still for 1 s with no gripper on it."""
    return in_holder(pipette) and ledger.tip_volume < 1e-9 and let_go


def grade(ledger: Ledger, pipette: ToolState, let_go: bool | None = None) -> float:
    """The score as the outcome table states it. ``let_go``: the pipette has been still for 1 s with no gripper on
    it (by default, as tracked through the run)."""
    let_go = settled() if let_go is None else let_go
    if went_back(ledger) or stock_spoiled(ledger) or stock_taken(ledger) > STOCK_LIMIT + 1e-6:
        return 0.0
    n = cups_in_order(ledger)
    score = PER_CUP * n + (0.1 if n == 3 and put_away(ledger, pipette, let_go) else 0.0)
    if ledger.spilled > 1e-9 or ledger.high > 1e-9:
        score = min(score, SPILL_CAP)
    return round(score, 6)


def success(objects) -> bool:
    """All three wells right, nothing spilled, no going back, and the pipette standing let go in its holder."""
    return grade(LEDGER, objects["pipette"]) >= 1.0


def score(objects) -> float:
    return grade(LEDGER, objects["pipette"])


def distance(objects) -> float:
    """Metres from the pipette's tip to the well the protocol needs next (or the holder once all three are made)."""
    tip = objects["pipette"].point("tip")
    n = cups_in_order(LEDGER)
    goal = (*HOLDER_XY, TOP) if n == 3 else (*WELL_XY[ORDER[n + 1]], FLOOR[ORDER[n + 1]])
    return float(np.linalg.norm(tip - np.array(goal)))


# -- the scripted demo (left arm) -----------------------------------------------------------------------------------
GRIP_ABOVE_CENTRE = 0.004  # the TCP sits this far above the bulb's centre when holding it
HOLD = -(REST_WIDTH - 2 * 0.0015) / 2  # finger target: sides pushed in 1.5 mm, enough to hold, short of a press
SQUEEZE = -(REST_WIDTH - 2 * SQUEEZE_TRAVEL - 0.002) / 2  # past full travel, so the sides bottom out
TIP_TRAVEL = RIM_Z + 0.010  # tip height while moving: 1 cm over the plate's rims
PRIME_ABOVE = 0.012  # inside a well, above its liquid: squeeze here before dipping
# A real bulb draws only while its tip is under the liquid, and takes only what lies above the tip. A pipetteful is
# 3.6 mm deep in a 34.8 mm well, so a draw takes a well from 7.2 mm to 3.6 mm. The tip therefore draws 1.8 mm off the
# floor, 1.8 mm under the level left after the draw (and 1.2 mm over the floor film), and 8.5 mm off-centre, beside the
# stir bar (5 x 2 mm, spinning at the centre), clear of its sweep. The offset is along y: the demo's aim corrections
# are mostly along x, where the stem has 11 mm to the wall at the draw point (8 mm at its corner).
DIP_ABOVE = 0.0018
DRAW_OFFSET = 0.0085  # towards the robot's right (-y), from the well's centre
DISPENSE_ABOVE = 0.011  # over the well's centre, above even two pipettefuls (7.2 mm)
TIP_PARK = TOP + 0.004  # back in the holder
STIR_BAR = (0.005, 0.002)  # length, height: a PTFE micro stir bar at each of wells 1-3's centre


def tip_z(name: str, what: str) -> float:
    """The tip's height in well ``name`` for ``what``: "prime", "dip" or "dispense", from the well's floor."""
    return FLOOR[name] + {"prime": PRIME_ABOVE, "dip": DIP_ABOVE, "dispense": DISPENSE_ABOVE}[what]


def tip_xy(name: str, what: str) -> tuple[float, float]:
    """Where the tip goes in well ``name``: beside the stir bar to prime and draw, over the centre to dispense."""
    x, y = WELL_XY[name]
    return (x, y - DRAW_OFFSET) if what in ("prime", "dip") else (x, y)


# The arm's wrist settles a few degrees off vertical, which the 15 cm pipette turns into centimetres at the tip, and
# it differs by reach. The demo is open loop, so it aims off by the error measured at each spot (m, in x and y).
# Measured in the simulator on Nov 15, 2026, from the tip's traced position at each settled waypoint of the demo (the
# residual error with first-guess trims was 3 mm or less everywhere).
TRIM = {"stock": (0.0037, -0.0004), "cup1": (0.0058, -0.0006), "cup2": (0.0083, -0.0009), "cup3": (0.0100, -0.0009),
        "holder": (0.0130, -0.0010)}


def tcp_for_tip(x: float, y: float, tip_z: float) -> tuple[float, float, float]:
    return (x, y, tip_z - TIP_LOCAL[2] + GRIP_ABOVE_CENTRE)


def _aim(name: str, what: str = "dispense") -> tuple[float, float]:
    """Where to send the TCP so the tip arrives at ``what``'s spot in ``name`` (or over the holder)."""
    x, y = HOLDER_XY if name == "holder" else tip_xy(name, what)
    return (x + TRIM[name][0], y + TRIM[name][1])


STIR_WAIT_S = 8.0  # over the well just filled, so its next draw comes well after MIX_S


def _transfer(src: str, dst: str) -> list[Waypoint]:
    sx, sy = _aim(src, "dip")
    dx, dy = _aim(dst, "dispense")
    return [
        Waypoint(2.0, tcp_for_tip(sx, sy, TIP_TRAVEL), HOLD, f"over {src}"),
        Waypoint(0.8, tcp_for_tip(sx, sy, TIP_TRAVEL), HOLD, "settle"),
        Waypoint(1.5, tcp_for_tip(sx, sy, tip_z(src, "prime")), HOLD, "into the well"),
        Waypoint(0.8, tcp_for_tip(sx, sy, tip_z(src, "prime")), SQUEEZE, "squeeze in air"),
        Waypoint(1.5, tcp_for_tip(sx, sy, tip_z(src, "dip")), SQUEEZE, "dip"),
        Waypoint(1.0, tcp_for_tip(sx, sy, tip_z(src, "dip")), HOLD, "release: draw"),
        Waypoint(1.5, tcp_for_tip(sx, sy, TIP_TRAVEL), HOLD, "lift"),
        Waypoint(2.0, tcp_for_tip(dx, dy, TIP_TRAVEL), HOLD, f"over {dst}"),
        Waypoint(0.8, tcp_for_tip(dx, dy, TIP_TRAVEL), HOLD, "settle"),
        Waypoint(1.5, tcp_for_tip(dx, dy, tip_z(dst, "dispense")), HOLD, "into the well"),
        Waypoint(0.8, tcp_for_tip(dx, dy, tip_z(dst, "dispense")), SQUEEZE, "squeeze: dispense"),
        Waypoint(1.5, tcp_for_tip(dx, dy, TIP_TRAVEL), SQUEEZE, "lift, still squeezed"),
        Waypoint(0.8, tcp_for_tip(dx, dy, TIP_TRAVEL), HOLD, "release in air"),
        Waypoint(STIR_WAIT_S, tcp_for_tip(dx, dy, TIP_TRAVEL), HOLD, "wait for the stirrer"),
    ]


def demo():
    """Left arm: take the pipette from its holder, make the three transfers, stand it back and let go."""
    hx, hy = HOLDER_XY
    start_tip = PIPETTE_Z + TIP_LOCAL[2]
    plan = [
        Waypoint(1.5, tcp_for_tip(hx, hy, TIP_TRAVEL), FINGER_OPEN, "over the pipette"),
        Waypoint(1.5, tcp_for_tip(hx, hy, start_tip), FINGER_OPEN, "down to the bulb"),
        Waypoint(0.8, tcp_for_tip(hx, hy, start_tip), HOLD, "grip"),
        Waypoint(1.2, tcp_for_tip(hx, hy, TIP_TRAVEL), HOLD, "lift out"),
    ]
    plan += _transfer("stock", "cup1") + _transfer("cup1", "cup2") + _transfer("cup2", "cup3")
    plan += [
        Waypoint(2.0, tcp_for_tip(*_aim("holder"), TIP_TRAVEL), HOLD, "over the holder"),
        Waypoint(0.8, tcp_for_tip(*_aim("holder"), TIP_TRAVEL), HOLD, "settle"),
        Waypoint(2.0, tcp_for_tip(*_aim("holder"), TIP_PARK), HOLD, "into the holder"),
        Waypoint(1.0, tcp_for_tip(*_aim("holder"), TIP_PARK), FINGER_OPEN, "let go"),
        Waypoint(1.5, tcp_for_tip(*_aim("holder"), TIP_TRAVEL), FINGER_OPEN, "up"),
    ]
    return {"left": plan}
