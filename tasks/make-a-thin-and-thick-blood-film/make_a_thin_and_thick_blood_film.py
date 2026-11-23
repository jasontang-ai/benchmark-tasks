"""Make a thin and a thick blood film with a spreader slide (WHO MM-SOP-05A steps 10-14; CDC DPDx blood specimens).

On the bench: a specimen slide (25 x 75 x 1.0 mm, frosted end toward the robot) lies on a silicone mat with three 2 uL
drops for the thick film and one 3 uL drop for the thin film; one clean slide stands in a drying rack to the right.
The thin film: the spreader's edge goes down at 30-45 deg in front of the drop, is drawn back until the blood runs
along the edge, then is pushed in one quick smooth stroke so the film thins to a feathered end on the slide. The thick
film: the spreader turned onto one corner joins the three drops in a few strokes into a patch about 1 cm across.

On the kit: the slides and the rack are free rigid bodies at real size. Blood is drawn, not a fluid: the films are
labelled patches on the specimen slide recoloured as the rules below lay them (fidelity in task.json):
  - spread: while the edge is on the glass (both ends within 0.2 mm, the spreader 20-60 deg to the slide) and within
    1.5 mm of the drop's centre, the blood runs along the edge at 22 mm/s up to 22 mm;
  - push: the edge on and moving away from the frosted end lays film, using 5 um x (tan angle / tan 40 deg) of the
    drop per mm of width per mm travelled; the drop running out while the edge is still on = a feathered end; the edge
    off for 0.1 s first = a thick end line; slower than 10 mm/s for over 0.25 s mid-film = a band;
  - thick film: the spreader's corner in the blood (one end within 0.6 mm, the other over 3 mm up) wets the 2 mm cells
    within 1.8 mm of it when it starts inside the blood; strokes are counted at each turn of over 120 deg.
Scoring reads the film geometry these rules leave: what a person measures on a photo of the slide on a mm ruler.

Demo: the right hand takes the clean slide from the rack by its top end, tilts it to 45 deg, sets its edge on the
glass 4 mm in front of the drop, draws it back into the drop, waits 1.5 s, pushes 33 mm in 1 s, lifts, turns it onto a
corner, swirls it one and a half times round a 3 mm circle through the three drops, then lays it on the mat.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ourkit as k  # noqa: E402
from ourkit import demo as D  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.0
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Make a thin blood film from the single drop, then a thick film from the three drops, using the clean "
               "slide from the rack as the spreader and touching the specimen slide only by its edges or frosted end; "
               "then lay the spreader down on the mat.")
R = math.radians
HZ = 20.0
DT = 1.0 / HZ

# -- the slides (slide frame: origin at the centre, x along the length, z up; top face at +T/2) ------------------------
SL_L, SL_W, SL_T = 0.075, 0.025, 0.001  # 25 x 75 x 1.0 mm, ground edges
SL_KG_REAL = 0.0047  # soda-lime glass, 2.5 g/cm3
SL_KG = 0.012  # sim mass, raised for PhysX contact stability with the gripper (declared in fidelity)
FROST = 0.020  # the white frosted end, nearest the robot
X_END = SL_L / 2  # the slide's far end (slide frame)
X_FROST = -SL_L / 2 + FROST  # where the frosted end stops
X_DROP = 0.036 - SL_L / 2  # the 3 uL drop, 36 mm from the frosted end
X_THICK = 0.026 - SL_L / 2  # the three 2 uL drops, centred 26 mm from the frosted end
DROP_D, THICK_DROP_D = 0.0035, 0.0025
TRI = 0.004  # the three drops 4 mm apart
THICK_DROPS = [(X_THICK + TRI / math.sqrt(3), 0.0), (X_THICK - TRI / (2 * math.sqrt(3)), TRI / 2),
               (X_THICK - TRI / (2 * math.sqrt(3)), -TRI / 2)]

# -- the film rules (fidelity) ----------------------------------------------------------------------------------------
V_THIN = 3e-9  # m3: the 3 uL drop
SPREAD_MAX, SPREAD_RATE = 0.022, 0.022  # m, m/s: the drop runs along the edge to 22 mm over 1 s
H40 = 5.6e-6  # m of blood per m of width per m pushed at 40 deg (tuned: 20.5 mm laid at 45 deg, 35.5 at 30 deg, 22 mm wide)
ON_MM = 0.2  # edge "on the glass": within 0.2 mm (resting contact settles within about 0.1 mm)
IN_MM = 0.3  # ... and no deeper than this below the top face (the sim's contact give; off the slide is lower)
OVER_MM = 3.0  # the edge's middle within this of the slide's centre line (it lies across the slide)
CORNER_MM = 0.6  # a corner "in the blood": within 0.6 mm (a 2 uL drop stands about 0.6-0.8 mm tall)
TOUCH_MM = 1.5  # the edge within 1.5 mm of the drop's centre takes it up
ANGLE_LO, ANGLE_HI = R(20), R(60)
PUSH_MM_S = 10.0  # slower than this ...
BAND_S = 0.25  # ... for longer than this mid-film: a band
OFF_S = 0.1  # edge off the glass this long mid-film: the film ends in a thick line
CORNER_UP = 0.003  # the other end of the edge this high: on a corner
CELL = 0.002  # thick-film cells
WET_R = 0.0018  # a corner wets the cells this near it
TURN = R(120)  # a stroke ends at a turn of more than this
STROKE_STEP = 0.002  # m of corner travel per direction sample

# -- scoring --------------------------------------------------------------------------------------------------------
MOVE_MM, TURN_DEG, LIFT_MM = 3.0, 3.0, 2.0
START_S = 60.0  # no film started by then: 0 (WHO: do not pause between applying and spreading the drops)
FINGER_MM = 2.0  # a finger this near the top face outside the frosted end: 0
THIN1_MM, THIN1_W = 10.0, 12.0
THIN2_LO, THIN2_HI, THIN2_W, FEATHER_GAP = 20.0, 38.0, 15.0, 3.0
THICK_LO, THICK_HI, THICK_GAP, MAX_STROKES = 8.0, 12.0, 2.0, 8
THICK_FILL = 0.6  # the patch covers at least this share of its bounding box (round, not a cross)
CLEAR_MM, STILL_MM, STILL_S = 20.0, 1.0, 1.0
FLAT_MM = 3.0  # the spreader lies on the mat: every corner this low over the mat

# -- the layout -----------------------------------------------------------------------------------------------------
MAT = (0.200, 0.140)
MAT_C = (0.30, 0.0)
MAT_T = 0.002
MAT_TOP = TOP + MAT_T
SLIDE_C = (0.30, 0.0)
SLIDE_Z = MAT_TOP + SL_T / 2
RACK_C = (0.30, -0.14)
RACK = (0.120, 0.060, 0.030)  # x, y, z
RACK_FLOOR = 0.005
SLOT, PITCH = 0.0015, 0.010
RACK_KG = 0.060
SPAWN = 0.0005
SPREADER_C = (RACK_C[0], RACK_C[1], TOP + RACK_FLOOR + SL_L / 2 + SPAWN)

GLASS, FROSTC = (0.80, 0.88, 0.90), (0.97, 0.97, 0.96)
BLOOD, FILM, FEATHER = (0.55, 0.03, 0.05), (0.78, 0.22, 0.25), (0.93, 0.66, 0.67)
MATC, RACKC = (0.55, 0.57, 0.60), (0.93, 0.93, 0.95)


def _disc(prefix: str, c, d: float, color) -> list:
    """A drawn drop: an 8-sided disc 0.4 mm tall on the top face (no collider)."""
    w = d * math.tan(math.pi / 8)
    return [k.box((c[0], c[1], SL_T / 2 + 0.0002), (d, w, 0.0004), color, turn=(0, 0, i * math.pi / 4), ghost=True,
                  label=f"{prefix}{i}") for i in range(4)]


THIN_XS = np.arange(X_DROP + 0.001, X_END - 0.0005, 0.002)  # film strip centres, 2 mm long
LANE_W = 0.0022
LANES = (np.arange(10) - 4.5) * LANE_W  # 2.2 mm lanes across the 22 mm a full spread covers
TK_XS = np.arange(X_THICK - 0.006, X_THICK + 0.0061, CELL)
TK_YS = np.arange(-0.008, 0.0081, CELL)


def _slide_boxes() -> list:
    out = [k.box((0, 0, 0), (SL_L, SL_W, SL_T), GLASS),
           k.box((-SL_L / 2 + FROST / 2, 0, SL_T / 2 + 0.00005), (FROST, SL_W - 0.0004, 0.0001), FROSTC, ghost=True)]
    out += _disc("thin_drop", (X_DROP, 0.0), DROP_D, BLOOD)
    for j, c in enumerate(THICK_DROPS):
        out += _disc(f"tk_drop{j}_", c, THICK_DROP_D, BLOOD)
    for i, x in enumerate(THIN_XS):
        for j, y in enumerate(LANES):
            out.append(k.box((x, y, SL_T / 2 + 0.00003), (0.002, LANE_W, 0.00004), GLASS, ghost=True,
                             label=f"film_{i}_{j}"))
    for i, x in enumerate(TK_XS):
        for j, y in enumerate(TK_YS):
            out.append(k.box((x, y, SL_T / 2 + 0.00003), (CELL, CELL, 0.00004), GLASS, ghost=True,
                             label=f"tk_{i}_{j}"))
    return out


SLIDE_BOXES = _slide_boxes()
SLIDE = k.mech("slide", (SLIDE_C[0], SLIDE_C[1], SLIDE_Z + SPAWN), SLIDE_BOXES, SL_KG, color=GLASS, surface=0.9,
               contact_offset=0.0005)
# the spreader stands on its short end in the rack, faces toward the robot: local x across (25 mm), y through (1 mm),
# z along (75 mm); yaw -90 deg turns its face normal onto world x
SPREADER_BOXES = [k.box((0, 0, 0), (SL_W, SL_T, SL_L), GLASS),
                  k.box((0, -SL_T / 2 - 0.00005, SL_L / 2 - FROST / 2), (SL_W - 0.0004, 0.0001, FROST), FROSTC,
                        ghost=True)]
SPREADER = k.mech("spreader", SPREADER_C, SPREADER_BOXES, SL_KG, yaw=-math.pi / 2, color=GLASS, surface=0.8,
                  contact_offset=0.0005)


def _rack_boxes() -> list:
    """A polypropylene drying rack: a base and fins across x leaving 1.5 mm slots 10 mm apart (the middle one used)."""
    out = [k.box((0, 0, RACK_FLOOR / 2), (RACK[0], RACK[1], RACK_FLOOR), RACKC)]
    n = int(round(RACK[0] / PITCH))
    edges = [-RACK[0] / 2] + [(-n / 2 + i + 0.5) * PITCH + s * SLOT / 2 for i in range(n) for s in (-1, 1)] + \
        [RACK[0] / 2]
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a > 1e-4:
            out.append(k.box(((a + b) / 2, 0, (RACK_FLOOR + RACK[2]) / 2), (b - a, RACK[1], RACK[2] - RACK_FLOOR),
                             RACKC))
    return out


RACK_BOXES = _rack_boxes()
RACKM = k.mech("rack", (RACK_C[0] + PITCH / 2, RACK_C[1], TOP + SPAWN), RACK_BOXES, RACK_KG, color=RACKC, surface=0.4,
               contact_offset=0.0005)
MATBOX = Box("mat", pos=(MAT_C[0], MAT_C[1], TOP + MAT_T / 2), size=(MAT[0], MAT[1], MAT_T), color=MATC)
OBJECTS = [MATBOX, RACKM, SLIDE, SPREADER]

# -- the rules and the scorer ---------------------------------------------------------------------------------------
BAD = k.Sticky()
END = k.Settle(STILL_S, STILL_MM / 1000)
RUN: dict = {}
REC = k.run_record.RunRecord(__file__, "slide", "spreader")
_SPREAD_CORNERS = np.array([[sx * SL_W / 2, sy * SL_T / 2, sz * SL_L / 2] for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)])
_SPREAD_EDGE_PTS = np.array([[u * SL_W / 2, sy * SL_T / 2, sz * SL_L / 2] for u in np.linspace(-1, 1, 11)
                             for sy in (-1, 1) for sz in (-1, 1)]
                            + [[sx * SL_W / 2, sy * SL_T / 2, v * SL_L / 2] for v in np.linspace(-1, 1, 16)
                               for sx in (-1, 1) for sy in (-1, 1)])  # along every edge of the spreader
_TOP_PTS = np.array([[x, y, SL_T / 2] for x in np.arange(X_FROST + 0.002, X_END - 0.001, 0.004)
                     for y in np.arange(-0.009, 0.0091, 0.003)])  # the top face, 3.5 mm in from the long edges


def _fresh() -> dict:
    wet = {(i, j) for i, x in enumerate(TK_XS) for j, y in enumerate(TK_YS)
           if min(math.hypot(x - c[0], y - c[1]) for c in THICK_DROPS) <= THICK_DROP_D / 2 + 1e-4}
    return {"t": 0.0, "w": 0.0, "on": False, "done": False, "x0": X_DROP, "xe": X_DROP, "xl": None, "v": V_THIN,
            "width": 0.0, "feathered": False, "band": False, "slow": 0.0, "off": 0.0, "wet": wet, "strokes": 0,
            "anchor": None, "dir": None, "painted": set(), "slide0": None, "prev_x": None,
            "front": False, "smudged": False, "lifted": 0.0, "near": False, "c_on": False}


RUN.update(_fresh())


def slide_frame(st):
    s = st["slide"]
    return s.rotation(), np.asarray(s.pos, dtype=float)


def to_slide(st, pts_world) -> np.ndarray:
    r, o = slide_frame(st)
    return (np.atleast_2d(pts_world) - o) @ r


def spreader_world(st, local) -> np.ndarray:
    s = st["spreader"]
    return np.asarray(s.pos, dtype=float) + np.atleast_2d(local) @ s.rotation().T


def _edges() -> list:
    """The spreader's two short working edges (either may be used): four corner points each, spreader frame."""
    return [np.array([[sx * SL_W / 2, sy * SL_T / 2, sz * SL_L / 2] for sx in (-1, 1) for sy in (-1, 1)])
            for sz in (-1, 1)]


_EDGES = _edges()


def edge_state(st) -> dict:
    """The lower short edge of the spreader in the slide frame: end heights over the top face, centre, the angle to
    the slide, whether it leans back over the drop (its far end toward the frosted end) and lies across the slide."""
    cands = [to_slide(st, spreader_world(st, e)) for e in _EDGES]
    i = int(np.argmin([c[:, 2].mean() for c in cands]))
    p, q = cands[i], cands[1 - i]
    h = p[:, 2] - SL_T / 2
    ends = [float(h[:2].min()), float(h[2:].min())]  # the two ends of the edge
    axis = q.mean(axis=0) - p.mean(axis=0)  # from the working edge to the far end
    ang = math.asin(min(1.0, abs(float(axis[2])) / float(np.linalg.norm(axis))))
    back = float(axis[0]) < 0.0 and float(axis[2]) > 0.0  # leans back over the drop, toward the frosted end
    mid = p.mean(axis=0)
    over = abs(float(mid[1])) <= OVER_MM / 1000 and abs(float(mid[0])) <= SL_L / 2
    low = int(np.argmin(h))
    cxy = p[low, :2]
    on_slide = abs(float(cxy[0])) <= SL_L / 2 and abs(float(cxy[1])) <= SL_W / 2
    in_band = lambda v, top: -IN_MM / 1000 <= v <= top / 1000  # noqa: E731
    return {"x": float(mid[0]), "y": float(mid[1]), "ends": ends, "angle": ang, "back": back, "corner": cxy,
            "on": all(in_band(v, ON_MM) for v in ends) and ANGLE_LO <= ang <= ANGLE_HI and back and over,
            "corner_on": on_slide and in_band(min(ends), CORNER_MM) and max(ends) >= CORNER_UP}


def film_h(angle: float) -> float:
    return H40 * math.tan(angle) / math.tan(R(40))


FILM_START = X_DROP - DROP_D / 2  # the film's visible start: the drop's back edge


def thin_length() -> float:
    """The film as a photo shows it: from where the drop was (its back edge) to the film's far end."""
    return max(0.0, RUN["xe"] - FILM_START) if RUN["on"] or RUN["done"] else 0.0


def _component() -> set:
    """The wet cells joined to the first drop (4-neighbour)."""
    wet = RUN["wet"]
    seed = min(wet, key=lambda c: math.hypot(TK_XS[c[0]] - THICK_DROPS[0][0], TK_YS[c[1]] - THICK_DROPS[0][1]))
    seen, todo = {seed}, [seed]
    while todo:
        i, j = todo.pop()
        for n in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if n in wet and n not in seen:
                seen.add(n)
                todo.append(n)
    return seen


def thick_patch() -> dict:
    comp = _component()
    joined = all(any(math.hypot(TK_XS[i] - c[0], TK_YS[j] - c[1]) <= THICK_DROP_D / 2 + 1e-4 for i, j in comp)
                 for c in THICK_DROPS)
    xs = [TK_XS[i] for i, _ in comp]
    ys = [TK_YS[j] for _, j in comp]
    ex, ey = max(xs) - min(xs) + CELL, max(ys) - min(ys) + CELL
    return {"joined": joined, "ex": ex * 1000, "ey": ey * 1000, "xmax": max(xs) + CELL / 2,
            "fill": len(comp) * CELL * CELL / (ex * ey)}


def thin_ok_1() -> bool:
    return thin_length() * 1000 >= THIN1_MM - 1e-9 and RUN["width"] * 1000 >= THIN1_W - 1e-9


def thin_ok_2() -> bool:
    L = thin_length() * 1000
    return (RUN["done"] and RUN["feathered"] and not RUN["band"] and not RUN["smudged"]
            and THIN2_LO - 1e-9 <= L <= THIN2_HI + 1e-9
            and RUN["width"] * 1000 >= THIN2_W - 1e-9 and (X_END - RUN["xe"]) * 1000 >= FEATHER_GAP - 1e-9)


def thick_ok() -> bool:
    t = thick_patch()
    return (t["joined"] and THICK_LO - 1e-9 <= t["ex"] <= THICK_HI + 1e-9 and THICK_LO - 1e-9 <= t["ey"] <= THICK_HI + 1e-9
            and t["fill"] >= THICK_FILL - 1e-9 and RUN["strokes"] <= MAX_STROKES
            and (FILM_START - t["xmax"]) * 1000 >= THICK_GAP - 1e-9)


def spreader_down(st) -> bool:
    """The spreader lies flat on the mat (every corner low and inside the mat's outline), off the specimen slide."""
    c = spreader_world(st, _SPREAD_CORNERS)
    if float(c[:, 2].max()) > MAT_TOP + FLAT_MM / 1000:
        return False
    if np.any(np.abs(c[:, 0] - MAT_C[0]) > MAT[0] / 2) or np.any(np.abs(c[:, 1] - MAT_C[1]) > MAT[1] / 2):
        return False
    p = to_slide(st, c)
    return not bool(np.any((np.abs(p[:, 0]) <= SL_L / 2 + 0.002) & (np.abs(p[:, 1]) <= SL_W / 2 + 0.002)))


def grade(st) -> float:
    if BAD:
        return 0.0
    return k.ladder((0.25, thin_ok_1()), (0.5, thin_ok_2()), (0.75, thick_ok()), (1.0, spreader_down(st) and END.ok))


def score(states) -> float:
    return grade(states)


def success(states) -> bool:
    return grade(states) >= 1.0


def distance(states) -> float:
    e = _low_edge_world(states).mean(axis=0)
    return float(np.hypot(e[0] - SLIDE_C[0], e[1] - SLIDE_C[1]))


def _paint(label: str, color) -> None:
    if label not in RUN["painted"]:
        RUN["painted"].add(label)
        k.recolor(label, color)


def _draw() -> None:
    if RUN["on"] or RUN["done"]:
        for i, x in enumerate(THIN_XS):
            if x - 0.001 >= RUN["xe"]:
                break
            near_end = RUN["feathered"] and x > RUN["xe"] - 0.004
            for j, y in enumerate(LANES):
                if abs(y) <= RUN["width"] / 2 + 1e-6:
                    _paint(f"film_{i}_{j}", FEATHER if near_end else FILM)
    for i, j in RUN["wet"]:
        _paint(f"tk_{i}_{j}", BLOOD)


def _thin(e: dict) -> None:
    x = e["x"]
    speed = 0.0 if RUN["prev_x"] is None else (x - RUN["prev_x"]) / DT * 1000
    RUN["prev_x"] = x
    if RUN["done"]:
        return
    if not RUN["on"]:
        if not e["on"]:
            RUN["lifted"] += DT
            if RUN["lifted"] >= OFF_S - 1e-9:  # lifted before the push: the blood on the edge is lost, start again
                RUN.update(w=0.0, front=False, near=False)
            return
        RUN["lifted"] = 0.0
        if x >= X_DROP + TOUCH_MM / 1000 and RUN["w"] == 0:
            RUN["front"] = True  # the edge came down in front of the drop (WHO: then drawn back into it)
        at_drop = math.hypot(x - X_DROP, e["y"]) <= TOUCH_MM / 1000
        if RUN["front"] and at_drop:
            RUN["w"] = min(SPREAD_MAX, RUN["w"] + SPREAD_RATE * DT)
            RUN["near"] = True
        if RUN["w"] > 0 and RUN["near"] and x > X_DROP + TOUCH_MM / 1000:
            # the push leaves the drop with the edge still on the glass: the film starts at the drop
            RUN.update(on=True, width=RUN["w"], xl=X_DROP + TOUCH_MM / 1000, xe=X_DROP + TOUCH_MM / 1000,
                       v=V_THIN - TOUCH_MM / 1000 * RUN["w"] * film_h(e["angle"]))
        else:
            return
    if e["on"]:
        RUN["off"] = 0.0
        dx = max(0.0, x - RUN["xl"])
        use = dx * RUN["width"] * film_h(e["angle"])
        if use >= RUN["v"] and dx > 0:
            RUN["xe"] = RUN["xl"] + dx * RUN["v"] / use
            RUN.update(v=0.0, done=True, feathered=True)
            return
        RUN["v"] -= use
        RUN["xe"] = max(RUN["xe"], x)
        RUN["xl"] = max(RUN["xl"], x)
        RUN["slow"] = RUN["slow"] + DT if speed < PUSH_MM_S else 0.0
        if RUN["slow"] > BAND_S + 1e-9:
            RUN["band"] = True
        if x >= X_END:
            RUN.update(xe=X_END, done=True)  # ran off the slide's end
    else:
        RUN["off"] += DT
        if RUN["off"] >= OFF_S - 1e-9:
            RUN["done"] = True  # lifted: a thick end line


def _smudge(st) -> None:
    """Any part of the spreader back on the finished thin film spoils it (any lean, edge, corner or face)."""
    if not RUN["done"] or RUN["xe"] <= FILM_START:
        return
    p = to_slide(st, spreader_world(st, _SPREAD_EDGE_PTS))
    h = p[:, 2] - SL_T / 2
    hit = (h <= CORNER_MM / 1000) & (p[:, 0] >= FILM_START) & (p[:, 0] <= RUN["xe"]) & \
        (np.abs(p[:, 1]) <= RUN["width"] / 2)
    if bool(np.any(hit)):
        RUN["smudged"] = True


def _thick(e: dict) -> None:
    if not e["corner_on"]:
        RUN["anchor"] = None
        RUN["c_on"] = False
        return
    if not RUN["c_on"]:  # each set-down of the corner on the drops is a stroke (a dab counts)
        RUN["c_on"] = True
        RUN["strokes"] += 1
        RUN["dir"] = None
    c = np.asarray(e["corner"], dtype=float)
    wet = RUN["wet"]
    inside = any(math.hypot(TK_XS[i] - c[0], TK_YS[j] - c[1]) <= WET_R + CELL / 2 for i, j in wet)
    if inside:
        for i, x in enumerate(TK_XS):
            for j, y in enumerate(TK_YS):
                if math.hypot(x - c[0], y - c[1]) <= WET_R:
                    wet.add((i, j))
    if RUN["anchor"] is None:
        RUN["anchor"] = c
        return
    d = c - RUN["anchor"]
    if float(np.hypot(*d)) < STROKE_STEP:
        return
    RUN["anchor"] = c
    if RUN["dir"] is not None and math.acos(float(np.clip(d @ RUN["dir"] / (np.hypot(*d) * np.hypot(*RUN["dir"])), -1, 1))) > TURN:
        RUN["strokes"] += 1
    RUN["dir"] = d


@k.on_reset
def _reset(states) -> None:
    for s in (BAD, END):
        s.reset()
    RUN.clear()
    RUN.update(_fresh())
    REC.reset()


@k.on_step
def _watch(st) -> None:
    RUN["t"] += DT
    r, o = slide_frame(st)
    if RUN["slide0"] is None:
        RUN["slide0"] = (r.copy(), o.copy())
    r0, o0 = RUN["slide0"]
    if float(np.hypot(*(o - o0)[:2])) * 1000 > MOVE_MM:
        BAD.trip("the specimen slide moved")
    if math.degrees(math.acos(float(np.clip(r[:, 0] @ r0[:, 0], -1, 1)))) > TURN_DEG:
        BAD.trip("the specimen slide turned")
    if (o[2] - o0[2]) * 1000 > LIFT_MM:
        BAD.trip("the specimen slide lifted")
    top = o + _TOP_PTS @ r.T
    grips = k.score.grippers()
    if grips and k.score.clearance(top, grips, beyond=0.05) * 1000 < FINGER_MM:
        BAD.trip("a finger touched the specimen slide's top face")
    if RUN["t"] > START_S and not (RUN["on"] or RUN["done"]):
        BAD.trip("no film started within 60 s")
    e = edge_state(st)
    _thin(e)
    _smudge(st)
    _thick(e)
    _draw()
    pts = np.vstack([top, spreader_world(st, _SPREAD_CORNERS)])
    END.push(o, k.hands_clear(pts, CLEAR_MM))
    REC.tick(st, lambda: f"edge x {e['x'] * 1000:+.1f} ends {e['ends'][0] * 1000:.2f}/{e['ends'][1] * 1000:.2f} mm "
                         f"ang {math.degrees(e['angle']):.0f} spread {RUN['w'] * 1000:.1f} film "
                         f"{thin_length() * 1000:.1f} mm feathered {RUN['feathered']} band {RUN['band']} strokes "
                         f"{RUN['strokes']} score {grade(st):g}")


# -- the demo ---------------------------------------------------------------------------------------------------------
THETA = R(45)
U = np.array([math.cos(THETA), 0.0, -math.sin(THETA)])  # the spreader's axis, top to edge, in the push grip
N = np.array([math.sin(THETA), 0.0, math.cos(THETA)])  # its face normal = the closing axis
G_PICK = D.Grip.from_matrix(np.column_stack([(1.0, 0, 0), (0, -1.0, 0), (0, 0, -1.0)]))
G_PUSH = D.Grip.from_matrix(np.column_stack([N, np.cross(U, N), U]))


def _turn(axis, a: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


CORNER_TURN = R(25)
G_CORNER = D.Grip.from_matrix(_turn(N, CORNER_TURN) @ G_PUSH.rotation)
UC = np.asarray(G_CORNER.approach)
HOLD_DEPTH = 0.015  # m of the spreader's top end between the pads
EDGE_FROM_TCP = D.P.PAD_TIP - HOLD_DEPTH + SL_L  # the edge's centre this far past the TCP along the approach
PRESS = 0.0006  # m: the edge commanded this far into the glass
CORNER_PRESS = -0.0003  # m: the corner carried 0.3 mm over the glass, through the blood (no press)
STROKE = 0.0030  # m: the radius of the corner's swirl round the drops' centre
TOP_W = SPREADER_C[2] + SL_L / 2  # the standing spreader's top


def slide_pt(st, x: float, y: float = 0.0, dz: float = 0.0) -> np.ndarray:
    r, o = slide_frame(st)
    return o + r @ np.array([x, y, SL_T / 2 + dz])


def _low_edge_world(st) -> np.ndarray:
    ws = [spreader_world(st, e) for e in _EDGES]
    return ws[int(np.argmin([w[:, 2].mean() for w in ws]))]


def edge_mid(st) -> np.ndarray:
    return _low_edge_world(st).mean(axis=0)


def low_corner(st) -> np.ndarray:
    p = _low_edge_world(st)
    return p[int(np.argmin(p[:, 2]))]


S0 = np.array([SLIDE_C[0], SLIDE_C[1], SLIDE_Z + SL_T / 2])


def spread_plan(how: str) -> D.Plan:
    p = D.Plan("right")
    grip_at = np.array([SPREADER_C[0], SPREADER_C[1], TOP_W + D.P.PAD_TIP - HOLD_DEPTH])
    p.move(2.0, tuple(grip_at + (0, 0, 0.06)), grip=G_PICK, finger=D.gap_finger(0.014), label="over the spreader")
    p.move(1.5, tuple(grip_at), label="down round its top end")
    p.fingers(0.0, 1.0, "pinch the spreader")
    p.move(1.5, tuple(grip_at + (0, 0, 0.07)), label="lift it out of the rack")
    front = S0 + np.array([X_DROP + 0.004, 0.0, 0.015])
    p.move(3.0, D.remember(p, "front0", lambda st: front - EDGE_FROM_TCP * U), grip=G_PUSH,
           fallback=tuple(front - EDGE_FROM_TCP * U), label="tilt it to 45 deg over the slide")
    D.corrected(p, 2.0, lambda st: slide_pt(st, X_DROP + 0.004, 0.0, 0.004), edge_mid, key="above", max_fix=0.03,
                label="edge just over the glass in front of the drop", fallback=tuple(front - EDGE_FROM_TCP * U))
    D.corrected(p, 1.5, lambda st: slide_pt(st, X_DROP + 0.004, 0.0, -PRESS), edge_mid, key="down", max_fix=0.01,
                label="edge onto the glass")
    if how == "early":
        p.move(1.0, lambda st: np.asarray(p.memo["down"]) + (0.031, 0, 0), fallback=tuple(front), label="push at once")
    else:
        D.corrected(p, 1.0, lambda st: slide_pt(st, X_DROP - 0.0005, 0.0, -PRESS), edge_mid, key="back0",
                    max_fix=0.012, label="draw it back into the drop")
        D.corrected(p, 0.6, lambda st: slide_pt(st, X_DROP - 0.0005, 0.0, -PRESS), edge_mid, key="back",
                    max_fix=0.004, label="edge in the drop")
        p.hold(1.5, "let the blood run along the edge")
        if how == "lift":
            p.move(0.4, lambda st: np.asarray(p.memo["back"]) + (0.012, 0, 0), fallback=tuple(front),
                   label="push part way")
            p.move(0.4, lambda st: np.asarray(p.memo["back"]) + (0.014, 0, 0.012), fallback=tuple(front),
                   label="lift off mid-film")
        else:
            p.move(1.0, lambda st: np.asarray(p.memo["back"]) + (0.033, 0, 0), fallback=tuple(front),
                   label="push in one quick, smooth stroke")
    p.move(1.0, lambda st: np.asarray(p.memo["_last"]) + (0, 0, 0.025), fallback=tuple(front), label="lift off")
    if how != "nothick":
        thick = S0 + np.array([X_THICK, 0.0, 0.025])
        p.move(2.5, D.remember(p, "thick0", lambda st: thick - EDGE_FROM_TCP * UC), grip=G_CORNER,
               fallback=tuple(thick - EDGE_FROM_TCP * UC), label="turn it onto a corner over the drops")
        D.corrected(p, 2.0, lambda st: slide_pt(st, X_THICK, 0.0, 0.004), low_corner, key="cabove", max_fix=0.04,
                    label="corner over the drops", fallback=tuple(thick - EDGE_FROM_TCP * UC))
        p.hold(0.8, "settle over the drops")
        D.corrected(p, 1.0, lambda st: slide_pt(st, X_THICK, 0.0, 0.004), low_corner, key="cabove2", max_fix=0.01,
                    label="corner over the drops, again")
        p.hold(0.5, "settle")
        D.corrected(p, 1.0, lambda st: slide_pt(st, X_THICK, 0.0, -CORNER_PRESS), low_corner, key="c00", max_fix=0.006,
                    label="corner onto the drops")
        D.corrected(p, 0.6, lambda st: slide_pt(st, X_THICK, 0.0, -CORNER_PRESS), low_corner, key="c0", max_fix=0.003,
                    label="corner on the drops' centre")
        for i in range(13):  # one and a half turns of a 3 mm circle: the drops swirled together (WHO step 14)
            a = i * math.pi / 4
            p.move(0.35, lambda st, a=a: np.asarray(p.memo["c0"]) + (STROKE * math.cos(a), STROKE * math.sin(a), 0),
                   fallback=tuple(thick), label="swirl the corner through the drops")
        p.move(1.0, lambda st: np.asarray(p.memo["c0"]) + (0, 0, 0.03), fallback=tuple(thick), label="lift off")
    put = np.array([0.300, -0.050, MAT_TOP + 0.004])  # the spreader falls back onto the mat, inside its outline
    p.move(2.5, D.remember(p, "put0", lambda st: put - EDGE_FROM_TCP * U + (0, 0, 0.03)), grip=G_PUSH,
           fallback=tuple(put - EDGE_FROM_TCP * U + (0, 0, 0.03)), label="over the mat beside the slide")
    D.corrected(p, 1.5, lambda st: put, edge_mid, key="put", max_fix=0.03, label="edge down to the mat",
                fallback=tuple(put - EDGE_FROM_TCP * U))
    if how == "held":
        p.hold(3.0, "keep holding it")
        return p
    p.fingers(0.02, 0.8, "let it go")
    p.move(1.0, lambda st: np.asarray(p.memo["put"]) - 0.05 * U + (0, 0, 0.05), fallback=tuple(put + (0, 0, 0.1)),
           label="back off")
    p.home()
    return p


def demo_plans(how: str) -> dict:
    return D.together(D.idle_plan("left", 1.0), spread_plan(how))


def demo():
    return D.choose(demo=lambda: demo_plans("demo"), early=lambda: demo_plans("early"),
                    lift=lambda: demo_plans("lift"), nothick=lambda: demo_plans("nothick"),
                    held=lambda: demo_plans("held"))


# -- what the template tests read ---------------------------------------------------------------------------------
TEXT_NUMBERS = [("25 x 75 x 1.0 mm", SL_T), ("4.7 g", SL_KG_REAL), ("12 g", SL_KG), ("20 mm long", FROST), ("26 mm from the end", 0.026), ("36 mm from the end", 0.036),
                ("3 mm", MOVE_MM / 1000), ("3 deg", TURN_DEG), ("60 s", START_S), ("10 mm long", THIN1_MM / 1000),
                ("12 mm wide", THIN1_W / 1000), ("15 mm wide", THIN2_W / 1000),
                ("20 mm clear", CLEAR_MM / 1000), ("1 s", STILL_S)]
CHEATS = {"none": 0.0, "early": 0.0, "lift": 0.25, "nothick": 0.5, "held": 0.75}
HOLD: dict = {}
