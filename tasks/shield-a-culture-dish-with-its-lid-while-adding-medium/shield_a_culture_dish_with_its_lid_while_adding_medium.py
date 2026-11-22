"""Add 5 mL of medium to a culture dish in a safety cabinet, its lid held over it as a shield (scorer at the top)."""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ourkit as k  # noqa: E402
from ledger import Ledger, Vessel  # noqa: E402
from ourkit import demo as D  # noqa: E402
from ourkit.scene import rpy  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

P = D.P
Grip = D.Grip
SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.0
TOP = TABLE_TOP[SCENE]
DT = 0.05
INSTRUCTION = ("Add one pipetteful of fresh medium from the reservoir to the culture dish, holding the dish's lid over it "
               "as a shield with the opening toward the back wall, then put the lid back on, stand the pipette in its "
               "holder and let go.")

# == the scoring numbers
DOSE_ML = 5.0  # one full squeeze of the bulb
DOSE_TOL = 0.3
TUBE_ML = 40.0  # in the reservoir
OPEN_MM = 5.0  # the lid is off its seat when any rim point is this far above its seated height
CENTRE_MM = 25.0  # shield: the lid's centre within this of the dish axis (horizontal)
LOW_MM = 20.0  # shield: the lid's lowest rim point at most this far above the dish rim
TILT_MAX = 45.0  # shield: the lid tilted at most this far
AZ_MAX = 60.0  # shield: the lid's highest rim point within this of the direction of the back wall
AZ_TILT = 10.0  # ... once the lid is tilted more than this
OPEN_S = 30.0  # the dish open at most this long in all
WALL_MM = 8.0  # the squeeze: the tip below the rim and within this of the side wall
TOUCH_MM = 0.02  # contact
MOVED_MM = 5.0  # the dish slid more than this: 0
SEAT_MM = 2.0  # seated: every rim point within this of its seated height
SEAT_CENTRE_MM = 3.0
HOLDER_TILT = 20.0
CLEAR_MM = 20.0
STILL_MM = 1.0
STILL_S = 1.0
DROP, DROP_T = 0.020, 0.25
FRICTION = 0.5

# == the dish: Corning 430166, 60 mm, tissue-culture treated (15 mm tall with its lid, growth floor 52.1 mm across)
DX, DY = 0.250, -0.010
R_IN = 0.02605  # the growth floor's radius
WALL_T = 0.0012
R_OUT = R_IN + WALL_T
H_BASE = 0.014  # rim height above the work surface
FLOOR_T = 0.001
R_LID = 0.030
SKIRT_T = 0.001
SKIRT = 0.008  # from the lid's top down
PLATE_T = 0.001
H_ALL = H_BASE + PLATE_T  # 15 mm with the lid on
DISH_M, LID_M = 0.009, 0.006
N = 12
CLEAR = (0.90, 0.92, 0.95)
PINK = (0.93, 0.55, 0.62)
GREY = (0.55, 0.56, 0.58)
WHITE = (0.95, 0.95, 0.95)


def _ring(name, r0, r1, z0, z1, color, ghost=False):
    """A 12-sided tube wall from radius r0 to r1 (apothems), z0 to z1, in its part's frame; flats face +-x and +-y."""
    rm, t = (r0 + r1) / 2, r1 - r0
    side = 2 * r1 * math.tan(math.pi / N)
    return [k.box((rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), (t, side, z1 - z0), color,
                  turn=(0.0, 0.0, a), label=f"{name}_{i}", ghost=ghost)
            for i, a in enumerate(2 * math.pi * np.arange(N) / N)]


def _disc(name, a, z0, z1, color, ghost=False):
    """A 12-sided disc of apothem ``a``: six strips through the centre, 30 deg apart."""
    w = 2 * a * math.tan(math.pi / N)
    return [k.box((0.0, 0.0, (z0 + z1) / 2), (2 * a, w, z1 - z0), color, turn=(0.0, 0.0, i * math.pi / 6),
                  label=f"{name}_{i}", ghost=ghost) for i in range(6)]


DISH_BOXES = [*_disc("dish_floor", R_OUT, 0.0, FLOOR_T, CLEAR), *_ring("dish_wall", R_IN, R_OUT, 0.0, H_BASE, CLEAR),
              k.box((0.0, 0.0, FLOOR_T + 0.0002), (0.034, 0.034, 0.0004), CLEAR, ghost=True, label="dish_medium")]
DISH = k.mech("dish", (DX, DY, TOP + 0.0002), DISH_BOXES, DISH_M, surface=FRICTION)
# the lid's frame: origin at the centre of its top plate (mid-thickness)
LID_BOXES = [*_disc("lid_plate", R_LID, -PLATE_T / 2, PLATE_T / 2, CLEAR),
             *_ring("lid_skirt", R_LID - SKIRT_T, R_LID, PLATE_T / 2 - SKIRT, PLATE_T / 2, CLEAR)]
LID_Z0 = TOP + 0.0002 + H_BASE + PLATE_T / 2  # seated, resting on the dish's rim
GAP_DEG = -40.0  # the shield gap's bearing (back and toward the right arm)
LID_YAW = math.radians((GAP_DEG + 90.0) % 30.0)  # two skirt flats face the lid hand's fingers
LID = k.mech("lid", (DX, DY, LID_Z0 + 0.0003), LID_BOXES, LID_M, surface=FRICTION, yaw=LID_YAW)

# == the medium: a Corning 4870 sterile 50 mL reagent reservoir (145 x 70 x 28 mm), white polystyrene
TX, TY = 0.200, -0.150
RES = (0.145, 0.070, 0.028)
RES_T = 0.002
RES_FLOOR = 0.003  # the U-bottom taken as a flat floor
RES_M = 0.015
RES_IN = (RES[0] - 2 * RES_T, RES[1] - 2 * RES_T)
RACK_BOXES = [k.box((0, 0, RES_FLOOR / 2), (RES[0], RES[1], RES_FLOOR), WHITE, label="res_floor"),
              k.box(((RES[0] - RES_T) / 2, 0, RES[2] / 2), (RES_T, RES[1], RES[2]), WHITE, label="res_wall_a"),
              k.box((-(RES[0] - RES_T) / 2, 0, RES[2] / 2), (RES_T, RES[1], RES[2]), WHITE, label="res_wall_b"),
              k.box((0, (RES[1] - RES_T) / 2, RES[2] / 2), (RES[0] - 2 * RES_T, RES_T, RES[2]), WHITE, label="res_wall_c"),
              k.box((0, -(RES[1] - RES_T) / 2, RES[2] / 2), (RES[0] - 2 * RES_T, RES_T, RES[2]), WHITE, label="res_wall_d"),
              k.box((0, 0, RES_FLOOR + 0.0015), (RES_IN[0] - 0.004, RES_IN[1] - 0.004, 0.003), PINK, ghost=True,
                    label="res_medium")]
RACK = k.mech("rack", (TX, TY, TOP + 0.0002), RACK_BOXES, RES_M + TUBE_ML * 1e-3, surface=FRICTION)

# == the transfer pipette: sterile, 5 mL large bulb, graduated; its bulb's squeeze faces point +-y
BULB = (0.020, 0.008, 0.030)
STEM = 0.006
STEM_LEN = 0.120
PAD = (0.020, 0.010, 0.026)
SQUEEZE_TRAVEL = 0.009
SIDE_RATE = 350.0
TIP_LOCAL = np.array([0.0, 0.0, -BULB[2] / 2 - STEM_LEN])
PIPETTE_M = 0.030
REST_WIDTH = BULB[1] + 2 * PAD[1]
PIPETTE_BOXES = [k.box((0, 0, 0), BULB, WHITE, label="bulb"),
                 k.box((0, 0, -BULB[2] / 2 - STEM_LEN / 2), (STEM, STEM, STEM_LEN), (0.90, 0.91, 0.93), label="stem")]
SIDES = [k.slider("side_l", (0.0, BULB[1] / 2 + PAD[1] / 2, 0.0), "Y", -SQUEEZE_TRAVEL, 0.0,
                  [k.box((0, 0, 0), PAD, WHITE, label="pad_l")], rate=SIDE_RATE, mass=0.004),
         k.slider("side_r", (0.0, -BULB[1] / 2 - PAD[1] / 2, 0.0), "Y", 0.0, SQUEEZE_TRAVEL,
                  [k.box((0, 0, 0), PAD, WHITE, label="pad_r")], rate=SIDE_RATE, mass=0.004)]
# the holder: a printed block with a 12 mm square well 50 mm deep, fixed
HX, HY = 0.210, -0.235
HOLDER_IN, HOLDER_H, HW = 0.024, 0.065, 0.004
PIPETTE_Z0 = TOP + 0.003 - TIP_LOCAL[2]
PIP_YAW = -math.pi / 3  # the bulb's squeeze faces turned to suit the right hand
R_PIP = rpy((0.0, 0.0, PIP_YAW))
PIPETTE = k.mech("pipette", (HX, HY, PIPETTE_Z0), PIPETTE_BOXES, PIPETTE_M, joints=SIDES, surface=1.0, yaw=PIP_YAW,
                 points=[("tip", tuple(TIP_LOCAL))])


def _holder() -> list:
    o, h = HOLDER_IN / 2 + HW / 2, HOLDER_H
    z = TOP + h / 2
    return [Box("holder_a", pos=(HX + o, HY, z), size=(HW, HOLDER_IN + 2 * HW, h), color=GREY),
            Box("holder_b", pos=(HX - o, HY, z), size=(HW, HOLDER_IN + 2 * HW, h), color=GREY),
            Box("holder_c", pos=(HX, HY + o, z), size=(HOLDER_IN, HW, h), color=GREY),
            Box("holder_d", pos=(HX, HY - o, z), size=(HOLDER_IN, HW, h), color=GREY)]


# == the biological safety cabinet the robot works in (Thermo Scientific 1300 Series A2, 4-foot)
BSC_W, BSC_H, BSC_D = 1.199, 0.780, 0.630
BSC_FRONT = -0.150
BSC_BACK = BSC_FRONT + BSC_D
FRONT_GRILLE, REAR_GRILLE = 0.100, 0.050
SASH_OPEN = 0.254
SS, GRILLE = (0.88, 0.89, 0.90), (0.22, 0.23, 0.25)


def _slats(name: str, x0: float, depth: float, pitch: float = 0.010, bar: float = 0.005) -> list:
    n = int(round(depth / pitch))
    return [Box(f"{name}_slat_{i}", pos=(x0 + (i + 0.5) * pitch, 0.0, TOP + 0.00325), size=(bar, BSC_W, 0.0005),
                color=SS) for i in range(n)]


def _cabinet() -> list:
    w2, t, rail = BSC_W / 2, 0.010, 0.030
    cz = TOP + BSC_H / 2
    sx = BSC_FRONT - FRONT_GRILLE
    return [Box("bsc_work_surface", pos=(sx + (BSC_BACK - sx) / 2, 0.0, TOP + 0.0001), size=(BSC_BACK - sx, BSC_W, 0.0004),
                color=SS),
            Box("bsc_back_wall", pos=(BSC_BACK + t / 2, 0.0, cz), size=(t, BSC_W, BSC_H), color=SS),
            Box("bsc_left_wall", pos=(BSC_FRONT + BSC_D / 2, w2 + t / 2, cz), size=(BSC_D, t, BSC_H), color=SS),
            Box("bsc_right_wall", pos=(BSC_FRONT + BSC_D / 2, -w2 - t / 2, cz), size=(BSC_D, t, BSC_H), color=SS),
            Box("bsc_rear_grille", pos=(BSC_BACK - REAR_GRILLE / 2, 0.0, TOP + 0.0015), size=(REAR_GRILLE, BSC_W, 0.003),
                color=GRILLE),
            Box("bsc_front_grille", pos=(BSC_FRONT - FRONT_GRILLE / 2, 0.0, TOP + 0.0015),
                size=(FRONT_GRILLE, BSC_W, 0.003), color=GRILLE),
            Box("bsc_left_front", pos=(sx + FRONT_GRILLE / 2, w2 + t / 2, cz), size=(FRONT_GRILLE, t, BSC_H), color=SS),
            Box("bsc_right_front", pos=(sx + FRONT_GRILLE / 2, -w2 - t / 2, cz), size=(FRONT_GRILLE, t, BSC_H), color=SS),
            Box("bsc_sash_rail", pos=(sx - rail / 2, 0.0, TOP + SASH_OPEN + rail / 2), size=(rail, BSC_W, rail), color=SS),
            Box("bsc_sash_header", pos=(sx - rail / 2, 0.0, TOP + BSC_H - rail / 2), size=(rail, BSC_W, rail), color=SS),
            *_slats("bsc_rear", BSC_BACK - REAR_GRILLE, REAR_GRILLE), *_slats("bsc_front", sx, FRONT_GRILLE)]


OBJECTS = [DISH, LID, RACK, PIPETTE, *_holder(), *_cabinet()]
WORK_IN = 0.1016
GRILLE_Z = 0.030
SASH_X = BSC_FRONT - FRONT_GRILLE
ARM_EVERY = 4

# == the liquid
ML = 1e-6


def _vessels() -> list:
    return [Vessel("tube", (TX, TY), RES_IN, floor_z=TOP + RES_FLOOR, rim_z=TOP + RES[2],
                   height_per_aliquot=ML / (RES_IN[0] * RES_IN[1]), contents={"medium": TUBE_ML}),
            Vessel("dish", (DX, DY), (2 * R_IN, 2 * R_IN), floor_z=TOP + FLOOR_T, rim_z=TOP + H_BASE,
                   height_per_aliquot=ML / (math.pi * R_IN ** 2), contents={}, catch_above=0.010, radius=R_IN)]


LEDGER = Ledger(_vessels(), travel=SQUEEZE_TRAVEL, aliquot=DOSE_ML)


def squeeze(st) -> float:
    return float(np.mean([abs(st.q.get("side_l", 0.0)), abs(st.q.get("side_r", 0.0))]))


# == geometry the scorer reads
_ANG = 2 * math.pi * np.arange(24) / 24
LID_RIM = np.array([[R_LID * math.cos(a), R_LID * math.sin(a), PLATE_T / 2 - SKIRT] for a in _ANG])  # skirt bottom
LID_TOP_RIM = np.array([[R_LID * math.cos(a), R_LID * math.sin(a), PLATE_T / 2] for a in _ANG])
_ZS = np.linspace(0.0, STEM_LEN, 25)
_H = STEM / 2
STEM_PTS = np.array([[a * _H, b * _H, TIP_LOCAL[2] + z] for z in _ZS for a in (-1, 1) for b in (-1, 1)]
                    + [[0.0, 0.0, TIP_LOCAL[2]]])
_PIP_PTS = np.array([[a * BULB[0] / 2, b * 0.014, c] for a in (-1, 1) for b in (-1, 1)
                     for c in (-BULB[2] / 2, BULB[2] / 2)] + [list(TIP_LOCAL)])


def world(st, local) -> np.ndarray:
    return np.asarray(st.pos, float) + np.asarray(local) @ st.rotation().T


def dish_frame(st):
    return st.rotation(), np.asarray(st.pos, float) - np.array([0.0, 0.0, 0.0002])


def lid_in_dish(lid, dish):
    """The lid's rim points (skirt bottom, plate top) and centre, in the dish's frame."""
    Rd, od = dish_frame(dish)
    to = lambda p: (p - od) @ Rd  # noqa: E731
    return to(world(lid, LID_RIM)), to(world(lid, LID_TOP_RIM)), to(np.asarray(lid.pos, float)[None])[0]


def lid_metrics(lid, dish) -> dict:
    rim, top, c = lid_in_dish(lid, dish)
    seated_rim = H_BASE + PLATE_T - SKIRT  # the skirt bottom's height when seated
    lift = rim[:, 2] - seated_rim
    hi = top[int(np.argmax(top[:, 2]))]
    tilt = math.degrees(math.acos(float(np.clip((dish.rotation().T @ lid.rotation())[2, 2], -1, 1))))
    return {"lift_max": float(lift.max()) * 1000, "lift_min": float(lift.min()) * 1000,
            "centre": float(np.hypot(c[0], c[1])) * 1000, "low": float(rim[:, 2].min() - H_BASE) * 1000,
            "tilt": tilt, "az": math.degrees(math.atan2(hi[1] - c[1], hi[0] - c[0]))}


def is_open(m) -> bool:
    return m["lift_max"] > OPEN_MM


def shield_ok(m) -> bool:
    return (m["centre"] <= CENTRE_MM and m["low"] <= LOW_MM and m["tilt"] <= TILT_MAX
            and (abs(m["az"]) <= AZ_MAX or m["tilt"] <= AZ_TILT))


def seated(m) -> bool:
    return m["lift_max"] <= SEAT_MM and m["lift_min"] >= -SEAT_MM and m["centre"] <= SEAT_CENTRE_MM


def touch_lid(pts_dish_frame_lid, lid) -> bool:
    """Any stem point within 0.02 mm of (or inside) the lid's plate or skirt, from the lid's own frame."""
    p = pts_dish_frame_lid
    r = np.hypot(p[:, 0], p[:, 1])
    e = TOUCH_MM / 1000
    plate = (np.abs(p[:, 2]) <= PLATE_T / 2 + e) & (r <= R_LID + e)
    skirt = (r >= R_LID - SKIRT_T - e) & (r <= R_LID + e) & (p[:, 2] >= PLATE_T / 2 - SKIRT - e) & (p[:, 2] <= PLATE_T / 2 + e)
    return bool(np.any(plate | skirt))


def touch_dish(p) -> bool:
    """A stem point on the dish's floor (the cell layer), its rim's top or its outside face (dish frame)."""
    r = np.hypot(p[:, 0], p[:, 1])
    e = TOUCH_MM / 1000
    floor = (r <= R_IN) & (p[:, 2] <= FLOOR_T + e)
    rim = (r >= R_IN - e) & (r <= R_OUT + e) & (p[:, 2] >= H_BASE - 0.0003 - e) & (p[:, 2] <= H_BASE + e)
    outside = (r >= R_OUT - 0.0003) & (r <= R_OUT + e) & (p[:, 2] <= H_BASE + e)
    return bool(np.any(floor | rim | outside))


def in_holder(pip) -> bool:
    tip = pip.point("tip")
    tilt = math.degrees(math.acos(float(np.clip(pip.rotation()[2, 2], -1, 1))))
    return (abs(tip[0] - HX) <= HOLDER_IN / 2 and abs(tip[1] - HY) <= HOLDER_IN / 2 and tip[2] < TOP + HOLDER_H
            and tilt <= HOLDER_TILT)


# == the scorer
RUN: dict = {}
SHIELD, TOUCHED, SPILL, MOVED, DROPPED, CABINET = (k.Sticky() for _ in range(6))
END = k.Settle(STILL_S, STILL_MM / 1000.0)
REC = k.run_record.RunRecord(__file__, "lid", "pipette")


def _reset_run(states=None) -> None:
    RUN.clear()
    RUN.update(k=0, dbg="", open_s=0.0, wall_ok=None, hist={"lid": [], "pipette": []}, dish0=np.array([DX, DY]), m=None,
               tint=-1, ml=0.0)
    if states and "dish" in states:
        RUN["dish0"] = np.asarray(states["dish"].pos[:2], float).copy()


_reset_run()


def grade(ml, wall_ok, zeroed, lid_seated, open_s, put_away) -> float:
    """The ladder."""
    if zeroed or not wall_ok or abs(ml - DOSE_ML) > DOSE_TOL + 1e-9:
        return 0.0
    if not lid_seated or open_s > OPEN_S + 1e-9:
        return 0.5
    return 1.0 if put_away else 0.75


def score(states) -> float:
    if "lid" not in states or "dish" not in states or "pipette" not in states:
        return 0.0
    m = lid_metrics(states["lid"], states["dish"])
    zeroed = any(bool(s) for s in (SHIELD, TOUCHED, SPILL, MOVED, DROPPED, CABINET))
    return grade(RUN["ml"], bool(RUN["wall_ok"]), zeroed, seated(m), RUN["open_s"],
                 in_holder(states["pipette"]) and END.ok)


def success(states) -> bool:
    return score(states) >= 1.0


def distance(states) -> float:
    return abs(DOSE_ML - float(RUN.get("ml", 0.0)))


@k.on_reset
def _reset(states) -> None:
    for s in (SHIELD, TOUCHED, SPILL, MOVED, DROPPED, CABINET):
        s.reset()
    END.reset()
    LEDGER.reset()
    _reset_run(states)
    REC.reset()


@k.on_step
def _watch(states) -> None:
    if any(n not in states for n in ("lid", "dish", "pipette", "rack")):
        return
    lid, dish, pip, rack = states["lid"], states["dish"], states["pipette"], states["rack"]
    RUN["k"] += 1
    t = RUN["k"] * DT
    Rd, od = dish_frame(dish)
    dv = LEDGER.vessels["dish"]
    dv.xy = (float(od[0]), float(od[1]))
    tv = LEDGER.vessels["tube"]
    tv.xy = (float(rack.pos[0]), float(rack.pos[1]))
    n_before = len(LEDGER.events)
    tip = pip.point("tip")
    LEDGER.update(tip, squeeze(pip))
    for e in LEDGER.events[n_before:]:
        if e.kind == "spill":
            SPILL.trip("medium squeezed out over no vessel", t)
        if e.kind == "dispense" and e.vessel == "dish":
            tl = (tip - od) @ Rd
            at_wall = tl[2] < H_BASE and math.hypot(tl[0], tl[1]) >= R_IN - WALL_MM / 1000
            RUN["wall_ok"] = at_wall if RUN["wall_ok"] is None else (RUN["wall_ok"] and at_wall)
    RUN["ml"] = round(dv.volume, 3)
    m = lid_metrics(lid, dish)
    RUN["m"] = m
    RUN["dbg"] = " ".join(f"{a[:4]} {v:.1f}" for a, v in m.items()) + " " + "; ".join(
        f"{n}: {x.reason[:60]} @{x.t:.1f}" for n, x in (("shield", SHIELD), ("touch", TOUCHED), ("spill", SPILL),
                                                         ("moved", MOVED), ("drop", DROPPED), ("cabinet", CABINET)) if x)
    if is_open(m):
        RUN["open_s"] += DT
        if not shield_ok(m):
            SHIELD.trip(f"lid off its seat but not a shield: {m}", t)
    stem = world(pip, STEM_PTS)
    if touch_lid((stem - np.asarray(lid.pos, float)) @ lid.rotation(), lid):
        TOUCHED.trip("pipette on the lid", t)
    if touch_dish((stem - od) @ Rd):
        TOUCHED.trip("pipette on the dish's floor, rim or outside", t)
    if np.linalg.norm(od[:2] - RUN["dish0"]) > MOVED_MM / 1000:
        MOVED.trip("the dish slid", t)
    for name, st in (("lid", lid), ("pipette", pip)):
        hist = RUN["hist"][name]
        hist.append(float(st.pos[2]))
        del hist[:-int(round(DROP_T / DT)) - 1]
        if max(hist) - hist[-1] > DROP:
            DROPPED.trip(f"{name} dropped", t)
    still = np.concatenate([world(lid, LID_RIM[::6]), world(pip, _PIP_PTS)])
    clear = k.hands_clear(np.concatenate([world(lid, LID_TOP_RIM), world(pip, _PIP_PTS)]), CLEAR_MM)
    END.push(still.ravel(), clear)
    tint = int(math.ceil(10 * min(RUN["ml"] / DOSE_ML, 1.0)))
    if tint != RUN["tint"]:
        RUN["tint"] = tint
        f = tint / 10
        k.recolor("dish_medium", tuple(float(c) for c in f * np.array(PINK) + (1 - f) * np.array(CLEAR)))
    REC.tick(states, lambda: f"ml {RUN['ml']:.2f} open {RUN['open_s']:.1f} lift {m['lift_max']:.1f} "
                             f"low {m['low']:.1f} tilt {m['tilt']:.1f} az {m['az']:.0f} c {m['centre']:.1f} "
                             f"sq {1000 * squeeze(pip):.1f} ptilt {pip.tilt_deg():.1f} held {int(in_holder(pip))} end {int(END.ok)} "
                             f"score {score(states):g} {RUN['dbg'][-90:]}")


# == the cabinet's own rules (sticky zeros): the work zone, the grilles and the sash
def arm_points():
    e = k.scene.env()
    if e is None:
        return None
    from yam_arena import analysis, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = []
    for arm in ("left", "right"):
        robot = e.unwrapped.scene[f"{arm}_robot"]
        names = list(robot.joint_names)
        j = k.scene._np(robot.data.joint_pos)[0]
        q = {n: float(j[names.index(n)]) for n in ARM_JOINTS} | {f: float(j[names.index(f)]) for f in FINGER_JOINTS}
        for _, h in analysis.world_hulls(analysis.fk_link_poses_world(q, station.BASES[arm]), None):
            out.append(np.asarray(getattr(h, "vertices", getattr(h, "points", None)), dtype=float))
    return np.concatenate(out) if out else None


@k.on_step
def _cabinet_rules(states) -> None:
    if CABINET or "dish" not in states:
        return
    t = RUN.get("k", 0) * DT
    for name, boxes in (("dish", DISH_BOXES), ("lid", LID_BOXES), ("pipette", PIPETTE_BOXES), ("rack", RACK_BOXES)):
        if name not in states:
            continue
        pts = k.score.surface_points(states[name], "", boxes, 0.02)
        if np.any(pts[:, 0] < BSC_FRONT + WORK_IN):
            CABINET.trip(f"{name} within 4 in of the front grille", t)
            return
        if np.any((pts[:, 2] < TOP + GRILLE_Z) & (pts[:, 0] > BSC_BACK - REAR_GRILLE)):
            CABINET.trip(f"{name} on the rear grille", t)
            return
    if RUN.get("k", 0) % ARM_EVERY:
        return
    a = arm_points()
    if a is None:
        return
    if np.any((a[:, 0] < SASH_X) | (a[:, 0] > BSC_BACK) | (np.abs(a[:, 1]) > BSC_W / 2) | (a[:, 2] > TOP + BSC_H)):
        CABINET.trip("an arm left the cabinet's interior (sash plane, back or side wall, ceiling)", t)


# == the demo: two servoed arms
from servo import FEED, SRV, Servo, _WayPos, way_wp  # noqa: E402,F401

k.on_step(FEED.update)
k.on_reset(FEED.reset)

UP = np.array([0.0, 0.0, 1.0])
TILT = math.radians(40.0)  # the lid's tilt, back edge up
SHIFT = 0.006  # the lid drawn this far toward the robot
FRONT_Z = 0.004  # the lid's front skirt bottom this far above the dish rim
LIFT = 0.011  # first lifted straight up this far, the skirt clear of the rim
TIP_R = 0.0205  # the tip's distance from the dish axis, toward the back wall
PSI = math.radians(GAP_DEG)
TIP_Z = 0.006  # the tip's height above the work surface while it squeezes
STAGE_Z = 0.035
VIA_R, VIA_AZ = 0.065, math.radians(-50.0)  # the pipette comes in over this point, clear of the lid  # the tip's height while it waits over the dish
M_GRIP0 = np.column_stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # closing +y, approach down
M_PIP = R_PIP @ M_GRIP0
M_LID = rpy((0.0, 0.0, PSI + math.pi)) @ M_GRIP0  # closing along the tilt axis (bearing PSI - 90 deg)
HOLD_D = 0.009  # the lid's TCP sits this far above the middle of its skirt
X_LID = np.array([0.0, 0.0, PLATE_T / 2 - SKIRT / 2 + HOLD_D])
R_LID0 = rpy((0.0, 0.0, LID_YAW))
LS = Servo("left", "lid", R_LID0.T @ M_LID, R_LID0.T @ X_LID, gain=0.05, cap_p=0.006, cap_r=math.radians(2.5))
LS.read_once = True
RS = Servo("right", "pipette", M_GRIP0, np.array([0.0, 0.0, 0.004]), gain=0.06, cap_p=0.008, cap_r=math.radians(3.0))
HOLD_F = P.finger_for_gap(REST_WIDTH - 0.005)
SQUEEZE_F = P.finger_for_gap(REST_WIDTH - 2 * SQUEEZE_TRAVEL - 0.002)
_PLAN: dict = {}


def lid_pose(tilt, shift, front_z=FRONT_Z, lift=0.0):
    """(R, origin) of the lid tilted ``tilt`` about the horizontal axis at bearing PSI + 90 deg (its edge at bearing
    PSI up), its opposite skirt bottom drawn ``shift`` out past the dish's rim and ``front_z`` above it."""
    Rz = rpy((0.0, 0.0, PSI))
    Rt = Rz @ rpy((0.0, -tilt, 0.0)) @ Rz.T
    u = np.array([math.cos(PSI), math.sin(PSI), 0.0])
    F = -R_LID * u + np.array([0.0, 0.0, PLATE_T / 2 - SKIRT])
    fw = np.array([DX, DY, 0.0]) - (R_LID + shift) * u + np.array([0.0, 0.0, TOP + 0.0002 + H_BASE + front_z + lift])
    if tilt == 0.0:
        fw[2] = TOP + 0.0002 + H_BASE + PLATE_T - SKIRT + lift
    return Rt @ R_LID0, fw - Rt @ F


def on_dish(pose):
    """fn(states) -> the lid pose ``pose`` moved with the dish's live position, and the planned pose."""
    R, o = pose
    d0 = np.asarray(DISH.pos, float)
    return (lambda st: (R, o + (np.asarray(st["dish"].pos, float) - d0) * np.array([1.0, 1.0, 0.0]))), pose


def pip_pose(x, y, tip_z):
    return R_PIP, np.array([x, y, tip_z]) - R_PIP @ TIP_LOCAL


@k.on_reset
def _servo_reset(states) -> None:
    RS.reset()
    LS.reset()
    SRV.clear()
    _set_times()


@k.on_step
def _servo_step(states) -> None:
    t = RS.steps * DT
    RS.update(states, t + DT)
    LS.update(states, t + DT)


def _set_times() -> None:
    seen = set()
    for arm in ("right", "left"):
        t = 0.0
        for w in _PLAN.get(arm, []):
            if isinstance(w.pos, _WayPos) and w.pos.way not in seen:
                seen.add(w.pos.way)
                w.pos.way.t0, w.pos.way.t1 = t, t + w.duration
            t += w.duration


def _fixed(pose):
    return (lambda _st: pose), pose


def _release(servo, pose, grip, opened, back, label):
    last = servo.ways[-1]
    at = P.LivePoint(lambda st: tuple(float(v) for v in last.resolve()[1]), FEED,
                     tuple(float(v) for v in servo.tcp(*pose, bias=False)[1]))
    up = P.LivePoint(lambda st: tuple(float(v) for v in np.asarray(at.resolve()) + back), FEED,
                     tuple(float(v) for v in servo.tcp(*pose, bias=False)[1] + back))
    return [Waypoint(P.T_RELEASE, at, opened, f"let go of {label}", grip),
            Waypoint(1.2, up, opened, f"up off {label}", grip)]


def plan(how: str = "demo") -> dict:
    """The two-arm plan: right fills the pipette and waits over the dish's back; left lifts the lid into the shield;
    right goes down through the back gap, squeezes at the wall and comes out; left puts the lid back; right stands the
    pipette in its holder."""
    RS.ways.clear()
    LS.ways.clear()
    g = Grip.from_matrix(M_PIP)
    # -- right: take the pipette, fill it
    p_hold = pip_pose(HX, HY, TOP + 0.003)
    take = P.grasp_tilted(tuple(p_hold[1] + R_PIP @ np.array([0.0, 0.0, 0.004])), REST_WIDTH, grip=g, squeeze=0.0015, approach=0.060, retreat=None,
                          label="the bulb", unproven=True)
    right = list(take[:2]) + [Waypoint(0.4, take[1].pos, take[1].finger, "settle before closing", g),
                              Waypoint(0.8, take[1].pos, HOLD_F, "grip the bulb", g),
                              Waypoint(0.5, take[1].pos, HOLD_F, "settle the grip", g)]
    RS.held_from = P.duration(right) - 0.3
    travel = TOP + RES[2] + 0.012
    right.append(way_wp(RS, 1.5, *_fixed(pip_pose(HX, HY, TOP + HOLDER_H + 0.010)), HOLD_F, "lift out", integrate=False))
    right.append(way_wp(RS, 2.0, *_fixed(pip_pose(TX, TY, travel)), HOLD_F, "over the reservoir"))
    right.append(way_wp(RS, 0.8, *_fixed(pip_pose(TX, TY, travel)), SQUEEZE_F, "squeeze in air"))
    right.append(way_wp(RS, 1.5, *_fixed(pip_pose(TX, TY, TOP + RES_FLOOR + 0.0025)), SQUEEZE_F, "dip"))
    right.append(way_wp(RS, 1.0, *_fixed(pip_pose(TX, TY, TOP + RES_FLOOR + 0.0025)), HOLD_F, "release: draw 5 mL"))
    right.append(way_wp(RS, 0.5, *_fixed(pip_pose(TX, TY, TOP + RES_FLOOR + 0.0025)), HOLD_F, "wait for the draw"))
    right.append(way_wp(RS, 1.5, *_fixed(pip_pose(TX, TY, travel)), HOLD_F, "lift out of the reservoir"))
    sx, sy = DX + TIP_R * math.cos(PSI), DY + TIP_R * math.sin(PSI)
    if how == "centre":
        sx = DX
    via = (DX + VIA_R * math.cos(VIA_AZ), DY + VIA_R * math.sin(VIA_AZ))
    right.append(way_wp(RS, 1.0, *_fixed(pip_pose(TX, TY, travel)), HOLD_F, "wait over the reservoir"))
    t_staged = P.duration(right) + 2.0

    # -- left: take the lid, wait, lift it into the shield
    gl = Grip.from_matrix(M_LID)
    l0 = lid_pose(0.0, 0.0)
    grip_pt = l0[1] + l0[0] @ (X_LID - np.array([0, 0, HOLD_D]))
    takel = P.grasp_tilted(tuple(grip_pt), 2 * R_LID, grip=gl, depth=HOLD_D, squeeze=0.002, approach=0.050,
                           retreat=None, label="the lid", unproven=True)
    hold_l = takel[-1].finger
    left = list(takel[:2]) + [Waypoint(0.4, takel[1].pos, takel[1].finger, "settle before closing", gl)] + takel[2:]
    left.append(Waypoint(0.5, takel[-1].pos, hold_l, "settle the grip", gl))
    LS.held_from = P.duration(left) - 0.3
    t_open = 3.5
    wait = max(0.5, t_staged - t_open - P.duration(left))
    left.append(way_wp(LS, wait, *_fixed(l0), hold_l, "hold the lid on until the pipette is ready", integrate=False))
    if how == "lid_aside":
        aside = (l0[0], l0[1] + np.array([-0.020, 0.090, 0.0]))
        left.append(way_wp(LS, 1.0, *_fixed(lid_pose(0.0, 0.0, lift=0.020)), hold_l, "lift the lid off"))
        left.append(way_wp(LS, 2.0, *_fixed((aside[0], aside[1] + 0.020 * UP)), hold_l, "carry it aside"))
        left.append(way_wp(LS, 1.0, *_fixed(aside), hold_l, "set it down beside the dish"))
        shield = aside
    elif how == "level_high":
        shield = lid_pose(0.0, 0.0, lift=0.045)
        left.append(way_wp(LS, t_open, *_fixed(shield), hold_l, "lift the lid straight up"))
    else:
        left.append(way_wp(LS, 1.0, *_fixed(lid_pose(0.0, 0.0, lift=LIFT)), hold_l, "lift the lid level, clear of the rim"))
        shield = lid_pose(TILT, SHIFT)
        left.append(way_wp(LS, t_open - 1.0, *_fixed(shield), hold_l, "tilt into the shield"))
        left.append(way_wp(LS, 0.5, *_fixed(shield), hold_l, "settle the shield"))
    t_shield = P.duration(left)

    # -- right: down through the gap, squeeze at the wall, out
    right.append(way_wp(RS, max(0.5, t_shield - 2.0 - P.duration(right)), *_fixed(pip_pose(TX, TY, travel)),
                        HOLD_F, "wait for the shield"))
    right.append(way_wp(RS, 2.0, *_fixed(pip_pose(*via, TOP + STAGE_Z)), HOLD_F, "toward the dish's back"))
    right.append(way_wp(RS, 1.5, *_fixed(pip_pose(sx, sy, TOP + STAGE_Z)), HOLD_F, "over the gap"))
    right.append(way_wp(RS, 2.0, *_fixed(pip_pose(sx, sy, TOP + TIP_Z + 0.006)), HOLD_F, "down through the gap"))
    right.append(way_wp(RS, 0.6, *_fixed(pip_pose(sx, sy, TOP + TIP_Z)), HOLD_F, "tip at the wall"))
    right.append(way_wp(RS, 0.8, *_fixed(pip_pose(sx, sy, TOP + TIP_Z)), SQUEEZE_F, "squeeze: dispense"))
    right.append(way_wp(RS, 0.5, *_fixed(pip_pose(sx, sy, TOP + TIP_Z)), SQUEEZE_F, "hold the squeeze"))
    right.append(way_wp(RS, 2.0, *_fixed(pip_pose(sx, sy, TOP + STAGE_Z)), SQUEEZE_F, "out through the gap"))
    right.append(way_wp(RS, 0.6, *_fixed(pip_pose(sx, sy, TOP + STAGE_Z)), HOLD_F, "release in air"))
    right.append(way_wp(RS, 1.5, *_fixed(pip_pose(*via, TOP + STAGE_Z)), HOLD_F, "away from the dish"))
    t_out = P.duration(right) + 1.0
    right.append(way_wp(RS, 3.0, *_fixed(pip_pose(HX, HY, TOP + HOLDER_H + 0.010)), HOLD_F, "over the holder"))

    # -- left: the lid back on and let go
    left.append(way_wp(LS, max(0.5, t_out - P.duration(left)), *_fixed(shield), hold_l, "hold the shield"))
    if how not in ("lid_left_open", "lid_aside"):
        left.append(way_wp(LS, 2.0, *on_dish(lid_pose(0.0, 0.0, lift=LIFT)), hold_l, "level over the dish"))
        left.append(way_wp(LS, 1.2, *on_dish(lid_pose(0.0, 0.0, lift=LIFT)), hold_l, "centre over the dish"))
        left.append(way_wp(LS, 1.2, *on_dish(lid_pose(0.0, 0.0, lift=0.001)), hold_l, "lower it straight down"))
        left.append(way_wp(LS, 0.6, *on_dish(l0), hold_l, "lid on", integrate=False))
        left.append(way_wp(LS, 0.4, *_fixed(l0), hold_l, "settle on the rim", integrate=False))
        left += _release(LS, l0, gl, P.finger_for_gap(2 * R_LID + 0.016), 0.050 * UP, "the lid")
        hp, hg = P.home("left")
        left.append(Waypoint(P.HOME_S, hp, P.FINGER_OPEN, "home", hg))
    elif how == "lid_aside":
        left += _release(LS, shield, gl, P.finger_for_gap(2 * R_LID + 0.016), 0.050 * UP, "the lid")
        hp, hg = P.home("left")
        left.append(Waypoint(P.HOME_S, hp, P.FINGER_OPEN, "home", hg))

    # -- right: the pipette back in its holder
    if how != "still_held":
        right.append(way_wp(RS, 1.2, *_fixed(pip_pose(HX, HY, TOP + HOLDER_H + 0.010)), HOLD_F, "line up over the holder"))
        right.append(way_wp(RS, 3.0, *_fixed(pip_pose(HX, HY, TOP + 0.004)), HOLD_F, "into the holder"))
        right.append(way_wp(RS, 0.5, *_fixed(pip_pose(HX, HY, TOP + 0.004)), HOLD_F, "settle in the holder",
                            integrate=False))
        right += _release(RS, pip_pose(HX, HY, TOP + 0.004), g, P.finger_for_gap(REST_WIDTH + 0.016), 0.060 * UP,
                          "the pipette")
        hp, hg = P.home("right")
        right.append(Waypoint(P.HOME_S, hp, P.FINGER_OPEN, "home", hg))
    out = P.sync({"left": left, "right": right})
    for ws in out.values():
        e = ws[-1]
        ws.append(Waypoint(P.FINAL_HOLD_S, e.pos, e.finger, "hold still", e.yaw))
    _PLAN.clear()
    _PLAN.update(out)
    _set_times()
    return out


def demo():
    return D.choose(demo=lambda: plan("demo"), lid_aside=lambda: plan("lid_aside"),
                    level_high=lambda: plan("level_high"), centre=lambda: plan("centre"),
                    lid_left_open=lambda: plan("lid_left_open"), still_held=lambda: plan("still_held"))


# == what the template tests read
TEXT_NUMBERS = [("15 mm tall", H_ALL), ("52.1 mm across", 2 * R_IN), ("54.5 mm across", 2 * R_OUT),
                ("1.2 mm walls", WALL_T), ("rim 14 mm up", H_BASE), ("9 g", DISH_M), ("60 mm across", 2 * R_LID),
                ("8 mm skirt", SKIRT), ("1 mm top", PLATE_T), ("6 g", LID_M), ("x 25.0 cm", DX), ("y -1.0 cm", DY),
                ("70 x 28 mm", RES[2]), ("x 20.0 cm", TX), ("y -15.0 cm", TY),
                ("give 9 mm each", SQUEEZE_TRAVEL),
                ("6 mm across", STEM), ("120 mm long", STEM_LEN), ("24 mm square well", HOLDER_IN),
                ("65 mm deep", HOLDER_H), ("x 21.0 cm", HX), ("y -23.5 cm", HY), ("more than 5 mm above", OPEN_MM / 1000),
                ("within 25 mm", CENTRE_MM / 1000), ("at most 20 mm above", LOW_MM / 1000), ("at most 45 deg", TILT_MAX),
                ("more than 10 deg", AZ_TILT), ("within 60 deg", AZ_MAX), ("0.02 mm", TOUCH_MM / 1000),
                ("more than 5 mm; the lid", MOVED_MM / 1000), ("more than 20 mm within 0.25 s", DROP),
                ("within 0.25 s", DROP_T), ("within 8 mm of the side wall", WALL_MM / 1000),
                ("more than 30 s", OPEN_S), ("within 2 mm of seated", SEAT_MM / 1000), ("centre within 3 mm", SEAT_CENTRE_MM / 1000),
                ("at most 20 deg", HOLDER_TILT), ("within 20 mm of lid", CLEAR_MM / 1000), ("more than 1 mm for 1 s", STILL_MM / 1000),
                ("350 N/m", SIDE_RATE), ("15 g", RES_M), ("38 g", PIPETTE_M + 0.008)]
CHEATS = {"none": 0.0, "lid_aside": 0.0, "level_high": 0.0, "centre": 0.0, "lid_left_open": 0.5, "still_held": 0.75}
HOLD: dict = {}
