"""Move a cantilever fridge shelf up one slot: tip the front up so the rear hooks swing in, then lower it level.

On the station table, straight ahead of the robot: a full-size fridge section (back wall, two side walls and a top,
open at the front) with two slotted steel shelf tracks screwed upright to its back wall. A half-width tempered-glass
cantilever shelf hangs from the tracks by two stamped steel brackets. Each bracket's rear edge carries an upper hook
(a neck through a slot window, a tip turned down behind the track's face) and, one window lower, a short straight tab.

- The hook only goes in tipped: carried level, either the hook tip meets the rung under its window or the lower tab
  meets the rung above the window below, at every height (the hook needs 8-15 mm of lift in its window, the tab can
  take at most 1 mm). Tipped up 9 degrees or more, the tab swings off the face while the hook goes in and drops; then
  lowering the front swings the tab into its window and the bracket's rear edge onto the face.
- Both brackets are on one rigid shelf 340 mm apart, so one hook a row off leaves the shelf hanging crooked.

Nothing is moved, latched or held by code: the hooks hold by contact geometry under gravity. The fridge section is a
heavy free body standing on the table (it stands in for the clamps); the shelf is one free rigid body.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import ourkit as k  # noqa: E402
from ourkit import demo as D  # noqa: E402
from ourkit import scene as S  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.0  # the 1.0 state already includes 1 s released and still
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Move the glass shelf up one slot, to the blue-taped slots, level side to side, and let go."

# == the fridge section (x forward from the arm bases' midpoint, y left, z up from the table top; metres) ==============
BOX_FRONT, BOX_BACK = 0.200, 0.545  # front edges of the walls; back wall's inner face
BOX_HALF_W, BOX_H = 0.210, 0.470  # interior half-width and height
BACK_T, SIDE_T, TOP_T = 0.018, 0.012, 0.012  # 3/4 in back wall, 1/2 in sides and top
TRACK_Y = 0.170  # track centre lines at y = +-0.170
TRACK_W, TRACK_D = 0.025, 0.012  # channel section
FACE_X = BOX_BACK - TRACK_D  # 0.533: the tracks' front faces
FACE_T = 0.002  # face sheet (collider; the real sheet is about 1.2 mm)
FLANGE_T = 0.002
TRACK_Z = (0.025, 0.460)
WIN_W, WIN_H, PITCH = 0.008, 0.020, 0.038  # slot windows: width, height, row pitch
Z_START = 0.185  # lower edge of the start row's windows
ROWS = range(-4, 7)  # window rows on each track, row 0 the start row
TARGET = 1  # the taped row: one slot up
TAPE_W, TAPE_L = 0.024, 0.050
TAPE_Y = 0.1965  # tape centred between each track and its side wall
MAT = (0.210, 0.530, 0.200, 0.013)  # foam mat: x from, x to, half-width, thickness
FRIDGE_KG = 40.0  # the section is clamped to the table; in the simulator a 40 kg free body stands in


def row_z(row: int) -> float:
    return Z_START + row * PITCH


# == the shelf (shelf frame: x from the trim's front face toward the back, y left, z up from the glass underside) ======
REAR = 0.333  # brackets' rear bearing edges: on the track faces when hung
SHELF_W, GLASS_T = 0.378, 0.004
TRIM_D, TRIM_Z, TRIM_WALL = 0.020, (-0.014, 0.016), 0.0025  # ABS front trim, a U channel open to the back
BR_Y, BR_T = 0.170, 0.003  # brackets: plate centre lines and thickness
THROAT = 0.0015  # gap between the hook tip and the back of the face when the neck sits on the window's edge
RELIEF = 0.005  # the rear edge is cut back 5 mm under the hook, so the bracket can rock on its hook tip
NECK = (REAR - RELIEF, REAR + FACE_T + THROAT, 0.000, 0.005)  # x0, x1, z0, z1
TIP = (NECK[1], NECK[1] + 0.003, -0.008, 0.005)  # the downturned tip: 8 mm below the neck's underside
TAB = (REAR, REAR + FACE_T + 0.001, -0.027, -0.022)  # lower tab, 3 mm long, 22 mm below the neck
PLATE = (0.300, REAR, -0.055, -0.020)  # rear plate's lower part: its rear edge bears on the face
PLATE_UP = (0.300, REAR - RELIEF, -0.020, 0.005)  # its upper part, under the hook
RAIL = (0.020, 0.300, -0.030, 0.000)  # bracket rail under the glass
SHELF_KG = 1.63  # see fidelity: about 1.5 kg real, raised so the hand moment matches the real centre of mass
SPAWN_GAP = 0.0005

GLASS, ABS, STEEL, PLY, TAPE_BLUE, FOAM, TRACK = (0.78, 0.88, 0.86), (0.93, 0.93, 0.92), (0.55, 0.57, 0.60), \
    (0.95, 0.95, 0.94), (0.25, 0.55, 0.88), (0.90, 0.90, 0.88), (0.85, 0.86, 0.87)


def _track(s: int) -> list:
    y0, y1 = s * TRACK_Y - TRACK_W / 2, s * TRACK_Y + TRACK_W / 2
    w0, w1 = s * TRACK_Y - WIN_W / 2, s * TRACK_Y + WIN_W / 2
    fx = (FACE_X, FACE_X + FACE_T)
    z0, z1 = TRACK_Z
    sp = k.span
    out = [sp((fx[0], y0, z0), (fx[1], w0, z1), TRACK), sp((fx[0], w1, z0), (fx[1], y1, z1), TRACK),
           sp((fx[1], y0, z0), (BOX_BACK, y0 + FLANGE_T, z1), TRACK),
           sp((fx[1], y1 - FLANGE_T, z0), (BOX_BACK, y1, z1), TRACK)]
    edges = [z0] + [e for r in ROWS for e in (row_z(r), row_z(r) + WIN_H)] + [z1]
    for a, b in zip(edges[0::2], edges[1::2]):  # rungs: the face between windows
        out.append(sp((fx[0], w0, a), (fx[1], w1, b), TRACK))
    # tape beside the target row, on the back wall (drawn only)
    ty = s * TAPE_Y
    out.append(k.box((BOX_BACK - 0.0003, ty, row_z(TARGET) + WIN_H / 2), (0.0006, TAPE_W, TAPE_L), TAPE_BLUE,
                     ghost=True, label=f"tape_{'left' if s > 0 else 'right'}"))
    return out


def _fridge_boxes() -> list:
    sp = k.span
    hw = BOX_HALF_W
    out = [sp((BOX_BACK, -hw - SIDE_T, 0), (BOX_BACK + BACK_T, hw + SIDE_T, BOX_H + TOP_T), PLY),
           sp((BOX_FRONT, hw, 0), (BOX_BACK, hw + SIDE_T, BOX_H), PLY),
           sp((BOX_FRONT, -hw - SIDE_T, 0), (BOX_BACK, -hw, BOX_H), PLY),
           sp((BOX_FRONT, -hw - SIDE_T, BOX_H), (BOX_BACK, hw + SIDE_T, BOX_H + TOP_T), PLY),
           sp((MAT[0], -MAT[2], 0), (MAT[1], MAT[2], MAT[3]), FOAM)]
    return out + _track(1) + _track(-1)


FRIDGE = k.mech("fridge", (0.0, 0.0, TOP + SPAWN_GAP), _fridge_boxes(), FRIDGE_KG, color=PLY, surface=0.3)


def _bracket(s: int) -> list:
    y = s * BR_Y
    out = []
    for (x0, x1, z0, z1), name in ((RAIL, "rail"), (PLATE, "plate"), (PLATE_UP, "plate"), (NECK, "neck"), (TIP, "tip"),
                                    (TAB, "tab")):
        out.append(k.span((x0, y - BR_T / 2, z0), (x1, y + BR_T / 2, z1), STEEL))
    return out


def _shelf_boxes() -> list:
    hw = SHELF_W / 2
    z0, z1 = TRIM_Z
    sp = k.span
    out = [sp((0.006, -hw + 0.002, 0.0), (REAR - 0.015, hw - 0.002, GLASS_T), GLASS, label="glass"),
           sp((0.0, -hw, z0), (TRIM_WALL, hw, z1), ABS, label="trim_front"),
           sp((0.0, -hw, z1 - TRIM_WALL), (TRIM_D, hw, z1), ABS, label="trim_top"),
           sp((0.0, -hw, z0), (TRIM_D, hw, z0 + TRIM_WALL), ABS, label="trim_bottom")]
    return out + _bracket(1) + _bracket(-1)


SHELF_BOXES = _shelf_boxes()
HANG_SLOPE = -1.57  # deg: hung, the shelf leans 1.57 deg front-down (its hook tip on the back of the face)
HANG_NECK_X = FACE_X - 0.0005  # where the neck's mid-face point then sits: the top 1.5 mm forward of upright


def _hang_pos() -> tuple:
    """The body origin for the shelf hung in the start row as it settles (spawned so, it does not drop)."""
    r = S.rpy((0.0, math.radians(HANG_SLOPE), 0.0))
    neck = np.array([HANG_NECK_X, 0.0, TOP + Z_START + SPAWN_GAP])
    return tuple(neck - r @ np.array([REAR + FACE_T / 2, 0.0, NECK[2]]))


SHELF = k.mech("shelf", _hang_pos(), SHELF_BOXES, SHELF_KG, pitch=math.radians(HANG_SLOPE), color=GLASS,
               surface=0.4, contact_offset=0.001)
OBJECTS = [FRIDGE, SHELF]

# == scoring ===========================================================================================================
SEAT_TOL = 0.003  # neck's underside within 3 mm above its window's lower edge (1 mm below allowed for contact)
CATCH_DEPTH = 0.004  # the tip reaches at least 4 mm below the window's lower edge, behind the face
LEVEL_TOL = 0.003  # the trim's two ends within 3 mm of each other in height
SLOPE_TOL = 2.0  # deg: the glass's front-to-back slope within 2 deg of the start's
CLEAR_MM = 30.0  # released: no gripper within 30 mm of the shelf
STILL_S, STILL_TOL = 1.0, 0.001  # still: no point of the shelf moves more than 1 mm over the last 1 s
HELD_MM = 10.0  # held: a gripper within 10 mm of the shelf
DROP_STEP = 0.045  # a fall: any point of the shelf drops more than 45 mm in one 50 ms step (0.9 m/s)
BOX_MOVE = 0.005  # the fridge section moved more than 5 mm: score 0 (it is clamped on the real bench)

NECK_PT = lambda s: (REAR + FACE_T / 2, s * BR_Y, NECK[2])  # noqa: E731 - neck underside, in the face's plane
TIP_PT = lambda s: ((TIP[0] + TIP[1]) / 2, s * BR_Y, TIP[2])  # noqa: E731 - the tip's lowest edge
ENDS = [(0.010, 0.185, 0.001), (0.010, -0.185, 0.001)]  # the trim's ends
WATCH = ENDS + [NECK_PT(1), NECK_PT(-1), (0.170, 0.0, GLASS_T)]


def slope_deg(st) -> float:
    """The glass's front-to-back slope, deg, positive with the front up."""
    r = st.rotation()
    return math.degrees(math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])))


def hook_row(st, fr, s: int):
    """The row whose window this bracket's hook is engaged in (through it and hooked over its lower edge), or None."""
    inv = fr.rotation().T
    n = inv @ (st.to_world(NECK_PT(s)) - fr.pos)  # in the fridge's frame (its origin at the table, as built)
    t = inv @ (st.to_world(TIP_PT(s)) - fr.pos)
    row = int(round((n[2] - Z_START) / PITCH))
    if row not in ROWS or abs(n[1] - s * TRACK_Y) > (WIN_W - BR_T) / 2 + 0.001:
        return None
    seated = -0.001 <= n[2] - row_z(row) <= SEAT_TOL
    caught = t[0] >= FACE_X + FACE_T - 0.0002 and t[2] <= row_z(row) - CATCH_DEPTH
    return row if seated and caught else None


FELL, MOVED, END = k.Sticky(), k.Sticky(), k.Settle(STILL_S, STILL_TOL)
RUN = {"slope0": None, "prev": None, "box0": None, "held": False, "rows": (None, None)}
REC = k.run_record.RunRecord(__file__, "shelf", "fridge")


def grade(states) -> float:
    st, fr = states["shelf"], states["fridge"]
    if FELL or MOVED:
        return 0.0
    rows = [hook_row(st, fr, 1), hook_row(st, fr, -1)]
    ends = [st.to_world(e)[2] for e in ENDS]
    level = abs(ends[0] - ends[1]) <= LEVEL_TOL + 1e-9
    slope0 = RUN["slope0"] if RUN["slope0"] is not None else slope_deg(st)
    lowered = abs(slope_deg(st) - slope0) <= SLOPE_TOL + 1e-9
    both = rows[0] is not None and rows[0] == rows[1]
    if both and rows[0] == TARGET and level and lowered:
        return 1.0 if END.ok else 0.75
    if both and rows[0] != 0:
        return 0.5
    if None not in rows and rows[0] != rows[1]:  # engaged in two different rows: hanging crooked
        return 0.25
    if 0 not in rows and RUN["held"]:  # lifted off the start row and held up by a gripper
        return 0.25
    return 0.0


def score(states) -> float:
    return grade(states)


def success(states) -> bool:
    return grade(states) >= 1.0


def distance(states) -> float:
    st = states["shelf"]
    return abs(st.to_world(NECK_PT(1))[2] - (TOP + row_z(TARGET))) + abs(st.to_world(NECK_PT(-1))[2] - (TOP + row_z(TARGET)))


@k.on_reset
def _reset(states) -> None:
    FELL.reset(), MOVED.reset(), END.reset()
    st = states["shelf"]
    RUN.update(slope0=slope_deg(st), prev=None, box0=np.asarray(states["fridge"].pos, dtype=float), held=False)
    REC.reset()


@k.on_step
def _watch(states) -> None:
    st, fr = states["shelf"], states["fridge"]
    pts = np.array([st.to_world(p) for p in WATCH])
    if RUN["prev"] is not None and float(np.max(RUN["prev"][:, 2] - pts[:, 2])) > DROP_STEP:
        FELL.trip("the shelf fell")
    RUN["prev"] = pts
    if RUN["box0"] is not None and np.linalg.norm(np.asarray(fr.pos) - RUN["box0"]) > BOX_MOVE:
        MOVED.trip("the fridge section moved")
    surf = k.score.surface_points(st, "", SHELF_BOXES, 0.03)
    grips = k.score.grippers()
    clear = k.hands_clear(surf, CLEAR_MM, grips)
    RUN["held"] = bool(grips) and not k.hands_clear(surf, HELD_MM, grips)
    END.push(pts.ravel(), clear)
    REC.tick(states, lambda: f"rows {hook_row(st, fr, 1)}/{hook_row(st, fr, -1)} necks "
                             f"{(st.to_world(NECK_PT(1))[2] - TOP) * 1000:.1f}/{(st.to_world(NECK_PT(-1))[2] - TOP) * 1000:.1f} mm "
                             f"slope {slope_deg(st):+.2f} cmd {CTL.slope_cmd} off "
                             f"{ {a: np.round(np.asarray(v) * 1000, 1).tolist() for a, v in CTL.off.items()} } "
                             f"held {RUN['held']} score {grade(states):g}")


# == the demo: both hands on the front trim, closed loop on the shelf's pose ==========================================
# The hands hold the trim with the fingers pointing into the fridge and the jaws closing up and down across it. Every
# loaded move is a servo on a wanted pose of the shelf (its slope, and the point midway between the two necks'
# undersides), read back from the shelf's pose every control step: each step a fraction of the shelf's error goes into
# the hands' offsets and the wrists' slope offset, so the arms' lag under the shelf's 16 N (about 47 mm and 8 deg at
# each hand at this reach: the wrists' 10 N m/rad drives) is led out as it builds, as the load moves between the
# hooks and the hands. Before letting go the offsets are eased back to zero with the shelf hanging on its hooks.
GRIP_X = TRIM_D + 0.004 - D.P.PAD_TIP  # TCP in the shelf frame: the fingertips 4 mm past the trim's back
GRIP_Y, GRIP_Z = 0.130, (TRIM_Z[0] + TRIM_Z[1]) / 2
OPEN_GAP, SQUEEZE = 0.060, 0.009  # 9 mm into each lip: about 45 N a finger on its 5,000 N/m drive (gripper max 50 N)
STANDOFF = 0.025  # fingertips 2 mm in front of the trim before they slide over it (the wrist is near its limit further back)
HOOK_REF = np.array([REAR + FACE_T / 2, 0.0, NECK[2]])  # midway between the necks' undersides
TILT = 16.0  # deg front up while the hooks go in and out (the tab needs 8)
LIFT = 0.011  # neck raised this far in its window: tipped 16 deg, the tip clears the lower edge above 8.4 mm and the
# neck clears the upper one below 13.7 mm
DRAW = 0.012  # drawn this far forward to bring the hooks out of the windows
PUSH_IN = 0.001  # the hook reference this far in front of the hang while the shelf is tipped and lifted on its hooks
HOOK_IN = 0.003  # the hooks go in this far behind the hang: tipped 16 deg, the tip then drops behind the face (needs
# 2.1 mm or more) while the tab still clears its front (up to 4.0 mm)
SET_DOWN = 0.003  # (slam mode) commanded this far below the lower edge to set the necks down
MAX_FIX = 0.040  # (slam mode) largest correction a move
SERVO_DT = 0.15  # s per servo waypoint (3 control steps); the target is re-read every step
SERVO_GAIN = 0.15  # fraction of the shelf's pose error added to the hands' offsets each control step
SERVO_MAX = 0.090  # m: largest hand offset on any axis (the lag at full load is about 50 mm)
MAX_SLOPE_FIX = 30.0  # deg: the wrists lead the shelf's pitch by up to this much under its moment
RELAX = 0.85  # offsets kept per control step while the hands ease off before letting go


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def _r0(arm: str) -> np.ndarray:  # fingers along +x, jaws closing across z (left -z, right +z: both reach)
    c = np.array([0.0, 0.0, -1.0 if arm == "left" else 1.0])
    a = np.array([1.0, 0.0, 0.0])
    return np.column_stack([c, np.cross(a, c), a])


def _tcp_local(arm: str) -> np.ndarray:
    return np.array([GRIP_X, GRIP_Y if arm == "left" else -GRIP_Y, GRIP_Z])


START_POS = np.array(SHELF.pos)


def _wanted_tcp(arm, slope, href) -> np.ndarray:
    """Where the TCP would be with the shelf at ``slope`` deg and its hook reference at ``href``, held exactly."""
    r = _ry(slope)
    return np.asarray(href) + r @ (_tcp_local(arm) - HOOK_REF)


class LiveGrip(D.Grip):
    """A waypoint grip whose rotation is worked out from the live states when its move starts (like ``D.Live`` for
    positions), then kept for the rest of the trial; with ``t1`` (the move's end, s) it is re-read every control step
    until then, like a tracking ``D.Live``. Before the run steps it is ``fallback``."""

    def __new__(cls, fn, fallback, t1=None):
        g = cls._make(float(fallback), fallback.rotation, None)
        g.fn, g.fb, g.value, g.trial, g.t1, g.seen = fn, fallback, None, None, t1, -1
        return g

    def __tiltgrip_rotation__(self) -> np.ndarray:
        if not D.started():
            return np.array(self.fb.rotation)
        if self.trial != S.LATEST.get("resets"):
            self.value, self.trial, self.seen = None, S.LATEST.get("resets"), -1
        k = S.LATEST["steps"]
        track = self.t1 is not None and D.now() <= self.t1 + 1e-9
        if self.value is None or (track and k != self.seen):
            self.value, self.seen = np.asarray(self.fn(S.LATEST["states"]), dtype=float), k
        return np.array(self.value)


class Shelf:
    """The two hands' shared controller. ``aim(key, slope, href, fix)`` returns {arm: (TCP, slope command)} for a move,
    worked out once per trial when that move starts. ``href`` is relative to the hook reference at the start, ``slope``
    relative to the start slope. With ``fix`` the hands' offsets (command minus where the held shelf puts the TCP) and
    the slope offset are measured afresh; otherwise the last ones are kept."""

    def __init__(self):
        self.trial, self.cache = None, {}
        self.cmd, self.off, self.slope_cmd, self.slope_off, self.h0, self.s0 = {}, {}, None, 0.0, None, 0.0
        self.step = None

    def _fresh(self) -> None:
        if self.trial != S.LATEST.get("resets"):
            self.trial, self.cache = S.LATEST.get("resets"), {}
            self.cmd, self.off, self.slope_cmd, self.slope_off, self.h0 = {}, {}, None, 0.0, None
            self.step = None

    def servo(self, slope, dh, axes="xyz", wrist="servo", ref="hook"):
        """{arm: (TCP, slope command)} for a servo waypoint on the wanted pose (``slope``, ``dh`` relative to the
        start). Once per control step each hand's offset takes in ``SERVO_GAIN`` of the shelf's error (``ref`` "hook":
        at the neck on that hand's side, so the hooks go where they must even if the slope lags; "tcp": at the hand,
        where the shelf hangs on its hooks and only the front's height can change) on the ``axes`` it is free to move
        along (on its hooks only the height: they fix it front to back), and the wrists' slope offset does the same
        for the slope error (``wrist`` "servo"), is kept ("hold") or eases back to zero ("ease": the hooks carry the
        rear). ``axes`` "" with ``wrist`` "ease" eases every offset back to zero before the hands let go."""
        self._fresh()
        st = S.LATEST["states"]["shelf"]
        if self.h0 is None:
            self.h0, self.s0 = st.to_world(HOOK_REF), slope_deg(st)
        want_slope = self.s0 + slope
        href = self.h0 + np.asarray(dh, dtype=float)
        k = S.LATEST["steps"]
        if self.step != k:  # one update a control step, whichever waypoint or arm asks first
            self.step = k
            if wrist == "servo":
                self.slope_off = float(np.clip(self.slope_off + SERVO_GAIN * (want_slope - slope_deg(st)),
                                               -MAX_SLOPE_FIX, MAX_SLOPE_FIX))
            elif wrist == "ease":
                self.slope_off *= RELAX
            mask = np.array([c in axes for c in "xyz"], dtype=float)
            for arm in ("left", "right"):
                off = np.asarray(self.off.get(arm, np.zeros(3)), dtype=float)
                if not axes:
                    off = RELAX * off
                else:
                    if ref == "hook":
                        side = np.array(NECK_PT(1 if arm == "left" else -1))
                        err = href + _ry(want_slope) @ (side - HOOK_REF) - st.to_world(side)
                    else:
                        err = _wanted_tcp(arm, want_slope, href) - st.to_world(_tcp_local(arm))
                    off = np.clip(off + SERVO_GAIN * mask * err, -SERVO_MAX, SERVO_MAX)
                self.off[arm] = off
        out = {arm: (_wanted_tcp(arm, want_slope, href) + self.off.get(arm, 0.0), want_slope + self.slope_off)
               for arm in ("left", "right")}
        self.cmd = {arm: v[0] for arm, v in out.items()}
        self.slope_cmd = want_slope + self.slope_off
        return out

    def aim(self, key, slope, dh, fix: bool, lead: float = 0.0):
        self._fresh()
        if key in self.cache:
            return self.cache[key]
        st = S.LATEST["states"]["shelf"]
        if self.h0 is None:
            self.h0, self.s0 = st.to_world(HOOK_REF), slope_deg(st)
        want_slope = self.s0 + slope
        href = self.h0 + np.asarray(dh, dtype=float)
        if fix and self.slope_cmd is not None:
            self.slope_off = float(np.clip(self.slope_cmd - slope_deg(st), -MAX_SLOPE_FIX, MAX_SLOPE_FIX))
        self.slope_off = float(np.clip(self.slope_off - lead, -MAX_SLOPE_FIX, MAX_SLOPE_FIX))
        out = {}
        for arm in ("left", "right"):
            if fix and arm in self.cmd:
                have = st.to_world(_tcp_local(arm))
                self.off[arm] = np.clip(self.cmd[arm] - have, -MAX_FIX, MAX_FIX)
            pos = _wanted_tcp(arm, want_slope, href) + self.off.get(arm, 0.0)
            out[arm] = (pos, want_slope + self.slope_off)
        for arm, (pos, sl) in out.items():
            self.cmd[arm] = pos
        self.slope_cmd = want_slope + self.slope_off
        self.cache[key] = out
        return out


CTL = Shelf()
_NOMINAL_H0 = START_POS + _ry(HANG_SLOPE) @ HOOK_REF  # planning-time stand-ins for the live start


def _go(p: D.Plan, seconds: float, key: str, slope: float, dh, fix: bool = False, label: str = "move",
        finger=None, lead: float = 0.0) -> D.Plan:
    arm = p.arm
    fb_pos = tuple(_wanted_tcp(arm, HANG_SLOPE + slope, _NOMINAL_H0 + np.asarray(dh, dtype=float)))
    fb_grip = D.Grip.from_matrix(_ry(HANG_SLOPE + slope) @ _r0(arm))
    grip = LiveGrip(lambda st: _ry(CTL.aim(key, slope, dh, fix, lead)[arm][1]) @ _r0(arm), fb_grip)
    return p.move(seconds, lambda st: CTL.aim(key, slope, dh, fix, lead)[arm][0], fallback=fb_pos, grip=grip, label=label,
                  finger=finger)


def _path(p: D.Plan, seconds: float, slope, dh, label: str, axes: str = "xyz", wrist: str = "servo",
          ref: str = "hook", finger=None) -> D.Plan:
    """A servo move: the wanted pose goes from where the last servo move left it (``p.memo``; the start pose at first)
    to (``slope``, ``dh``) along a smoothstep, as waypoints of ``SERVO_DT`` whose targets are re-read every step."""
    arm = p.arm
    s0, d0 = p.memo.get("want", (0.0, (0.0, 0.0, 0.0)))
    d0, d1 = np.asarray(d0, dtype=float), np.asarray(dh, dtype=float)
    n = max(1, int(round(seconds / SERVO_DT)))
    for i in range(n):
        u = (i + 1) / n
        f = u * u * (3.0 - 2.0 * u)  # smoothstep
        sl, d = s0 + f * (slope - s0), d0 + f * (d1 - d0)
        fb_pos = tuple(_wanted_tcp(arm, HANG_SLOPE + sl, _NOMINAL_H0 + d))
        fb_grip = D.Grip.from_matrix(_ry(HANG_SLOPE + sl) @ _r0(arm))
        t1 = p.duration + seconds / n
        grip = LiveGrip(lambda st, sl=sl, d=d: _ry(CTL.servo(sl, d, axes, wrist, ref)[arm][1]) @ _r0(arm), fb_grip, t1=t1)
        p.move(seconds / n, lambda st, sl=sl, d=d: CTL.servo(sl, d, axes, wrist, ref)[arm][0], fallback=fb_pos, grip=grip,
               label=label, finger=finger if i == 0 else None, track=True)
    p.memo["want"] = (float(slope), tuple(float(v) for v in d1))
    return p


def _clear(arm: str) -> tuple:
    """Out in front of the fridge section, beside it: the start pose is level with its side walls."""
    return (0.160, 0.250 if arm == "left" else -0.250, TOP + 0.250)


def _retreat(arm: str) -> tuple:
    """In front of the section's side wall, the start pose's grip: the way out of and back to the start pose."""
    return (0.170, 0.300 if arm == "left" else -0.300, TOP + 0.260)


def _clear_grip(arm: str):
    return D.Grip.from_matrix(_ry(30.0) @ _r0(arm))  # fingers forward and 30 deg down: the wrist turns over here


def _hands(how: str) -> dict:
    """Both arms' plans. ``how``: "demo"; "slam" (the natural move: lifted level, carried up a slot and pushed in
    level, then let go); "held" (lifted out of the start row and still held at the end)."""
    plans = {}
    for arm in ("left", "right"):
        p = D.Plan(arm)
        close = D.gap_finger(TRIM_Z[1] - TRIM_Z[0] - 2 * SQUEEZE)
        o = D.gap_finger(OPEN_GAP)
        p.move(1.5, _retreat(arm), label="back from beside the fridge", grip=D.P.home(arm)[1], finger=o)
        p.move(2.0, _clear(arm), label="out in front of the fridge", grip=_clear_grip(arm))
        _go(p, 2.5, "front", 0.0, (-STANDOFF, 0, 0), label="in front of the trim", finger=o)
        _go(p, 2.0, "on", 0.0, (0, 0, 0), label="fingers over and under the trim")
        p.fingers(TRIM_Z[1] - TRIM_Z[0] - 2 * SQUEEZE, 1.0, "pinch the trim").hold(0.5)
        if how == "slam":
            _go(p, 1.5, "lift", 0.0, (0, 0, LIFT), label="lift it level", finger=close)
            _go(p, 1.2, "lift fix", 0.0, (0, 0, LIFT), fix=True, label="correct")
            _go(p, 1.5, "out", 0.0, (-DRAW, 0, LIFT), label="draw it forward")
            _go(p, 3.0, "up", 0.0, (-DRAW, 0, LIFT + PITCH), label="up one slot")
            _go(p, 1.2, "up fix", 0.0, (-DRAW, 0, LIFT + PITCH), fix=True, label="correct")
            _go(p, 2.0, "in", 0.0, (0, 0, LIFT + PITCH), label="push it in level")
            _go(p, 1.5, "down", 0.0, (0, 0, PITCH - SET_DOWN), label="lower it")
            p.fingers(OPEN_GAP, 1.0, "let go")
        else:
            p.memo["want"] = (0.0, (0.0, 0.0, 0.0))
            # on the hooks only the height is servoed; carried, every axis and the wrists' slope lead
            _path(p, 3.0, TILT, (-PUSH_IN, 0, 0), "tip the front up about the hooks", "z", "hold", "tcp", finger=close)
            _path(p, 1.5, TILT, (-PUSH_IN, 0, 0), "hold it tipped", "z", "hold", "tcp")
            _path(p, 2.0, TILT, (-PUSH_IN, 0, LIFT), "lift the hooks off the lower edges", "z", "servo", "tcp")
            _path(p, 2.5, TILT, (-PUSH_IN, 0, LIFT), "hold the hooks up", "z", "servo", "tcp")
            _path(p, 1.5, TILT, (-DRAW, 0, LIFT), "draw the hooks out of the windows")
            _path(p, 1.0, TILT, (-DRAW, 0, LIFT), "hold")
            if how == "held":
                _path(p, 3.0, TILT, (-DRAW, 0, LIFT + 0.020), "hold it up")
            else:
                _path(p, 3.0, TILT, (-DRAW, 0, LIFT + PITCH), "up one slot")
                _path(p, 1.5, TILT, (-DRAW, 0, LIFT + PITCH), "hold")
                _path(p, 2.0, TILT, (HOOK_IN, 0, LIFT + PITCH), "hooks in through the windows")
                _path(p, 1.0, TILT, (HOOK_IN, 0, LIFT + PITCH), "hold")
                _path(p, 1.5, TILT, (HOOK_IN, 0, PITCH), "set the necks on the lower edges")
                _path(p, 1.0, TILT, (HOOK_IN, 0, PITCH), "hold")
                _path(p, 3.0, 0.0, (0, 0, PITCH), "lower the front", "z", "ease", "tcp")
                _path(p, 1.5, 0.0, (0, 0, PITCH), "let the tabs and rear edges seat", "z", "ease", "tcp")
                _path(p, 1.5, 0.0, (0, 0, PITCH), "ease the hands off", "", "ease", "tcp")
                p.fingers(OPEN_GAP, 1.2, "let go")
        if how != "held":
            p.back(STANDOFF, 1.5, "back out of the fridge")
            p.move(2.0, _clear(arm), label="clear of the fridge", grip=_clear_grip(arm))
            p.move(2.0, _retreat(arm), label="beside the fridge", grip=D.P.home(arm)[1])
            p.home()
        else:
            p.hold(3.0)
        plans[arm] = p
    return D.together(plans["left"], plans["right"])


def demo():
    return D.choose(demo=lambda: _hands("demo"), slam=lambda: _hands("slam"), held=lambda: _hands("held"))


# == what the template tests read ======================================================================================
def _com_x() -> float:
    """The shelf's centre of mass behind the trim's front, as the simulator spreads its mass (by volume)."""
    v = np.array([np.prod(bx[1]) for bx in SHELF_BOXES])
    return float((v * np.array([bx[0][0] for bx in SHELF_BOXES])).sum() / v.sum())


def _trim_start_z() -> float:
    return float(SHELF.pos[2] + (_ry(HANG_SLOPE) @ np.array([0.010, 0.0, GRIP_Z]))[2] - TOP)


TEXT_NUMBERS = [("38 mm pitch", PITCH), ("20 mm tall on", WIN_H), ("8 mm wide", WIN_W), ("1.63 kg", SHELF_KG),
                ("3.3 cm up", row_z(min(ROWS))), ("18.5 cm up", Z_START), ("22.3 cm up", row_z(TARGET)),
                ("34 cm apart", 2 * BR_Y), ("333 mm behind", REAR), ("5 mm tall through", NECK[3] - NECK[2]),
                ("8 mm below the neck", -TIP[2]), ("1.5 mm behind", THROAT), ("22 to 27 mm", -TAB[2]),
                ("cut back 5 mm", RELIEF), ("about 17.7 cm up", _trim_start_z(), 0.01),
                ("about 1.6 deg", -HANG_SLOPE, 0.03), ("within 3 mm above", SEAT_TOL),
                ("at least 4 mm below", CATCH_DEPTH), ("within 3 mm of each other", LEVEL_TOL),
                ("within 2 deg", SLOPE_TOL), ("within 30 mm", CLEAR_MM / 1000), ("more than 1 mm over", STILL_TOL),
                ("over the last 1 s", STILL_S), ("45 mm in one", DROP_STEP), ("more than 5 mm off", BOX_MOVE),
                ("within 10 mm", HELD_MM / 1000), ("148 mm behind", _com_x()), ("40 kg", FRIDGE_KG),
                ("Track face 2 mm", FACE_T), ("4 mm catch", CATCH_DEPTH)]
CHEATS = {"none": 0.0, "slam": 0.0, "held": 0.25}  # mode: the most it may score
HOLD = {}  # no joints; the shelf and the section are watched as bodies
