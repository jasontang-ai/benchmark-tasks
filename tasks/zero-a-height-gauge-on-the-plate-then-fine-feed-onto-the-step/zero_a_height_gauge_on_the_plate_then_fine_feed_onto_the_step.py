"""Zero a digital height gauge on a granite surface plate, then measure a steel block with it and hold the reading.

On the station table: a 12 x 12 x 3 in granite surface plate; on it an INSIZE 1150-300 digital height gauge (cast-iron
base, a 300 mm beam, a slider carrying the scriber and the display with its ZERO and HOLD buttons) and a 1-2-3 block.

- The display is a rule on real geometry: a ZERO or HOLD press registers when its button travels 1.5 mm, and counts only
  if the scriber's measuring face is resting on its surface at that moment (the plate for ZERO, the block's top for
  HOLD). The reading is the slider's height minus its height at the last valid ZERO.
- Nothing is moved or latched by code. The slider rides the beam against 3 N of friction and stays where it is put.
"""

from __future__ import annotations

import collections
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
SUCCESS_HOLD_S = 0.0  # the 1.0 state already includes 1 s still
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Zero the height gauge on the surface plate, then measure the height of the steel block with it and leave "
               "the reading held on its display.")
HZ = 20

# == the plate (Standridge 12 x 12 x 3 in grade A granite) ===========================================================
PLATE = (0.305, 0.305, 0.076)
PX, PY = 0.330, -0.020
PLATE_KG = 40.0  # held as fixed: a heavy free body (the real plate is about 21 kg and does not move)
PLATE_TOP = TOP + PLATE[2]
GRANITE, IRON, STEEL, CARBIDE, LCD, BTN_C, NUT_C = ((0.12, 0.12, 0.13), (0.30, 0.32, 0.36), (0.72, 0.74, 0.76),
                                                   (0.55, 0.55, 0.58), (0.62, 0.70, 0.60), (0.85, 0.20, 0.20),
                                                   (0.20, 0.20, 0.22))

# == the gauge (INSIZE 1150-300; frame: origin at the base's underside centre, x toward the beam, scriber toward -x) ===
GAUGE_KG = 2.5
BASE = (0.080, 0.110, 0.030)
BEAM_X, BEAM_W, BEAM_H = (0.025, 0.040), 0.024, 0.300
SLIDER_KG, SLIDER_FRICTION = 0.30, 3.0
SCRIBER_L, SCRIBER_W, SCRIBER_T = 0.085, 0.008, 0.006
SLIDER_FACE = -0.005  # the display face's x
SCRIBER0 = 0.060  # scriber face start height above the plate
BTN, BTN_PITCH, BTN_PROUD, BTN_TRAVEL, BTN_FIRE = 0.009, 0.015, 0.003, 0.002, 0.0015
BTN_Z = 0.095  # button centres above the scriber's face (the fingers then clear the base's top)
BTN_RATE = 1000.0  # N/m: 2 N at full travel (a light press: at the button's height about 8 N would tip the gauge)
ZERO_Y, HOLD_Y = BTN_PITCH / 2, -BTN_PITCH / 2  # ZERO on the robot's left (+y), HOLD on its right
GX0, GY0 = 0.360, 0.060  # gauge start (base centre)

SCRIBER_TIP = (SLIDER_FACE - SCRIBER_L + 0.006, 0.0, 0.0)  # a point on the measuring face near the tip, slider frame


def _gauge_root() -> list:
    sp = k.span
    hx, hy, h = BASE[0] / 2, BASE[1] / 2, BASE[2]
    return [sp((-hx, -hy, 0.0), (hx, hy, h), IRON, label="base"),
            sp((BEAM_X[0], -BEAM_W / 2, h), (BEAM_X[1], BEAM_W / 2, h + BEAM_H), STEEL, label="beam")]


def _slider_boxes() -> list:
    sp = k.span
    return [sp((SLIDER_FACE, -0.030, SCRIBER_T), (0.045, 0.030, 0.135), IRON, label="slider_body"),
            sp((SLIDER_FACE - SCRIBER_L, -SCRIBER_W / 2, 0.0), (SLIDER_FACE, SCRIBER_W / 2, SCRIBER_T), CARBIDE,
               label="scriber"),
            sp((SLIDER_FACE - 0.0006, -0.026, 0.104), (SLIDER_FACE, 0.026, 0.130), LCD, ghost=True, label="lcd"),
            k.box((0.030, 0.034, 0.030), (0.020, 0.008, 0.020), NUT_C, ghost=True, label="fine_nut")]


def _button(y: float, tag: str) -> list:
    return [k.span((SLIDER_FACE - BTN_PROUD, y - BTN / 2, BTN_Z - BTN / 2), (SLIDER_FACE, y + BTN / 2, BTN_Z + BTN / 2),
                   BTN_C, label=f"btn_{tag}")]


GAUGE = k.mech("gauge", (GX0, GY0, PLATE_TOP + 0.0005), _gauge_root(), GAUGE_KG - SLIDER_KG - 0.02, color=IRON,
               surface=0.2, joints=[
                   # the joint's frame sits at the start height, so the start is q 0 (Isaac resets every joint to 0)
                   k.slider("slider", (0, 0, SCRIBER0), "Z", -0.005 - SCRIBER0, 0.25 - SCRIBER0, _slider_boxes(),
                            friction=SLIDER_FRICTION, damping=5.0, mass=SLIDER_KG, weightless=True),
                   k.slider("zero", (0, 0, SCRIBER0), "X", 0.0, BTN_TRAVEL, _button(ZERO_Y, "zero"), rate=BTN_RATE,
                            target=0.0, damping=2.0, mass=0.01, parent="slider"),
                   k.slider("hold", (0, 0, SCRIBER0), "X", 0.0, BTN_TRAVEL, _button(HOLD_Y, "hold"), rate=BTN_RATE,
                            target=0.0, damping=2.0, mass=0.01, parent="slider")])

# == the block (hardened 1-2-3 block, lying on its broad face) ========================================================
BLOCK_W, BLOCK_L, BLOCK_H = 0.0508, 0.0762, 0.0254
BLOCK_KG = 0.77
BX, BY = 0.265, -0.060
BLOCK = k.mech("block", (BX, BY, PLATE_TOP + 0.0005), [k.span((-BLOCK_W / 2, -BLOCK_L / 2, 0.0),
                                                              (BLOCK_W / 2, BLOCK_L / 2, BLOCK_H), STEEL, label="blk")],
               BLOCK_KG, color=STEEL, surface=0.2)
PLATE_M = k.mech("plate", (PX, PY, TOP + 0.0005), [k.span((-PLATE[0] / 2, -PLATE[1] / 2, 0.0),
                                                          (PLATE[0] / 2, PLATE[1] / 2, PLATE[2]), GRANITE,
                                                          label="granite")], PLATE_KG, color=GRANITE, surface=0.2)
OBJECTS = [PLATE_M, BLOCK, GAUGE]

# == scoring ===========================================================================================================
REST_GAP = 0.00002  # touching: the scriber's face within 0.02 mm of its surface (real: a 0.02 mm feeler will not pass)
PEN = 0.002  # contact may sink up to 2 mm in the simulator's solver
BAND_COARSE, BAND_FINE = 0.0005, 0.0001
CORNER_LIFT = 0.0005
TIP_DEG = 10.0
BLOCK_MOVE = 0.005
CLEAR_MM = 30.0
STILL_S, STILL_TOL = 1.0, 0.001
CORNERS = [(sx * BASE[0] / 2, sy * BASE[1] / 2, 0.0) for sx in (1, -1) for sy in (1, -1)]


def scriber_face(gs) -> np.ndarray:
    return gs.to_world(SCRIBER_TIP, "slider")


def plate_gap(gs) -> float:
    return float(scriber_face(gs)[2] - PLATE_TOP)


def block_gap(gs, bs):
    """The scriber's gap above the block's top, or None if the face point is not over the block's top."""
    p = bs.rotation().T @ (scriber_face(gs) - np.asarray(bs.pos))
    if abs(p[0]) > BLOCK_W / 2 - 0.002 or abs(p[1]) > BLOCK_L / 2 - 0.002:
        return None
    return float(p[2] - BLOCK_H)


def resting(gap) -> bool:
    return gap is not None and -PEN <= gap <= REST_GAP + 1e-9


def base_flat(gs) -> bool:
    return all(gs.to_world(c)[2] - PLATE_TOP <= CORNER_LIFT + 1e-9 for c in CORNERS)


def true_height(bs) -> float:
    return float(bs.to_world((0.0, 0.0, BLOCK_H))[2] - PLATE_TOP)


ZERO_SET, ZEROED = k.Ever(), {"q": None}
HELD = {"value": None}
FAIL = k.Sticky()
END = k.Settle(STILL_S, STILL_TOL)
RUN = {"zero_dn": False, "hold_dn": False, "block0": None, "t": 0.0}
REC = k.run_record.RunRecord(__file__, "gauge", "block")


def grade(states) -> float:
    if FAIL:
        return 0.0
    gs, bs = states["gauge"], states["block"]
    h = true_height(bs)
    held = HELD["value"]
    coarse = held is not None and abs(held - h) <= BAND_COARSE + 1e-9
    fine = held is not None and abs(held - h) <= BAND_FINE + 1e-9
    done = fine and base_flat(gs) and resting(block_gap(gs, bs)) and END.ok
    return k.ladder((0.25, bool(ZERO_SET)), (0.5, coarse), (0.75, fine), (1.0, done))


def score(states) -> float:
    return grade(states)


def success(states) -> bool:
    return grade(states) >= 1.0


def distance(states) -> float:
    held = HELD["value"]
    return 1.0 if held is None else abs(held - true_height(states["block"]))


@k.on_reset
def _reset(states) -> None:
    ZERO_SET.reset(), FAIL.reset(), END.reset()
    ZEROED["q"], HELD["value"] = None, None
    RUN.update(zero_dn=False, hold_dn=False, block0=np.asarray(states["block"].pos, dtype=float), t=0.0)
    REC.reset()


def presses(q, zero_dn, hold_dn, plate, block, t) -> None:
    """The display's rule at each press edge: ZERO counts with the scriber touching the plate, HOLD after a valid zero
    with the scriber touching the block (gaps in m; ``block`` None when the face is not over the block)."""
    if zero_dn and not RUN["zero_dn"]:  # a ZERO press
        if resting(plate):
            ZEROED["q"] = q
            ZERO_SET.update(True, t)
        else:
            ZEROED["q"] = None  # zero set in the air: every later reading is off
        HELD["value"] = None
    if hold_dn and not RUN["hold_dn"]:  # a HOLD press
        HELD["value"] = (q - ZEROED["q"]) if (ZEROED["q"] is not None and resting(block)) else None
    RUN["zero_dn"], RUN["hold_dn"] = zero_dn, hold_dn


@k.on_step
def _watch(states) -> None:
    RUN["t"] += 1.0 / HZ
    t = RUN["t"]
    gs, bs = states["gauge"], states["block"]
    q = gs.q["slider"]
    presses(q, gs.q["zero"] >= BTN_FIRE, gs.q["hold"] >= BTN_FIRE, plate_gap(gs), block_gap(gs, bs), t)
    if gs.tilt_deg() > TIP_DEG:
        FAIL.trip("the gauge tipped")
    c = np.asarray(gs.pos)
    if abs(c[0] - PX) > PLATE[0] / 2 or abs(c[1] - PY) > PLATE[1] / 2 or c[2] < PLATE_TOP - 0.02:
        FAIL.trip("the gauge left the plate")
    if RUN["block0"] is not None and np.linalg.norm(np.asarray(bs.pos) - RUN["block0"]) > BLOCK_MOVE:
        FAIL.trip("the block moved")
    grips = k.score.grippers()
    pts = np.concatenate([k.score.surface_points(gs, "", GAUGE.boxes, 0.03),
                          k.score.surface_points(gs, "slider", [b for b in _slider_boxes() if not S.look_of(b).ghost],
                                                 0.02)])
    clear = k.hands_clear(pts, CLEAR_MM, grips)
    END.push(np.concatenate([np.asarray(gs.pos), [q], np.asarray(bs.pos)]), clear)
    REC.tick(states, lambda: f"slider {(q + SCRIBER0) * 1000:.2f} mm plate_gap {plate_gap(gs) * 1000:.2f} block_gap "
                             f"{(block_gap(gs, bs) or 9.999) * 1000:.2f} zero {ZEROED['q']} held {HELD['value']} "
                             f"H {true_height(bs) * 1000:.2f} tilt {gs.tilt_deg():.2f} btn "
                             f"{gs.q['zero'] * 1000:.1f}/{gs.q['hold'] * 1000:.1f} score {grade(states):g}")


# == the demo ==========================================================================================================
# The right hand works horizontally (fingers along +x, closing across y): it pinches the slider's sides to move it, and
# with the fingers spread presses one button with one fingertip. The left hand, straight down, pushes the base along -y.
R_GRIP = D.Grip.deg(pitch=90)
L_GRIP = D.Grip.deg(yaw=90)  # straight down, closing along x (the hand's long side along x, clear of the slider)
PINCH_Z = 0.100  # TCP height on the slider above the scriber's face: the fingers (about 49 mm below the TCP) clear the base
PINCH_X = 0.010  # TCP x in the gauge frame: pads over the slider's front half
OPEN_PINCH, SQUEEZE = 0.080, 0.060 - 2 * 0.006
OPEN_PRESS = 0.094
PRESS_IN = 0.010  # fingertip past the button's face (2 mm travel, then the arm's give)
FINGER_HALF = 0.005  # a finger's half-thickness: its middle is this far outside its pad face
BACK = 0.060
PAST = 0.003  # slider commanded this far below its surface
SLIDE = GY0 - BY  # 0.12: the gauge slides this far toward -y to bring the scriber over the block
VIA_R = (0.22, -0.12, TOP + 0.26)
VIA_L = (0.30, 0.16, TOP + 0.20)


def gauge_pt(st, local, part="") -> np.ndarray:
    return st["gauge"].to_world(local, part)


def _slider_tcp(st, dz=0.0, dx=0.0) -> np.ndarray:
    return gauge_pt(st, (PINCH_X + dx, 0.0, PINCH_Z + dz), "slider")


def _btn_tcp(st, y_btn, depth) -> np.ndarray:
    """TCP for pressing the button at y_btn with the finger on its side of the spread hand."""
    y_tcp = y_btn + OPEN_PRESS / 2 + FINGER_HALF  # the -y finger's middle on the button; the +y finger away from the block
    x_tip = SLIDER_FACE - BTN_PROUD + depth
    return gauge_pt(st, (x_tip - D.P.PAD_TIP, y_tcp, BTN_Z), "slider")


def _fb(local, part="slider", q=SCRIBER0, dy=0.0) -> tuple:
    return (GX0 + local[0], GY0 + dy + local[1], PLATE_TOP + q + local[2])


def _move_slider(p: D.Plan, to_surface: str, label: str) -> None:
    """Pinch the slider, move it so the scriber's face reaches ``to_surface`` ('plate', 'block' or 'up'), let go."""
    p.move(1.5, lambda st: _slider_tcp(st, dx=-BACK), fallback=_fb((PINCH_X - BACK, 0, PINCH_Z)), grip=R_GRIP,
           finger=D.gap_finger(OPEN_PINCH), label=f"in front of the slider ({label})")
    p.move(1.2, lambda st: _slider_tcp(st), fallback=_fb((PINCH_X, 0, PINCH_Z)), grip=R_GRIP, label="round the slider")
    p.fingers(SQUEEZE, 0.8, "pinch the slider")

    if to_surface == "up":
        p.move(2.0, lambda st: _slider_tcp(st, dz=BLOCK_H + RAISE - plate_gap(st["gauge"])),
               fallback=_fb((PINCH_X, 0, PINCH_Z + BLOCK_H + RAISE - SCRIBER0)), grip=R_GRIP,
               label="raise the slider")
    else:
        # servo the scriber down onto the surface: each step a share of the remaining gap (plus a small press) goes into
        # the hand's offset, so the arm's vertical sag under the grip is led out until the face rests on the surface
        n = int(round(LOWER_S / 0.15))
        for i in range(n):
            p.move(LOWER_S / n, lambda st: LOWER.tcp(st, to_surface), fallback=_fb((PINCH_X, 0, PINCH_Z)),
                   grip=R_GRIP, label=f"slider to the {to_surface}", track=True)
    p.hold(0.5)
    p.fingers(OPEN_PINCH, 0.8, "let go of the slider")
    p.move(1.0, lambda st: _slider_tcp(st, dx=-BACK), fallback=_fb((PINCH_X - BACK, 0, PINCH_Z)), grip=R_GRIP,
           label="back off the slider")


RAISE = 0.050  # raise the scriber 50 mm over the block's top height (keeps the slider low for the base push)
LOWER_S = 4.0
LOWER_GAIN, LOWER_MAX = 0.12, 0.090


class Lower:
    """The hand's servo for lowering the pinched slider until the scriber rests on its surface."""

    def __init__(self):
        self.key, self.off, self.step, self.z0 = None, 0.0, None, None

    def tcp(self, st, surface: str) -> np.ndarray:
        gs = st["gauge"]
        key = (S.LATEST.get("resets"), surface)
        if self.key != key:
            self.key, self.off, self.step = key, 0.0, None
            self.z0 = float(_slider_tcp(st)[2])
        if surface == "plate":
            gap = plate_gap(gs)
        else:
            gap = block_gap(gs, st["block"])
            gap = plate_gap(gs) - BLOCK_H if gap is None else gap
        k_ = S.LATEST["steps"]
        if self.step != k_:
            self.step = k_
            self.off = float(np.clip(self.off - LOWER_GAIN * (gap + PAST), -LOWER_MAX, 0.0))
        x, y = _slider_tcp(st)[:2]
        return np.array([x, y, self.z0 + self.off])


LOWER = Lower()


class Press:
    """Push one fingertip on a button until the button's live travel passes its fire point (a servo on q)."""

    def __init__(self):
        self.key, self.off, self.step = None, 0.0, None

    def tcp(self, st, y_btn: float, joint: str) -> np.ndarray:
        key = (S.LATEST.get("resets"), joint, round(st["gauge"].pos[1], 2) if self.key is None else self.key[2])
        if self.key is None or self.key[:2] != key[:2]:
            self.key, self.off, self.step = key, 0.0, None
        k_ = S.LATEST["steps"]
        if self.step != k_:
            self.step = k_
            q = st["gauge"].q[joint]
            self.off = float(np.clip(self.off + PRESS_GAIN * (PRESS_Q - q) + 0.0002, 0.0, PRESS_MAX))
        ax = st["gauge"].rotation()[:, 0]
        return _btn_tcp(st, y_btn, 0.0) + ax * self.off


PRESS = Press()
PRESS_Q, PRESS_GAIN, PRESS_MAX = 0.0017, 0.25, 0.020


def _press(p: D.Plan, y_btn: float, label: str) -> None:
    joint = "zero" if y_btn == ZERO_Y else "hold"
    p.move(1.2, lambda st: _btn_tcp(st, y_btn, -0.020), fallback=_fb((SLIDER_FACE - 0.02 - D.P.PAD_TIP, 0, BTN_Z)),
           grip=R_GRIP, finger=D.gap_finger(OPEN_PRESS), label=f"line up on {label}")
    n = int(round(2.4 / 0.1))
    for i in range(n):
        p.move(0.1, lambda st: PRESS.tcp(st, y_btn, joint), fallback=_fb((SLIDER_FACE - D.P.PAD_TIP, 0, BTN_Z)),
               grip=R_GRIP, label=f"press {label}", track=True)
    p.move(0.8, lambda st: _btn_tcp(st, y_btn, -0.030), fallback=_fb((SLIDER_FACE - 0.03 - D.P.PAD_TIP, 0, BTN_Z)),
           grip=R_GRIP, label=f"off {label}")
    PRESS.key = None


def _right(how: str) -> D.Plan:
    p = D.Plan("right")
    p.move(2.0, VIA_R, grip=R_GRIP, finger=D.gap_finger(OPEN_PINCH), label="out in front")
    if how == "airzero":
        _press(p, ZERO_Y, "ZERO")  # zero with the scriber parked in the air
    else:
        _move_slider(p, "plate", "zero")
        _press(p, ZERO_Y, "ZERO")
    _move_slider(p, "up", "raise")
    p.move(1.5, VIA_R, grip=R_GRIP, label="clear while the base slides")
    p.hold(SLIDE_S + 8.0, "wait for the base")
    _move_slider(p, "block", "measure")
    if how == "hover":  # pinch the slider again and lift the scriber 3 mm off the block before HOLD
        p.move(1.2, lambda st: _slider_tcp(st), fallback=_fb((PINCH_X, 0, PINCH_Z)), grip=R_GRIP,
               label="(hover) round the slider")
        p.fingers(SQUEEZE, 0.8, "(hover) pinch")
        p.move(1.0, lambda st: _slider_tcp(st, dz=0.003 + 0.015), fallback=_fb((PINCH_X, 0, PINCH_Z + 0.018)),
               grip=R_GRIP, label="(hover) lift the scriber off the block")
        p.fingers(OPEN_PINCH, 0.8, "(hover) let go")
        p.move(1.0, lambda st: _slider_tcp(st, dx=-BACK), fallback=_fb((PINCH_X - BACK, 0, PINCH_Z)), grip=R_GRIP,
               label="(hover) back off")
    _press(p, HOLD_Y, "HOLD")
    p.move(1.5, VIA_R, grip=R_GRIP, label="away")
    p.home()
    return p


SLIDE_S = 6.0
PUSH_Z = 0.015  # fingertip height on the base's side above the plate
PUSH_GAIN, PUSH_MAX = 0.06, 0.025
PUSH_X = -0.030  # push at the base's front corner, clear of the slider and beam
PUSH_DY = 0.006  # fingertip's middle this far further out (more room between the hand and the slider)


class Push:
    def __init__(self):
        self.trial, self.off, self.step, self.y0 = None, 0.0, None, None

    def tcp(self, st, frac: float) -> np.ndarray:
        if self.trial != S.LATEST.get("resets"):
            self.trial, self.off, self.step, self.y0 = S.LATEST.get("resets"), 0.0, None, float(st["gauge"].pos[1])
        k_ = S.LATEST["steps"]
        want_y = self.y0 - frac * SLIDE
        if self.step != k_:
            self.step = k_
            self.off = float(np.clip(self.off + PUSH_GAIN * (want_y - st["gauge"].pos[1]), -PUSH_MAX, PUSH_MAX))
        gx = float(st["gauge"].pos[0]) + PUSH_X
        return np.array([gx, want_y + BASE[1] / 2 + 0.006 + PUSH_DY + self.off, PLATE_TOP + PUSH_Z + D.P.PAD_TIP])


PUSH = Push()


TILT_L = math.radians(30.0)
_A = np.array([0.0, -math.sin(TILT_L), -math.cos(TILT_L)])  # approach: down and inward (-y), the wrist leans out (+y)
_C = np.array([-1.0, 0.0, 0.0])  # closing across the base's 80 mm (x); this sign keeps joint 6 off its limit
L_HOLD = D.Grip.from_matrix(np.column_stack([_C, np.cross(_A, _C), _A]))
GRIP_IN = 0.040  # grip centre this far in from the base's +y end... (y in the gauge frame: BASE[1]/2 - 0.015)
GRIP_Y = BASE[1] / 2 - 0.015
GRIP_Z = 0.015
BASE_OPEN, BASE_SQUEEZE = 0.094, BASE[0] - 2 * 0.006
SLIDE_GAIN, SLIDE_MAX = 0.10, 0.040


class Slide:
    def __init__(self):
        self.trial, self.off, self.step, self.p0, self.rel = None, np.zeros(2), None, None, None

    def tcp(self, st, frac: float) -> np.ndarray:
        g = st["gauge"]
        if self.trial != S.LATEST.get("resets"):
            self.trial, self.off, self.step = S.LATEST.get("resets"), np.zeros(2), None
            self.p0 = np.asarray(g.pos, dtype=float)
            self.rel = g.to_world((0.0, GRIP_Y, GRIP_Z)) - self.p0
        want = self.p0 + np.array([0.0, -frac * SLIDE, 0.0])
        k_ = S.LATEST["steps"]
        if self.step != k_:
            self.step = k_
            err = want[:2] - np.asarray(g.pos[:2], dtype=float)
            self.off = np.clip(self.off + SLIDE_GAIN * err, -SLIDE_MAX, SLIDE_MAX)
        return want + self.rel + np.array([self.off[0], self.off[1], 0.0])


SLIDER_SERVO = Slide()


def _left(t_start: float, t_release: float) -> D.Plan:
    p = D.Plan("left")
    p.hold(t_start, "wait")
    p.move(2.0, VIA_L, grip=L_HOLD, finger=D.gap_finger(BASE_OPEN), label="above the base's end")
    grip_pt = lambda st: st["gauge"].to_world((0.0, GRIP_Y, GRIP_Z))  # noqa: E731
    fb = np.array([GX0, GY0 + GRIP_Y, PLATE_TOP + GRIP_Z])
    p.move(1.5, lambda st: grip_pt(st) - _A * 0.06, fallback=tuple(fb - _A * 0.06), grip=L_HOLD,
           label="lined up over the base's end")
    p.move(1.2, grip_pt, fallback=tuple(fb), grip=L_HOLD, label="fingers either side of the base")
    p.fingers(BASE_SQUEEZE, 0.8, "grip the base").hold(0.3)
    n = int(round(SLIDE_S / 0.15))
    for i in range(n):
        u = (i + 1) / n
        fr = u * u * (3 - 2 * u)
        p.move(SLIDE_S / n, lambda st, fr=fr: SLIDER_SERVO.tcp(st, fr), fallback=tuple(fb - np.array([0, fr * SLIDE, 0])),
               grip=L_HOLD, label="slide the base along the plate", track=True)
    p.hold(0.5)
    p.fingers(BASE_OPEN, 0.8, "let go of the base")
    p.move(1.0, lambda st: grip_pt(st) - _A * 0.07, fallback=tuple(fb - np.array([0, SLIDE, 0]) - _A * 0.07),
           grip=L_HOLD, label="back off the base")
    p.move(1.5, VIA_L, grip=L_HOLD, label="away")
    p.home()
    return p


def _plan(how: str) -> dict:
    r = _right(how)
    t_slide = sum(w.duration for w in r.ws[: next(i for i, w in enumerate(r.ws) if w.label == "wait for the base")])
    i_off = max(i for i, w in enumerate(r.ws) if w.label == "off HOLD")
    t_release = sum(w.duration for w in r.ws[: i_off + 1]) + 0.3
    left = _left(t_slide, t_release)
    return D.together(left, r)


def demo():
    return D.choose(demo=lambda: _plan("demo"), airzero=lambda: _plan("airzero"), hover=lambda: _plan("hover"))


TEXT_NUMBERS = [("305 x 305 x 76 mm", PLATE[2]), ("80 x 110 x 30 mm", BASE[2]), ("300 mm steel beam", BEAM_H),
                ("3 N of friction", SLIDER_FRICTION), ("85 x 8 x 6 mm", SCRIBER_T), ("60 mm above the plate", SCRIBER0),
                ("9 mm square", BTN), ("15 mm apart", BTN_PITCH), ("95 mm above", BTN_Z), ("travels 2 mm", BTN_TRAVEL),
                ("reaching 2 N", BTN_RATE * BTN_TRAVEL), ("at 1.5 mm", BTN_FIRE), ("2.5 kg", GAUGE_KG),
                ("0.77 kg", BLOCK_KG), ("25.4 mm tall", BLOCK_H), ("76.2 mm along y", BLOCK_L),
                ("within 0.02 mm in the simulator", REST_GAP), ("more than 10 deg", TIP_DEG), ("more than 5 mm", BLOCK_MOVE),
                ("within 0.50 mm of H", BAND_COARSE), ("within 0.10 mm of H", BAND_FINE), ("over 0.5 mm", CORNER_LIFT),
                ("within 30 mm", CLEAR_MM / 1000), ("for 1 s", STILL_S)]
CHEATS = {"none": 0.0, "airzero": 0.25, "hover": 0.25}
HOLD = {"gauge.slider": 0.5, "gauge.zero": 0.5, "gauge.hold": 0.5}
