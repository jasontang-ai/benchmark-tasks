"""The scripted demo for lower_the_shade_to_the_tape_line.py: ``plan(M)`` builds both arms' waypoints from the task
module's namespace ``M``. Kept apart so the task file stays short; the task file's ``demo()`` calls it.

The left arm first parks out of the shade's sweep (its start pose sits where the hem bar and fabric pass on the way
down). The right hand, turned so its fingers point along the bar (+y) and its jaws close front to back, comes in line
with the hem bar beyond its right end and slides on to 30 mm from the end, so its palm stays clear of the fabric. It
closes on the bar and pulls it down past the line's catch. The arm is a soft spring: under the shade's 6 N it sits about
30 mm above where it is sent, so the demo reads where the bar really is (a scripted demo may read the simulator's state;
nothing is moved by script) and corrects the pull to 26 mm below the line's catch, less than one catch spacing. Then it
eases the bar up at about 6 mm/s, slower than the lock's catch speed, sent 4 mm past the line's catch, so the lock
takes the bar there. It rests the hand on the caught bar, lets go, slides off the bar's end and goes home.
"""

from __future__ import annotations

import numpy as np
from tiltgrip import Grip
from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint

GRIP = Grip.deg(yaw=90, pitch=90)  # fingers point +y, along the bar; jaws close along x; palm beyond the bar's end
IN_FROM_END = 0.030  # m: the TCP this far in from the bar's right end
APPROACH = 0.080  # m back along -y before sliding on: the fingertips start 37 mm beyond the bar's end
OPEN_GAP = 0.039  # m between the pads round the 19 mm bar (10 mm a side)
SQUEEZE = 0.002  # m commanded into each side of the bar
BELOW_LINE = 0.026  # m: the pull goes this far below the line's catch (36 mm above the next one down)
LAG_GUESS = 6.0 / 0.45 / 1000.0  # m: first guess of the hand's lag under 6 N (0.45 N/mm); then read live
EASE_V = 0.006  # m/s mean while easing up (the move's smooth profile peaks at 1.5 times this)
EASE_PAST = 0.004  # m the ease is sent past the line's catch
MAX_FIX, MAX_LAG = 0.040, 0.050  # m: live corrections are kept within these, so a slipped bar can't send the hand down
NEAR_V = 0.040  # m/s: the fastest mean pace near the shade
PARK = (0.200, 0.305, 0.300)  # left TCP, m (z above the table): out of the shade's sweep
HOME_S, HOLD_S = 3.0, 3.0


def gap_finger(gap: float) -> float:
    """The finger joint target that puts the pads ``gap`` m apart."""
    return -float(np.clip(gap, 0.0, -2.0 * FINGER_OPEN)) / 2.0


def home(arm: str) -> tuple:
    """(TCP position, grip) of the arm's start pose."""
    pos, rot = ik.tcp_pose(np.asarray(READY, dtype=float), station.BASES[arm])
    return tuple(float(v) for v in pos), Grip.from_matrix(rot)


class Live:
    """A Waypoint position (world, m) given by ``fn(states)``, worked out the first time it is read once the run has
    stepped (when the scripted arm starts toward it) and kept. Read while planning (no step yet), it gives
    ``fallback`` and keeps nothing, so offline checks see the planned path."""

    def __init__(self, M, fn, fallback):
        self.M, self.fn, self.fallback, self.value = M, fn, tuple(float(v) for v in fallback), None

    def resolve(self) -> tuple:
        if self.value is not None:
            return self.value
        if not self.M.LATEST["steps"] or self.M.LATEST["states"] is None:
            return self.fallback
        self.value = tuple(float(v) for v in self.fn(self.M.LATEST["states"]))
        return self.value

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.resolve(), dtype=dtype or float)

    def __iter__(self):
        return iter(self.resolve())

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return self.resolve()[i]


def duration(ws) -> float:
    return float(sum(w.duration for w in ws))


def plan(M) -> dict:
    """Both arms' waypoints (``M``: the task module's namespace)."""
    top = M.TOP + M.SPAWN_GAP
    y = M.BAR_Y[0] + IN_FROM_END
    cmd: dict = {}

    def at(z) -> tuple:
        return (M.BAR_X, y, float(z))

    def bar_z(st) -> float:  # the hem bar's centre (world z), where the simulator has it
        return float(st["shade"].pos[2]) + M.BAR_Z0 - M.REST_DROP + float(st["shade"].sliders["hembar"])

    def base(st) -> float:  # the table-top reference of the shade's travel, where the wall section has settled
        return float(st["shade"].pos[2])

    def live(key, fn, fallback) -> Live:
        def keep(st):
            cmd[key] = tuple(float(v) for v in fn(st))
            return cmd[key]

        return Live(M, keep, fallback)

    z0 = top + M.BAR_Z0
    target = top + M.BAR_Z0 + M.C_LINE - BELOW_LINE  # the bar's centre 26 mm below the line's catch
    pull_z = target - LAG_GUESS
    opened, hold = gap_finger(OPEN_GAP), gap_finger(M.BAR_D - 2 * SQUEEZE)
    ease_fb = at(pull_z + BELOW_LINE + EASE_PAST)
    ease_d = BELOW_LINE + EASE_PAST

    def fix(st):  # where to send the hand so the bar sits at the target depth (a correction of at most 40 mm)
        return at(pull_z + float(np.clip(target - bar_z(st), -MAX_FIX, MAX_FIX)))

    def ease(st):  # the line's catch, sent 4 mm past, less the hand's lag as read now (bar minus command, 0-50 mm)
        lag = float(np.clip(bar_z(st) - cmd["fix"][2], 0.0, MAX_LAG))
        cmd["lag"] = (0.0, 0.0, lag)
        return at(base(st) + M.BAR_Z0 + M.C_LINE - lag + EASE_PAST)

    right = [
        Waypoint(2.0, (M.BAR_X, y - APPROACH, z0), opened, "in line with the hem bar", GRIP),
        Waypoint(max(1.5, APPROACH / NEAR_V), at(z0), opened, "slide onto the hem bar", GRIP),
        Waypoint(0.8, at(z0), hold, "grip the hem bar", GRIP),
        Waypoint(max(2.0, (z0 - pull_z) / NEAR_V), at(pull_z), hold, "pull the bar down past the line", GRIP),
        Waypoint(1.0, at(pull_z), hold, "hold the bar down", GRIP),
        Waypoint(1.0, live("fix", fix, at(pull_z)), hold, "pull the bar to its depth", GRIP),
        Waypoint(1.0, live("fixed", lambda st: cmd["fix"], at(pull_z)), hold, "hold the bar down", GRIP),
        Waypoint(ease_d / EASE_V, live("ease", ease, ease_fb), hold, "ease the bar up to the line", GRIP),
        Waypoint(1.0, live("eased", lambda st: cmd["ease"], ease_fb), hold, "let the catch take the bar", GRIP),
        Waypoint(1.0, live("rest", lambda st: at(bar_z(st)), ease_fb), hold, "rest the hand on the caught bar", GRIP),
        Waypoint(1.0, live("open", lambda st: cmd["rest"], ease_fb), opened, "let go of the bar", GRIP),
        Waypoint(2.0, live("off", lambda st: np.subtract(cmd["rest"], (0.0, APPROACH, 0.0)),
                           np.subtract(ease_fb, (0.0, APPROACH, 0.0))), opened, "slide off the bar's end", GRIP),
    ]
    pos, g = home("right")
    right += [Waypoint(HOME_S, pos, FINGER_OPEN, "back to the start pose", g),
              Waypoint(HOLD_S, pos, FINGER_OPEN, "hold still", g)]
    park = (PARK[0], PARK[1], M.TOP + PARK[2])
    left_grip = home("left")[1]
    left = [Waypoint(2.0, park, FINGER_OPEN, "park clear of the shade", left_grip)]
    left.append(Waypoint(duration(right) - duration(left), park, FINGER_OPEN, "hold still", left_grip))
    return {"right": right, "left": left}


def total_s(M) -> float:
    return max(duration(ws) for ws in plan(M).values())
