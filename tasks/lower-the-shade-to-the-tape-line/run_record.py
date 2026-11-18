"""The run record for lower_the_shade_to_the_tape_line.py, kept apart so the task file stays short: what the task
file's hooks append to ``~/runs/<task file stem>-trace.txt`` on the simulator machine (only where that folder exists;
nothing changes in the trial). ``g`` is the task module's namespace."""

from __future__ import annotations

import pathlib

import numpy as np
from benchkit import _np, joint_name

RUN_RECORD = pathlib.Path.home() / "runs"  # the simulator machine's run folder (collect.sh copies it back); absent elsewhere
EVERY = 10  # control steps between lines: twice a second at 20 Hz


def _record(g: dict, line: str) -> None:
    if RUN_RECORD.is_dir():
        with open(RUN_RECORD / f"{pathlib.Path(g['__file__']).stem}-trace.txt", "a") as f:
            f.write(line + "\n")


def reset(g: dict) -> None:
    """At each reset: the hem bar's and the start catch's joints as the simulator holds them."""
    if RUN_RECORD.is_dir():
        _record(g, f"reset: {joint_physics(g)}")


def tick(g: dict, states) -> None:
    """After each control step: every half second, a line (``trace_line``). Never lets the record end a trial."""
    if g["LOG"].step % EVERY or not RUN_RECORD.is_dir():
        return
    try:
        _record(g, trace_line(g, states))
    except Exception as err:  # the record is diagnostic
        _record(g, f"{g['LOG'].step / g['CONTROL_HZ']:.1f}s trace failed: {type(err).__name__}: {err}")


def _array(x) -> np.ndarray:
    try:
        return _np(x)
    except Exception:  # a warp array without a torch view
        return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=float)


def joint_physics(g: dict) -> str:
    """The hem bar's and the start catch's stiffness, damping, friction and targets as the simulator holds them."""
    env = g["_ENV"]["env"]
    if env is None:
        return "no simulator"
    try:
        asset = env.unwrapped.scene[g["SHADE"].name]
        names = list(asset.joint_names)
        out = []
        for s in (g["HEMBAR"], g["SHADE"].sliders[1 + list(g["CATCH_K"]).index(0)]):
            j = names.index(joint_name(s))
            vals = []
            for attr in ("joint_stiffness", "joint_damping", "joint_friction_coeff", "joint_dynamic_friction_coeff",
                         "joint_pos_target", "joint_pos_limits"):
                value = getattr(asset.data, attr, None)
                if value is not None:
                    vals.append(f"{attr.removeprefix('joint_')} {np.round(_array(value)[0][j], 4).tolist()}")
            out.append(f"{s.name}: " + ", ".join(vals))
        return " | ".join(out)
    except Exception as err:  # the record is diagnostic: never let it end a trial
        return f"unreadable ({type(err).__name__}: {err})"


def trace_line(g: dict, states) -> str:
    """Half a second of the run: the bar's bottom against the tape (mm), its speed, the engaged catch, the deepest
    pull, the nearest gripper, the wall section's shift, the last lock events and the score so far."""
    log = g["LOG"]
    shade = states["shade"]
    q = float(shade.sliders["hembar"])
    off = (g["bottom"](q) - g["TAPE_TOP_Z"]) * 1000
    shift = (np.asarray(shade.pos) - log.wall0) * 1000 if log.wall0 is not None else np.zeros(3)
    locked = "-" if log.locked is None else f"{log.locked * 1000:.0f}"
    ev = " ".join("%s@%.2f:q%.0f,v%.0f" % e for e in log.events[-3:])
    return (f"{log.step / g['CONTROL_HZ']:.1f}s bar {off:+.1f} from tape, v {log.v * 1000:+.1f}/s, catch {locked}, "
            f"deepest {log.deepest * 1000:.0f}, clear {log.clear_mm}, wall {np.round(shift, 1).tolist()}, {ev}, "
            f"score {g['grade'](states):g}")
