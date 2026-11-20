"""The kit's 7 tests for zero_a_height_gauge_on_the_plate_then_fine_feed_onto_the_step (written by sync_kit.py --with-tests; edit freely below the line).

    PYTHONPATH=<yam-isaac-lab-arena> PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

The task file provides ``TEXT_NUMBERS`` [(snippet as task.json says it, the code's value in SI)], ``CHEATS``
{demo mode: the most it may score} and ``HOLD`` {"mech.joint": the most it may move with no command, mm or deg}.
Tests 4-6 read physcheck runs (``.planning/kit/kit_check.sh <task>``) and run holdcheck; they skip where our pipeline
isn't (a public copy of the task).
"""

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ourkit import testkit as T  # noqa: E402
from yam_arena import task_file  # noqa: E402

TASK = task_file.load(T.task_file_of(HERE))
M = TASK.module
SPEC = T.spec(HERE)
README = (HERE / "README.md").read_text() if (HERE / "README.md").exists() else ""
PIPE = T.pipeline_dir(HERE)
needs_pipeline = pytest.mark.skipif(PIPE is None, reason="no .planning/pipeline (public copy)")


def test_1_text_matches_code():
    assert getattr(M, "TEXT_NUMBERS", None), "list TEXT_NUMBERS in the task file"
    assert T.check_text_numbers(SPEC, M.TEXT_NUMBERS) == []


def test_2_blind_text():
    assert T.check_blind(SPEC, README) == []


def test_3_json_fields_and_run_limit():
    assert T.check_json(SPEC, M) == []


@needs_pipeline
@pytest.mark.parametrize("mode", sorted(getattr(M, "CHEATS", {"none": 0.0})))
def test_4_cheats_score_low(mode):
    cap = getattr(M, "CHEATS", {"none": 0.0})[mode]
    r, problem = T.physcheck_result(HERE, mode)
    assert r is not None and not problem, problem
    assert r["final_score"] <= cap + 1e-9, f"{mode} scored {r['final_score']} (cap {cap})"


@needs_pipeline
def test_5_demo_scores_1_in_physcheck():
    r, problem = T.physcheck_result(HERE, "demo")
    assert r is not None and not problem, problem
    assert r["final_score"] == 1.0 and r["success"], r["final_score"]
    assert r["steps"] <= T.run_limit_steps(SPEC)
    assert not r["arm_arm"], f"arms touched: {r['arm_arm']}"


@needs_pipeline
def test_6_state_holds_without_command():
    h = T.run_hold(HERE, seconds=5.0)
    caps = getattr(M, "HOLD", {})
    for joint, row in h["joints"].items():
        assert row["drift"] <= caps.get(joint, 0.5), f"{joint} moved {row['drift']} {row['unit']} with no command"
    assert all(v <= 2.0 for v in h["body_mm"].values()), h["body_mm"]
    assert (h["score"] or 0.0) == 0.0


def test_7_text_limits():
    assert T.check_limits(SPEC) == []


# -- task-specific -------------------------------------------------------------------------------------------------


def _fake_states(face_z):
    """A gauge whose scriber face sits ``face_z`` above the plate, flat on it; the block at its start, untouched."""
    import numpy as np

    class Gauge:
        pos = (M.GX0, M.GY0, M.PLATE_TOP)

        def to_world(self, local, part=""):
            if part == "slider":
                return np.array([M.BX, M.BY, M.PLATE_TOP + face_z])
            return np.array([0.0, 0.0, M.PLATE_TOP])  # base corners on the plate

    class Block:
        pos = np.array([M.BX, M.BY, M.PLATE_TOP])

        def rotation(self):
            return np.eye(3)

        def to_world(self, local, part=""):
            return np.array([M.BX, M.BY, M.PLATE_TOP + local[2]])

    return {"gauge": Gauge(), "block": Block()}


def _two_presses(zero_gap, hold_gap):
    """ZERO with the face ``zero_gap`` over the plate, then HOLD with it ``hold_gap`` over the block's top."""
    M.ZERO_SET.reset(), M.FAIL.reset(), M.END.reset()
    M.ZEROED["q"], M.HELD["value"] = None, None
    M.RUN.update(zero_dn=False, hold_dn=False, t=0.0)
    q_zero, q_hold = zero_gap, M.BLOCK_H + hold_gap
    M.presses(q_zero, True, False, zero_gap, None, 1.0)
    M.presses(q_zero, False, False, zero_gap, None, 1.1)
    M.presses(q_hold, False, True, M.BLOCK_H + hold_gap, hold_gap, 2.0)
    return M.grade(_fake_states(M.BLOCK_H + hold_gap))


def test_7_hovering_presses_score_at_most_a_quarter():
    """A press counts only touching (0.02 mm): presses hovering 0.1 mm at both surfaces cancel in the reading but score
    0; a touching zero and a hovering HOLD score 0.25; touching at both reads H (0.75 before the still window)."""
    assert _two_presses(0.0001, 0.0001) <= 0.25
    assert _two_presses(0.0, 0.0001) == 0.25
    assert _two_presses(0.0001, 0.0) == 0.0
    assert _two_presses(0.0, 0.0) == 0.75
    assert abs(M.HELD["value"] - M.BLOCK_H) < 1e-9
