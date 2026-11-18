"""The task file without a simulator: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

Needs ``yam_arena`` importable (put a clone of yam-isaac-lab-arena on PYTHONPATH). Checks the props against the real
items, the lock rule, every outcome row and loophole, the latch, the milestone and release tests, the text against the
code, and the demo: reach, the run limit and the final hold.
"""

import json
import math
import pathlib
import re
import sys
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from benchkit import tool_state  # noqa: E402
from yam_arena import station, task_file  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402

TASK = task_file.load(HERE / "lower_the_shade_to_the_tape_line.py")
M = TASK.module
HZ = 20.0
SPEC = json.loads((HERE / "task.json").read_text())
README = (HERE / "README.md").read_text()
START = M.REST_DROP  # slider travel at the start catch
LINE = M.CATCHES[-M.LINE_K]  # slider travel at the line's catch


# -- building states by hand ------------------------------------------------------------------------------------------


def shade(q=START, shift=(0.0, 0.0, 0.0)):
    """The shade with its hem bar at slider travel ``q``, the wall section moved ``shift`` m."""
    return {"shade": tool_state(M.SHADE, np.add(M.SHADE.pos, shift), (0.0, 0.0, 0.0, 1.0),
                                sliders={"hembar": q, **{M.catch_name(k): 0.0 for k in M.CATCH_K}})}


def cube_hull(centre, half=0.01):
    c = np.asarray(centre, dtype=float)
    normals = np.vstack([np.eye(3), -np.eye(3)])
    return SimpleNamespace(planes=np.c_[normals, normals @ c + half])


FAR = [[("gripper", cube_hull((0.10, 0.40, 1.10)))], [("gripper", cube_hull((0.10, -0.40, 1.10)))]]


def on_bar(q, gap=0.0):
    """A gripper (2 cm cube) whose face is ``gap`` m in front of the hem bar, 30 mm in from its right end."""
    z = M.TOP + M.SPAWN_GAP + M.BAR_Z0 - M.REST_DROP + q
    return [FAR[0], [("gripper", cube_hull((M.BAR_X - M.BAR_D / 2 - gap - 0.01, 0.0, z)))]]


def run(path, grips=FAR, shift_at=None):
    """Feed a path of slider travels (one per control step) through the hooks from a fresh reset; ``grips`` is the
    grippers' hulls, or a function of the step's travel. Returns the score at the last step."""
    M._reset(shade())
    st = shade()
    for i, q in enumerate(path):
        shift = shift_at(i) if shift_at else (0.0, 0.0, 0.0)
        st = shade(q, shift)
        M._watch(st, grips=grips(q) if callable(grips) else grips)
    return M.grade(st["shade"] and st)


def ramp(a, b, speed):
    """Travels from ``a`` to ``b`` at ``speed`` m/s, one per step (``b`` included)."""
    n = max(1, int(math.ceil(abs(b - a) / (speed / HZ))))
    return list(np.linspace(a, b, n + 1)[1:])


def hold(q, seconds):
    return [q] * int(round(seconds * HZ))


def pull_to(depth):
    """From the start catch, pulled down to ``depth`` m of travel below it at 50 mm/s."""
    return ramp(START, START - depth, 0.05)


def ease_to_line():
    """Pulled 26 mm below the line's catch, then eased up at 6 mm/s until the lock takes it at the line's catch."""
    return pull_to(START - LINE + 0.026) + hold(LINE - 0.026, 1.0) + ramp(LINE - 0.026, LINE, 0.006)


# -- props --------------------------------------------------------------------------------------------------------


def test_the_scene_loads():
    assert TASK.scene == "station"
    assert [o.name for o in M.OBJECTS][0] == "shade"


@pytest.mark.parametrize("sim, real", [
    (M.BAR_Y[1] - M.BAR_Y[0], 0.6096),  # 24 in shade
    (M.BAR_D, 0.019),  # 3/4 in round hem bar
    (2 * M.ROLLER_R, 0.038 + 0.010),  # 38 mm tube plus wound fabric (about 5 mm a side)
    (M.OPEN_Y[1] - M.OPEN_Y[0], 0.560),
    (M.OPEN_Z[1] - M.OPEN_Z[0], 0.640),
    (M.PANEL_Y[1] - M.PANEL_Y[0], 0.900),
    (M.PLY_T, 0.019), (M.TRIM_W, 0.089), (M.TAPE_W, 0.024), (M.RULE_Z[1] - M.RULE_Z[0], 0.300),
    (M.SPACER_X[1] - M.SPACER_X[0], 0.038),
])
def test_every_prop_is_full_size(sim, real):
    assert 0.9 <= sim / real <= 1.1


def test_the_masses_are_not_lighter_than_the_real_items():
    assert M.HEMBAR.mass >= M.BAR_MASS and M.SHADE.mass >= 10.0


def test_the_bar_hangs_from_the_roller_in_front_of_the_window():
    assert M.BAR_X < M.WALL_X - M.TRIM_T - 0.05  # 60 mm in front of the casing
    assert M.OPEN_Y[0] - M.TRIM_W < M.BAR_Y[0] < M.OPEN_Y[0] and M.OPEN_Y[1] < M.BAR_Y[1] < M.OPEN_Y[1] + M.TRIM_W


def test_the_shade_reaches_down_to_the_table_and_up_to_the_roller():
    assert M.bottom(M.Q_LOW) == pytest.approx(0.005)
    assert M.BAR_Z0 + M.Q_TOP - M.REST_DROP + M.BAR_D / 2 < M.ROLLER_Z - M.ROLLER_R
    assert M.ROLLER_Z - M.bottom(M.Q_LOW) < 0.914  # the 36 in fabric reaches


def test_a_catch_every_half_turn_of_the_38_mm_roller():
    assert M.SPACING == pytest.approx(math.pi * (0.038 + 0.001) / 2, abs=0.002)
    assert sorted(M.CATCHES.values())[0] > M.Q_LOW and sorted(M.CATCHES.values())[-1] < M.Q_TOP


def test_each_stop_sits_on_the_bars_top_at_its_catch():
    for k in M.CATCH_K:
        s = M.SHADE.sliders[1 + list(M.CATCH_K).index(k)]
        under = s.rest[2] - s.size[2] / 2
        bar_top = M.BAR_Z0 - M.REST_DROP + M.CATCHES[k] + M.BAR_D / 2
        assert under == pytest.approx(bar_top)


def test_the_start_stop_rests_in_and_every_other_rests_out():
    for k in M.CATCH_K:
        s = M.SHADE.sliders[1 + list(M.CATCH_K).index(k)]
        inner_y = s.rest[1] - s.size[1] / 2
        if k == 0:
            assert inner_y < M.BAR_Y[1] - 0.010  # over the bar's last 18 mm
        else:
            assert inner_y >= M.BAR_Y[1] + 0.004  # 4 mm clear of its end
        assert s.weightless and s.hidden


def test_the_bar_rests_a_mm_under_its_start_stop_and_the_spring_lifts_it_on():
    assert M.HEMBAR.rest[2] == pytest.approx(M.BAR_Z0 - 0.001)
    assert M.RATE * (M.PRELOAD - M.CATCHES[0]) == pytest.approx(M.F0)
    assert M.RATE * (M.PRELOAD - LINE) == pytest.approx(6.0, abs=0.01)


def test_the_fabric_collider_stops_short_of_the_stops():
    cloth = [b for b in M.BAR_BOXES if b[2] == M.FABRIC and not (len(b) > 3 and b[3].ghost)]
    hi = max(M.MID_Y + c[1] + s[1] / 2 for c, s, *_ in cloth)
    assert hi <= M.STOP_Y - M.STOP[1] / 2 - 0.005


def test_the_stops_target_each_end_of_their_travel():
    assert M.stop_target(0, True) < 0.0 < M.STOP_TRAVEL < M.stop_target(0, False)
    assert M.stop_target(-2, True) < -M.STOP_TRAVEL < 0.0 < M.stop_target(-2, False)


# -- the lock -------------------------------------------------------------------------------------------------------


def lock_events():
    return [e[0] for e in M.LOG.events]


def test_pulling_the_bar_five_mm_below_its_catch_frees_it():
    run(pull_to(0.004))
    assert M.LOG.locked == pytest.approx(START)
    run(pull_to(0.006))
    assert M.LOG.locked is None and "freed" in lock_events()


def test_a_slow_rise_is_caught_at_the_next_catch_up():
    run(ease_to_line())
    assert M.LOG.locked == pytest.approx(LINE) and lock_events()[-1] == "caught"


@pytest.mark.parametrize("speed", [0.015, 0.0195])
def test_a_rise_just_under_the_catch_speed_still_catches(speed):
    run(pull_to(0.150) + ramp(START - 0.150, LINE, speed))
    assert M.LOG.locked == pytest.approx(LINE)


@pytest.mark.parametrize("speed", [0.021, 0.10, 1.0])
def test_a_fast_rise_runs_past_every_catch(speed):
    run(pull_to(0.150) + ramp(START - 0.150, M.Q_TOP, speed))
    assert M.LOG.locked is None and "ran past" in lock_events()


def test_going_down_never_catches():
    run(ramp(START, M.Q_LOW, 0.005))
    assert M.LOG.locked is None and lock_events().count("caught") == 1  # only the start


# -- outcome table ---------------------------------------------------------------------------------------------------


def test_doing_nothing_scores_zero():
    assert run(hold(START, 10.0)) == 0.0


def test_pulled_less_than_five_cm_scores_zero():
    assert run(pull_to(0.045) + hold(START - 0.045, 4.0), grips=on_bar) == 0.0


def test_held_at_the_line_to_the_end_scores_a_quarter():
    assert run(pull_to(START - LINE) + hold(LINE, 5.0), grips=on_bar) == 0.25


def test_let_go_fast_and_run_up_to_the_top_scores_zero():
    assert run(pull_to(0.150) + ramp(START - 0.150, M.Q_TOP, 1.0) + hold(M.Q_TOP, 4.0)) == 0.0


def test_back_at_the_start_catch_scores_zero():
    assert run(pull_to(0.080) + ramp(START - 0.080, START, 0.005) + hold(START, 4.0)) == 0.0


def test_caught_at_the_line_released_and_still_scores_one():
    assert run(ease_to_line() + hold(LINE, 3.1)) == 1.0


@pytest.mark.parametrize("k", [-1, -3, -4])
def test_caught_at_another_catch_below_the_start_scores_a_half(k):
    c = M.CATCHES[k]
    path = pull_to(START - c + 0.020) + hold(c - 0.020, 1.0) + ramp(c - 0.020, c, 0.006) + hold(c, 3.1)
    assert M.LOG is not None and run(path) == 0.5


def test_caught_above_the_start_scores_zero():
    c = M.CATCHES[1]
    assert run(pull_to(0.010) + ramp(START - 0.010, c, 0.006) + hold(c, 4.0)) == 0.0


@pytest.mark.parametrize("off, score", [(-0.0299, 1.0), (0.0299, 1.0), (-0.0301, 0.5), (0.0301, 0.5)])
def test_the_full_score_band_is_three_cm_either_side_of_the_tape(off, score):
    q = LINE + off  # a bar left still there (as if caught) after a pull
    assert run(pull_to(0.160) + ramp(START - 0.160, q, 0.05) + hold(q, 3.1)) == score


def test_the_band_holds_only_the_lines_catch():
    for k in M.CATCH_K:
        inside = abs(M.bottom(M.CATCHES[k]) - M.TAPE_TOP_Z) <= M.LINE_BAND
        assert inside == (k == -M.LINE_K)


def test_every_honest_partial_is_ordered():
    held = run(pull_to(START - LINE) + hold(LINE, 5.0), grips=on_bar)
    wrong = run(pull_to(0.090) + ramp(START - 0.090, M.CATCHES[-1], 0.006) + hold(M.CATCHES[-1], 3.1))
    right = run(ease_to_line() + hold(LINE, 3.1))
    assert 0.0 < held < wrong < right == 1.0


# -- latches, milestones and release ----------------------------------------------------------------------------------


def test_the_wall_moved_more_than_five_mm_scores_zero_even_when_put_back():
    path = ease_to_line() + hold(LINE, 3.1)
    assert run(path, shift_at=lambda i: (0.006, 0.0, 0.0) if i == 10 else (0.0, 0.0, 0.0)) == 0.0


def test_the_wall_shifting_four_mm_costs_nothing():
    assert run(ease_to_line() + hold(LINE, 3.1), shift_at=lambda i: (0.004, 0.0, 0.0)) == 1.0


def test_caught_at_the_line_then_pulled_down_and_held_loses_the_credit():
    path = ease_to_line() + hold(LINE, 3.1)
    first = run(path)
    again = run(path + ramp(LINE, LINE - 0.040, 0.05) + hold(LINE - 0.040, 4.0), grips=on_bar)
    assert first == 1.0 and again == 0.25


def test_caught_then_run_up_after_being_freed_scores_zero():
    path = ease_to_line() + hold(LINE, 3.1) + ramp(LINE, LINE - 0.010, 0.05) + ramp(LINE - 0.010, M.Q_TOP, 1.0)
    assert run(path + hold(M.Q_TOP, 4.0)) == 0.0


def test_a_run_up_then_pulled_down_again_and_caught_scores_one():
    path = pull_to(0.150) + ramp(START - 0.150, M.Q_TOP, 1.0) + hold(M.Q_TOP, 2.0)
    path += ramp(M.Q_TOP, LINE - 0.026, 0.05) + ramp(LINE - 0.026, LINE, 0.006) + hold(LINE, 3.1)
    assert run(path) == 1.0


def test_a_gripper_still_on_the_caught_bar_scores_a_quarter():
    assert run(ease_to_line() + hold(LINE, 4.0), grips=on_bar) == 0.25


@pytest.mark.parametrize("gap, score", [(0.025, 0.25), (0.035, 1.0)])
def test_a_gripper_must_be_three_cm_clear_for_the_last_three_s(gap, score):
    assert run(ease_to_line() + hold(LINE, 3.1), grips=lambda q: on_bar(q, gap)) == score


def test_released_but_not_yet_three_s_still_scores_a_quarter():
    assert run(ease_to_line() + hold(LINE, 2.5)) == 0.25


def test_a_bar_still_moving_more_than_two_mm_scores_a_quarter():
    path = ease_to_line() + hold(LINE, 1.0) + ramp(LINE, LINE - 0.004, 0.002) + hold(LINE - 0.004, 1.0)
    assert run(path) == 0.25


def test_success_is_the_full_score():
    run(ease_to_line() + hold(LINE, 3.1))
    st = shade(LINE)
    assert M.success(st) and M.distance(st) < 0.001


def test_blind_policies_score_zero_or_a_quarter():
    assert run(hold(START, 30.0)) == 0.0  # never touch it
    assert run(pull_to(0.200) + hold(START - 0.200, 30.0), grips=on_bar) == 0.25  # pull down and hold
    assert run(pull_to(0.124) + ramp(START - 0.124, M.Q_TOP, 1.0) + hold(M.Q_TOP, 10.0)) == 0.0  # let go at the line


# -- text and code ---------------------------------------------------------------------------------------------------


def mm(x):
    return f"{x * 1000:g}"


def test_text_matches_code():
    setup, scoring, fid = SPEC["setup"], SPEC["scoring"], SPEC["fidelity"]
    for s in ["19 mm", "38 mm", "24 mm", "300 mm", "56 x 64 cm", "90 x 90 cm", "1x4", "19 x 89 mm",
              f"x {M.BAR_X * 100:g}", f"x {M.WALL_X * 100:g}", f"{M.ROLLER_Z * 100:g} cm",
              f"{(M.BAR_Y[1] - M.BAR_Y[0]) * 100:.0f} cm", f"y {M.BAR_Y[0] * 100:g} to {M.BAR_Y[1] * 100:g}",
              f"{M.START_BOTTOM * 100:.2f} cm", f"{M.BAR_Z0 * 100:.1f} cm", f"{(M.START_BOTTOM - 0.100) * 1000:.1f} mm",
              f"{M.TAPE_TOP_Z * 100:.2f} cm", f"{(M.TAPE_TOP_Z - 0.100) * 1000:.1f} mm", f"{M.SPACING * 100:.1f} cm",
              f"{-M.C_LINE * 100:.1f} cm", f"{M.F0:.1f} N", f"{M.F0 + M.RATE * M.SPACING * M.LINE_K:.1f} N",
              f"{M.RULE_Z[0] * 100:.0f} cm above the table", "7 mm outside", "y 0 to 56", "8 to 72"]:
        assert s in setup, s
    assert f"{M.OPEN_Y[1] * 100:g}" in setup and f"{M.OPEN_Z[1] * 100:g}" in setup
    for s in [f"{M.STILL_MM * 1000:g} mm", f"{M.STILL_S:g} s", f"{M.CLEAR_GAP * 100:g} cm", f"{M.WALL_MOVE * 1000:g} mm",
              f"{M.PULL_DEPTH * 100:g} cm below its start", f"{M.BELOW_START * 100:g} cm below its start",
              f"within {M.LINE_BAND * 100:g} cm of the tape", f"{M.SPACING * 100:.1f} cm apart"]:
        assert s in scoring, s
    assert re.search(r"0\.25 if .*0\.5 if .*1\.0 if", scoring)
    for s in [f"{M.V_CATCH * 1000:g} mm/s", f"{M.FREE_BELOW * 1000:g} mm below", f"{M.HEMBAR.mass:g} kg",
              f"{M.WALL_SIM_KG:g} kg", f"{M.SPACING * 100:.1f} cm per half turn", "19 mm diamond"]:
        assert s in fid, s
    assert SPEC["instruction"] == M.INSTRUCTION


def test_the_run_limit_text_matches_its_steps():
    steps = int(re.match(r"([\d,]+) steps", SPEC["run_limit"]).group(1).replace(",", ""))
    assert steps == 2400 and "2 minutes at 20 Hz" in SPEC["run_limit"]


def test_no_other_task_is_named_in_the_text():
    text = json.dumps(SPEC) + README
    assert not re.search(r"#\d{2,}|\bPR\b|TK-[A-Z0-9]", text)


def test_the_text_is_blind():
    text = (json.dumps(SPEC) + README).lower()
    for word in ("jason", "robocurve team", "/users/", "bluedot", "nanzheng", "github.com/"):
        assert word not in text


def test_the_required_fields_are_filled():
    for k in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
              "reset_notes", "fidelity", "scores", "media"):
        assert SPEC.get(k), k
    assert SPEC["category"] in ("Impact on jobs", "Impact on AI timelines", "Safety and alignment",
                                "Societal relevance", "Societal relatability", "Other")
    assert len(SPEC["setup"]) <= 3000 and len(SPEC["scoring"]) <= 2500


def test_the_readme_and_main_file_fit_the_reviewers_excerpt():
    assert len(README) + len((HERE / "lower_the_shade_to_the_tape_line.py").read_text()) < 40000


def test_the_instruction_states_the_goal_only():
    low = SPEC["instruction"].lower()
    for word in ("slow", "past", "catch", "lock", "let go", "if ", "don't", "never"):
        assert word not in low


# -- the demo --------------------------------------------------------------------------------------------------------

PLAN = M.demo()


def plans():
    """The demo as yam_scripted plays it, planned offline (no simulator states: live points use their fallbacks)."""
    M._reset({})
    out = {}
    for arm, ws in M.demo().items():
        p = ArmPlan(list(ws), station.BASES[arm])
        for k in range(int(math.ceil(p.duration * HZ)) + 2):
            p.action7((k + 1) / HZ)
        out[arm] = p
    return out


def test_every_demo_waypoint_is_reachable():
    for arm, p in plans().items():
        assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 2.0, arm


def test_the_demo_fits_the_run_limit():
    steps = max(sum(w.duration for w in ws) for ws in PLAN.values()) * HZ + 3 * HZ
    limit = 2400
    assert steps <= limit <= 20 * steps


def test_both_arms_end_together_and_hold_still_for_two_seconds():
    ends = {arm: sum(w.duration for w in ws) for arm, ws in PLAN.items()}
    assert abs(ends["left"] - ends["right"]) < 1e-6
    for ws in PLAN.values():
        assert ws[-1].duration >= 2.0 and tuple(ws[-1].pos) == tuple(ws[-2].pos)


def test_the_left_arm_parks_out_of_the_shades_sweep_first():
    left = PLAN["left"][0]
    assert left.duration <= 2.0 and left.pos[0] < M.BAR_X - 0.10


def test_the_right_hand_comes_onto_the_bar_from_beyond_its_end():
    first, on = PLAN["right"][0], PLAN["right"][1]
    assert first.pos[1] < M.BAR_Y[0] - 0.030 and on.pos[1] == pytest.approx(M.BAR_Y[0] + 0.030)


def test_the_ease_is_slower_than_the_catch_speed():
    ease = next(w for w in PLAN["right"] if w.label.startswith("ease"))
    import shade_demo as D

    assert 1.5 * D.EASE_V < 0.5 * M.V_CATCH and ease.duration * D.EASE_V == pytest.approx(D.BELOW_LINE + D.EASE_PAST)


def test_the_pull_goes_below_the_line_by_less_than_one_catch_spacing():
    import shade_demo as D

    assert 0.0 < D.BELOW_LINE < M.SPACING - 0.025


def test_the_run_record_is_written_only_where_the_runs_folder_exists(tmp_path, monkeypatch):
    import run_record

    monkeypatch.setattr(run_record, "RUN_RECORD", tmp_path / "runs")
    run(ease_to_line()[:40])
    assert not (tmp_path / "runs").exists()
    (tmp_path / "runs").mkdir()
    run(ease_to_line()[:40])
    lines = (tmp_path / "runs" / "lower_the_shade_to_the_tape_line-trace.txt").read_text().splitlines()
    assert lines and "from tape" in lines[-1]
