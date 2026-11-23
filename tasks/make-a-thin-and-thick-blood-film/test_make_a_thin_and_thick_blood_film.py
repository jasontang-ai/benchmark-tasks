"""The kit's 7 tests for make_a_thin_and_thick_blood_film (written by sync_kit.py --with-tests; edit freely below the line).

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


# -- below the line: the film rules against the reviewer's scenarios ----------------------------------------------------
import math as _m  # noqa: E402


def _run(path):
    """Feed the thin-film rule a list of (edge x in mm, on, angle deg) samples, one per step."""
    M.RUN.clear()
    M.RUN.update(M._fresh())
    for x, on, ang in path:
        M._thin({"x": x / 1000, "y": 0.0, "on": on and M.ANGLE_LO <= _m.radians(ang) <= M.ANGLE_HI, "angle": _m.radians(ang)})
    return M.RUN


def _good(ang=45.0, speed=33.0, wait=30, pause_at=None):
    x0 = M.X_DROP * 1000
    path = [(x0 + 4, True, ang)] + [(x0 - 0.5, True, ang)] * wait
    x = x0 - 0.5
    while x < 36:
        x += speed / 20
        path.append((x, True, ang))
        if pause_at is not None and abs(x - pause_at) < speed / 20:
            path += [(x, True, ang)] * 8
    return path


def test_rule_good_stroke_feathers():
    r = _run(_good())
    assert r["feathered"] and not r["band"] and M.THIN2_LO <= M.thin_length() * 1000 <= M.THIN2_HI
    assert M.thin_ok_2()


def test_rule_flat_drag_lays_nothing():
    r = _run(_good(ang=5.0))
    assert not r["on"] and not r["done"] and not M.thin_ok_1()


def test_rule_upright_90_lays_nothing():
    r = _run(_good(ang=88.0))
    assert not M.thin_ok_1()


def test_rule_pause_mid_stroke_is_a_band():
    r = _run(_good(pause_at=M.X_DROP * 1000 + 8))
    assert r["band"] and not M.thin_ok_2()


def test_rule_push_before_spread_is_narrow():
    r = _run(_good(wait=2))
    assert r["width"] * 1000 < M.THIN1_W and not M.thin_ok_1()


def test_rule_lift_mid_stroke_no_feather():
    x0 = M.X_DROP * 1000
    path = [(x0 + 4, True, 45.0)] + [(x0 - 0.5, True, 45.0)] * 30 + [(x0 + i * 1.6, True, 45.0) for i in range(1, 8)] + [(x0 + 14, False, 45.0)] * 5
    r = _run(path)
    assert r["done"] and not r["feathered"] and not M.thin_ok_2()


def test_rule_shallow_angle_runs_off_the_end():
    r = _run(_good(ang=25.0))
    assert not M.thin_ok_2()


# -- the edge-contact rule against the reviewers' loopholes (off the slide, wrong lean, either edge) --------------------
import numpy as _np  # noqa: E402
from ourkit.scene import mech_state as _ms  # noqa: E402


def _quat(r):
    w = _m.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        return ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w)
    i = int(_np.argmax(_np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = _m.sqrt(max(0.0, 1 + r[i, i] - r[j, j] - r[k, k])) * 2
    q = [0.0, 0.0, 0.0]
    q[i] = s / 4
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return (*q, (r[k, j] - r[j, k]) / s)


def _states(edge_xyz, lean_back=True, deg=45.0, flip=False):
    """Slide at its start; the spreader with its lower short edge centred at edge_xyz (world), at deg to the slide."""
    c, s = _m.cos(_m.radians(deg)), _m.sin(_m.radians(deg))
    zl = _np.array([-c if lean_back else c, 0.0, s])  # from the working edge to the far end
    xl = _np.array([0.0, 1.0, 0.0])
    if flip:  # the other short edge works: local +z points down to it
        zl, xl = -zl, xl
    yl = _np.cross(zl, xl)
    r = _np.column_stack([xl, yl, zl])
    sign = -1.0 if not flip else 1.0
    pos = _np.asarray(edge_xyz) - r @ _np.array([0, 0, sign * M.SL_L / 2])
    return {"slide": _ms(M.SLIDE, (M.SLIDE_C[0], M.SLIDE_C[1], M.SLIDE_Z)),
            "spreader": _ms(M.SPREADER, tuple(pos), _quat(r))}


_TOPZ = M.SLIDE_Z + M.SL_T / 2 + 0.0004  # the edge centre where its lowest corner rests on the glass at 45 deg


def test_edge_on_the_glass_counts():
    e = M.edge_state(_states((M.SLIDE_C[0] + M.X_DROP, 0.0, _TOPZ)))
    assert e["on"] and e["back"]


def test_edge_on_the_mat_beside_the_slide_does_not_count():
    e = M.edge_state(_states((M.SLIDE_C[0] + M.X_DROP, 0.03, M.MAT_TOP)))
    assert not e["on"]


def test_edge_below_the_top_face_does_not_count():
    e = M.edge_state(_states((M.SLIDE_C[0] + M.X_DROP, 0.0, _TOPZ - 0.0012)))
    assert not e["on"]


def test_wrong_lean_does_not_count():
    e = M.edge_state(_states((M.SLIDE_C[0] + M.X_DROP, 0.0, _TOPZ), lean_back=False))
    assert not e["on"] and not e["back"]


def test_either_short_edge_works():
    e = M.edge_state(_states((M.SLIDE_C[0] + M.X_DROP, 0.0, _TOPZ), flip=True))
    assert e["on"]


def test_rule_no_front_touch_no_spread():
    """Set down on the drop (not in front of it, then drawn back): the blood is not taken up along the edge."""
    x0 = M.X_DROP * 1000
    r = _run([(x0, True, 45.0)] * 30 + [(x0 + i * 1.6, True, 45.0) for i in range(1, 25)])
    assert r["w"] == 0 and not M.thin_ok_1()


def test_rule_lift_then_set_down_ahead_lays_no_film():
    """Wick the blood, lift the spreader, set it down 25 mm ahead with no push: no film (the reviewers' case)."""
    x0 = M.X_DROP * 1000
    path = [(x0 + 4, True, 45.0)] + [(x0 - 0.5, True, 45.0)] * 30 + [(x0 + 5, False, 45.0)] * 6 \
        + [(x0 + 25, True, 45.0)] * 10
    r = _run(path)
    assert not r["on"] and not M.thin_ok_1()


def test_rule_short_blip_off_the_glass_keeps_the_spread():
    x0 = M.X_DROP * 1000
    r = _run([(x0 + 4, True, 45.0)] + [(x0 - 0.5, True, 45.0)] * 30 + [(x0 - 0.5, False, 45.0)]
             + [(x0 - 0.5 + i * 1.6, True, 45.0) for i in range(1, 30)])
    assert r["feathered"] and M.thin_ok_2()


def test_rule_thirty_and_fortyfive_degrees_both_pass():
    for ang in (30.0, 45.0):
        _run(_good(ang=ang))
        assert M.thin_ok_2(), ang


def test_spreader_back_on_the_finished_film_smudges_it():
    _run(_good())
    assert M.thin_ok_2()
    M._smudge(_states((M.SLIDE_C[0] + M.X_DROP + 0.008, 0.0, _TOPZ), lean_back=False))
    assert M.RUN["smudged"] and not M.thin_ok_2()


def test_each_corner_set_down_is_a_stroke():
    M.RUN.clear()
    M.RUN.update(M._fresh())
    c = (M.X_THICK, 0.0)
    for _ in range(10):
        M._thick({"corner_on": True, "corner": c})
        M._thick({"corner_on": False, "corner": c})
    assert M.RUN["strokes"] == 10 and not M.thick_ok()


def test_rule_lift_then_push_from_the_drop_discards_the_spread():
    """Wick the blood, lift the spreader clear, set it back on the drop and push: the spread was lost, no film."""
    x0 = M.X_DROP * 1000
    path = [(x0 + 4, True, 45.0)] + [(x0 - 0.5, True, 45.0)] * 30 + [(x0 - 0.5, False, 45.0)] * 6 \
        + [(x0 - 0.5 + i * 1.6, True, 45.0) for i in range(0, 30)]
    r = _run(path)
    assert r["w"] == 0 and not r["on"] and not M.thin_ok_1()
