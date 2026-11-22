"""The kit's 7 tests for spread_medium_evenly_over_a_tripleflasks_three_layers (written by sync_kit.py --with-tests; edit freely below the line).

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


# == the revision's points (PR review: "upright with the cap pointing up"; "visual measurement for real life trials") ==
import math  # noqa: E402

import numpy as np  # noqa: E402

from ourkit import scene as S  # noqa: E402


def _rx(deg):
    t = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])


def _ry(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])


def _run(path, settle=300):
    med = M.Medium()
    for r in list(path) + [path[-1]] * settle:
        med.step(M.up_in_compartments(r), 0.01)
    return med


def _smooth(n):
    return [3 * u * u - 2 * u ** 3 for u in (i / n for i in range(1, n + 1))]


def test_8_starts_upright_cap_up():
    """Review point 1: the flask starts standing on its closed end, the cap pointing up, the closed end on the table."""
    st = S.mech_state(M.FLASK, M.FLASK.pos, S.quat_xyzw(M.FLASK.start_rotation()))
    cap = st.to_world(M.CAP_C)
    assert np.allclose(st.rotation() @ np.array([0.0, -1.0, 0.0]), [0, 0, 1], atol=1e-9)  # cap end up
    assert cap[2] - M.TOP == pytest.approx(M.BODY_L + M.NECK_L + M.CAP_L / 2, abs=0.001)
    lowest = min(st.to_world(c)[2] for c in M.CORNERS) - M.TOP
    assert 0.0 <= lowest <= 0.001  # standing on the closed end
    assert "stands upright on its closed end, cap up" in SPEC["setup"]


def test_9_laying_it_on_its_side_keeps_the_layers_even():
    """Turned from standing onto its side about the closed end's edge, the layers stay side by side and level."""
    s0 = M.FLASK.start_rotation()
    for n in (50, 300):
        med = _run([_rx(90 * u) @ s0 for u in _smooth(n)])
        assert np.allclose(med.V, M.EVEN_ML, atol=0.2) and med.neck == 0.0


def test_10_tipped_straight_over_from_standing_is_not_even():
    """The natural wrong move from the new start: tipped straight over onto its bottom face in 0.5-3 s, the bottom
    layer ends over the 1.0 band (the medium model, no arm)."""
    s0 = M.FLASK.start_rotation()
    for secs in (0.5, 1.0, 3.0):
        med = _run([_ry(90 * u) @ s0 for u in _smooth(int(secs * 100))])
        assert max(abs(v - M.EVEN_ML) for v in med.V) > M.EVEN_ML * M.BAND_FULL


def test_11_photo_depth_bands_equal_the_volume_bands():
    """Review point 2: the real trial reads each layer's depth by eye on a printed mm scale; the depth bands in the text
    are the scorer's volume bands over the 170 x 98 mm floor."""
    per_mm = M.FLOOR_CM2 / 10  # mL per mm of depth
    for lo, hi, band in ((3.6, 4.4, M.BAND_FULL), (3.0, 5.0, M.BAND_HALF)):
        assert lo * per_mm == pytest.approx(M.EVEN_ML * (1 - band), abs=0.2)
        assert hi * per_mm == pytest.approx(M.EVEN_ML * (1 + band), abs=0.2)
        assert f"{lo}-{hi} mm" in SPEC["scoring"]
    assert "photo" in SPEC["scoring"] and "printed millimetre scale" in SPEC["scoring"]
    assert "serological" not in SPEC["scoring"]


def test_12_relative_photo_check_equals_the_scorer():
    """The real trial compares each layer's band with the three bands' mean (perspective and the wall's offset scale all
    three alike, so they cancel). With all 200 mL in the layers that mean is an even split, so 'within 10% / 25% of the
    mean' picks exactly the scorer's volume bands, whatever the photo's scale."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        v = rng.dirichlet((4, 4, 4)) * M.MEDIUM_ML
        scale = rng.uniform(0.7, 1.3)  # any photo magnification
        band = v / M.FLOOR_CM2 * 10 * scale
        for frac in (M.BAND_FULL, M.BAND_HALF):
            by_photo = all(abs(b - band.mean()) <= frac * band.mean() + 1e-9 for b in band)
            by_scorer = all(abs(x - M.EVEN_ML) <= M.EVEN_ML * frac + 1e-6 for x in v)
            assert by_photo == by_scorer
    assert "three layers' mean" in SPEC["scoring"] and "edge-on" in SPEC["scoring"] and "6 are averaged" in SPEC["scoring"]
