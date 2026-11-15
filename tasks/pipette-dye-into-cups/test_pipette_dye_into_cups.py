"""The task file without a simulator: python -m pytest -q test_pipette_dye_into_cups.py

Needs ``yam_arena`` importable (it is in the simulator's environment; elsewhere put a clone of yam-isaac-lab-arena
on PYTHONPATH). Checks the scene loads, every demo waypoint is reachable, the outcome table, and a kinematic replay
of the demo: the pipette follows the gripper rigidly while held, which is enough to score the liquid moves.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from benchkit import tool_state  # noqa: E402
from yam_arena import station, task_file  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402

TASK = task_file.load(HERE / "pipette_dye_into_cups.py")
M = TASK.module
UPRIGHT = (0.0, 0.0, 0.0, 1.0)
HZ = 20.0


def pipette_at(tip_xyz, squeeze=0.0, quat=UPRIGHT):
    origin = np.asarray(tip_xyz, dtype=float) - np.array(M.TIP_LOCAL)
    return tool_state(M.PIPETTE, origin, quat, sliders={"side_l": -squeeze, "side_r": squeeze})


def feed(steps, grips=()):
    """Run (tip position, squeeze) pairs through the task's step hook. ``grips``: the grippers' bounds each step
    (none by default, so the pipette counts as untouched)."""
    M._reset({})
    for tip, sq in steps:
        M._track({"pipette": pipette_at(tip, sq)}, grips=list(grips))


def rest(pipette, steps=None, grips=()):
    """Leave ``pipette`` where it is for ``steps`` control steps (1 s by default)."""
    for _ in range(M.REST_STEPS if steps is None else steps):
        M._track({"pipette": pipette}, grips=list(grips))


def xyz(well, z):
    return (*M.WELL_XY[well], z)


def at(well, what):
    return (*M.tip_xy(well, what), M.tip_z(well, what))


def deep(well):
    """The tip on the well's floor at the draw point: a real bulb draws only the liquid above its tip, so a policy
    taking a second draw from a well holding one pipetteful has to go all the way down. The demo draws 1.8 mm up."""
    return (*M.tip_xy(well, "dip"), M.FLOOR[well])


def transfer_steps(src, dst, dip_dst=False, wait=True):
    """One transfer, then (by default) a wait over ``dst`` long enough for the stirrer to mix it, as the demo leaves."""
    full = M.SQUEEZE_TRAVEL
    steps = [
        (at(src, "prime"), 0.0), (at(src, "prime"), full), (deep(src), full),
        (deep(src), 0.0), (xyz(src, M.TIP_TRAVEL), 0.0), (xyz(dst, M.TIP_TRAVEL), 0.0),
        (deep(dst) if dip_dst else at(dst, "dispense"), 0.0), (at(dst, "dispense"), full),
        (xyz(dst, M.TIP_TRAVEL), full), (xyz(dst, M.TIP_TRAVEL), 0.0),
    ]
    return steps + ([(xyz(dst, M.TIP_TRAVEL), 0.0)] * int(M.MIX_S * M.HZ) if wait else [])




PARKED = pipette_at((*M.HOLDER_XY, M.TIP_PARK))
HELD_UP = pipette_at((*M.HOLDER_XY, M.TIP_TRAVEL))


def test_the_task_loads_with_its_objects_hooks_and_docs():
    assert TASK.scene == "station"
    assert {"success", "score", "distance", "demo"} <= {k for k in ("success", "score", "distance", "demo")
                                                         if getattr(TASK, k) is not None}
    assert TASK.free_objects == ("pipette",)
    assert "transfer pipette" in TASK.docs


def test_the_full_protocol_scores_one():
    feed(transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3"))
    rest(PARKED)
    assert M.cups_in_order(M.LEDGER) == 3
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(1.0)
    assert M.success({"pipette": PARKED})


def test_doing_nothing_scores_zero():
    M._reset({})
    rest(PARKED, 3 * M.REST_STEPS)
    assert M.grade(M.LEDGER, PARKED) == 0.0


def test_partial_credit_counts_wells_in_order():
    feed(transfer_steps("stock", "cup1"))
    assert M.grade(M.LEDGER, HELD_UP) == pytest.approx(0.3)
    feed(transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2"))
    assert M.grade(M.LEDGER, HELD_UP) == pytest.approx(0.6)
    feed(transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3"))
    assert M.grade(M.LEDGER, HELD_UP) == pytest.approx(0.9)  # the pipette not back in its holder


def test_skipping_a_well_breaks_the_chain():
    feed(transfer_steps("stock", "cup2"))  # well 2 gets a 1:2, not 1:4; well 1 untouched
    assert M.cups_in_order(M.LEDGER) == 0


def test_stock_into_every_well_is_not_a_dilution_series():
    feed(transfer_steps("stock", "cup1") + transfer_steps("stock", "cup2"))
    assert M.grade(M.LEDGER, HELD_UP) == 0.0


def test_going_back_to_an_earlier_well_scores_zero():
    feed(transfer_steps("stock", "cup1", dip_dst=True) + transfer_steps("stock", "cup2"))
    assert M.went_back(M.LEDGER)
    assert M.grade(M.LEDGER, HELD_UP) == 0.0


def test_a_spill_caps_the_score():
    steps = transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3")
    full = M.SQUEEZE_TRAVEL
    bench = (0.35, -0.05, M.TOP + 0.05)
    steps += [(at("cup3", "prime"), 0.0), (at("cup3", "prime"), full), (at("cup3", "dip"), full),
              (at("cup3", "dip"), 0.0), (bench, 0.0), (bench, full)]
    feed(steps)
    assert M.LEDGER.spilled > 0
    assert M.grade(M.LEDGER, PARKED, let_go=True) == pytest.approx(M.SPILL_CAP)


def test_a_pipette_lying_down_is_not_in_its_holder():
    lying = pipette_at((*M.HOLDER_XY, M.TIP_PARK), quat=(0.7071068, 0.0, 0.0, 0.7071068))
    assert not M.in_holder(lying)
    assert M.in_holder(PARKED)


def plan():
    return ArmPlan(list(M.demo()["left"]), station.BASES["left"])


def test_every_demo_waypoint_is_reachable_within_the_joint_limits():
    arm = plan()
    for t in np.arange(0.0, arm.duration + 0.5, 1.0 / HZ):
        arm.action7(t)
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1


def test_the_demo_fits_the_run_limit():
    assert plan().duration * HZ + 3 * HZ < 2400


def test_a_kinematic_replay_of_the_demo_scores_one(monkeypatch):
    """Hold the pipette rigidly under the TCP while the fingers are narrower than the bulb; drop it where it is
    let go. Squeeze follows the finger gap. The replay has no wrist sag, so it runs without the trims that offset it."""
    monkeypatch.setattr(M, "TRIM", {k: (0.0, 0.0) for k in M.TRIM})
    arm = plan()
    origin = np.array(M.PIPETTE.pos, dtype=float)
    held = False
    M._reset({})
    pipette = pipette_at(origin + np.array(M.TIP_LOCAL))
    for t in np.arange(0.0, arm.duration + 1.5, 1.0 / HZ):
        tcp, _, finger, _ = arm.target(t)
        q, _ = arm.action7(t)
        grip = M.gripper_bounds(q[:6], abs(finger), station.BASES["left"])
        width = -2.0 * finger
        touching = width < M.REST_WIDTH
        near = np.linalg.norm(tcp - (origin + np.array([0, 0, M.GRIP_ABOVE_CENTRE]))) < 0.005
        if touching and (held or near):
            held = True
            origin = np.asarray(tcp) - np.array([0, 0, M.GRIP_ABOVE_CENTRE])
        else:
            held = False
        side = min(max((M.REST_WIDTH - width) / 2, 0.0), M.SQUEEZE_TRAVEL) if held else 0.0
        pipette = tool_state(M.PIPETTE, origin, UPRIGHT, sliders={"side_l": -side, "side_r": side})
        M._track({"pipette": pipette}, grips=[grip])  # the right arm stays at its ready pose, far to the right
    assert not M.went_back(M.LEDGER) and M.LEDGER.spilled == 0.0
    assert [e.kind for e in M.LEDGER.events if e.kind != "dip"] == ["aspirate", "dispense"] * 3
    assert M.settled(), "the left gripper must be clear of the pipette for 1 s by the end"
    assert M.grade(M.LEDGER, pipette) == pytest.approx(1.0)


def test_the_hold_grip_does_not_count_as_a_squeeze():
    side_at_hold = (M.REST_WIDTH + 2 * M.HOLD) / 2
    assert 0 < side_at_hold / M.SQUEEZE_TRAVEL < 0.3
    side_at_squeeze = min((M.REST_WIDTH + 2 * M.SQUEEZE) / 2, M.SQUEEZE_TRAVEL)
    assert side_at_squeeze / M.SQUEEZE_TRAVEL >= 0.7


def test_holding_the_pipette_in_its_holder_is_not_putting_it_away():
    feed(transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3"))
    on_it = [(PARKED.low - 0.01, PARKED.high + 0.01)]
    rest(PARKED, 3 * M.REST_STEPS, grips=on_it)
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(0.9)
    assert not M.success({"pipette": PARKED})


def test_a_full_pipette_put_away_is_not_full_marks():
    steps = transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3")
    full = M.SQUEEZE_TRAVEL
    # Dip into well 3 again and draw a pipetteful up, then park with it.
    steps += [(at("cup3", "prime"), full), (at("cup3", "dip"), full), (at("cup3", "dip"), 0.0)]
    feed(steps)
    assert M.LEDGER.tip_volume > 0
    assert M.grade(M.LEDGER, PARKED, let_go=True) < 1.0


def test_squeezing_just_over_a_well_lands_in_it():
    full = M.SQUEEZE_TRAVEL
    over = xyz("cup1", M.RIM_Z + 0.01)
    feed([(at("stock", "prime"), full), (at("stock", "dip"), full), (at("stock", "dip"), 0.0),
          (over, 0.0), (over, full), (over, 0.0)])
    assert M.LEDGER.spilled == 0.0 and M.cups_in_order(M.LEDGER) == 1


def test_two_pipettefuls_of_stock_into_well_1_is_wrong_by_volume():
    feed(transfer_steps("stock", "cup1") + transfer_steps("stock", "cup1"))
    well = M.LEDGER.vessels["cup1"]
    assert well.volume == pytest.approx(3.0)
    assert M.cups_in_order(M.LEDGER) == 0


def test_strength_band_matches_two_fold_reference_cuvettes():
    lo, hi = M.CONC_BAND
    assert lo * 0.5 == pytest.approx(0.5 / 2**0.5) and hi * 0.5 == pytest.approx(0.5 * 2**0.5)


@pytest.mark.parametrize("well", ["stock", "cup1", "cup2"])
def test_a_real_bulb_draws_a_full_pipetteful_beside_the_stir_bar(well):
    """Each well the demo draws from holds two pipettefuls then, and one after the draw. A real bulb draws only while
    its tip is under the liquid, so the tip must stay under the level that is left; it draws beside the stir bar,
    clear of its sweep, and clear of the wall."""
    floor = M.FLOOR[well]
    assert floor < M.tip_z(well, "dip") <= floor + M.LIQUID_PER_ALIQUOT - 0.0015
    bar_sweep = M.STIR_BAR[0] / 2
    assert M.DRAW_OFFSET - M.STEM / 2 - bar_sweep >= 0.0019  # beside the spinning bar
    assert M.WELL_R - M.DRAW_OFFSET - M.STEM / 2 >= 0.004  # clear of the wall across the offset
    assert (M.WELL_R**2 - M.DRAW_OFFSET**2) ** 0.5 - M.STEM / 2 >= 0.010  # along x, where the aim corrections are
    assert M.tip_z(well, "dip") - (floor + M.FILM) >= 0.001  # clear of the floor film


@pytest.mark.parametrize("well", ["stock", "cup1", "cup2", "cup3"])
def test_dispensing_and_priming_stay_above_the_liquid_and_inside_the_well(well):
    floor = M.FLOOR[well]
    assert M.tip_z(well, "dispense") > floor + 2 * M.LIQUID_PER_ALIQUOT + 0.003
    assert floor + 2 * M.LIQUID_PER_ALIQUOT < M.tip_z(well, "prime") < M.RIM_Z
    assert M.TIP_TRAVEL >= M.RIM_Z + 0.0099  # 1 cm over every rim while moving


def test_the_stem_clears_the_well_walls_with_room_to_spare():
    assert M.WELL_R - M.STEM / 2 >= 0.013  # dispensing over the centre


def test_task_json_matches_the_task_file():
    import json

    spec = json.loads((HERE / "task.json").read_text())
    assert spec["instruction"] == M.INSTRUCTION
    assert len(spec["fidelity"]) <= 500 and len(spec["scores"]) <= 300
    readme = (HERE / "README.md").read_text()
    assert readme.count(M.INSTRUCTION) == 2


def test_two_stock_pipettefuls_then_the_normal_chain_scores_zero():
    feed(transfer_steps("stock", "cup1") + transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2")
         + transfer_steps("cup2", "cup3"))
    assert M.grade(M.LEDGER, PARKED) == 0.0


def test_carrying_liquid_back_up_the_row_scores_zero():
    steps = transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3")
    feed(steps + transfer_steps("cup3", "cup2"))
    assert M.went_back(M.LEDGER) and M.grade(M.LEDGER, PARKED) == 0.0


# -- let go: still for 1 s with no gripper on any part of the pipette (revision, Nov 14) ------------------------------
def chain():
    return transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3")


def test_a_gripper_on_the_stem_alone_still_counts_as_holding():
    """The old let-go test read only the bulb's sides, so a grip on the stem scored 1.0. Bounds catch any grip."""
    stem_lo = PARKED.point("tip") + np.array([-0.006, -0.006, 0.01])
    stem_hi = PARKED.point("tip") + np.array([0.006, 0.006, 0.05])
    assert M.touched(PARKED, [(stem_lo, stem_hi)])
    feed(chain())
    rest(PARKED, 3 * M.REST_STEPS, grips=[(stem_lo, stem_hi)])
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(0.9)


def test_a_gripper_beside_the_pipette_is_not_on_it():
    beside = [(PARKED.high + 0.01, PARKED.high + 0.2)]
    assert not M.touched(PARKED, beside) and M.touched(PARKED, None)


def test_the_pipette_must_rest_untouched_for_a_full_second():
    feed(chain())
    M._track({"pipette": PARKED}, grips=[(PARKED.low, PARKED.high)])  # the hand lets go here
    rest(PARKED, M.REST_STEPS - 1)
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(0.9)
    rest(PARKED, 1)
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(1.0)


def test_a_touch_or_a_nudge_starts_the_count_again():
    feed(chain())
    rest(PARKED)
    M._track({"pipette": PARKED}, grips=[(PARKED.low, PARKED.high)])
    assert M.RUN["settled"] == 0
    moving = tool_state(M.PIPETTE, PARKED.pos, UPRIGHT, lin_vel=(0.05, 0, 0))
    M._track({"pipette": moving}, grips=[])
    assert M.RUN["settled"] == 0 and M.grade(M.LEDGER, PARKED) == pytest.approx(0.9)


def test_unreadable_arms_never_count_as_let_go():
    M._reset({})
    for tip, sq in chain():
        M._track({"pipette": pipette_at(tip, sq)}, grips=None)
    for _ in range(3 * M.REST_STEPS):
        M._track({"pipette": PARKED}, grips=None)
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(0.9)


# -- the stock: one pipetteful out, net of any put back --------------------------------------------------------------
def test_a_draw_put_back_and_drawn_again_scores_normally():
    full = M.SQUEEZE_TRAVEL
    redo = [(at("stock", "prime"), 0.0), (at("stock", "prime"), full), (at("stock", "dip"), full),
            (at("stock", "dip"), 0.0), (at("stock", "dispense"), 0.0), (at("stock", "dispense"), full),
            (xyz("stock", M.TIP_TRAVEL), full), (xyz("stock", M.TIP_TRAVEL), 0.0)]
    feed(redo + chain())
    rest(PARKED)
    assert M.stock_taken(M.LEDGER) == pytest.approx(1.0)
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(1.0)


def test_two_pipettefuls_out_of_the_stock_score_zero_even_if_one_is_spilled():
    full = M.SQUEEZE_TRAVEL
    bench = (0.12, 0.30, M.TOP + 0.05)
    spill = [(at("stock", "prime"), 0.0), (at("stock", "prime"), full), (deep("stock"), full),
             (deep("stock"), 0.0), (bench, 0.0), (bench, full), (bench, 0.0)]
    feed(spill + chain())
    assert M.stock_taken(M.LEDGER) == pytest.approx(2.0)
    assert M.grade(M.LEDGER, PARKED, let_go=True) == 0.0


# -- the stirrer and the plate --------------------------------------------------------------------------------------
def test_the_wells_sit_over_the_stirrers_points_in_a_real_plate():
    """Corning 3516: 127.8 x 85.5 mm, wells 34.8 mm across, 17.4 mm deep, 39 mm apart; on the 2mag MIXdrive 6 MTP."""
    assert M.PLATE_SIZE[:2] == pytest.approx((0.1278, 0.0855)) and 2 * M.WELL_R == pytest.approx(0.0348)
    assert M.WELL_DEPTH == pytest.approx(0.0174) and M.PITCH == pytest.approx(0.039)
    assert M.STIRRER_SIZE == pytest.approx((0.140, 0.102, 0.026))
    xs = sorted({xy[0] for xy in list(M.WELL_XY.values()) + M.EMPTY_XY})
    ys = sorted({xy[1] for xy in list(M.WELL_XY.values()) + M.EMPTY_XY})
    assert np.diff(xs) == pytest.approx([M.PITCH] * 2) and np.diff(ys) == pytest.approx([M.PITCH])
    px, py = M.PLATE_XY
    for x, y in list(M.WELL_XY.values()) + M.EMPTY_XY:  # every well inside the plate, the plate on the stirrer
        assert abs(x - px) + M.WELL_R <= M.PLATE_SIZE[0] / 2 and abs(y - py) + M.WELL_R <= M.PLATE_SIZE[1] / 2
    assert M.PLATE_SIZE[0] <= M.STIRRER_SIZE[0] and M.PLATE_SIZE[1] <= M.STIRRER_SIZE[1]
    assert M.FLOOR_Z == pytest.approx(M.TOP + M.STIR_H + M.PLATE_SIZE[2] - M.WELL_DEPTH)


def test_the_holder_stands_clear_of_the_stirrer():
    sy = M.STIRRER_XY[1] + M.STIRRER_SIZE[1] / 2
    holder_near = M.HOLDER_XY[1] - M.HOLDER_INNER / 2 - M.WALL
    assert holder_near > sy + 0.01


def test_each_well_wall_is_round_enough():
    """Eight panels per well; each overlaps its neighbours, so the octagon is closed."""
    names = {o.name for o in M.OBJECTS}
    for well in M.ORDER:
        assert all(f"{well}_wall{k}" in names for k in range(8))
        assert f"{well}_liquid0" in names and f"{well}_liquid1" in names


# -- the text says what the code does ------------------------------------------------------------------------------
def test_the_scoring_and_setup_text_state_the_codes_numbers():
    import json

    spec = json.loads((HERE / "task.json").read_text())
    scoring, setup = spec["scoring"], spec["setup"]
    low, high = (v * M.PIPETTEFUL_ML for v in M.VOLUME_RANGE)
    assert f"{low:g} and {high:g} mL" in scoring
    assert "1 s" in scoring and "net of any squeezed back" in scoring
    assert "MIXdrive 6 MTP" in setup and "14 x 10.2 x 2.6 cm" in setup and "3.9 cm apart" in setup
    assert "Corning 3516" in setup and "34.8 mm" in setup
    for well in ("cup1", "cup2", "cup3", "stock"):
        x, y = M.WELL_XY[well]
        assert f"x {x * 100:g}" in setup or f"{x * 100:g}" in setup


# -- mixing: a draw before the stirrer has had MIX_S to mix a well counts as unmixed ------------------------------------
def test_drawing_from_well_1_straight_after_filling_it_breaks_the_chain():
    feed(transfer_steps("stock", "cup1", wait=False) + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3"))
    rest(PARKED)
    assert [e.vessel for e in M.unmixed_draws(M.LEDGER)] == ["cup1"] and "cup2" in M.spoiled(M.LEDGER)
    assert M.cups_in_order(M.LEDGER) == 1 and M.grade(M.LEDGER, PARKED) == pytest.approx(0.3)


def test_waiting_for_the_stirrer_scores_normally():
    feed(chain())
    assert not M.unmixed_draws(M.LEDGER)
    assert M.cups_in_order(M.LEDGER) == 3


def test_the_demo_waits_for_the_stirrer_before_each_draw():
    """From the end of each dispense to the end of the next draw from that well, in the scripted plan."""
    plan = list(M.demo()["left"])
    t, ends = 0.0, []
    for w in plan:
        t += w.duration
        ends.append((w.label, t))
    dispenses = [e for lbl, e in ends if lbl == "squeeze: dispense"]
    draws = [e for lbl, e in ends if lbl == "release: draw"]
    for done, drawn in zip(dispenses[:2], draws[1:3]):
        assert drawn - done >= M.MIX_S + 2.0




# -- a real bulb draws only the liquid above its tip -------------------------------------------------------------------
def test_a_shallow_draw_takes_only_the_liquid_above_the_tip_and_breaks_the_chain():
    """Drawing with the tip 6 mm up in the 7.2 mm-deep stock takes about a third of a pipetteful, then air."""
    full = M.SQUEEZE_TRAVEL
    shallow = (*M.tip_xy("stock", "dip"), M.FLOOR["stock"] + 0.006)
    steps = [(at("stock", "prime"), 0.0), (at("stock", "prime"), full), (shallow, full), (shallow, 0.0),
             (xyz("stock", M.TIP_TRAVEL), 0.0), (xyz("cup1", M.TIP_TRAVEL), 0.0), (at("cup1", "dispense"), 0.0),
             (at("cup1", "dispense"), full), (xyz("cup1", M.TIP_TRAVEL), full), (xyz("cup1", M.TIP_TRAVEL), 0.0)]
    feed(steps)
    drawn = [e.volume for e in M.LEDGER.events if e.kind == "aspirate"]
    expected = (2 * M.LIQUID_PER_ALIQUOT - 0.006) / M.LIQUID_PER_ALIQUOT
    assert drawn == [pytest.approx(expected)] and expected < 0.4
    assert not M.cup_right(M.LEDGER, "cup1")  # too little liquid, too weak


def test_the_demo_draw_height_takes_a_full_pipetteful():
    full = M.SQUEEZE_TRAVEL
    steps = []
    for src, dst in (("stock", "cup1"), ("cup1", "cup2"), ("cup2", "cup3")):
        steps += [(at(src, "prime"), 0.0), (at(src, "prime"), full), (at(src, "dip"), full), (at(src, "dip"), 0.0),
                  (xyz(src, M.TIP_TRAVEL), 0.0), (xyz(dst, M.TIP_TRAVEL), 0.0), (at(dst, "dispense"), full),
                  (xyz(dst, M.TIP_TRAVEL), full), (xyz(dst, M.TIP_TRAVEL), 0.0)]
        steps += [(xyz(dst, M.TIP_TRAVEL), 0.0)] * int(M.MIX_S * M.HZ)
    feed(steps)
    assert [e.volume for e in M.LEDGER.events if e.kind == "aspirate"] == [pytest.approx(1.0)] * 3
    assert M.cups_in_order(M.LEDGER) == 3


def squeeze_back(well):
    """Draw from ``well`` and squeeze it straight back in: a correction, or mixing by pipetting."""
    full = M.SQUEEZE_TRAVEL
    return [(at(well, "prime"), 0.0), (at(well, "prime"), full), (deep(well), full), (deep(well), 0.0),
            (at(well, "dispense"), 0.0), (at(well, "dispense"), full), (xyz(well, M.TIP_TRAVEL), full),
            (xyz(well, M.TIP_TRAVEL), 0.0)]


def test_an_early_draw_squeezed_back_and_mixing_by_pipetting_taint_nothing():
    """Review, Nov 14: one early draw used to spoil the rest of the trial even when it went straight back."""
    steps = transfer_steps("stock", "cup1", wait=False) + squeeze_back("cup1") + transfer_steps("cup1", "cup2")
    feed(steps + transfer_steps("cup2", "cup3"))
    rest(PARKED)
    assert not M.spoiled(M.LEDGER) and M.cups_in_order(M.LEDGER) == 3
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(1.0)


def test_splitting_one_stock_pipetteful_between_wells_1_and_2_breaks_the_chain():
    """Review, Nov 14: two partial stock draws, one into well 1 and one into well 2, then well 2 on to well 3, left every
    colour in range and scored 1.0 without the well 1 to well 2 transfer. Cup 2 must come from well 1."""
    full = M.SQUEEZE_TRAVEL

    def part(src, depth, dst):
        z = xyz(src, M.FLOOR[src] + depth)
        return [(at(src, "prime"), 0.0), (at(src, "prime"), full), (z, full), (z, 0.0), (xyz(src, M.TIP_TRAVEL), 0.0),
                (xyz(dst, M.TIP_TRAVEL), 0.0), (at(dst, "dispense"), 0.0), (at(dst, "dispense"), full),
                (xyz(dst, M.TIP_TRAVEL), full), (xyz(dst, M.TIP_TRAVEL), 0.0)] + [(xyz(dst, M.TIP_TRAVEL), 0.0)] * 120
    h = M.LIQUID_PER_ALIQUOT  # the tip at (level - wanted) takes just the wanted pipettefuls
    feed(part("stock", (2.0 - 0.6) * h, "cup1") + part("stock", (1.4 - 0.4) * h, "cup2") + part("cup2", (1.4 - 0.5) * h, "cup3"))
    assert [round(e.volume, 3) for e in M.LEDGER.events if e.kind == "aspirate"] == [0.6, 0.4, 0.5]
    rest(PARKED)
    assert M.stock_taken(M.LEDGER) <= 1.0 + 1e-6
    assert all(M.cup_right(M.LEDGER, c) for c in ("cup1", "cup2", "cup3"))  # every colour and volume in range...
    assert "cup2" in M.spoiled(M.LEDGER) and M.cups_in_order(M.LEDGER) == 1  # ...but well 2 didn't come from well 1
    assert M.grade(M.LEDGER, PARKED) <= 0.3


def test_every_well_must_come_from_the_one_before():
    feed(transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + transfer_steps("cup1", "cup3"))
    assert "cup3" in M.spoiled(M.LEDGER) and M.cups_in_order(M.LEDGER) <= 2


# -- review 3, Nov 14: a token squeeze-back doesn't mix; the colour settles over the stirrer's time -------------------
def test_a_token_squeeze_back_does_not_cancel_the_wait():
    full = M.SQUEEZE_TRAVEL
    M._reset({})
    for tip, sq in transfer_steps("stock", "cup1", wait=False):
        M._track({"pipette": pipette_at(tip, sq)}, grips=[])
    just_under = xyz("cup1", M.LEDGER.vessels["cup1"].surface_z - 0.00005)
    for tip, sq in [(at("cup1", "prime"), full), (just_under, full), (just_under, 0.0), (at("cup1", "dispense"), 0.0),
                    (at("cup1", "dispense"), full), (xyz("cup1", M.TIP_TRAVEL), full), (xyz("cup1", M.TIP_TRAVEL), 0.0)]:
        M._track({"pipette": pipette_at(tip, sq)}, grips=[])
    assert M.LEDGER.events[-1].kind == "dispense" and M.transfers(M.LEDGER)[-1][2][0].volume < 0.05
    for tip, sq in transfer_steps("cup1", "cup2", wait=False):
        M._track({"pipette": pipette_at(tip, sq)}, grips=[])
    assert "cup2" in M.spoiled(M.LEDGER)


def test_a_real_squeeze_back_mixes_the_well_at_once():
    M._reset({})
    for tip, sq in transfer_steps("stock", "cup1", wait=False) + squeeze_back("cup1"):
        M._track({"pipette": pipette_at(tip, sq)}, grips=[])
    assert "cup1" not in M.SETTLING
    for tip, sq in transfer_steps("cup1", "cup2", wait=False):
        M._track({"pipette": pipette_at(tip, sq)}, grips=[])
    assert "cup2" not in M.spoiled(M.LEDGER)


def test_a_wells_colour_settles_over_the_stirrers_time():
    M._reset({})
    steps = transfer_steps("stock", "cup1", wait=False)
    for tip, sq in steps:
        M._track({"pipette": pipette_at(tip, sq)}, grips=[])
    well = M.LEDGER.vessels["cup1"]
    mid = M.shown_fraction("cup1", well)
    assert 0.0 <= mid < 0.5 * 0.9  # still mostly the water it was
    for _ in range(int(M.MIX_S * M.HZ)):
        M._track({"pipette": pipette_at(xyz("cup1", M.TIP_TRAVEL), 0.0)}, grips=[])
    assert M.shown_fraction("cup1", well) == pytest.approx(0.5)


def test_a_clean_dip_before_the_first_draw_is_not_going_back():
    """Review 3, Nov 14: a clean, empty pipette dipped into well 3's water first carried no dye back, but scored 0."""
    full = M.SQUEEZE_TRAVEL
    first = [(at("cup3", "prime"), 0.0), (deep("cup3"), 0.0), (at("cup3", "prime"), 0.0)]
    feed(first + chain())
    rest(PARKED)
    assert not M.went_back(M.LEDGER) and M.grade(M.LEDGER, PARKED) == pytest.approx(1.0)


def test_going_back_after_the_work_has_started_still_scores_zero():
    feed(transfer_steps("stock", "cup1") + [(at("cup1", "prime"), 0.0), (deep("cup1"), 0.0), (at("cup1", "prime"), 0.0)]
         + [(at("stock", "prime"), 0.0), (deep("stock"), 0.0), (at("stock", "prime"), 0.0)]
         + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3"))
    rest(PARKED)
    assert M.went_back(M.LEDGER) and M.grade(M.LEDGER, PARKED) == 0.0


def test_water_drawn_and_returned_before_the_stock_is_not_going_back():
    """Review 4, Nov 14: drawing well 3's water and squeezing it back before touching the stock carried no dye, but scored 0."""
    feed(squeeze_back("cup3") + chain())
    rest(PARKED)
    assert not M.went_back(M.LEDGER) and M.grade(M.LEDGER, PARKED) == pytest.approx(1.0)


def test_a_squeeze_back_under_three_quarters_does_not_mix():
    assert M.MIX_BACK == pytest.approx(0.75)
    import json
    assert "three-quarters of a pipetteful" in json.loads((HERE / "task.json").read_text())["scoring"]


def test_squeezing_into_a_spare_well_is_a_discard_not_a_spill():
    """Review 5, Nov 15: the empty wells weren't vessels, so a discard into one counted as a spill (capped at 0.5)."""
    full = M.SQUEEZE_TRAVEL
    spare = (*M.WELL_XY["empty1"], M.FLOOR_Z + M.DISPENSE_ABOVE)
    discard = [(at("cup3", "prime"), 0.0), (at("cup3", "prime"), full), (deep("cup3"), full), (deep("cup3"), 0.0),
               (xyz("cup3", M.TIP_TRAVEL), 0.0), (spare, 0.0), (spare, full), (spare, 0.0)]
    feed(chain() + discard)
    rest(PARKED)
    assert M.LEDGER.spilled == 0.0 and M.cups_in_order(M.LEDGER) == 3
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(1.0)


def test_a_spare_well_cant_stand_in_for_the_stock():
    """Stock into a spare well, then from there into well 1: well 1 didn't come from the stock."""
    full = M.SQUEEZE_TRAVEL
    spare = (*M.WELL_XY["empty0"], M.FLOOR_Z + M.DISPENSE_ABOVE)
    via = [(at("stock", "prime"), 0.0), (at("stock", "prime"), full), (deep("stock"), full), (deep("stock"), 0.0),
           (xyz("stock", M.TIP_TRAVEL), 0.0), (spare, 0.0), (spare, full), (spare, 0.0)]
    via += [((*M.WELL_XY["empty0"], M.FLOOR_Z + 0.012), full), ((*M.WELL_XY["empty0"], M.FLOOR_Z), full),
            ((*M.WELL_XY["empty0"], M.FLOOR_Z), 0.0), (xyz("cup1", M.TIP_TRAVEL), 0.0), (at("cup1", "dispense"), full),
            (xyz("cup1", M.TIP_TRAVEL), 0.0)] + [(xyz("cup1", M.TIP_TRAVEL), 0.0)] * 120
    feed(via)
    assert "cup1" in M.spoiled(M.LEDGER) and M.cups_in_order(M.LEDGER) == 0


def test_liquid_landing_in_an_empty_spare_well_shows_its_own_colour_at_once():
    """Review 6, Nov 15: stock squeezed into an empty spare well showed the water colour while 'settling'."""
    full = M.SQUEEZE_TRAVEL
    spare = (*M.WELL_XY["empty0"], M.FLOOR_Z + M.DISPENSE_ABOVE)
    feed([(at("stock", "prime"), 0.0), (at("stock", "prime"), full), (deep("stock"), full), (deep("stock"), 0.0),
          (xyz("stock", M.TIP_TRAVEL), 0.0), (spare, 0.0), (spare, full)])
    assert "empty0" not in M.SETTLING
    assert M.shown_fraction("empty0", M.LEDGER.vessels["empty0"]) == pytest.approx(1.0)


def test_the_stock_limit_leaves_an_honest_margin():
    """Review 9, Nov 15: one correct draw leaves exactly 3.4 mL, so a hard 'one pipetteful' line put honest bench trials
    on the zero line. The limit is one and a quarter pipettefuls, the margin the wells get."""
    assert M.STOCK_LIMIT == pytest.approx(1.25)
    import json
    scoring = json.loads((HERE / "task.json").read_text())["scoring"]
    assert "2.55 mL left" in scoring and f"{M.STOCK_LIMIT * M.PIPETTEFUL_ML:g} mL" in scoring


def test_watering_the_stock_from_another_well_scores_zero():
    """Review 10, Nov 15: well 1's water squeezed into the stock before the first stock draw counted as 'returned',
    stretching a watered-down stock to two transfers, and scored 1.0."""
    full = M.SQUEEZE_TRAVEL
    into_stock = (*M.WELL_XY["stock"], M.FLOOR_Z + M.DISPENSE_ABOVE)
    water = [(at("cup1", "prime"), 0.0), (at("cup1", "prime"), full), (deep("cup1"), full), (deep("cup1"), 0.0),
             (xyz("cup1", M.TIP_TRAVEL), 0.0), (into_stock, 0.0), (into_stock, full), (into_stock, 0.0)]
    feed(water + chain())
    rest(PARKED)
    assert M.stock_spoiled(M.LEDGER) and M.grade(M.LEDGER, PARKED) == 0.0


def test_stock_squeezed_back_into_the_stock_is_still_a_correction():
    feed(transfer_steps("stock", "stock") + chain())
    rest(PARKED)
    assert not M.stock_spoiled(M.LEDGER) and M.grade(M.LEDGER, PARKED) == pytest.approx(1.0)


def test_liquid_squeezed_from_high_over_a_well_still_lands_in_it():
    """Review 11, Nov 15: a squeeze 4 cm over a well's rim counted as a spill, so liquid carried back that way vanished
    from the checks and scored 0.5. It lands in the well from any height; a high squeeze still caps the score."""
    full = M.SQUEEZE_TRAVEL
    high_over_1 = (*M.WELL_XY["cup1"], M.RIM_Z + 0.04)
    back = [(at("cup2", "prime"), 0.0), (at("cup2", "prime"), full), (deep("cup2"), full), (deep("cup2"), 0.0),
            (xyz("cup2", M.TIP_TRAVEL + 0.04), 0.0), (high_over_1, 0.0), (high_over_1, full)]
    feed(transfer_steps("stock", "cup1") + transfer_steps("cup1", "cup2") + back + transfer_steps("cup2", "cup3"))
    rest(PARKED)
    assert M.LEDGER.high > 0 and M.LEDGER.spilled == 0.0
    assert M.went_back(M.LEDGER) and M.grade(M.LEDGER, PARKED) == 0.0


def test_an_honest_high_squeeze_counts_but_is_capped():
    full = M.SQUEEZE_TRAVEL
    steps = transfer_steps("stock", "cup1")
    high_over_1 = (*M.WELL_XY["cup1"], M.RIM_Z + 0.04)
    steps = [s if s[0] != at("cup1", "dispense") else (high_over_1, s[1]) for s in steps]
    feed(steps + transfer_steps("cup1", "cup2") + transfer_steps("cup2", "cup3"))
    rest(PARKED)
    assert M.LEDGER.high > 0 and M.cups_in_order(M.LEDGER) == 3
    assert M.grade(M.LEDGER, PARKED) == pytest.approx(M.SPILL_CAP)
