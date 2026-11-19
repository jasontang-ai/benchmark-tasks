# Open a child-locked cupboard, key held over the latch

A low cupboard's door is shut by a magnetic child lock screwed inside it: nothing shows outside but a sticker where
the lock is. The robot must open the door wide and put the magnetic key back on its spot. The maker's method is the
knack: one hand keeps the key flat on the door over the hidden latch, riding with the door, while the other pulls the
knob until the latch's hook has cleared its catch (6 mm of travel at the lock); a key set on and taken off before the
pull lets the sprung latch drop back, and the door stops after 1 mm.

Instruction: "Open the cupboard door wide and put the key back on its spot."

## Run and score a trial

    # scripted demo (the task file's demo) on the Isaac Lab arena
    python -m yam_arena.run --task tasks/open-a-magnetic-child-locked-cupboard --policy yam_scripted --steps 2400
    # tests (kit tests 1-7)
    cd tasks/open-a-magnetic-child-locked-cupboard && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

- Runtime per trial: up to 2,400 steps (2 minutes of sim time at 20 Hz); on one GPU a full-length trial took about 11 minutes wall clock, the demo (750 steps) about 4.
- Seeds: one fixed layout, no randomisation (seed 0, object jitter 0).
- The score (`task_score`) is read at the end of the run by `score()` in `open_a_magnetic_child_locked_cupboard.py` (top of the
  file: the lock rule, then the scorer, then the demo).

## Scoring (from task.json)

0 anything else, or the cabinet moved more than 5 mm at any time; 0.25 door not past the catch but the key flat over
the lock at the end; 0.5 door past the catch (3 deg) but under 45 deg; 0.75 door at 45 deg or more but the key not
flat in its circle, a gripper within 40 mm or something moving; 1.0 door at 45 deg or more, key flat in its circle,
hands 40 mm clear, still for 1 s.

## Real trials

A real low cabinet (330 x 300 x 340 mm) screwed to the table, a Safety 1st magnetic lock and key fitted by the
maker's guide, a protractor taped at the hinge and a 60 mm circle drawn on the table. Before trials, set the key on
the door over the lock and let go: it must slide off (the task assumes the key does not cling by itself). Nothing in
the task is sharp, hot or under load; the door is light and swings freely, so trials are safe to run.

## Files and assets

- `open_a_magnetic_child_locked_cupboard.py`: scene, lock rule, scorer, scripted demo.
- `ourkit/`: the shared mechanism kit, copied unchanged.
- `test_open_a_magnetic_child_locked_cupboard.py`: the kit's tests.
- No third-party meshes or scans: every part is built from boxes and cylinders.
