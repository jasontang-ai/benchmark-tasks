# Zero a height gauge on the plate, then measure a block

**Instruction:** "Zero the height gauge on the surface plate, then measure the height of the steel block with it and leave the reading held on its display."

A digital height gauge stands on a granite surface plate beside a hardened 1-2-3 block. The robot must zero it with
the scriber resting on the plate, slide the gauge by its base until the scriber is over the block, set the scriber down
on the block and press HOLD. A press counts only with the scriber resting on its surface (within 0.02 mm: touching): zeroed in the air, or held
while hovering, the display holds a wrong number.

Sources: INSIZE 1150 digital height gage manual (https://m.insize.com/upfiles/sms/MN-1150-E%20V0.pdf): "ON/OFF
ZERO---short press ... to set zero", "HOLD---short press to keep the reading", zero with the measuring surface and the
base on the plate, "When move the height gage, please hold the base". O*NET 51-9061
(https://www.onetonline.org/link/summary/51-9061.00).

## Run and score a trial

From this folder, in the simulator's environment:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=zero_a_height_gauge_on_the_plate_then_fine_feed_onto_the_step.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Zero the height gauge on the surface plate, then measure the height of the steel block with it and leave the reading held on its display." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/gauge-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=zero_a_height_gauge_on_the_plate_then_fine_feed_onto_the_step.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Zero the height gauge on the surface plate, then measure the height of the steel block with it and leave the reading held on its display." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/gauge-demo
```

One fixed layout (`object_jitter=0`; the seed changes nothing). Runtime per trial: up to 2,400 steps (2 minutes at
20 Hz); the scripted demo reaches full score in about 55 s. The run prints `task_score`, scored on the final state.
Demo modes (`KIT_DEMO`): `demo`,
`airzero` (ZERO pressed with the scriber parked in the air), `hover` (HOLD pressed with the scriber hovering over the
block), `none`. Tests: `PYTHONPATH=<yam-isaac-lab-arena> PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.

## Scoring

| Score | Condition |
|---|---|
| 0 | never zeroed on the plate; or at any time the gauge tips more than 10 deg, leaves the plate or falls, or the block moves more than 5 mm |
| 0.25 | at least one ZERO press with the scriber resting on the plate (a later ZERO in the air voids the readings after it, not this rung) |
| 0.5 | a reading held with the scriber resting on the block, within 0.50 mm of its height |
| 0.75 | within 0.10 mm |
| 1.0 | 0.75, the base flat (no corner raised over 0.5 mm), the scriber still on the block, no gripper within 30 mm, still 1 s |

Real trial, after the robot stops: a photo of the held display; a 0.02 mm feeler that will not pass under the scriber
on the block; a 0.5 mm feeler at the base corners; then release HOLD, slide the gauge clear and lower the scriber onto
the plate by hand: within 0.02 mm of 0.00 shows the zero was set on the plate.
The display's 8 mm digits are read from the full-resolution video or a photo; the policy does not need to read them.

## Parts (real bench)

INSIZE 1150-300 digital height gage (about $660 at Zoro; a lower-cost substitute: iGaging 35-630, 0-300 mm, about $200 at Penn Tool Co, https://www.penntoolco.com/igaging-12-digital-height-gauge-w-absolute-origin-35-630/); Standridge 12 x 12 x 3 in
grade A black granite surface plate (about $213, Judge Tool); a hardened 1-2-3 block (any machinist's set-up block,
25.4 x 50.8 x 76.2 mm).

## Simulation and the real case

Rigid bodies at real size. The slider rides the beam against 3 N of static friction and holds where it is put (idle creep on the GPU: 0.002 mm/s); the
fine-adjustment nut and lock are drawn, not worked. The display is a rule on the real geometry at each press. Measuring
force and beam flex are not simulated.

## The scripted demo

The right hand works horizontally: it pinches the slider's sides and lowers it on a servo until the scriber rests on
the plate, then presses ZERO with one fingertip (servoed on the button's travel, so the press stays light: about 8 N
at the button's height would tip the gauge). The left hand, tilted 30 deg outward so its wrist clears the slider,
grips the base's end and slides the gauge 120 mm on a pose servo until the scriber is over the block. The right hand
lowers the scriber onto the block on the same servo and presses HOLD.

## Safety

Real trials are safe: no heat, no liquids and no powered parts beyond the gauge's button cell. The forces are light (2 N
buttons, a 3 N slider, a 2.5 kg gauge slid on a fixed plate). The carbide scriber's tip is blunt-edged; keep hands clear
of the scriber while the robot moves, and stop the run if the gauge tips toward the plate's edge.

## Licences

All geometry is built from boxes in the task file; no third-party meshes or scenes.
