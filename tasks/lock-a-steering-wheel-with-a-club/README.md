# Hook a Club across a steering wheel and pull it snug

Instruction: *Lock the steering wheel with the Club across it, its handle pointing up, and let go of it.*

A Winner International The Club 1000 steering-wheel lock (1.6 kg, 558 mm closed) stands on its two hooks in front of
a 14 in steering wheel on a plywood column stand, its face leaning back 25° like a car's, with a car's dash,
instrument hood and windshield behind and above it. The Club is two telescoping steel pieces with one hook each; the
hooks open toward the bar's ends, so they fit inside the rim and bite outward, and the inner bar slides out on a
self-locking ratchet that never slides back. The maker's guide says to put one hook on first and pull the other until
it fits snugly (the Club 900 guide on etrailer,
https://www.etrailer.com/Vehicle-Locks/Winner International/WI900YELLOW.html); either hook may go first. Because the
hooks open outward, the lower hook can carry the Club on the bottom of the grip, but the upper one can't: hook the top
first (as the demo does) and one hand has to hold the whole Club up with that hook seated while the other pulls the
inner bar down along the bar's own line; hook the bottom first and the Club rests on it while the outer piece is
pushed up the bar's line. Either way the knack is the same: draw the halves apart along the bar's line, the first
hook kept seated, right through to the tooth at the grip. Stop more than 2.4 mm before the second hook meets the grip
and the ratchet holds the bar a tooth short and it rattles loose. The Club's construction (two pieces, the smaller sliding inside, one hook
each) is as Wikipedia describes it (https://en.wikipedia.org/wiki/The_Club_(automotive)).

## Files

| File | What |
|---|---|
| `lock_a_steering_wheel_with_a_club.py` | The task file: props, the ratchet rule, the scorer, `demo()` |
| `ourkit/` | The shared mechanism kit (boxes, sliders, rules, closed-loop demo moves), copied unchanged |
| `test_lock_a_steering_wheel_with_a_club.py` | Offline tests (no GPU) |

No third-party assets: every prop is built from boxes in the task file.

## Run a trial

From this folder, in the simulator's environment:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=lock_a_steering_wheel_with_a_club.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Lock the steering wheel with the Club across it, its handle pointing up, and let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir yam-eval/club-dream

inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=lock_a_steering_wheel_with_a_club.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Lock the steering wheel with the Club across it, its handle pointing up, and let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir yam-eval/club-demo
```

Offline tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` with `yam_arena` importable.

## Runtime and seeds

- Runs in the simulator (Isaac Lab Arena, the commands above): the scripted demo scored 1.0, the trial ending at
  step 760 (38.0 s) once the Club was locked, released and still; the inner bar clicked through 12 teeth during the
  pull (57.6 mm out, 2.4 mm of play). Dream-1 scored 0.0: it ran the full 2,400 steps and tipped the Club over on
  the table but never lifted it.
- The scripted demo plans 44.1 s (882 steps at 20 Hz) and stops early at 1.0. In a MuJoCo check of the scene it
  reached 1.0 at step 752 (37.6 s).
- Policies run up to the 2,400-step limit (2 minutes) unless they reach 1.0 first.
- One fixed layout: `-E object_jitter=0`; the seed does not change the start.

## Scoring

| Score | End state (released: grippers 30 mm clear; still: under 1 mm over 1 s) |
|---|---|
| 0 | the wheel stand moved more than 10 mm at any time; or the Club held, dropped, on the table, untouched |
| 0.25 | one hook on the grip, released and still |
| 0.5 | both hooks on, released and still, but loose: play at the upper hook over 4.8 mm |
| 0.75 | both hooks on and snug (play 4.8 mm or less), released and still, handle more than 30° from 12 o'clock |
| 1.0 | both hooks on, snug, handle within 30° of 12 o'clock, released and still |

Hook on: the grip passes through the hook's U (lip behind the grip, bar in front, the stem within 10 mm of the grip's
inner edge). Play: the 304 mm inner diameter less the hooks' inside spacing, which is what a steel rule reads at the
upper hook with the Club resting on its lower hook. Pulled until the lower hook meets the grip, this Club rests on the
tooth that leaves 2.4 mm; a tooth short leaves 7.2 mm, so the 4.8 mm line has 2.4 mm of margin either side.

### Measuring it in a real trial
1. Watch the video for release and stillness over the last second.
2. For each hook, pull that end 10 N toward the driver with a spring scale: on, it moves less than 5 mm.
3. Press the Club down onto its lower hook and read the gap at the upper hook with a steel rule.
4. Read the handle's angle from 12 o'clock with a protractor laid on the wheel's face.

## Layout (cm from the arm bases' midpoint, x forward, y left, heights above the table)

| Object | Type and material | Size, mass | Pose |
|---|---|---|---|
| Wheel stand | 18 mm birch plywood: base board, post, 100 mm box column, steel hub stub with a Grant 3196 installation kit (3-bolt adapter) | board 41 x 60 cm; 30 kg in sim with the dash | base x 45-86, clamped |
| Dash | 18 mm plywood boxes, dark grey | 60 cm across; top 4 cm above the top of the wheel | upper face x 52 (130 mm behind the grip at 12 o'clock) from 25 cm up; lower face x 56; back to x 85 |
| Instrument hood | plywood, black | 36 x 15 cm, top 8 cm above the top of the wheel | x 52-67, on the dash |
| Windshield | 6 mm clear acrylic | 80 x 50 cm | lower edge on the dash at x 82, 40 cm up (445 mm from the wheel's top), rising to x 42, 70 cm up at 37° |
| Steering wheel | Grant Elite GT 742, vinyl grip, aluminium spokes | 356 mm across, grip 26 mm, dish 3.75 in | centre (30, 0), 20 cm up, face 25° back |
| The Club 1000 | heat-treated steel, red vinyl coat | 558 mm closed, 1.6 kg (1.2 + 0.4) | on its hooks, middle at (15, 8), handle left, turned 30° |

## The Club's numbers (planned; to record on the bought unit)
Tubes 25 mm (outer) and 19 mm (inner); inner piece 0.4 kg; handle 30 cm past the upper hook (558 mm overall
closed, the retailer's figure); hooks 15 mm rod, 38 mm from bar to lip, 22 mm lip; 244 mm inside the hooks closed,
415 mm fully open (the maker's maximum); ratchet pitch 4.8 mm (3/16 in, the Club 900's step); extension drag 5 N
(20 N spring scale). Measure each with a ruler, a kitchen scale and the spring scale on receipt, then set the scene's
numbers and the snug line (one pitch) from them.

## Simulation and the real bench
- Rigid bodies. The grip is 40 straight pieces of square section with flat faces at 12 and 6 o'clock; the tubes are
  square. The hooks hold the grip by contact alone.
- The ratchet is one rule: after every control step, the inner bar's slide-back limit is set at the last 4.8 mm tooth
  it has passed. It slides out against 5 N of joint friction (static = sliding), which holds it against its own weight
  along the bar (3.6 N). Nothing else is moved or latched by code.
- The wheel is bolted fast to the stand, as a parked car's steering lock holds it, so there is no bearing or drag to
  set. The stand is a 30 kg free body standing for its two C-clamps.
- The car around the wheel is part of the stand's body, a simplified stand-in: a dash, an instrument hood and the
  lower half of a windshield, as flat panels at a car's distances. The hood's face is 130 mm behind the grip at
  12 o'clock (Oregon's vehicle equipment rule, OAR 735-124-0140, asks at least 2 in between the steering wheel and
  the instrument panel; set this far back, the bar's swing in clears it by about 5 cm); the
  windshield's lower edge is 445 mm from the top of the wheel (17.5 in, a tape measurement in a car); its 37° rake and
  the dash and hood heights are estimates, to measure on the car the bench copies. The maker says the Club must not
  rest against the windshield or other vehicle surfaces (https://winner-intl.com/?p=208): fitted with its handle up it
  clears the glass by about 4 cm and the hood by about 11 cm, so they bound the swing in, not the fit. No score reads
  a touch on them; knocking the stand more than 10 mm is the latched 0.
- The demo reads the simulator's object poses (allowed for scripted demos): both hands carry the Club and turn it to
  the wheel's face, the turn bowed to the left so the swinging handle passes in under the hood, clear of the dash, and
  the right hand never nearer the robot than x 10 cm, correct its pose twice, put the hooks through the wheel inside the grip and raise it until the
  upper hook's stem touches the grip at 12 o'clock. The left hand then servoes on that hook's live gap and depth while
  the right servoes the inner bar down the bar's line, keeping the lower hook mid-throat, until it is out to the grip;
  the arm's sag under the 1.6 kg load (tens of millimetres) is taken out by these loops, not by fixed offsets.

## Failure variants (MuJoCo check, `KIT_DEMO=<mode>`)

| Mode | What it does | Score |
|---|---|---|
| demo | as above | 1.0 |
| none | both arms stay home | 0 |
| slam | the right hand stops pulling 12 mm short of the grip (the ratchet locks it short) | 0.5 |
| wrong_order | the left hand lets go before the pull: the upper hook drops off the grip | 0.25 |

Physcheck ran with `--rigid-friction` (the 5 N drag held as PhysX holds joint friction). The arms' sag under the
Club is large in MuJoCo (the left hand ends up to about 85 mm below its command while it carries the Club); the GPU
run decides the real figure, and the demo's servos take out whatever it is.

## Safety
A 1.6 kg vinyl-coated bar on a clamped wooden stand at table height; no sharp edges, heat or stored energy beyond the
5 N ratchet drag. Run behind the cell's guard with the e-stop in reach.
