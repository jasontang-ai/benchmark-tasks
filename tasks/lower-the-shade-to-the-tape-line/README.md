# Lower the spring roller shade to the tape line

A 61 cm Draper Cordless FlexShade spring roller shade hangs in front of a small cased window, set in a plywood wall
section screwed to the table. It starts caught part way down. Blue painter's tape runs across both side casings two
catches lower. The robot must leave the shade caught with its hem bar at the tape. The roller locks only at each half
turn (6.2 cm of shade) and only when it turns slowly, so the robot must hold the bar against the spring's 5-6 N, pull
it below the line by less than one catch spacing, ease it up slowly until the lock takes it at the line's own catch,
and only then let go. Let go while it moves, and the shade runs up to the top. This is the knack anyone who has used
such a shade knows, and what the maker's instructions say: pull down, hesitate, then let it retract very slowly until
it locks.

## Files

| File | What |
|---|---|
| `lower_the_shade_to_the_tape_line.py` | The task file: scoring, the lock rule, the props, `demo()` |
| `shade_demo.py` | The scripted demo's waypoints |
| `benchkit.py` | Props built from boxes with sliders (the hem bar, the hidden stops), spring targets and per-step hooks |
| `run_record.py` | The simulator machine's trace (`~/runs`) |
| `tiltgrip.py` | Tilted and sideways grips for the demo's waypoints |
| `test_lower_the_shade_to_the_tape_line.py` | Offline tests: no GPU needed |

No third-party assets: every prop is built from boxes in the task file.

## Run a trial

In the simulator's environment (the Robocurve Brev launchable, or yam-isaac-lab-arena's `setup.sh`), with a Dream API
key configured, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=lower_the_shade_to_the_tape_line.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lower the shade so its bottom bar stays at the blue tape line." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/shade-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=lower_the_shade_to_the_tape_line.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Lower the shade so its bottom bar stays at the blue tape line." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/shade-demo

for log in ~/yam-eval/shade-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score`. The episode stops at once on 1.0 (which already includes 3 s at rest with the grippers
clear); otherwise it runs to the step limit and is scored on its final state.

Offline tests (no GPU), with `yam_arena` importable: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.

## Runtime and seeds

- The scripted demo plans 27.9 s (558 steps at 20 Hz). In a MuJoCo check of the scene the lock caught the bar at the
  line at 15.4 s and the score reached 1.0 at step 492 (24.6 s); the same at 1.0 with the spring 20% weaker or
  stronger, its damping halved or doubled, the bar's friction 20% lower or higher, and the catch speed 25% lower or
  higher.
- Dream-1 runs up to the 2,400-step limit (2 minutes of simulated time) unless it reaches 1.0 first.
- One fixed layout: `-E object_jitter=0`, so the seed does not change the start.

## Scoring

`task.json` gives the rule in full. The bar's height is the mean of the two rule readings at its bottom. "Still and
clear" means the reading changed less than 2 mm over the last 3 s, with no gripper within 3 cm of the bar or fabric.

| Outcome | Score |
|---|---|
| The wall section moved more than 5 mm, or the shade came out of a bracket, at any time | 0 |
| Never pulled 5 cm below its start | 0 |
| At the end, back at its start catch or run up to the top (bottom less than 3 cm below its start) | 0 |
| Pulled 5 cm down and at least 3 cm below its start at the end, but held, touched or moving | 0.25 |
| Still and clear, caught at another catch below the start (bottom more than 3 cm from the tape) | 0.5 |
| Still and clear, bottom within 3 cm of the tape's upper edge | 1.0 |

The catches are 6.2 cm apart, so the 3 cm band holds only the line's catch. Only the end counts: a shade that ran up
can be pulled down again within the run limit.

## The real shade

- **Product.** Draper Cordless FlexShade, regular roll, outside mount, 24 x 36 in (Draper's smallest listed width), with
  the small round exposed hem bar. Draper's submittal (form FlexShade_Cordless_Sub21) specifies a spring roller "with
  positive locking mechanism permitting shade to be stopped at each half turn of roller", an extruded aluminium roller
  tube of 1.5 in (38 mm) outer diameter, metal brackets, and spring tension adjustable at the brackets. The hem bar is
  listed as 3/4 in (19 mm) in the selection list and 5/8 in (16 mm) in the specification text: the delivered bar is
  measured and recorded here before trials.
- **Speed-dependent lock.** Draper's operating instructions for spring-operated shades (form ManualShadeOps_Inst19):
  to lower, pull down, "hesitate, then allow screen to retract very slowly until it locks in place"; to raise, pull it
  down and, while holding, let it retract slowly. A spring roller holds with a ratchet and pawls at one end of the roller
  (Skyco describes its spring roller locks as inertia or gravity driven): the pawls drop in only when the roller turns
  slowly, so a fast-turning roller runs past them. Draper publishes neither the catch speed nor the catch spacing, so both
  are measured before the first trial (reset notes): the spacing on the rules (half the roller's circumference with
  its wound fabric, about 6.2 cm), and the catch speed by raising the bar by hand through a catch at a steady pace,
  filmed at 60 fps against a rule. The sim uses 20 mm/s. The demo eases the bar at under 10 mm/s, so a real lock that
  catches at 10 mm/s or faster takes it the same way.
- **Spring.** The pull at the bar is read with a 10 N spring scale: 5.0 N at the start catch and 6.0 N at the line's.

## Fidelity

- **What is real and what is approximated.** The shade is an off-the-shelf catalogue product at its real size (61 x 91
  cm, a 38 mm roller on the maker's brackets), and the robot works it the way people do, by its hem bar alone: no
  handle, tab, lever or other added part. What the sim approximates is the inside of the roller (the spring and the
  half-turn lock) and the fabric's flex, below.
- **Fabric and roller.** The fabric is rigid: a 2 mm plate riding on the bar and a fixed plate below the roller. The
  bar slides straight up and down, 6.2 cm per half turn of the roller (half the circumference of the 38 mm tube with
  its wound fabric). The roller is drawn as a fixed square block (tube plus wound fabric) that does not turn; only the
  bar's travel stands for its turning. The spring is linear on that travel (5.0 N at the start catch to 6.0 N at the
  line's, the weight of the bar and fabric taken off), and the roller's inertia is added to the bar (0.5 kg moving).
- **The lock** is a rule standing for the roller's pawls. Each control step, a bar rising slower than 20 mm/s within
  3 mm under a catch engages it: a hidden stop slides over the bar's left end, and the bar stops at the catch. A bar
  rising faster runs past. Pulling a caught bar 5 mm down frees it. Draper documents a half-turn lock that catches
  when the shade retracts very slowly but publishes no catch speed: 20 mm/s is the sim's value, and the real shade's
  catch speed is measured before trials (reset notes). The robot does the real work: hold the spring, pull to the
  right depth, ease up slowly, let go only once caught.
- **Hem bar and wall.** The round bar collides (and is drawn) as a 19 mm diamond, two line contacts on the pads as a
  round bar gives. Draper lists the bar as 19 mm and as 16 mm across; the delivered bar is measured at reset, and the
  start reading, tape and rules are set from the measured bar and catches. The wall section is a free 30 kg body
  standing on four brace feet, in place of the braces screwed to the table.

## Design notes

- **The arm's give.** Under the spring's 6 N the YAM's hand sits about 31 mm above where it is sent (the arm is a
  soft spring). That is close to half a catch spacing, so a policy must watch where the bar really is, not where it
  sent the hand.
- **Unintended strategies, all tested:** hold the bar at the line to the end (0.25); let go at the line (runs up: 0);
  pull it to the table and let go (0); ease it up past the start (caught above its start or run up: 0); leave it caught
  one or more catches too low (0.5); catch it at the line, then pull it down again and hold (0.25); rest a gripper on
  the caught bar (0.25); never touch it (0). Nothing in the scene can wedge the bar: the nearest thing is the casing,
  6 cm behind it.
- **Reach.** The right hand turns its fingers along the bar and slides on from beyond the bar's right end, 30 mm in,
  so its palm stays clear of the fabric; the left arm parks out of the shade's sweep first. Every waypoint solves
  within 0.1 mm, and the arms stay more than 20 mm apart.
- **Safety.** A fabric shade and a 19 mm aluminium bar on a wall section screwed to the table; the spring pulls 5-6 N
  and a shade that runs up stops at its roller. No sharp, hot or powered part. The operator stays outside the arms'
  reach and reads the rules from the video.
