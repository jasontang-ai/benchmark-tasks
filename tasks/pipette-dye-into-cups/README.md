# Pipette dye into three wells of water

A six-well plate on a running magnetic stirrer: dark blue stock in one well, and wells 1, 2 and 3 with water in the
next row. The robot takes a squeeze-bulb transfer pipette from its holder and moves one pipetteful at a time from
well to well (stock to well 1, well 1 to well 2, well 2 to well 3). The stirrer mixes each well as liquid lands in it,
so the wells end at a half, a quarter and an eighth of the stock's strength. Lab staff call this a serial dilution.
Then it stands the pipette back in its holder, empty, and lets go.

The skill is real liquid-handling technique on a single arm: squeeze before dipping, release low in the liquid to
draw, lift, squeeze over the next well to dispense, and give the stirrer time before drawing from a well again. The
order is the protocol: each well is made from the one before it.

## Files

| File | What |
|---|---|
| `pipette_dye_into_cups.py` | The yam_arena task file: objects, `success()`, `score()`, `distance()` and the scripted `demo()` |
| `ledger.py` | The liquid ledger: volumes and contents of each well, moved by squeezes (pure Python) |
| `benchkit.py` | Tools built from several boxes with spring-loaded sliders (the bulb's sides), and per-step hooks |
| `test_ledger.py`, `test_pipette_dye_into_cups.py` | Offline tests: no GPU needed |

`benchkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the simulator's environment (the Robocurve Brev launchable, or yam-isaac-lab-arena's `setup.sh`), with a Dream API
key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=pipette_dye_into_cups.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use the transfer pipette to dilute the dark stock by half in each well. The stock is in the right-hand well nearest you, and wells 1, 2 and 3 run away from you along the left-hand row, each holding water. Move one pipetteful from the stock into well 1, then one from well 1 into well 2, then one from well 2 into well 3. Then stand the empty pipette back in its holder and let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dye-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=pipette_dye_into_cups.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Use the transfer pipette to dilute the dark stock by half in each well. The stock is in the right-hand well nearest you, and wells 1, 2 and 3 run away from you along the left-hand row, each holding water. Move one pipetteful from the stock into well 1, then one from well 1 into well 2, then one from well 2 into well 3. Then stand the empty pipette back in its holder and let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dye-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/dye-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. A trial ends at 1.0 once all three wells are right and the pipette has stood
still in its holder for 1 s; otherwise it scores the last step.

Offline tests (no GPU), with `yam_arena` importable:

```bash
python -m pytest -q test_ledger.py test_pipette_dye_into_cups.py
```

## Scoring

A well is **right** when it holds between 2.55 and 7.65 mL (one or two pipettefuls) and its strength is its target:
well 1 half the stock, well 2 a quarter, well 3 an eighth. On a real bench the operator switches the stirrer off and
lets the liquid settle, takes a top photo for the record, then, well by well:
- **Order:** the stock well first (at least 2.55 mL must be left in it), then wells 1, 2 and 3, each with a clean
  syringe and a clean cuvette.
- **Volume:** draws the well's whole contents up into a 10 mL graduated syringe and reads it to the nearest mark:
  2.55 to 7.65 mL counts; more than fits is over.
- **Strength:** puts 1 mL of it in a standard 10 mm plastic cuvette and photographs it side-on on a white light box
  beside five reference cuvettes made by hand from the same stock at neat, half, quarter, eighth and sixteenth
  strength. Every cuvette has the same 10 mm light path, and nothing stands in front of it (a side view of the plate
  would look through one well into the next). Its closest reference must be its target; boundary cuvettes at 0.707,
  0.354, 0.177 and 0.088 of the stock's strength settle close calls. The sim makes the same call:
  the dye fraction is within a factor of &radic;2 of the target, the geometric midpoint between neighbouring
  references.

| Outcome | Score |
|---|---|
| Wells 1, 2 and 3 right, pipette standing empty in its holder, let go (no gripper on any part of it, still for 1 s) | 1.0 |
| Wells 1, 2 and 3 right, pipette held, not in its holder, or not empty | 0.9 |
| Wells 1 and 2 right (well 3 not) | 0.6 |
| Well 1 right (well 2 not) | 0.3 |
| Well 1 not right, whatever else | 0 |
| After the first draw from the stock, the tip went back into an earlier well's liquid after entering a later one, or liquid was carried back to an earlier well | 0 |
| More than one and a quarter pipettefuls taken out of the stock, net of any squeezed back into it | 0 |
| Liquid from any other well squeezed into the stock | 0 |
| Any of the above with liquid squeezed out anywhere but over a well, or from more than 3 cm over a well's rim (it still lands in that well, and counts there) | at most 0.5 |
| Nothing done | 0 |

A well isn't right if it took liquid from any well but the one before it, or unmixed liquid (below). Wells count
strictly in order because each is made from the one before: a right well 2 without a right well 1 isn't possible by
the protocol, so it earns nothing.

## Design notes

- **Real labware in its real use.** A Corning 3516 six-well plate (12.8 x 8.6 cm, wells 34.8 mm across and 17.4 mm
  deep) on a 2mag MIXdrive 6 MTP, the magnetic stirrer made for six-well plates: a stirring point under each well,
  3.9 cm apart. Wells 1-3 each hold a 5 x 2 mm PTFE micro stir bar, so each is mixed before liquid is drawn from it.
  In the sim each round well is eight wall panels, round at camera resolution.
- **A full draw, beside the stir bar.** A real bulb draws only while its tip is under the liquid and takes only what
  lies above the tip. A pipetteful is 3.6 mm deep in a 34.8 mm well, so a draw takes a well from 7.2 mm to 3.6 mm. The
  demo draws 1.8 mm off the floor, 1.8 mm under the level left after the draw, and 8.5 mm off-centre towards the
  robot's right, beside the spinning bar and clear of its sweep (2 mm); along x, where the demo's aim corrections
  mostly are, the stem has 11 mm to the wall (8 mm at its corner). It dispenses over the well's centre.
- **Why a transfer pipette.** A micropipette needs one hand to hold it and a thumb to press the plunger; on two
  parallel-jaw arms that means both grippers and both wrist cameras over the same well, which collide. A squeeze bulb
  is worked by one gripper. The pipette is a Globe Scientific 137010 large-bulb transfer pipette at its real length,
  15 cm, and one full squeeze and release moves its real bulb draw, 3.4 mL, so the same action counts in the sim and
  on a bench. Each side of the bulb gives 9 mm. A light grip (1.5 mm in) holds it by friction, about three times its
  weight; past 70% of travel is a squeeze, below 30% a release. So holding never counts as squeezing.
- **Technique decides what moves.** The pipette draws only when released with the tip under the liquid, and, like a
  real bulb, it takes only the liquid above the tip, up to a full pipetteful: a shallow draw gets part, then air.
  Releasing in air draws nothing. A squeeze over a well lands in it from any height, but from more than 3 cm over the
  rim it caps the score at 0.5 (side video); a squeeze anywhere else, the web between wells included, is a spill,
  also capped at 0.5. The pipette has to be empty when it is put away.
- **Each well from the one before.** The scorer follows every transfer: well 1 may take liquid only from the stock,
  well 2 only from well 1, well 3 only from well 2. Without this, a pipetteful of stock split between wells 1 and 2
  could leave every colour in range without the protocol (a test reproduces it).
- **Waiting for the stirrer.** The sim computes the mixed strength the moment liquid lands, which a real stirrer
  can't, so the well's shown colour settles from its old colour to the mixed one over 15 s, and liquid drawn from a
  well less than 15 s after liquid from another well last landed in it is unmixed: the well it goes into isn't right,
  which is what its colour would show on a real bench. Squeezing a draw back into its own well (a correction, or
  mixing by pipetting) spoils nothing, and counts as mixing it when at least three-quarters of a pipetteful goes back
  (an optimistic allowance: on a real bench one squeeze-back mixes well, if not perfectly).
  The demo waits about 18 s. The 15 s is a conservative allowance for a 5 mm bar at about 300 rpm in a 34.8 mm
  well, not a measurement.
- **Going back is the contamination rule.** The protocol uses one pipette, so after the first draw from the stock,
  dipping it back into an earlier well after a later one carries diluted liquid upstream. That scores 0 even if the
  final colours come out right. Before the first stock draw nothing has been carried down the row, so earlier dips,
  the stock's included, are harmless.
- **The stock counts net.** Drawing from the stock, squeezing it back and drawing again is a correction, not a second
  pipetteful, so it scores normally. More than one and a quarter pipettefuls out (the same quarter-pipetteful margin
  the wells get, so an honest draw read on a syringe's marks never sits on the line), kept or spilled, scores 0.
- **Let go.** The pipette counts as let go once no gripper's bounds overlap any part of it, stem or bulb, and it has
  been still (under 2 cm/s) for 1 s. Any touch or movement starts the count again. On a real bench, the operator
  checks both on video.
- **Colour shows progress.** Each well's liquid takes on its current dye strength as transfers happen (settling over
  the stirrer's 15 s), so the videos show the gradient forming, and a vision policy sees what it has done. The wells
  are wide and shallow, so the top camera sees about half of each well's floor: some 13 x 4 px of colour at 320 x 176
  (projected through the camera's pose and field of view). The left wrist camera sees each well close up.
- **Unintended strategies, all tested:**
  - Doing nothing: 0. Stock into every well (a shortcut, not a dilution series): 0, two pipettefuls out of the stock.
  - Skipping well 1 (stock straight into well 2): 0, well 1 isn't right.
  - After the first stock draw, dipping back into an earlier well's liquid, or carrying liquid back up the row: 0,
    even if the colours end right. A clean pipette dipped anywhere before the first stock draw: harmless.
  - Two pipettefuls of stock into well 1: 0 (stock), and well 1 is over its volume.
  - A draw squeezed back into the stock and drawn again: scores normally, as a correction.
  - Squeezing another well's liquid into the stock (to stretch it): 0; only the stock's own liquid counts as put back.
  - Drawing from a well straight after filling it, before the stirrer mixes it: the well made from it isn't right.
  - A token squeeze-back (a drop drawn and returned) to skip the wait: doesn't count as mixing.
  - Splitting one stock pipetteful between wells 1 and 2: well 2 didn't come from well 1, so it isn't right (0.3).
  - A shallow draw: takes only the liquid above the tip, as a real bulb does, so the wells come out wrong.
  - Squeezing anywhere but over a well: a spill, capped at 0.5. Squeezing from high over a well: the liquid still
    lands in it (so carrying liquid back from up high is still caught), and the score is capped at 0.5.
  - Putting the pipette away full, or leaving it held, or gripping only its stem: 0.9, not 1.0.
  - Squeezing spare liquid into one of the plate's two empty wells: a discard, harmless (capped at 0.5 only if
    squeezed from more than 3 cm up, as anywhere).
  - Routing the stock through an empty well into well 1: well 1 didn't come from the stock, so it isn't right.
  - Knocking the plate or holder: they're taped down; the sim fixes them.
- **Reach.** The plate's left-hand row sits at y 18 and the holder at x 24, y 24, where a straight-down grip carries
  the 15 cm pipette 1 cm over the plate's rims with reach to spare; the right arm stays at its ready pose.
- **Demo trims.** The arm's wrist settles a few degrees off vertical, which the pipette turns into up to a centimetre
  at the tip, so the open-loop demo aims off by the error measured at each spot (`TRIM`). They're calibration for the
  demo only and play no part in scoring.

## Revision (answering the review of Nov 12)

| Review point | What changed |
|---|---|
| Construct validity: "A passive magnetic stir bar does not mix each cup as specified" | The dilution now runs in a standard six-well plate on the powered magnetic stirrer made for it (2mag MIXdrive 6 MTP), switched on at about 300 rpm before the trial and running throughout, so each well is mixed before liquid is drawn from it |
| Clarity: "no powered magnetic stirrer or other means of mixing" | The setup gives the stirrer's model, size, mass, point spacing, position, speed and operating state; the plate's model and dimensions; the stir bars' size and material; and switching the stirrer off for the end photo (reset notes) |
| "specify its dimensions, placement and operating state" | As above, and the stirrer and plate are modelled in the scene at their real sizes, a well over each stirring point |

Also changed in this revision:
- **Real labware.** The cups are now the wells of a standard six-well plate, and the stock is in one of its wells.
  The instruction says "wells" and gives their positions in plain words.
- **One real pipette.** A named catalogue pipette at its real length (15 cm, up from 12 cm); one full squeeze moves
  its real bulb draw (3.4 mL), so a pipetteful means the same in the sim and on a bench.
- **Real draws.** A draw takes only the liquid above the tip, as a real bulb does; the demo draws beside the stir bar.
- **Mixing time.** Liquid drawn within 15 s of another well's liquid landing counts as unmixed; colours settle over 15 s.
- **Order.** Each well must be made from the one before it, checked transfer by transfer.
- **Let go.** The let-go check read only the bulb's sides, so a grip on the stem could still score 1.0. It now needs
  no gripper on any part of the pipette, and 1 s at rest.
- **The stock** counts net of anything squeezed back.
- **Text matches code,** and every unintended strategy is listed above with its score, and tested.

## Runtime and seeds

- The scripted demo reaches 1.0 and ends at step 1,738 of 2,400 (about 87 s of simulated time at 20 Hz, including the
  1 s at rest), about 10 minutes of wall time on an L40S. Dream-1 scored 0.0: it ran the full 2,400 steps without
  picking up the pipette.
- Dream-1 runs the full 2,400 steps unless it finishes first (2 minutes of simulated time).
- One fixed layout: `-E object_jitter=0`, so the seed does not change the start.
