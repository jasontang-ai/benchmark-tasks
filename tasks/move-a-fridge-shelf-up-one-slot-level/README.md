# Move a fridge shelf up a slot, tipping it to hook in

**Instruction:** "Move the glass shelf up one slot, to the blue-taped slots, level side to side, and let go."

A half-width cantilever glass fridge shelf hangs by two stamped steel brackets from two slotted tracks on the back
wall of a full-size fridge section. Each bracket's rear edge has an upper hook (a neck through a slot window, a tip
turned down behind the track's face) and, one window lower, a short straight tab. Carried in level, the shelf cannot
be hung at any height: when the hook is high enough to clear its window's lower edge (8-15 mm up in the window) the
tab is above the window below, and the tab can only go in with the hook 1 mm or less above the edge. The front must
be tipped up (8 degrees or more; the demo uses 16) so the tabs swing off the face while the hooks go in and drop,
then lowered so the tabs swing into their windows and the brackets' rear edges come onto the faces. Both hooks are on
one rigid shelf 34 cm apart, so one a row off leaves the shelf hanging crooked. The robot holds only the front trim,
333 mm in front of the hooks, and cannot see them.

The makers' own steps: Whirlpool, "Moving or Adjusting the Shelves", type 2 shelves: tilt the front up and lift the
shelf out of the supports; to replace it, guide the rear hooks into the supports, tilt up until they drop in, lower
the front (https://producthelp.whirlpool.com/Refrigeration/Full-Size_Refrigerators/French_Door_Bottom_Freezer_Refrigerator/Interior_Storage/Moving_or_Adjusting_the_Shelves_-_Refrigerator).
LG's help library: tilt the front up, guide the hooks into the slots at the height wanted, lower the front, check the
shelf is level (https://www.lg.com/us/support/help-library/lg-refrigerator-how-to-remove-and-install-shelves-on-an-lg-refrigerator--20153109594851);
LG's owner's manual warns a shelf not level side to side may fall (https://www.manualslib.com/manual/977531/Lg-Lfx25973.html?page=36).

## Run and score a trial

In the simulator's environment (the Robocurve Brev launchable, or yam-isaac-lab-arena's `setup.sh`), with a Dream API
key configured, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=move_a_fridge_shelf_up_one_slot_level.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the glass shelf up one slot, to the blue-taped slots, level side to side, and let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/shelf-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=move_a_fridge_shelf_up_one_slot_level.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Move the glass shelf up one slot, to the blue-taped slots, level side to side, and let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/shelf-demo
```

The run prints `task_score`; it is scored on the final state (or stops once 1.0 has held).

- One fixed layout (`object_jitter=0`); the seed changes nothing. Runtime per trial: up to 2,400 steps (120 s of sim
  time at 20 Hz); the scripted demo reaches full score in about 42 s (844 steps), and the run then stops.
- Demo modes (`KIT_DEMO`): `demo` (default), `slam` (the natural level move: lifted level, carried up a slot, pushed
  in level, let go), `held` (lifted out of the start row and still held), `none` (arms stay home).
- Tests: `PYTHONPATH=<yam-isaac-lab-arena> PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` in this folder.

## Runs

In Isaac (PhysX), this task file, the commands above:

- Scripted demo: 1.0 at 42.2 s (844 steps). Hooks out at 17.5 s, seated in the taped row at 29.5 s, front lowered by
  34 s, released by 38 s; level, still, grippers clear.
- Dream-1: 0.0. The arms never move the shelf; it stays in the start row for all 2,400 steps.

## Scoring (the scorer is `grade()` near the top of the task file)

| Score | End state (first that fits) |
|---|---|
| 0 (sticky) | the shelf fell at any time (a point dropped more than 45 mm in one 50 ms step), or the section moved more than 5 mm |
| 1.0 | both hooks engaged in the taped row, trim ends level within 3 mm, slope within 2 deg of the start's, no gripper within 30 mm, nothing moving more than 1 mm over the last 1 s |
| 0.75 | the same, still held or not yet still |
| 0.5 | both hooks engaged in one row other than the start row (target row but not level or not lowered, or another row) |
| 0.25 | hooks engaged in two different rows (crooked), or neither hook in the start row with the shelf held up by a gripper |
| 0 | anything else: untouched, back in the start row, on the floor |

An engaged hook: neck through a window with its underside within 3 mm above the window's lower edge, tip behind the
face at least 4 mm below that edge.

**Measuring it in a real trial.** Hooks: a torch photo through the glass at each hook, the row counted from the tape;
then a 10 N pull toward the robot at the trim's middle with a spring scale moves the shelf less than 3 mm. Level: a
steel rule from the box floor to each end of the trim. Front lowered: a phone level on the glass against the start
slope noted before the trial. Released and still: the trial video.

## Layout

| Object | Type and material | Size | Mass | Pose |
|---|---|---|---|---|
| Fridge section | plywood box, open front, white | inside 420 x 345 x 470 mm | clamped (sim: 40 kg free body) | front edges x 0.200 m, centred on y 0 |
| Shelf tracks (2) | GE WR71X2588, painted steel channel | 25 x 12 mm, 11 windows 8 x 20 mm on 38 mm | fixed | y +-0.170, faces at x 0.533, 25-460 mm up |
| Shelf | GE WR32X10177 half cantilever glass shelf | 378 mm wide, 333 mm to the brackets' rear edges, trim 30 x 20 mm | 1.63 kg (sim) | hung in the 5th windows from the bottom (lower edges 185 mm up), leaning 1.57 deg front-down |
| Tape (2) | ScotchBlue 2090 | 24 x 50 mm | - | beside the 6th windows (223 mm up), y +-0.1965 |
| Foam mat | Uline S-19391 PE foam | 400 x 320 x 13 mm | - | on the box floor |

Bracket profile (estimates, to caliper on the purchased brackets): 3 mm plate; neck 5 mm tall, 3.5 mm past the rear
edge; tip 3 mm thick, from the neck's top to 8 mm below its underside, 1.5 mm behind the face when the neck sits on
the edge; tab 3 mm long, 22-27 mm below the neck; rear edge cut back 5 mm for 20 mm under the hook. A substitute shelf
works if its hook and tab clear the level approach the same way: hook lift needed > tab lift allowed.

## Parts (real bench)

- Shelf: GE WR32X10177 Refrigerator Half Cantilever Glass Shelf (GE Appliances, clearance stock:
  https://www.geappliances.com/appliance/Refrigerator-Half-Cantilever-Glass-Shelf-WR32X10177; size from
  https://www.trayco.com/Sku/3558580/ge-refrigerator-half-cantilever-glass-shelf-840383135322-wr32x10177). GE lists it
  as no longer manufactured; an in-production cantilever glass shelf of the same hook-and-tab type is Whirlpool
  WP2223288 (listed in stock: https://partselect.ca/PS11739792-Whirlpool-WP2223288-Cantilever-Shelf-with-Glass.htm);
  its hook and tab are to be checked against the profile above before use.
- Tracks: GE WR71X2588 cantilever shelf track, two (https://www.partselect.ca/PS306274-GE-WR71X2588-Cantilever-Shelf-Track.htm).
- Box: ProWood 3/4 in x 2 ft x 4 ft sanded plywood project panel (Home Depot 202093792) and 1/2 in sanded plywood
  (Home Depot 100017950), screwed and glued, painted white, clamped to the table.
- ScotchBlue Original 2090 painter's tape, 24 mm; Uline S-19391 plank foam.

## Simulation and the real case

- Rigid bodies; the hooks hold by contact under gravity, nothing is latched or moved by code. The scored state
  changes only through the arms.
- The shelf is one free body of boxes; the simulator spreads its mass by volume, which puts the centre of mass 148 mm
  behind the front against about 161 mm for the real shelf (glass, ABS trim, steel brackets). The mass is raised from
  about 1.5 kg to 1.63 kg so the hands bear the real moment (never lighter than real).
- The track's face is 2 mm thick (real sheet about 1.2 mm) for stable contact; the window and hook sizes are estimates
  to measure on the purchased parts.
- The clamped section is a 40 kg free body standing on the table; moving it more than 5 mm scores 0.
- Friction (set, not measured): shelf 0.4, track and box 0.3, static = dynamic.
- The real trial's 10 N pull is judged in simulation by its geometric condition: the tip at least 4 mm below the
  window's lower edge behind the face.

## The scripted demo

Both hands pinch the front trim (about 45 N a finger; the gripper's limit is 50 N). Every loaded move servoes the
shelf's pose, read every control step: 15% of its error goes into the hands' offsets and the wrists' slope, leading
out the arms' lag under the shelf (about 47 mm and 8 deg a hand). On the hooks only the front's height is servoed;
carried, the hooks are. Steps: grip, tip the front up 16 deg, lift the hooks 11 mm (tipped, the tip clears the lower
edge above 8.4 mm, the neck the upper below 13.7 mm), draw them 12 mm out, raise them 38 mm, push them in 3 mm deeper
than they hang, set the necks down, lower the front, ease off, let go, back out, home.

| Mode | Expected | physcheck (MuJoCo) |
|---|---|---|
| demo | 1.0 | 1.0 at 42.2 s; also 1.0 with the shelf at 1.0 and 1.3 kg |
| slam (level push) | 0 | 0: the tab stops the lift; the hooks never leave the start row |
| held | 0.25 | 0.25 (lifted out, still held) |
| none | 0 | 0 |

## Safety

Run behind the cell's guard with the e-stop in reach. The shelf is tempered glass on steel brackets; the foam mat on
the box floor catches a dropped shelf. No sharp, hot or high-force step: the grip is the gripper's own 50 N squeeze.

## Licences

All geometry is built from boxes in the task file; no third-party meshes, scans or scenes are used.
