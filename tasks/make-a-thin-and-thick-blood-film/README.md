# Make a thin and thick blood film with a spreader slide

A specimen slide (25 x 75 x 1.0 mm, frosted end toward the robot) lies on a silicone mat with one 3 uL drop of blood for
the thin film and three 2 uL drops for the thick film; one clean slide stands in a drying rack to the right. The
method is WHO's malaria microscopy SOP MM-SOP-05A, steps 10-14 (thin film: spreader at about 45 deg in front of the
drop, drawn back until the blood runs along its edge, then pushed in one quick, smooth stroke to a feathered end; thick
film: three to six quick corner strokes joining the drops into a patch about 1 cm across) and CDC DPDx's blood smear
procedure (30-45 deg; a thick film over newsprint lets you barely read the words).

Instruction: "Make a thin blood film from the single drop, then a thick film from the three drops, using the clean slide from the rack as the spreader and touching the specimen slide only by its edges or frosted end; then lay the spreader down on the mat."

## Run and score a trial

    # scripted demo (the task file's demo) on the Isaac Lab arena
    python -m yam_arena.run --task tasks/make-a-thin-and-thick-blood-film --policy yam_scripted --steps 2400
    # tests (kit tests 1-7)
    cd tasks/make-a-thin-and-thick-blood-film && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

- Runtime per trial: up to 2,400 steps (2 minutes of sim time at 20 Hz); the demo takes about 760 steps (755 in the MuJoCo check).
- Seeds: one fixed layout, no randomisation (seed 0, object jitter 0).
- The score (`task_score`) is read at the end of the run by `score()` in `make_a_thin_and_thick_blood_film.py`
  (top of the file: the slides and rack, then the film rules, the scorer and the demo).
- Blood is drawn, not simulated as a fluid: the films are patches on the specimen slide laid by the rules at the top of
  the task file from the spreader's pose (edge on the glass, angle, travel and speed; corner strokes). The scorer reads
  the geometry they leave: thin-film length, width and end, bands, thick-patch size and strokes.

## Real trials

Defibrinated sheep blood (stored at 2-8 C, brought to room temperature, mixed by gentle inversion, used within its
expiry; gloves, handled as BSL-1/2); slides from a new box; drops placed with a 2-20 uL micropipette over a printed
1 mm grid, the slide then set on the mat by its edges inside a taped outline; the trial starts within 20 s of the last
drop; a fresh specimen slide and spreader every trial.
Score from the video and one top photo, taken after the films air-dry flat (about 30 min) with the slide on the printed
1 mm grid beside a mm ruler: thin-film length from the drop's back edge, width and where it ends (feathered: the grid
readable through its last 2 mm and no straight end line; a band is a straight thick line across the film), the thick
patch across both ways on the ruler and its roundness, corner strokes counted on the video. Real trials are safe to run: no sharps (the lancet step is out of scope; the drops are pre-placed by pipette), no heat
or power, and low forces (the edge presses the glass with about 5-7 N, the pinch about 2 N); gloves throughout, the
blood handled at BSL-1/2; a cracked slide scores 0 and goes to the glass waste. Only the
spreader's glass touches blood; a finger that does scores 0 and its pad is wiped with 70% ethanol. Used slides go to
the glass waste.

Parts for one session of 20 trials (approximate list prices): a box of 72 frosted, ground-edge 25 x 75 x 1.0 mm
slides (about $20), 10 mL defibrinated sheep blood (about $30-40, sold for blood-agar plates), a 2-20 uL micropipette
and tips (about $100; a fixed 2 uL and 3 uL capillary pipette set is a cheaper option), a slide drying rack (about $15),
a silicone bench mat, a mm ruler, masking tape and a printed 1 mm grid.

## Files and assets

- `make_a_thin_and_thick_blood_film.py`: scene, film rules, scorer, demo.
- `ourkit/`: the shared mechanism kit, copied unchanged.
- `test_make_a_thin_and_thick_blood_film.py`: the kit's tests.
- No third-party meshes or scans: every part is built from boxes.
