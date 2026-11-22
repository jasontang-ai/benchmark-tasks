# Shield a culture dish with its lid while adding medium

**Instruction:** "Add one pipetteful of fresh medium from the reservoir to the culture dish, holding the dish's lid over it
as a shield with the opening toward the back wall, then put the lid back on, stand the pipette in its holder and let go."

A dry 60 mm tissue-culture dish sits closed in a Class II biological safety cabinet, beside a reagent reservoir of
medium (water dyed pink) and a sterile transfer pipette standing in its holder. The open dish is where contamination
gets in, so the lid is never set down: one hand lifts it clear of the rim and tilts it up at its back edge, low edge close
over the rim, while the other hand draws 5 mL, brings the pipette over the lid's back edge, down through the gap and
squeezes with the tip at the side wall (not onto the cell layer). The lid then goes straight back on and the pipette back
in its holder. The robot works inside a 4-foot Thermo Scientific 1300 Series A2 cabinet (interior 1199 x 780 x 630 mm),
the sash behind it at its 10 in working height; the cabinet's own rules (arms inside the walls, ceiling and sash plane,
objects 4 in from the front grille, rear grille clear) are sticky zeros. In this layout the work does not come near them.

Sources: CDC/NIH Biosafety in Microbiological and Biomedical Laboratories, 6th ed., Appendix A (hold Petri dish and
tissue culture plate lids above the open sterile surface, against direct impaction of the downflow;
https://www.cdc.gov/labs/bmbl/); Galloway Lab adherent cell culture protocol
(https://gallowaylabmit.github.io/protocols/en/latest/protocols/tc/tc-basics/adherent_cell_culture.html: lift the lid with
one hand while pipetting with the other; the lid covering as much as possible, just enough opening; the opening facing the
back wall; add liquid against the side wall, not onto the cells); Corning 430166 60 mm dish
(https://ecatalog.corning.com/life-sciences/b2c/US/en/Cell-Culture/Cell-Culture-Vessels/Dishes/Corning%C2%AE-Cell-Culture-Dishes/p/430166:
15 mm tall, growth surface 52.1 mm across, 21 cm2, 4.2-6.3 mL); Corning 4870 reagent reservoir (145 x 70 x 28 mm, sterile,
https://www.fishersci.com/shop/products/costar-sterile-disposable-reagent-reservoirs-4/07200127); sterile cell-culture
transfer pipettes (CellPro, https://www.avantorsciences.com/us/en/product/NA5652575/cellpro-transfer-pipettes); Thermo
Scientific 1300 Series A2 brochure
(https://assets.thermofisher.com/TFS-Assets/LED/brochures/1300-Series-A2-Biological-Safety-Cabinet.pdf).

## Run and score a trial

In the simulator's environment (the Robocurve Brev launchable, or yam-isaac-lab-arena's `setup.sh`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=shield_a_culture_dish_with_its_lid_while_adding_medium.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Add one pipetteful of fresh medium from the reservoir to the culture dish, holding the dish's lid over it as a shield with the opening toward the back wall, then put the lid back on, stand the pipette in its holder and let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir logs/shield-dream
```
The scripted two-arm demo: the same with `--policy yam_scripted -E cam_width=640 -E cam_height=352 --log-dir logs/shield-demo`.

The run prints `task_score`, read at the end. Runtime per trial: 2,400 steps at most (2 minutes simulated); the demo
takes about 37 s. One fixed layout (`object_jitter=0`), no seeds. Tests: `python3 -m pytest -q`.

## Scoring (the scorer is at the top of the task file)

The ladder is task.json's `scoring`. Real-trial checks: a phone overhead (the lid's centre and the bearing of its high
edge against marks on the work surface; the tip against a ring drawn on the dish's underside 8 mm inside the wall), a
phone at bench level with a 1 mm grid card behind the dish (heights and tilt), video timestamps (the open time), a frame
of the tip on the lid, rim or floor (a touch), and the dish weighed on a 0.01 g balance before and after (5.0 g +-0.3). The
simulator calls a touch at 0.02 mm (contact); on a bench it is judged from the video frames, and the 5 mm, 20 mm and 2 mm
bands are read on the grid card.

| Strategy (KIT_DEMO mode) | Cap | MuJoCo (physcheck) |
|---|---|---|
| demo | 1.0 | 1.0 (5.0 mL at the wall, open 18 s, lid seated within 0.2 mm) |
| none: both arms stay home | 0 | 0.0 |
| lid_aside: lid lifted off and set down beside the dish | 0 | 0.0 (shield rule) |
| level_high: lid lifted straight up, level | 0 | 0.0 (low edge over 20 mm) |
| centre: squeezed over the dish's centre, onto the cells | 0 | 0.0 (the pipette meets the lid) |
| lid_left_open: shield kept, lid not put back | 0.5 | 0.5 |
| still_held: lid back, pipette kept in the hand | 0.75 | 0.75 |

Isaac Lab (GPU): the demo scored 1.0; Dream-1 scored 0.0 (hit the 2,400-step limit).

## Simulation and the real case

- Free rigid bodies at catalogue size; the dish and lid are 12-sided; nothing is moved or latched by code.
- The medium is a ledger (`ledger.py`): a full squeeze empties the pipette into the vessel its tip is in (or over the
  dish within 10 mm of its rim); a release with the tip under the surface draws what lies above the tip, up to 5 mL; a
  squeeze over no vessel is a spill. A pink tint shows the dish's volume (cosmetic).
- The bulb's squeeze faces are on 350 N/m springs with 9 mm of travel; the reservoir's U-bottom is taken flat.
- Friction 0.5 (set, not measured), static = dynamic.
- Cabinet for repeated trials: a full-size plywood mock of the 4-foot interior, 1199 mm wide, 780 mm high and 630 mm
  deep (acrylic sash at 254 mm, grilles as tape bands 100 and 50 mm wide). Dishes, reservoirs and pipettes are consumables of about US$1-3 each.

## Safety of real trials

Safe to run: dyed water only (no cells or chemicals), plastic dish, reservoir and pipette, light grips; nothing sharp or hot.

## Demo

Both arms servo the held object's pose each step (`servo.py`). Right: takes the pipette, squeezes in air over the
reservoir, dips, releases to draw 5 mL, and waits there. Left: grips the lid across two flats, lifts it 11 mm level, then
tilts it 40 deg (high edge toward the back wall, 40 deg toward the right) and draws it 6 mm the other way. Right: comes in
over the lid's back edge, down through the gap to the wall, squeezes, comes out, releases in air and heads for the
holder; left: levels the lid over the dish's live position and lowers it straight on. Right stands the pipette in its
holder. Third-party assets: none (boxes only). `ourkit/` is the team's shared mechanism kit, copied unchanged.
