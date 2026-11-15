"""The liquid ledger: volumes and contents of every vessel, moved by a pipette. Pure Python (NumPy only).

The simulator has no liquids, so the task tracks them here. Each control step the task calls :meth:`Ledger.update`
with the pipette tip's position (world frame, m) and how far the pipette is pressed (m: the squeeze of a bulb or
the stroke of a plunger). The ledger follows a fixed-volume pipette:

- **Press** (squeeze the bulb): if the pipette holds liquid it is dispensed, into the vessel the tip is inside or
  just above (``catch_above``), or onto the bench (a spill) if it is over no vessel. Either way the pipette is now pressed.
- **Release**: if the tip is below the liquid surface of a vessel, one aliquot (or what is left) is drawn up
  from it. Released in air, nothing is drawn.

So the technique that works is the real one: squeeze in air, dip, release to draw, move, dip into the next vessel,
squeeze to dispense, lift out, then release in air. Releasing while still dipped draws liquid back up.

Volumes are in aliquots (one full squeeze). Contents are tracked per component (e.g. ``"dye"``,
``"water"``) so concentrations follow by mixing, assumed instantaneous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Press hysteresis, as fractions of the full travel: pressed past 70 %, released under 30 %.
PRESS_AT = 0.7
RELEASE_AT = 0.3


@dataclass
class Vessel:
    """A vessel centred on ``xy``: a round well of ``radius``, or square-walled with inner opening ``inner`` (m, x and
    y); its floor and rim heights."""

    name: str
    xy: tuple[float, float]
    inner: tuple[float, float]
    floor_z: float
    rim_z: float
    height_per_aliquot: float  # m of liquid one aliquot adds
    contents: dict[str, float] = field(default_factory=dict)
    catch_above: float = 0.0  # liquid squeezed out this far above the rim, over the opening, still lands inside
    radius: float = 0.0  # a round well of this radius (m), centred on ``xy``; 0: square, by ``inner``

    @property
    def volume(self) -> float:
        return float(sum(self.contents.values()))

    @property
    def surface_z(self) -> float:
        return self.floor_z + self.volume * self.height_per_aliquot

    def fraction(self, component: str) -> float:
        v = self.volume
        return self.contents.get(component, 0.0) / v if v > 1e-9 else 0.0

    def contains_xy(self, p, margin: float = 0.0) -> bool:
        if self.radius > 0:
            return bool(np.hypot(p[0] - self.xy[0], p[1] - self.xy[1]) <= self.radius - margin)
        return bool(
            abs(p[0] - self.xy[0]) <= self.inner[0] / 2 - margin and abs(p[1] - self.xy[1]) <= self.inner[1] / 2 - margin
        )

    def holds(self, p) -> bool:
        """The point is inside the vessel: within its opening and below its rim."""
        return self.contains_xy(p) and p[2] < self.rim_z

    def catches(self, p) -> bool:
        """Liquid squeezed out here lands in this vessel: inside it, or over its opening just above the rim."""
        return self.contains_xy(p) and p[2] < self.rim_z + self.catch_above

    def submerged(self, p) -> bool:
        return self.holds(p) and self.volume > 1e-9 and p[2] < self.surface_z


@dataclass
class Event:
    t: int  # control step
    kind: str  # "aspirate", "dispense", "spill", "dip"
    vessel: str | None
    volume: float = 0.0


def _mix(into: dict[str, float], add: dict[str, float]) -> None:
    for k, v in add.items():
        into[k] = into.get(k, 0.0) + v


class Ledger:
    """Vessels plus one pipette tip. ``order`` gives each vessel's place in the protocol for :attr:`went_back`."""

    def __init__(self, vessels: list[Vessel], travel: float, aliquot: float = 1.0, order: list[str] | None = None):
        self._initial = [(v.name, dict(v.contents)) for v in vessels]
        self.vessels = {v.name: v for v in vessels}
        self.travel = travel
        self.aliquot = aliquot
        self.order = {n: i for i, n in enumerate(order)} if order else {}
        self.reset()

    def reset(self) -> None:
        for name, contents in self._initial:
            self.vessels[name].contents = dict(contents)
        self.tip: dict[str, float] = {}
        self.pressed = False
        self.dipped_in: str | None = None
        self.step = 0
        self.events: list[Event] = []
        self.dips: list[str] = []  # vessels whose liquid the tip entered, in order
        self.spilled = 0.0
        self.high = 0.0  # liquid squeezed into a vessel from more than catch_above over its rim
        self.went_back = False  # the raw flag: the task's scorer applies its own going-back rule

    # -- queries ---------------------------------------------------------------------------------------------------

    def vessel_at(self, p) -> Vessel | None:
        for v in self.vessels.values():
            if v.holds(p):
                return v
        return None

    @property
    def tip_volume(self) -> float:
        return float(sum(self.tip.values()))

    # -- the step --------------------------------------------------------------------------------------------------

    def update(self, tip, press: float) -> None:
        """One control step: ``tip`` the tip's world position, ``press`` how far the pipette is pressed (m, 0 at
        rest, ``travel`` fully pressed)."""
        tip = np.asarray(tip, dtype=float)
        depth = min(max(press / self.travel, 0.0), 1.0) if self.travel > 0 else 0.0
        vessel = self.vessel_at(tip)
        catcher = next((v for v in self.vessels.values() if v.catches(tip)), None)
        # Liquid squeezed out over a vessel's opening falls into it from any height; only a squeeze within catch_above
        # of the rim is a clean one. A high squeeze is recorded, so a task can treat it as sloppy, not as lost liquid.
        high = None
        if catcher is None:
            high = next((v for v in self.vessels.values() if v.contains_xy(tip) and tip[2] >= v.rim_z), None)
        wet = vessel if vessel is not None and vessel.submerged(tip) else None

        # Entering a vessel's liquid: record the dip, and whether it goes back to an earlier vessel in the protocol.
        if wet is not None and self.dipped_in != wet.name:
            self.dips.append(wet.name)
            self.events.append(Event(self.step, "dip", wet.name))
            if self.order and wet.name in self.order:
                furthest = max((self.order[d] for d in self.dips[:-1] if d in self.order), default=-1)
                if self.order[wet.name] < furthest:
                    self.went_back = True
        self.dipped_in = wet.name if wet is not None else None

        if not self.pressed and depth >= PRESS_AT:
            self.pressed = True
            if self.tip_volume > 1e-9:
                if catcher is None and high is not None:
                    self.high += self.tip_volume
                    catcher = high
                if catcher is not None:
                    _mix(catcher.contents, self.tip)
                    self.events.append(Event(self.step, "dispense", catcher.name, self.tip_volume))
                    if self.order and catcher.name in self.order:
                        drawn = [e.vessel for e in self.events if e.kind == "aspirate" and e.vessel in self.order]
                        furthest = max((self.order[v] for v in drawn), default=-1)
                        if self.order[catcher.name] < furthest:  # liquid carried back up the row
                            self.went_back = True
                else:
                    self.spilled += self.tip_volume
                    self.events.append(Event(self.step, "spill", None, self.tip_volume))
                self.tip = {}
        elif self.pressed and depth <= RELEASE_AT:
            self.pressed = False
            if wet is not None:
                # A bulb draws only while its tip is under the liquid: it takes what lies above the tip, up to a full
                # aliquot, and then air, as a real transfer pipette does.
                above_tip = max(wet.surface_z - tip[2], 0.0) / wet.height_per_aliquot if wet.height_per_aliquot > 0 else wet.volume
                take = min(self.aliquot, wet.volume, above_tip)
                share = take / wet.volume
                drawn = {k: v * share for k, v in wet.contents.items()}
                for k, v in drawn.items():
                    wet.contents[k] -= v
                _mix(self.tip, drawn)
                self.events.append(Event(self.step, "aspirate", wet.name, take))
        self.step += 1
