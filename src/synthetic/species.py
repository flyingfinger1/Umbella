"""Calibrated parameter ranges for selected Apiaceae species.

Each `SpeciesSpec` captures empirical numeric ranges from botanical references
(see `sources`) for stem height, ray counts, ray lengths, pedicel counts and
lengths, plus a few qualitative diagnostic flags. `sample_params(spec, seed)`
draws a random `ApiaceaeParams` instance from the ranges, suitable for
generating one realistic plant.

Initial focus is on the safety-relevant Apiaceae set called out in the
test-set methodology (see RESEARCH.md): Wiesen-Bärenklau, Schierling, Wilde
Möhre, Wiesen-Kerbel, Hundspetersilie, Pastinak.

Numbers should be treated as **first-pass calibration**. Fine-tuning against
field photos and authoritative German Bestimmungsschlüssel (Rothmaler,
Schmeil-Fitschen) is still pending.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .apiaceae import ApiaceaeParams


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float

    def sample(self, rng) -> float:
        return float(rng.uniform(self.lo, self.hi))

    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)


@dataclass
class SpeciesSpec:
    name: str                 # binomial
    german_name: str

    stem_height_mm: Range
    n_primary_rays: Range
    primary_ray_length_mm: Range
    n_pedicels: Range
    pedicel_length_mm: Range

    primary_ray_half_angle_deg: Range
    pedicel_half_angle_deg: Range
    n_laterals: Range

    has_involucre: bool          # bracts at base of main umbel
    has_involucel: bool          # bracts at base of umbellets
    reflexed_involucel: bool     # bracteoles point downward (Aethusa diagnostic)

    # quantitative bract parameters; (0, 0) ranges mean "absent"
    n_involucre_bracts: Range = field(default_factory=lambda: Range(0, 0))
    involucre_bract_length_mm: Range = field(default_factory=lambda: Range(0, 0))
    involucre_bract_angle_deg: Range = field(default_factory=lambda: Range(70, 90))
    n_involucel_bracts: Range = field(default_factory=lambda: Range(0, 0))
    involucel_bract_length_mm: Range = field(default_factory=lambda: Range(0, 0))
    involucel_bract_angle_deg: Range = field(default_factory=lambda: Range(70, 90))

    # purple-brown stem speckles (Conium maculatum diagnostic). Density is the
    # fraction of stem surface points relabeled to render as speckles.
    stem_speckle_density: Range = field(default_factory=lambda: Range(0.0, 0.0))

    sources: list[str] = field(default_factory=list)

    def sample(self, seed: int | None = None) -> ApiaceaeParams:
        rng = np.random.default_rng(seed)
        return ApiaceaeParams(
            stem_height_mm=self.stem_height_mm.sample(rng),
            n_internodes=int(rng.integers(4, 8)),
            n_laterals=int(round(self.n_laterals.sample(rng))),
            n_primary_rays=max(3, int(round(self.n_primary_rays.sample(rng)))),
            primary_ray_length_mm=self.primary_ray_length_mm.sample(rng),
            primary_ray_half_angle_deg=self.primary_ray_half_angle_deg.sample(rng),
            n_pedicels=max(3, int(round(self.n_pedicels.sample(rng)))),
            pedicel_length_mm=self.pedicel_length_mm.sample(rng),
            pedicel_half_angle_deg=self.pedicel_half_angle_deg.sample(rng),
            n_involucre_bracts=int(round(self.n_involucre_bracts.sample(rng))),
            involucre_bract_length_mm=self.involucre_bract_length_mm.sample(rng),
            involucre_bract_angle_deg=self.involucre_bract_angle_deg.sample(rng),
            n_involucel_bracts=int(round(self.n_involucel_bracts.sample(rng))),
            involucel_bract_length_mm=self.involucel_bract_length_mm.sample(rng),
            involucel_bract_angle_deg=self.involucel_bract_angle_deg.sample(rng),
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Calibrated species — values from Wikipedia/botanical references; conservative
# midranges where sources disagree. Rough first pass.
# ---------------------------------------------------------------------------

SPECIES: dict[str, SpeciesSpec] = {
    "Heracleum_sphondylium": SpeciesSpec(
        name="Heracleum sphondylium",
        german_name="Wiesen-Bärenklau",
        stem_height_mm=Range(800, 2000),
        n_primary_rays=Range(12, 30),
        primary_ray_length_mm=Range(60, 125),
        n_pedicels=Range(15, 30),
        pedicel_length_mm=Range(5, 15),
        primary_ray_half_angle_deg=Range(45, 65),
        pedicel_half_angle_deg=Range(55, 75),
        n_laterals=Range(1, 9),
        has_involucre=False,
        has_involucel=True,
        reflexed_involucel=False,
        n_involucel_bracts=Range(5, 10),
        involucel_bract_length_mm=Range(3, 8),
        involucel_bract_angle_deg=Range(70, 90),
        sources=["https://en.wikipedia.org/wiki/Heracleum_sphondylium"],
    ),
    "Conium_maculatum": SpeciesSpec(
        name="Conium maculatum",
        german_name="Gefleckter Schierling",
        stem_height_mm=Range(800, 2500),
        n_primary_rays=Range(10, 20),
        primary_ray_length_mm=Range(10, 35),
        n_pedicels=Range(12, 18),
        pedicel_length_mm=Range(5, 10),
        primary_ray_half_angle_deg=Range(50, 70),
        pedicel_half_angle_deg=Range(60, 80),
        n_laterals=Range(2, 6),
        has_involucre=True,                # small leafy bracts
        has_involucel=True,
        reflexed_involucel=False,
        n_involucre_bracts=Range(3, 6),
        involucre_bract_length_mm=Range(3, 6),
        involucre_bract_angle_deg=Range(70, 95),
        n_involucel_bracts=Range(3, 5),
        involucel_bract_length_mm=Range(2, 4),
        involucel_bract_angle_deg=Range(70, 90),
        stem_speckle_density=Range(0.10, 0.20),    # diagnostic purple stem flecks
        sources=["https://en.wikipedia.org/wiki/Conium_maculatum"],
    ),
    "Daucus_carota": SpeciesSpec(
        name="Daucus carota",
        german_name="Wilde Möhre",
        stem_height_mm=Range(300, 1000),
        n_primary_rays=Range(30, 50),
        primary_ray_length_mm=Range(30, 70),
        n_pedicels=Range(10, 20),
        pedicel_length_mm=Range(2, 12),    # very short at anthesis, longer in fruit
        primary_ray_half_angle_deg=Range(60, 80),  # quite flat-topped at anthesis
        pedicel_half_angle_deg=Range(70, 88),
        n_laterals=Range(0, 3),
        has_involucre=True,                # multiple pinnate bracts (very diagnostic)
        has_involucel=True,
        reflexed_involucel=False,
        n_involucre_bracts=Range(5, 12),
        involucre_bract_length_mm=Range(15, 35),
        involucre_bract_angle_deg=Range(80, 110),  # spread wide, sometimes slightly reflexed
        n_involucel_bracts=Range(5, 8),
        involucel_bract_length_mm=Range(5, 12),
        involucel_bract_angle_deg=Range(75, 95),
        sources=["https://en.wikipedia.org/wiki/Daucus_carota"],
    ),
    "Anthriscus_sylvestris": SpeciesSpec(
        name="Anthriscus sylvestris",
        german_name="Wiesen-Kerbel",
        stem_height_mm=Range(600, 1700),
        n_primary_rays=Range(4, 10),
        primary_ray_length_mm=Range(15, 30),
        n_pedicels=Range(6, 12),
        pedicel_length_mm=Range(5, 10),
        primary_ray_half_angle_deg=Range(40, 60),
        pedicel_half_angle_deg=Range(60, 75),
        n_laterals=Range(2, 6),
        has_involucre=False,               # absent
        has_involucel=True,                # 5-8 small ovate bracteoles
        reflexed_involucel=False,
        n_involucel_bracts=Range(5, 8),
        involucel_bract_length_mm=Range(2, 5),
        involucel_bract_angle_deg=Range(70, 90),
        sources=["https://en.wikipedia.org/wiki/Anthriscus_sylvestris"],
    ),
    "Aethusa_cynapium": SpeciesSpec(
        name="Aethusa cynapium",
        german_name="Hundspetersilie",
        stem_height_mm=Range(300, 800),
        n_primary_rays=Range(10, 20),
        primary_ray_length_mm=Range(6, 26),
        n_pedicels=Range(15, 25),
        pedicel_length_mm=Range(3, 8),
        primary_ray_half_angle_deg=Range(50, 70),
        pedicel_half_angle_deg=Range(55, 75),
        n_laterals=Range(1, 4),
        has_involucre=False,
        has_involucel=True,
        reflexed_involucel=True,           # KEY DIAGNOSTIC: 3-4 long reflexed bracteoles
        n_involucel_bracts=Range(3, 5),
        involucel_bract_length_mm=Range(8, 18),    # long — clearly longer than the pedicels (3-8mm)
        involucel_bract_angle_deg=Range(125, 155), # >90 = strongly reflexed (downward)
        sources=["https://en.wikipedia.org/wiki/Aethusa_cynapium"],
    ),
    "Pastinaca_sativa": SpeciesSpec(
        name="Pastinaca sativa",
        german_name="Pastinak",
        stem_height_mm=Range(600, 1800),
        n_primary_rays=Range(15, 25),
        primary_ray_length_mm=Range(50, 100),
        n_pedicels=Range(12, 35),
        pedicel_length_mm=Range(20, 50),
        primary_ray_half_angle_deg=Range(55, 75),
        pedicel_half_angle_deg=Range(60, 80),
        n_laterals=Range(2, 5),
        has_involucre=False,
        has_involucel=False,
        reflexed_involucel=False,
        sources=["https://en.wikipedia.org/wiki/Parsnip"],
    ),
}
