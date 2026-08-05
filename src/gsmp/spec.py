"""SourceSpec -- the complete declaration of one source's behaviour.

Collapses four registries that used to live in make_parquet_v2.py:
    SOURCE_ITERS       -> the sources package (module per source)
    VALIDITY_THRESH    -> a_min / i_min fields
    SKIP_EMPTY_FILTER  -> the baseline field
    NC_SOURCES         -> license_repo field
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, Optional, Tuple

from gsmp import config
from gsmp.baseline import BaselineStrategy

_DOMAINS = ("real", "sim")
_GEL_VARIANTS = ("markered", "markerless", "mixed")
_LICENSE_REPOS = ("main", "nc")
_CHANNEL_MODES = ("auto", "rgb", "bgr", "mixed")


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    """Everything the runner needs to know about one source.

    `i_min` deliberately has no default. The legacy code carried three
    different values for it (10, 12, 15) and the published README told users
    a fourth (12 real / 10 sim), so no default can be justified. Each source
    declares the value recovered from the published release by
    tools/recover_imin.py.

    `rng_seed` seeds the runner's background-keep draw. Legacy seeded every
    iterator explicitly -- mostly `random.Random(0)`, but
    `make_parquet_v2.py:234` uses `random.Random(42)`. Reproducing a
    source's kept set requires its actual seed, so copy it from the legacy
    iterator rather than assuming 0.

    `touch_window_keep` is the per-frame Bernoulli keep probability applied
    inside a touch window. It is 1.0 everywhere except real_tactile_mnist,
    which keeps 0.30 (~2 frames per touch).
    """

    name: str
    domain: str
    gel_variant: str
    license_repo: str
    baseline: BaselineStrategy
    i_min: float
    a_min: int = 40
    channel_mode: str = "auto"
    phash_dist: Optional[int] = 4
    phash_lookback: int = 30
    budget: int = 200_000
    bg_keep_rate: float = 0.015
    rng_seed: int = 0
    touch_window_keep: float = 1.0
    resolution: Optional[Tuple[int, int]] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.domain not in _DOMAINS:
            raise ValueError(f"domain must be one of {_DOMAINS}: {self.domain!r}")
        if self.gel_variant not in _GEL_VARIANTS:
            raise ValueError(
                f"gel_variant must be one of {_GEL_VARIANTS}: {self.gel_variant!r}"
            )
        if self.license_repo not in _LICENSE_REPOS:
            raise ValueError(
                f"license_repo must be one of {_LICENSE_REPOS}: {self.license_repo!r}"
            )
        if self.channel_mode not in _CHANNEL_MODES:
            raise ValueError(
                f"channel_mode must be one of {_CHANNEL_MODES}: {self.channel_mode!r}"
            )
        if self.a_min <= 0:
            raise ValueError(f"a_min must be positive: {self.a_min}")
        if not 0.0 <= self.bg_keep_rate <= 1.0:
            raise ValueError(f"bg_keep_rate must be in [0,1]: {self.bg_keep_rate}")

    @property
    def dedupe_enabled(self) -> bool:
        return self.phash_dist is not None

    @property
    def markered(self) -> bool:
        return self.gel_variant == "markered"

    def out_dir(self) -> Path:
        return config.OUT_ROOT / self.name

    def published_dir(self) -> Path:
        return config.published_dir(self.name, self.license_repo)


_REGISTRY: Dict[str, SourceSpec] = {}


def register(spec: SourceSpec) -> SourceSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"source already registered: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> SourceSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registry() -> Dict[str, SourceSpec]:
    return dict(_REGISTRY)
