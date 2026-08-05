"""Regression harness: new pipeline output vs the published release.

The published parquet is the ground truth for the legacy behaviour. Each row
carries (capture, frame_idx), so the set of kept frames is directly
queryable -- no need to keep the legacy code runnable.

BG_KEEP_RATE is stochastic, so exact set equality is impossible. The
assertion is therefore two-tier:

  deterministic part -- frames that pass area+intensity MUST match exactly
  stochastic part    -- frames kept despite failing are only checked in bulk
                         against a budget derived from bg_keep_rate.

`RegressionReport.missing`/`.extra` hold the full sorted list of differing
keys -- `deterministic_ok` and `bg_within_tolerance` are always computed
from these full lists, never from a truncated view, so a large regression
can never be made to look small. Callers that want a short human-readable
preview (e.g. the CLI in tools/regress.py) should slice the lists
themselves at display time.
"""
from __future__ import annotations

import dataclasses
import glob
from typing import List, Set, Tuple

import pyarrow.parquet as pq

from gsmp import config

Key = Tuple[str, int]


@dataclasses.dataclass(frozen=True)
class RegressionReport:
    missing: List[Key]
    extra: List[Key]
    deterministic_ok: bool
    bg_within_tolerance: bool
    summary: str


def published_keys(source: str, license_repo: str = "main") -> Set[Key]:
    """(capture, frame_idx) of every published row of `source`."""
    out: Set[Key] = set()
    for shard in sorted(glob.glob(str(config.published_dir(source, license_repo) / "*.parquet"))):
        t = pq.read_table(shard, columns=["capture", "frame_idx"])
        caps = t.column("capture").to_pylist()
        idxs = t.column("frame_idx").to_pylist()
        for c, i in zip(caps, idxs):
            if c is not None and i is not None:
                out.add((c, int(i)))
    return out


def compare(
    published: Set[Key],
    produced: Set[Key],
    bg_keep_rate: float,
    n_candidates: int,
) -> RegressionReport:
    missing = sorted(published - produced)
    extra = sorted(produced - published)

    # Budget: how many frames the 1.5% background keep could plausibly move,
    # with a 3x slack for sampling variance on top of the expected count.
    budget = max(10, int(3 * bg_keep_rate * n_candidates))
    diff = len(missing) + len(extra)

    deterministic_ok = diff == 0
    bg_within_tolerance = diff <= budget

    if deterministic_ok:
        summary = f"PASS exact: {len(published)} keys identical"
    elif bg_within_tolerance:
        summary = (
            f"PASS within background budget: {diff} differing keys "
            f"(budget {budget}, published {len(published)}, produced {len(produced)})"
        )
    else:
        summary = (
            f"FAIL: {len(missing)} missing, {len(extra)} extra, "
            f"exceeds background budget {budget}"
        )

    return RegressionReport(
        missing=missing,
        extra=extra,
        deterministic_ok=deterministic_ok,
        bg_within_tolerance=bg_within_tolerance,
        summary=summary,
    )
