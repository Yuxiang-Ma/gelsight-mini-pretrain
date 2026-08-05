#!/usr/bin/env python3
"""Decide, per source, whether it can be regression-tested.

tier-1  published rows carry a usable (capture, frame_idx) join key
        -> migrate to the new abstraction, prove with tools/regress.py
tier-2  no join key
        -> wrap verbatim; do not touch the internal filtering logic

Note on the criterion: an earlier draft of this task required tier-1 to
also have "an unambiguous recovered i_min". That second clause is obsolete.
Task 10 established (see docs/imin_from_code.md) that i_min is
authoritatively readable straight from the legacy code for every one of the
13 sources -- including the ones whose *empirical* recovery is ambiguous
(docs/imin_recovered.md) -- so i_min is no longer a discriminator between
tiers. It is still recorded below (sourced from docs/imin_from_code.md) so
a Task 15/16 implementer has tier and threshold in one place, but it plays
no part in the tier decision. The join key, tested by
gsmp.schema.has_join_key(), is the sole criterion now.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gsmp import schema  # noqa: E402

SOURCES = [
    "gelslam", "tactile_tracking", "real_tactile_mnist", "feelanyforce",
    "threedcal", "tacquad", "unit", "sim_tactile_mnist", "sim_starstruck",
    "feats", "fota_labeled", "fota_unlabeled", "sparsh",
]

#: Must stay in sync with tools/recover_imin.py::SOURCE_LICENSE_REPO.
SOURCE_LICENSE_REPO = {"sparsh": "nc"}

#: i_min per source, as read from the legacy code (docs/imin_from_code.md).
#: Not a tiering input -- see module docstring. Kept here purely for
#: convenience so this table doubles as the tier+threshold reference.
I_MIN_FROM_CODE = {
    "gelslam": "10",
    "tactile_tracking": "10",
    "real_tactile_mnist": "15",
    "feelanyforce": "10",
    "sim_tactile_mnist": "15",
    "sim_starstruck": "15",
    "threedcal": "10",
    "tacquad": "12",
    "unit": "12 (weakly-evidenced, see docs/imin_from_code.md)",
    "sparsh": "12",
    "fota_labeled": "10",
    "fota_unlabeled": "10",
    "feats": "N/A (force-gated)",
}


def main() -> int:
    print("| source | repo | published cols | join key | i_min (code) | tier |")
    print("|---|---|---:|---|---|---|")
    for src in SOURCES:
        repo = SOURCE_LICENSE_REPO.get(src, "main")
        imin = I_MIN_FROM_CODE.get(src, "?")
        try:
            cols = schema.published_columns(src, repo)
            join = schema.has_join_key(src, repo)
        except FileNotFoundError:
            print(f"| {src} | {repo} | - | MISSING | {imin} | skip |")
            continue
        tier = "tier-1" if join else "tier-2"
        print(f"| {src} | {repo} | {len(cols)} | {'yes' if join else 'NO'} | {imin} | {tier} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
