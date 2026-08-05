#!/usr/bin/env python3
"""Compare a migrated source against the published release.

Usage:
    python tools/regress.py --source gelslam
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gsmp import regress, spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap candidate frames (0 = all)")
    args = ap.parse_args()

    try:
        mod = importlib.import_module(f"gsmp.sources.{args.source}")
    except ModuleNotFoundError as e:
        print(
            f"error: no gsmp.sources.{args.source} module "
            f"(gsmp.sources.{args.source} not implemented yet -- "
            f"regression testing requires the migrated source module's "
            f"dry_run_keys() to compare against the published release): {e}",
            file=sys.stderr,
        )
        return 2

    s = spec.get(args.source)

    produced, n_candidates = mod.dry_run_keys(limit=args.limit or None)

    published = regress.published_keys(s.name, s.license_repo)
    rep = regress.compare(published, produced, s.bg_keep_rate, n_candidates)

    print(rep.summary)
    if rep.missing:
        print(f"  first missing: {rep.missing[:5]}")
    if rep.extra:
        print(f"  first extra:   {rep.extra[:5]}")
    return 0 if rep.bg_within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
