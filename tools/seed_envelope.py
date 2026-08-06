#!/usr/bin/env python3
"""Verify a source whose legacy producer had an unrecoverable RNG seed.

Three producers seed from `random.Random(hash(<str>) & 0xFFFFFFFF)`, and
CPython randomises str hash per process. The seed behind the release was never
recorded, so `tools/regress.py` can never report PASS exact for them --
including if you re-ran the original script.

This measures the honest thing instead: run the port N times under different
process hash seeds, compare its run-to-run spread against its gap to the
release, and report whether the release sits inside that spread.

    python tools/seed_envelope.py --source sparsh --runs 3
    python tools/seed_envelope.py --source tacquad --runs 3 --limit 40

Read-only: touches only the published parquet and the raw tree.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gsmp import determinism, spec  # noqa: E402
from gsmp.regress import published_keys  # noqa: E402

_WORKER = """
import importlib, json, sys
m = importlib.import_module("gsmp.sources.%s")
k, n = m.dry_run_keys(%s)
print("@@" + json.dumps({"keys": sorted(list(k)), "n_seen": n}))
"""


def one_run(source: str, limit, hashseed: int):
    """Run dry_run_keys in a subprocess with a specific PYTHONHASHSEED."""
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    code = _WORKER % (source, "limit=%d" % limit if limit else "")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    if out.returncode != 0:
        raise RuntimeError(f"run failed (seed {hashseed}):\n{out.stderr[-2000:]}")
    line = [l for l in out.stdout.splitlines() if l.startswith("@@")][-1]
    payload = json.loads(line[2:])
    return {tuple(x) for x in payload["keys"]}, payload["n_seen"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if determinism.producer_is_reproducible(args.source):
        print(
            f"{args.source}: producer is reproducible -- use tools/regress.py "
            f"and require PASS exact. A seed envelope would understate the bar."
        )
        return 1

    print(f"{args.source}: {determinism.why_not_reproducible(args.source)}\n")

    runs = []
    for i in range(args.runs):
        keys, n_seen = one_run(args.source, args.limit or None, hashseed=1000 + i)
        print(f"  run {i + 1}/{args.runs} (PYTHONHASHSEED={1000 + i}): "
              f"kept {len(keys)} of {n_seen} seen")
        runs.append(keys)

    s = spec.get(args.source)
    published = published_keys(args.source, s.license_repo)
    env = determinism.evaluate_envelope(args.source, runs, published)

    print()
    print(f"  own run-to-run symdiffs : {env.self_symdiffs}")
    print(f"  gap to published        : {env.published_symdiff}")
    print(f"  produced (mean)         : {env.mean_produced:.0f}   "
          f"published: {env.n_published}")
    print(f"\n  {env.verdict}")
    return 0 if env.consistent else 2


if __name__ == "__main__":
    raise SystemExit(main())
