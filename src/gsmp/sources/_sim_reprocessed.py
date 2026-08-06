"""Shared body for sim_tactile_mnist + sim_starstruck.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.

**IMPORTANT DEVIATION FROM THE TASK BRIEF.** The task brief points at
`legacy/make_parquet_v2.py:370-432` (`_iter_sim_parquet_filtered`, called
from `iter_sim_tactile_mnist`/`iter_sim_starstruck`) as "the specification",
and `docs/imin_from_code.md` correspondingly records `I_MIN=15` for both
sources, citing that function. That is the code that was intended to
produce these two sources, but it is **not what actually produced the
`sim_tactile_mnist`/`sim_starstruck` parquet currently sitting under
`mini_data_parquet/`**. This module reproduces the script that actually
did, `/home/yxma/MultimodalData/reprocess_upstream.py` (also present in
this repo's checkout indirectly -- it is not under `legacy/`, since it was
written and run *after* the `legacy/` snapshot was taken). Evidence, all
independently checked against the live published parquet and raw upstream
before writing this module:

1. **Capture format.** `_iter_sim_parquet_filtered` emits
   `capture=f"r{round_id}_t{tj}"` (leading `"r"`); so does
   `legacy/parallel_sim.py:94` (a parallelised copy of the same function).
   Every one of the 316,705 published rows across both sources instead has
   `capture` of the form `f"{row_id}_t{fi}"` with NO leading `"r"`
   (e.g. `"00000_000_t1"`) -- exactly `reprocess_upstream.py:270`:
   `capture=f"{row_id}_t{fi}" if row_id else f"row{global_idx}_t{fi}"`.
   Checked exhaustively over every published shard, both sources, both
   splits: zero rows match the `"r..."` pattern.
2. **Row counts are exact-target fingerprints.** Published `train` counts
   are `sim_tactile_mnist: 102,000` and `sim_starstruck: 150,000` -- exactly
   `cap(200_000) * split_alloc["train"]` (`0.51` and `0.75` respectively) in
   `reprocess_upstream.py:76-104`. This is the fingerprint of that script's
   final step, "if oversubscribed, `np.linspace` down to exactly
   `target_count`" (`reprocess_upstream.py:294-296`) -- a mechanism that
   does not exist anywhere in `make_parquet_v2.py` or `parallel_sim.py`.
   `test` counts (`48,601` / `16,104`) are below target, i.e.
   undersubscribed, which the same script leaves alone -- also consistent.
3. **Timeline.** `reprocess_upstream.py` was written (per its own mtime)
   2026-05-19 15:40, after `make_parquet_v2.py`/`parallel_sim.py`
   (2026-05-18) and after the *earlier* stats snapshot
   `stats_v2_sim_tactile_mnist.json`/`stats_v2_sim_starstruck.json`
   (2026-05-17, `n_kept=200,000` both -- the BUDGET-capped output of an
   earlier run, since superseded). The published parquet's own mtimes
   (2026-05-19 16:27-17:07) postdate `reprocess_upstream.py`'s write time
   and postdate the stale stats snapshots, and `reprocess_upstream.py`
   fully rewrites its output directory from scratch
   (`os.makedirs`+`pq.write_table` to a fixed `{split}-00000-of-00001.parquet`
   name, no in-place patching) -- so whatever superseded the 2026-05-17
   BUDGET-capped state must be a full rewrite, and this is the only script
   in the tree that performs one and matches the resulting row counts and
   capture format. `reprocess_upstream.py`'s own docstring says as much:
   "our aggregated ... parquets were each made by a *single-frame-per-touch*
   extractor before the area+intensity contact filter was tuned. ... Re-
   ingest ... from the raw upstream parquets instead."
4. `docs/imin_from_code.md` and `PIPELINE.md` do not mention
   `reprocess_upstream.py` at all -- they were written against the
   `legacy/` snapshot and are stale for exactly these two sources. This
   module intentionally does NOT use `i_min=15`; it uses `i_min=10`, the
   literal value at `reprocess_upstream.py:78,93`.

**What could not be independently recovered: the exact RNG seed.**
`reprocess_upstream.py` seeds each *subset* (not split) with
`args.seed + i`, `args.seed` defaulting to `20260519`, `i` the subset's
0-based position in the CLI's `subsets` argument list (default order
`["real_tactile_mnist", "sim_tactile_mnist", "sim_starstruck"]` if no
subsets were named). No log, shell history, or other artifact records the
actual invocation. `finalize.sh`'s comment ("run after RTM video extract +
**sim rerun** complete", i.e. a step distinct from and named separately to
RTM) is the only corroborating signal, and points at a 2-subset invocation
of just the two sim sources -- `python reprocess_upstream.py
sim_tactile_mnist sim_starstruck`, giving seed `20260519` (i=0) for
sim_tactile_mnist and `20260520` (i=1) for sim_starstruck. That is the seed
pair below. It is a single well-evidenced hypothesis, tried once and not
iterated on -- if it is wrong, both regressions will show the SAME
non-degenerate failure mode (train count exactly on target but wrong keys,
since the target-count arithmetic doesn't depend on the seed at all), and
the correct seed cannot be recovered by search without that being exactly
the "tune a threshold until it matches" behaviour the task explicitly
prohibits.

Algorithm (per split, mirroring `reprocess_upstream.py:176-303`
decision-for-decision):

  1. List raw parquet files matching `split_globs[split]` (a glob, so
     `"train-*.parquet"` does NOT match `"printed_train-*.parquet"` --
     "Skip printed_train per upstream conventions" is a real behavioural
     difference from `_iter_sim_parquet_filtered`, which globs
     `data/*.parquet` and buckets by substring, i.e. WOULD include
     `printed_train-*` under `split="train"`).
  2. Baseline: `_compute_baseline` samples `n_rows = min(max(8, 100 //
     frames_per_row + 1), total_rows)` ROWS uniformly at random
     (`random.Random(seed).sample`), then for each sampled row draws ONE
     random frame index (`rng.randint`) from that row's 32 touches, decodes
     it, and takes the median grey-centre-crop over however many decoded
     successfully (<=8 for frames_per_row=32, since `n_rows` caps at 8 long
     before the `len(grays) >= n(=100)` early-return could ever fire). This
     is a SINGLE global-per-split baseline, not the per-row cross-touch
     median `_iter_sim_parquet_filtered` uses.
  3. Row scan: a SEPARATE `random.Random(seed)` instance (same seed value,
     independent draw sequence) walks every row across the split's files in
     glob-sorted order, at `global_idx % force_stride == 0` (2 for
     sim_tactile_mnist, 3 for sim_starstruck -- fixed, not computed).
     Every frame of a scanned row is decoded and filtered
     (area>=40, intensity>=i_min against the split's single baseline);
     passing frames are kept unconditionally, failing frames draw
     `rng.random() < BG_RATE(=0.015)`.
  4. Early exit: after finishing an entire file, if the running kept count
     exceeds `target_count * 1.5`, stop scanning (remaining files in this
     split are never touched).
  5. Cap: if the final kept count still exceeds `target_count`
     (`= cap(200_000) * split_alloc[split]`), `np.linspace(0, len-1,
     target_count, dtype=np.int64)` picks evenly-spaced indices INTO THE
     KEPT LIST IN SCAN ORDER -- not a re-sort, not a fresh random sample.
     If under target, everything scanned is kept as-is (this is the
     `test` split's fate for both sources).

Because filtering, baseline, stride, and the final linspace cap all happen
inline here, `SPEC` declares `baseline=NoBaseline()` (`gsmp.runner.run`
never buffers frames or computes its own baseline, `is_empty` stays False
for every record) and `phash_dist=None` (`reprocess_upstream.py` never
dedupes). `dry_run_keys` still routes through `gsmp.runner.run()` purely
for its (capture, frame_idx) bookkeeping, mirroring `unit.py`/`threedcal.py`.
"""
from __future__ import annotations

import glob
import io
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from gsmp import config
from gsmp.filters import PIXEL_THRESH, grey_center
from gsmp.runner import FrameRecord

A_MIN = 40
BG_RATE = 0.015
N_BASELINE = 100          # reprocess_upstream.py:202 compute_baseline(..., n=100, ...)
CAP = 200_000             # reprocess_upstream.py SOURCES[*]["cap"]
FRAMES_PER_ROW = 32       # reprocess_upstream.py SOURCES[*]["frames_per_row"]
_WANTED_COLS = ["sensor_image", "label", "object_id", "id", "info.run_id"]

_RAW_BASE = config.RAW_ROOT / "markerless"


class SimConfig:
    """One source's slice of reprocess_upstream.py's SOURCES dict."""

    def __init__(
        self,
        raw_subdir: str,
        i_min: float,
        obj_pattern: str,
        digit_class: bool,
        split_alloc: Dict[str, float],
        force_stride: int,
        seed: int,
    ) -> None:
        self.raw_dir = _RAW_BASE / raw_subdir
        self.i_min = i_min
        self.obj_pattern = obj_pattern
        self.digit_class = digit_class
        self.split_alloc = split_alloc
        self.force_stride = force_stride
        self.seed = seed
        self.split_globs = {"train": "train-*.parquet", "test": "test-*.parquet"}


# reprocess_upstream.py:63-104, restricted to the two sim entries, plus the
# seed hypothesis from the module docstring (args.seed=20260519 default,
# subset index within a 2-subset [sim_tactile_mnist, sim_starstruck] CLI call).
CONFIGS: Dict[str, SimConfig] = {
    "sim_tactile_mnist": SimConfig(
        raw_subdir="SimTactileMNIST",
        i_min=10,
        obj_pattern="digit_{label}",
        digit_class=True,
        split_alloc={"train": 0.51, "test": 0.49},
        force_stride=2,
        seed=20260519,
    ),
    "sim_starstruck": SimConfig(
        raw_subdir="SimStarStruck",
        i_min=10,
        obj_pattern="starstruck",
        digit_class=False,
        split_alloc={"train": 0.75, "test": 0.25},
        force_stride=3,
        seed=20260520,
    ),
}


def _decode(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def _compute_baseline(paths: List[str], seed: int) -> Optional[np.ndarray]:
    """reprocess_upstream.py:127-162, verbatim decision-for-decision."""
    rng = random.Random(seed)
    counts = [pq.read_metadata(p).num_rows for p in paths]
    total_rows = sum(counts)
    if total_rows == 0:
        return None
    n_rows = min(max(8, N_BASELINE // FRAMES_PER_ROW + 1), total_rows)
    row_idxs = sorted(rng.sample(range(total_rows), n_rows))

    grays: List[np.ndarray] = []
    cum = 0
    rii = iter(row_idxs)
    nxt = next(rii, None)
    for p, c in zip(paths, counts):
        if nxt is None:
            break
        if nxt >= cum + c:
            cum += c
            continue
        local: List[int] = []
        while nxt is not None and nxt < cum + c:
            local.append(nxt - cum)
            nxt = next(rii, None)
        if local:
            t = pq.read_table(p, columns=["sensor_image"])
            for li in local:
                imgs = t.column("sensor_image")[li].as_py()
                idx = rng.randint(0, len(imgs) - 1)
                try:
                    rgb = _decode(imgs[idx]["bytes"])
                    grays.append(grey_center(rgb))
                    if len(grays) >= N_BASELINE:
                        return np.median(np.stack(grays), axis=0)
                except Exception:
                    pass
        cum += c
    if not grays:
        return None
    return np.median(np.stack(grays), axis=0)


class _KeptFrame:
    """Lightweight record held during the scan -- raw JPEG bytes, not a
    decoded array, so the pre-cap `kept` list (up to target_count * 1.5)
    doesn't blow up memory the way holding decoded uint8 arrays would."""

    __slots__ = ("image_bytes", "capture", "obj_name", "split", "episode",
                 "frame_idx", "digit_class")

    def __init__(self, image_bytes: bytes, capture: str, obj_name: str,
                 split: str, episode: str, frame_idx: int,
                 digit_class: Optional[int]) -> None:
        self.image_bytes = image_bytes
        self.capture = capture
        self.obj_name = obj_name
        self.split = split
        self.episode = episode
        self.frame_idx = frame_idx
        self.digit_class = digit_class


def _scan_split(cfg: SimConfig, split: str, target_count: int) -> List[_KeptFrame]:
    """reprocess_upstream.py:176-303 (process_split), minus writing."""
    paths = sorted(glob.glob(str(cfg.raw_dir / "data" / cfg.split_globs[split])))
    if not paths:
        return []
    counts = [pq.read_metadata(p).num_rows for p in paths]
    stride = cfg.force_stride

    baseline = _compute_baseline(paths, cfg.seed)
    if baseline is None:
        return []

    rng = random.Random(cfg.seed)
    kept: List[_KeptFrame] = []
    cum = 0
    for p, c in zip(paths, counts):
        t = None
        for local in range(c):
            global_idx = cum + local
            if global_idx % stride != 0:
                continue
            if t is None:
                schema_names = pq.read_schema(p).names
                wanted = [w for w in _WANTED_COLS if w in schema_names]
                t = pq.read_table(p, columns=wanted)
            row_imgs = t.column("sensor_image")[local].as_py()
            label = (t.column("label")[local].as_py()
                     if "label" in t.column_names else None)
            object_id = (t.column("object_id")[local].as_py()
                         if "object_id" in t.column_names else None)
            row_id = (t.column("id")[local].as_py()
                      if "id" in t.column_names else "")
            run_id = (t.column("info.run_id")[local].as_py()
                      if "info.run_id" in t.column_names else "")

            for fi, frame_struct in enumerate(row_imgs):
                if frame_struct is None:
                    continue
                b = frame_struct["bytes"]
                try:
                    rgb = _decode(b)
                except Exception:
                    continue
                g = grey_center(rgb)
                diff = np.abs(g - baseline)
                mask = diff > PIXEL_THRESH
                area = int(mask.sum())
                inten = float(diff[mask].mean()) if area > 0 else 0.0
                passes = area >= A_MIN and inten >= cfg.i_min

                keep = False
                if passes:
                    keep = True
                elif rng.random() < BG_RATE:
                    keep = True
                if not keep:
                    continue

                obj_name = (cfg.obj_pattern.format(label=label)
                            if "{label}" in cfg.obj_pattern else cfg.obj_pattern)
                capture = f"{row_id}_t{fi}" if row_id else f"row{global_idx}_t{fi}"
                kept.append(_KeptFrame(
                    image_bytes=b,
                    capture=capture,
                    obj_name=obj_name,
                    split=split,
                    episode=str(object_id) if object_id is not None else run_id,
                    frame_idx=fi,
                    digit_class=(label if cfg.digit_class else None),
                ))
        cum += c
        if t is not None:
            del t
        if len(kept) > target_count * 1.5:
            break

    if len(kept) > target_count:
        idx = np.linspace(0, len(kept) - 1, target_count, dtype=np.int64)
        kept = [kept[int(i)] for i in idx]

    return kept


def _frame_records(source_tag: str, limit: Optional[int]) -> Iterator[FrameRecord]:
    cfg = CONFIGS[source_tag]
    n_yielded = 0
    for split, alloc in cfg.split_alloc.items():
        target_count = int(CAP * alloc)
        for kf in _scan_split(cfg, split, target_count):
            yield FrameRecord(
                rgb=_decode(kf.image_bytes),
                capture=kf.capture,
                obj_name=kf.obj_name,
                split=kf.split,
                episode=kf.episode,
                frame_idx=kf.frame_idx,
                extra=dict(digit_class=kf.digit_class),
            )
            n_yielded += 1
            if limit is not None and n_yielded >= limit:
                return


def iter_units(
    source_tag: str, limit: Optional[int] = None
) -> Iterator[Tuple[str, Iterator[FrameRecord]]]:
    """Yield a single (source_tag, frame_generator) unit.

    Both splits are fully computed (baseline + stride scan + linspace cap)
    before anything is yielded, since the linspace cap needs the full
    per-split kept count up front -- `limit` therefore only truncates the
    final yielded sequence, it cannot skip the underlying scan.
    """
    yield source_tag, _frame_records(source_tag, limit)
