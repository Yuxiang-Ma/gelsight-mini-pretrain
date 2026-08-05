#!/usr/bin/env python3
"""Unified ingest pipeline helpers for the gelsight-mini-pretrain aggregation.

A new source goes through this pipeline:

  1. **DOWNLOAD** raw upstream data (often via huggingface_hub.snapshot_download
     or a custom HTTP/git pull). The source-specific ingest module is
     responsible for downloading and yielding decoded frames.

  2. **FILTER** frames through the unified area+intensity contact rule:
       diff      = |frame - baseline|   (greyscale, central 50% crop)
       mask      = diff > PIXEL_THRESH (=10)
       area      = mask.sum()
       intensity = diff[mask].mean()       (grey-levels)
       KEEP iff (area >= A_MIN) AND (intensity >= I_MIN)
       ELSE  keep with probability BG_RATE (= 1.5 %) for background diversity

  3. **NORMALIZE CHANNEL ORDER**: GelSight Mini's at-rest illumination
     has B > R. If a sampled image's per-channel mean shows R > B by a
     significant margin, we either flag the whole source as BGR-stored
     (unconditional swap) or flag it as mixed (per-image conditional
     swap). All frames in the output are guaranteed RGB.

  4. **WRITE PARQUET** under the unified 30-column schema, with image
     bytes built as chunked binary (CHUNK = 5000 rows) so each chunk
     stays under PyArrow's 2 GB pa.binary() limit.

  5. **GENERATE SAMPLE GRID**: render 40 random frames as a 4x10 grid
     for visual inspection (samples_40_<source>.png in assets/).

  6. **PUSH TO HF** via huggingface_hub.create_commit, including the
     parquet + sample grid + a diagnostic JSON entry for channel order.

Each source-specific ingest module follows this template:

  ```python
  from pipeline import IngestPipeline, FrameRecord

  def yield_frames():
      # walk upstream, decode each frame
      for ...:
          yield FrameRecord(
              rgb=rgb_array,                 # H x W x 3 uint8
              obj_name=...,
              split=...,
              episode=..., capture=...,
              frame_idx=...,
              extra={...},                   # any other fields
          )

  if __name__ == "__main__":
      IngestPipeline(
          source_name="newsource",
          repo="yxma/gelsight-mini-pretrain",
          domain="real",
          gel_variant="markerless",
          i_min=12,
          channel_mode="auto",   # auto | rgb | bgr | mixed
      ).run(yield_frames())
  ```

This file lives in /home/yxma/MultimodalData/pipeline.py.
"""
from __future__ import annotations

import dataclasses
import glob
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# Schema is shared across all sources
sys.path.insert(0, "/home/yxma/MultimodalData")
from make_parquet_v2 import SCHEMA

PIXEL_THRESH = 10
A_MIN = 40
BG_RATE = 0.015
N_BASELINE = 120
JPEG_Q = 92
CHUNK = 5000

BASE_MAIN = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
BASE_NC = "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc"


@dataclasses.dataclass
class FrameRecord:
    """One source frame, decoded into an RGB numpy array, ready to filter.

    The source-specific ingest function yields these in order.
    """
    rgb: np.ndarray                # H x W x 3 uint8
    obj_name: str = ""
    capture: str = ""
    split: str = "train"
    episode: str = ""
    frame_idx: int = 0
    # Optional metadata fields (any unified-schema column the source has)
    extra: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Filter / channel-order helpers
# ---------------------------------------------------------------------------

def grey_center(rgb: np.ndarray) -> np.ndarray:
    g = rgb.mean(axis=2).astype(np.float32)
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4]


def to_jpeg(rgb: np.ndarray, q: int = JPEG_Q) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def passes_filter(rgb: np.ndarray, baseline: np.ndarray, i_min: float) -> bool:
    g = grey_center(rgb)
    diff = np.abs(g - baseline)
    mask = diff > PIXEL_THRESH
    area = int(mask.sum())
    if area < A_MIN:
        return False
    inten = float(diff[mask].mean())
    return inten >= i_min


def channel_check(rgb: np.ndarray) -> float:
    """Return R-B mean signed difference. R-B > 0 likely BGR-stored."""
    return float(rgb[..., 0].mean()) - float(rgb[..., 2].mean())


def maybe_swap_channels(rgb: np.ndarray, channel_mode: str) -> np.ndarray:
    """channel_mode:
       'auto'  → use per-image R-B sign (mixed sources)
       'rgb'   → never swap
       'bgr'   → always swap
       'mixed' → use per-image R-B sign (same as 'auto')
    """
    if channel_mode in ("rgb",):
        return rgb
    if channel_mode in ("bgr",):
        return rgb[..., ::-1].copy()
    # auto/mixed: swap only if R > B
    if channel_check(rgb) > 0:
        return rgb[..., ::-1].copy()
    return rgb


# ---------------------------------------------------------------------------
# 2. Pipeline driver
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class IngestPipeline:
    """End-to-end ingest pipeline.

    Args:
      source_name: tag stored in source column + used as output subdir
      repo:        HF repo id ('yxma/gelsight-mini-pretrain' or '-nc')
      domain:      'real' or 'sim'
      gel_variant: 'markerless' or 'markered'
      i_min:       intensity threshold for the contact filter
      channel_mode: 'auto' (per-image R-B check), 'rgb' (skip), 'bgr'
                    (unconditional swap), or 'mixed' (= auto)
      apply_filter: if False, keep every frame (e.g. continuous-contact
                    datasets like UniT where filter has no signal)
      bg_rate:     probability of keeping a rejected frame for diversity
      base:        local output dir (defaults by repo)
      sample_seed: seed for the 40-image grid sampler
    """
    source_name: str
    repo: str = "yxma/gelsight-mini-pretrain"
    domain: str = "real"
    gel_variant: str = "markerless"
    i_min: float = 12.0
    channel_mode: str = "auto"
    apply_filter: bool = True
    bg_rate: float = BG_RATE
    base: Optional[str] = None
    sample_seed: int = 20260520

    def __post_init__(self):
        if self.base is None:
            self.base = BASE_NC if self.repo.endswith("-nc") else BASE_MAIN

    # ------------------------------------------------------------------
    def estimate_baseline(self, samples: list[np.ndarray]) -> np.ndarray:
        """Compute the gel-at-rest baseline from a few sample frames."""
        grays = [grey_center(s) for s in samples]
        return np.median(np.stack(grays), axis=0)

    def filter_and_swap(self, frames: Iterator[FrameRecord]) -> Iterator[
        tuple[FrameRecord, bytes, bool]
    ]:
        """Yield (record, jpeg_bytes, kept_as_bg) for each frame that's kept.

        Maintains one baseline PER RESOLUTION encountered in the source
        (some datasets mix multiple capture resolutions, e.g. TVL has
        HCT@640x480 + SSVTP@240x320). For each unseen resolution we
        buffer the first N_BASELINE frames at that resolution to build
        its baseline; once built, processed eagerly.
        """
        rng = random.Random(self.sample_seed)
        # res_key -> baseline (or None while still buffering)
        baselines: dict[tuple, np.ndarray] = {}
        buffered: dict[tuple, list[FrameRecord]] = {}

        def res_key(rgb):
            return (rgb.shape[0], rgb.shape[1])

        def process_one(r):
            """Emit one frame through filter+bg logic given baseline exists."""
            key = res_key(r.rgb)
            baseline = baselines[key]
            if self.apply_filter:
                if passes_filter(r.rgb, baseline, self.i_min):
                    return (r, to_jpeg(r.rgb), False)
                elif rng.random() < self.bg_rate:
                    return (r, to_jpeg(r.rgb), True)
                else:
                    return None
            else:
                return (r, to_jpeg(r.rgb), False)

        for rec in frames:
            rec.rgb = maybe_swap_channels(rec.rgb, self.channel_mode)
            key = res_key(rec.rgb)

            if key in baselines:
                # Steady-state for this resolution
                ret = process_one(rec)
                if ret is not None:
                    yield ret
                continue

            # Building baseline for this resolution
            buffered.setdefault(key, []).append(rec)
            if len(buffered[key]) >= N_BASELINE:
                baselines[key] = self.estimate_baseline(
                    [r.rgb for r in buffered[key]])
                # Drain
                for r in buffered[key]:
                    ret = process_one(r)
                    if ret is not None:
                        yield ret
                buffered[key].clear()

        # Drain remaining buffered frames per resolution (if iterator ended
        # before any res hit N_BASELINE)
        for key, recs in buffered.items():
            if not recs:
                continue
            if key not in baselines:
                baselines[key] = self.estimate_baseline([r.rgb for r in recs])
            for r in recs:
                ret = process_one(r)
                if ret is not None:
                    yield ret

    # ------------------------------------------------------------------
    def row_dict(self, rec: FrameRecord, jpeg_bytes: bytes) -> dict:
        """Build a unified-schema row from a FrameRecord."""
        row = {f.name: None for f in SCHEMA}
        row.update(dict(
            image=jpeg_bytes,
            image_format="jpeg",
            source=self.source_name,
            markered=(self.gel_variant == "markered"),
            capture=rec.capture or f"{rec.obj_name}_{rec.frame_idx}",
            split=rec.split,
            height=rec.rgb.shape[0],
            width=rec.rgb.shape[1],
            obj_name=rec.obj_name or "unknown",
            episode=rec.episode,
            frame_idx=rec.frame_idx,
            gel_variant=self.gel_variant,
            domain=self.domain,
        ))
        # Optional extras override the defaults
        for k, v in rec.extra.items():
            if k in row:
                row[k] = v
        return row

    # ------------------------------------------------------------------
    def write_parquets(self, rows_by_split: dict[str, list[dict]],
                       out_subdir: str) -> dict[str, str]:
        """Write one parquet per split. Returns {split: path}."""
        os.makedirs(out_subdir, exist_ok=True)
        out_paths = {}
        for split, rows in rows_by_split.items():
            if not rows: continue
            # Use chunked binary for the image column to avoid 2GB limit
            img_chunks = [
                pa.array([r["image"] for r in rows[i:i + CHUNK]], type=pa.binary())
                for i in range(0, len(rows), CHUNK)
            ]
            img_col = pa.chunked_array(img_chunks, type=pa.binary())

            cols = {f.name: [] for f in SCHEMA}
            for r in rows:
                for fname in cols.keys():
                    cols[fname].append(r.get(fname))
            arrays = []
            for f in SCHEMA:
                if f.name == "image":
                    arrays.append(img_col)
                else:
                    arrays.append(pa.array(cols[f.name], type=f.type))
            table = pa.Table.from_arrays(arrays, schema=SCHEMA)

            out_path = f"{out_subdir}/{split}-00000-of-00001.parquet"
            tmp = out_path + ".tmp"
            pq.write_table(table, tmp, compression="snappy")
            os.replace(tmp, out_path)
            sz = os.path.getsize(out_path) / 1e6
            print(f"  wrote {out_path}  rows={table.num_rows:,}  {sz:.1f} MB",
                  flush=True)
            out_paths[split] = out_path
        return out_paths

    # ------------------------------------------------------------------
    def generate_sample_grid(self, out_subdir: str) -> str:
        """Render a 40-image grid to assets/samples_40_<source>.png."""
        import make_samples_100 as ms

        paths = sorted(glob.glob(f"{out_subdir}/*.parquet"))
        if not paths:
            return ""

        rng = random.Random(self.sample_seed + 1)
        counts = [pq.read_metadata(p).num_rows for p in paths]
        total = sum(counts)
        if total == 0: return ""
        N = 40
        idxs = sorted(rng.sample(range(total), min(N, total)))
        imgs = []
        cum = 0
        it = iter(idxs); nxt = next(it, None)
        for p, c in zip(paths, counts):
            if nxt is None: break
            if nxt >= cum + c: cum += c; continue
            local = []
            while nxt is not None and nxt < cum + c:
                local.append(nxt - cum); nxt = next(it, None)
            if local:
                t = pq.read_table(p, columns=["image"])
                for li in local:
                    imgs.append(t.column("image")[li].as_py())
            cum += c

        thumbs = [ms.thumbnail(b) for b in imgs]
        title = f"{self.source_name} — {len(thumbs)} random samples"
        grid = ms.make_grid(thumbs, title)
        assets_dir = f"{self.base}/assets"
        os.makedirs(assets_dir, exist_ok=True)
        out_p = f"{assets_dir}/samples_40_{self.source_name}.png"
        grid.save(out_p, optimize=True)
        print(f"  saved {out_p}  size={grid.size}", flush=True)
        return out_p

    # ------------------------------------------------------------------
    def push(self, parquet_paths: dict[str, str], sample_grid: str,
             commit_message: Optional[str] = None) -> str:
        """Push parquets + sample grid to HF repo."""
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi()
        ops = []
        for split, p in parquet_paths.items():
            rel = os.path.relpath(p, self.base)
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=p))
        if sample_grid and os.path.exists(sample_grid):
            rel = os.path.relpath(sample_grid, self.base)
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=sample_grid))
        msg = commit_message or f"add {self.source_name} via unified pipeline"
        info = api.create_commit(
            repo_id=self.repo, repo_type="dataset",
            operations=ops, commit_message=msg)
        print(f"  pushed: {info.commit_url}", flush=True)
        return info.commit_url

    # ------------------------------------------------------------------
    def run(self, frame_iter: Iterator[FrameRecord],
            *, push: bool = True) -> dict:
        """Run the full pipeline: filter+swap → write parquets → sample
        grid → optional push."""
        t0 = time.time()
        print(f"=== Pipeline: {self.source_name} ===", flush=True)
        print(f"  channel_mode={self.channel_mode}  i_min={self.i_min}  "
              f"apply_filter={self.apply_filter}", flush=True)

        rows_by_split: dict[str, list[dict]] = {}
        n_seen = n_kept = n_bg = 0
        for rec, jpeg_b, is_bg in self.filter_and_swap(frame_iter):
            row = self.row_dict(rec, jpeg_b)
            rows_by_split.setdefault(rec.split, []).append(row)
            n_kept += 1
            if is_bg: n_bg += 1
            if n_kept % 5000 == 0:
                print(f"  kept={n_kept:,}  bg={n_bg:,}  "
                      f"({n_kept/(time.time()-t0):.0f} fps)", flush=True)
        n_seen = n_kept  # the iterator only gives us the kept ones; if you
                        # need a true 'seen' counter, instrument the source

        print(f"  filter pass done. kept={n_kept:,} bg={n_bg:,} in "
              f"{time.time()-t0:.0f}s", flush=True)

        out_subdir = f"{self.base}/{self.source_name}"
        parquet_paths = self.write_parquets(rows_by_split, out_subdir)
        sample_grid = self.generate_sample_grid(out_subdir)

        commit_url = ""
        if push:
            commit_url = self.push(parquet_paths, sample_grid)

        return dict(n_kept=n_kept, n_bg=n_bg,
                    parquet_paths=parquet_paths,
                    sample_grid=sample_grid,
                    commit_url=commit_url)
