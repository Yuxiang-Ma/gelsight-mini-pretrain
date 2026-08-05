# Pipeline Reference (v4)

The canonical reference for **how every frame in `yxma/gelsight-mini-pretrain`
got there**. Source code: [`scripts/make_parquet_v2.py`](scripts/make_parquet_v2.py).

This file is the single source of truth — if the pipeline code disagrees
with this doc, the doc wins and the code should be updated to match.

---

## What this corpus is — and what views each frame has

This is a **single-view tactile RGB corpus** for GelSight Mini VAE/SSL
pretraining. Each row stores **one tactile image** (`image` column, JPEG)
from a GelSight Mini sensor.

**Other camera views (external scene cameras, depth maps, or paired
sensors) are NOT carried through.** This is deliberate: the corpus
goal is tactile representation learning, not multimodal alignment.

### What multi-view raw data actually exists upstream

If we ever want to extend to multi-view, here's what's available
(verified by inspection of `mini_data/`):

| Source | Tactile views in raw | Non-tactile views in raw | Currently kept |
|---|---|---|---|
| `gelslam` | 1 (`gelsight.avi`) | — | tactile only |
| `tactile_tracking` | 1 (normalflow .avi) | — | tactile only |
| `real_tactile_mnist` | 1 (sensor_video column) | — | tactile only |
| `sim_tactile_mnist` | 1 (sensor_image column) | — | tactile only |
| `sim_starstruck` | 1 (sensor_image column) | — | tactile only |
| `feats` | 1 (.tar shards) | — | tactile only |
| `fota_labeled` | 2 separate captures: `_left`, `_right` | — | each side as its own row |
| `fota_unlabeled` | 2 separate captures: `_left`, `_right` | — | each side as its own row |
| `threedcal` | 1 (probe_images/) | — | tactile only |
| `feelanyforce` | 1 (tactile/) + 1 (tactile_nobg/) | **1 (depth/ .npy)** | tactile only |
| `unit` | 1 (gelsight zarr) | — | tactile only |
| **`tacquad_mini`** | **4** (`gelsight`, `digit`, `duragel`, `tac3d`) | **4 external scene cams** (`img_*`) | **only `gelsight` tactile** |

### Multi-view extension scope (if/when requested)

The only source where adding views materially changes the corpus is
**TacQuad**: 4 tactile sensors + 4 external scene cameras per touch.
Adding these would:

- Mix 3 non-Mini tactile sensors (DIGIT, DuraGel, Tac3D) — **physically
  different** sensors with different gel chemistry, lighting, optics.
  These are out-of-distribution for a "Mini pretraining" corpus.
- Add 4 external RGB scene cams of the gripper-on-object — useful for
  cross-modal pretraining (vision↔tactile) but NOT for tactile-only VAE.

Recommended scope additions, in order of value-to-Mini-pretraining:

1. **TacQuad `img_gelsight`** (scene camera paired with the Mini view):
   gives vision-tactile pairs for joint training. ✓ in-scope for Mini.
2. **FAF depth `.npy`** (per-frame depth map from indenter rig): gives a
   tactile→depth supervision signal. ✓ in-scope for Mini.
3. **FoTA left+right pairing**: link `_left` and `_right` rows of the
   same (object, init_pose) into one paired row. ✓ purely Mini.
4. **TacQuad non-Mini tactile** (`digit`, `duragel`, `tac3d`): only
   if we redefine the corpus as "multi-tactile-sensor pretraining."
   ✗ probably out-of-scope.

A concrete multi-view schema is sketched at the bottom of this doc; see
"Proposed multi-view schema (not yet implemented)."

---

## TL;DR — the validity rule (single-view)

Every frame in the dataset has passed the **unified area + intensity filter**
against a per-capture baseline:

```
1. Per-capture/round/episode baseline  =  source-specific gel-at-rest reference
2. pixel_diff      =  | frame - baseline |          (central 50% crop, greyscale)
3. mask            =  pixel_diff > PIXEL_THRESH     (10 grey-levels, sensor noise)
4. contact_area    =  mask.sum()                    (lit-pixel count)
5. contact_int     =  pixel_diff[mask].mean()       (avg deformation in lit pixels)

KEEP iff   contact_area >= A_MIN  AND  contact_int >= I_MIN
ELSE       keep with probability BG_KEEP_RATE       (background-diversity)
```

| Constant | Value | What it does |
|---|---|---|
| `PIXEL_THRESH` | **10** grey-levels | floor for what counts as a "lit" pixel; drops sensor noise |
| `A_MIN` | **40** pixels | smallest imprint we'll accept (~0.2% of the central crop) |
| `I_MIN` | **15** grey-levels (10 for FoTA) | average per-lit-pixel deformation that constitutes a real imprint |
| `BG_KEEP_RATE` | **0.015** (1.5%) | of frames that FAIL the filter, kept anyway for VAE-diversity |

Every kept row also carries:
- `domain` ∈ `{real, sim}`
- `markered` ∈ `{True, False}` — gel has tracking dots?

---

## Baseline per source

The baseline is **the median of frames from the same capture/round/episode
where the gel is most likely to be at rest**. Source-specific recipe:

| Source | Capture unit | Baseline method |
|---|---|---|
| `fota_labeled` | one (object × pose × side) capture | cross-frame median of a random 30-frame sample |
| `fota_unlabeled` | same | same |
| `threedcal` | global | cross-image median over a random 200-frame sample |
| `feats` | per-indenter-shape | cross-capture median within same `indenter`+`indenter_param` group |
| `gelslam` | one tracking/recon episode | median of first 10 frames of the .avi |
| `tactile_tracking` | one trial | same |
| `real_tactile_mnist` | one parquet row (256 touch videos) | per-touch median of first 5 frames; falls back to cross-touch |
| `feelanyforce` | global | cross-object median (3 imgs x 42 objs) |
| `sim_tactile_mnist` | one digit object row (32 touches) | median across the 32 rendered touches |
| `sim_starstruck` | same | same |
| `tacquad_mini` | per data_* split | cross-touch median (60-frame random sample) |
| `unit` | per zarr episode | first-5-frame median |

**Caveat — markered sources (FEATS).** Tracking dots make pixel diffing
unreliable. FEATS uses force-based filter `|f_z| >= 0.4 N` + 1.5% bg-keep
instead.

---

## Beyond the validity filter — additional curation

### Per-frame Bernoulli sampling inside a "touch window"

| Source | Window | K (sampling probability) |
|---|---|---|
| `real_tactile_mnist` | upstream `touch_start..touch_end` (~6 frames) | **0.30** -> ~2 frames per touch |
| `gelslam`, `tactile_tracking` | full video post-validity | 1.0 |
| all others | already 1 frame per "event" | 1.0 |

### Perceptual-hash dedupe

| Source | PHASH_DIST | LOOKBACK | Notes |
|---|---:|---:|---|
| `gelslam`, `tactile_tracking` | 4 | 30 | strict |
| `fota_labeled` | 4 | 30 | strict |
| `fota_unlabeled` | **1** | **5** | **loose** to retain 200K out of 516K raw |
| sim / RTM | n/a | n/a | each upstream frame already distinct |

### Per-source frame budget

Cap each source at **`BUDGET = 200,000` rows**. Above-budget sources are
stride-subsampled uniformly across the post-validity, post-dedupe stream.

---

## Image encoding

JPEG quality 92, **original native resolution**:

| Source | Resolution |
|---|---|
| FoTA (labeled / unlabeled) | 640 x 480 |
| All others | 320 x 240 |

No resizing.

---

## Fast pipeline (cv2-based, since v4)

For raw-JPEG sources (FoTA), we swapped PIL -> cv2 + crop-first greyscale:

| Stage | Pillow (v1-v3) | cv2 (v4) | Speedup |
|---|---:|---:|---:|
| JPEG decode | 2.85 ms | 1.40 ms | 2.0x |
| Grey-center conversion | 6.80 ms | 0.70 ms | 9.7x |
| JPEG encode | 2.22 ms | 1.20 ms | 1.9x |
| **Per-frame total** | **~12 ms** | **~3.3 ms** | **3.6x** |

Worker count is 8 (one per hyperthread on 4-core i7-6700K) with each
subprocess pinned to single-thread cv2/numpy via:

```python
cv2.setNumThreads(1)
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
```

`fota_unlabeled` full reprocess (516K raw -> 200K kept): ~32 min.

---

## Safety: pickle the row list before parquet write

`pyarrow` has a 2 GB binary-chunk overflow when concatenating ~470K
rows x ~50 KB JPEGs. Mitigation:

1. Cap each shard at `SHARD_ROWS = 60_000` (~1.5-2 GB).
2. **Pickle `all_rows` to disk BEFORE the parquet write.** If write
   crashes (f-string bug, schema mismatch, OOM), pickle is a complete
   recovery point.
3. Delete `_all_rows.pkl` only after every shard is on disk.

---

## Unified schema (current, single-view, 30 columns)

| Column | Type | Description |
|---|---|---|
| `image` | binary | JPEG bytes (the **one** tactile view) |
| `image_format` | string | always `"jpeg"` |
| `source` | string | which subset |
| `domain` | string | `"real"` or `"sim"` |
| `markered` | bool | gel has tracking dots? |
| `gel_variant` | string | FEATS gel sub-type |
| `capture` | string | per-source capture id |
| `split` | string | upstream split |
| `height`, `width` | int32 | image dimensions |
| `obj_name`, `init_pose`, `side` | str/int/str | object x pose x left/right |
| `x_mm`, `y_mm`, `z_mm` | float32 | probe / EEF position |
| `quat_x..w` | float32 | EEF orientation |
| `indenter`, `indenter_param` | string | FEATS probe |
| `f_x`, `f_y`, `f_z` | float32 | FEATS / FAF force vector |
| `grid_z_max`, `grid_z_mean` | float32 | FEATS depth-grid summary |
| `episode` | string | episode id |
| `frame_idx` | int32 | frame index within source video |
| `digit_class` | int32 | RTM / sim digit 0-9 |
| `sequence_id`, `frame_in_seq`, `sequence_length`, `fps` | str/int/int/float | **video subset only** |

---

## Proposed multi-view schema (NOT YET IMPLEMENTED)

If we extend to carry paired views, recommended approach:

### Option A - nullable secondary-view columns

```
image                 : tactile RGB (Mini)            [current]
image_format          : "jpeg"                        [current]

# NEW
secondary_image       : binary or null
secondary_image_format: string or null   ("jpeg" | "png" | "npy_f32")
secondary_view_name   : string or null   ("scene_cam" | "depth" | "ext_rgb")
secondary_height      : int32 or null
secondary_width       : int32 or null
```

Pros: one-row paired reads. Cons: schema bloat (5 null cols for ~95% of rows).

### Option B - multi-row + `multiview_group_id` [RECOMMENDED]

Each upstream timestep emits N rows, all sharing a `multiview_group_id`.
`view_name` distinguishes views within a group.

```
# NEW
multiview_group_id    : string  ("<source>:<capture>:<frame_idx>")
view_name             : string  ("tactile_mini" | "scene_cam" | "depth_npy")
```

Pros: flexible (1..N views), backwards-compatible (filter
`view_name == "tactile_mini"` for current behavior). Cons: requires
self-join for paired access.

### Initial multi-view additions if we extend

| Source | Views to emit | Added rows |
|---|---|---:|
| TacQuad (Mini-only) | `tactile_mini` + `scene_cam` | +~4K |
| FAF | `tactile_mini` + `depth_npy` | +~48K |
| FoTA | `tactile_mini` only | 0 |
| All others | `tactile_mini` only | 0 |

Total growth: **~52K rows = ~5% of current 1.1M**.

---

## Where the code lives

```
scripts/
  make_parquet_v2.py        # main pipeline
  redo_fota_unlabeled.py    # v4 fast cv2 reprocess (raw JPEG -> parquet)
  reprocess_fota.py         # v3 in-place post-filter
  dedupe_cap_fota.py        # phash dedupe + budget cap, streaming
  parallel_rtm.py           # multiprocessing RTM
  parallel_sim.py           # multiprocessing sim_*
  make_parquet_video.py     # video subset (sequence-preserving)
  make_samples_100.py       # sample grids for assets/
  make_analytical_plots.py
  make_pie_charts.py
  probe_*.py
```

---

## Reproducibility checklist

Before pushing any new release:

- [ ] every row has `domain` set
- [ ] every row has `markered` set
- [ ] every kept frame passes area+intensity rule (or is in 1.5% bg bucket)
- [ ] sample grids show real contact (no empty-gel grids)
- [ ] `composition.png`, `summary_pies.png`, `combined_overview.png` reflect current counts
- [ ] `SOURCES.md` per-source counts match parquet files
- [ ] `README.md` totals match
- [ ] safety pickles `_all_rows.pkl` deleted after successful shard writes
