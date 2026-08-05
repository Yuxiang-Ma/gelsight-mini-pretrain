# Pipeline Reference

The canonical reference for **how every frame in `yxma/gelsight-mini-pretrain`
got there**. The code that produced the published release is
[`legacy/make_parquet_v2.py`](../legacy/make_parquet_v2.py) plus the
standalone ingest/repair scripts alongside it; it is being migrated
task-by-task into [`src/gsmp/`](../src/gsmp/) (see "Where the code lives"
and "Migration status" below).

The code is the source of truth. This document describes what
`src/gsmp/` (and, for not-yet-migrated sources, `legacy/`) actually does;
where they disagree, the code is right and this document is stale — fix
the document.

Per-source parameters are NOT duplicated here. They live in the
`SourceSpec` at the top of each `src/gsmp/sources/<name>.py`. The previous
version of this file listed thresholds inline, which is how i_min came to
have four different documented values.

This doc previously labelled itself "v4", a version number that stopped
being updated years before `archive/finalize_v9.sh` shipped. It no longer
carries a version number for that reason — check `git log` for this file
and `archive/README.md` for what actually happened between releases.

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
| `I_MIN` | per-source, see each `SourceSpec` | authoritative values are read from the legacy code, file:line cited, in `docs/imin_from_code.md`; `tools/recover_imin.py` / `docs/imin_recovered.md` is an independent empirical cross-check from the published data, not the primary source. Historically this doc alone documented three different global values (10, 12, 15) that matched none of the 13 sources exactly |
| `BG_KEEP_RATE` | **0.015** (1.5%) | of frames that FAIL the filter, kept anyway for VAE-diversity |

Every kept row also carries:
- `domain` ∈ `{real, sim}`
- `markered` ∈ `{True, False}` — gel has tracking dots?

---

## Baseline per source

> This table is a human-readable summary, kept intentionally free of the
> numeric thresholds that belong in each `SourceSpec` (see the I_MIN row
> above). It has been wrong before — two rows below were corrected while
> writing this doc (see the notes under the table) — so treat it as a
> map, not the ground truth. For a migrated (tier-1) source, the
> authoritative description is the `BaselineStrategy` passed to its
> `SourceSpec` in `src/gsmp/sources/<name>.py`; for everything still in
> `legacy/`, it's the iterator that actually produced the published shard
> (see `docs/imin_from_code.md` for which script that is, per source —
> more than one source has a dead second implementation that looks live
> at a glance).

Most sources build the baseline as **the median of frames from the same
capture/round/episode where the gel is most likely to be at rest**.
Source-specific recipe:

| Source | Capture unit | Baseline method |
|---|---|---|
| `fota_labeled` | one (object × pose × side) capture | cross-frame median of a random 30-frame sample |
| `fota_unlabeled` | same | same |
| `threedcal` | global | **not a median** — the single `blank_images/blank.png` shipped with the upstream release, read once (`legacy/make_parquet_v2.py:529-547`) |
| `feats` | per-indenter-shape | cross-capture median within same `indenter`+`indenter_param` group (as read by `legacy/make_parquet_v2.py`'s `iter_feats`; the published data was actually produced by `legacy/convert_feats.py` (raw npy -> parquet, no filter) followed by the force-gated `archive/reprocess_feats.py` (re-extract + filter), which has no pixel baseline at all — see the FEATS caveat below) |
| `gelslam` | one tracking/recon episode | median of first 10 frames of the .avi |
| `tactile_tracking` | one trial | same |
| `real_tactile_mnist` | one parquet row (256 touch videos) | per-touch median of first 5 frames; falls back to cross-touch |
| `feelanyforce` | global | cross-object median (3 imgs x 42 objs) |
| `sim_tactile_mnist` | one digit object row (32 touches) | median across the 32 rendered touches |
| `sim_starstruck` | same | same |
| `tacquad` | per domain (`data_indoor` / `data_outdoor` / `data_fine`) | cross-object median (100-frame random sample), `legacy/ingest_tacquad_full.py`. Not `tacquad_mini`: that's a second, dead `iter_tacquad_mini()` in `make_parquet_v2.py` that never produced published output — see `docs/imin_from_code.md`'s "tacquad conflict" |
| `unit` | global (one zarr array, not per-episode) | median of 120 frames sampled uniformly at random from the whole 11,340-frame array (`random.Random(20260520)`), not a first-N-frames median — see `src/gsmp/sources/unit.py`'s docstring for why this matters for reproducing the published set exactly |

**Caveat — markered sources (FEATS).** Tracking dots make pixel diffing
unreliable. The script that actually produced the published `feats` data
uses a force-based filter `|f_z| >= 0.4 N` + 1.5% bg-keep instead, with no
pixel baseline at all (`archive/reprocess_feats.py:18,113`).

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

## Unified schema (single-view, 30 columns)

> **Not actually unified.** `fota_labeled` and `fota_unlabeled` were
> published with 26 columns, missing `episode`, `frame_idx`, `digit_class`
> and `gel_variant` (93,155 frames, ~11% of the corpus). The released
> README's `concatenate_datasets` example fails because of this.
> `gsmp.schema.LEGACY_26_SOURCES` records it and
> `tests/test_schema.py::test_fota_is_known_to_deviate` pins it.
> Remediation requires republishing those two subsets — see Task 19.

Authoritative definition: `gsmp.schema.SCHEMA` in
[`src/gsmp/schema.py`](../src/gsmp/schema.py). `gsmp.schema.published_columns()`
reads a source's actual shard schema for comparison; use it, don't trust
this table blindly.

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

That's the full 30. The sequence-preserving video subset
(`mini_data_parquet_video/`, built by `legacy/make_parquet_video.py`) is a
**separate schema**, not a superset of this one: it adds `sequence_id`,
`frame_in_seq`, `sequence_length`, `fps` and is not modeled by
`gsmp.schema.SCHEMA` at all. See "Not migrated" in the top-level
`README.md`.

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

All of it lives in one git repository (`gelsight-mini-pretrain`), mirrored
to GitHub:

| Directory | What's there |
|---|---|
| `src/gsmp/` | The package under active migration: `runner.py` (generic frame-decision engine), `sources/<name>.py` (one `SourceSpec` + iterator per source), `spec.py`, `schema.py`, `baseline.py`, `filters.py`, `encode.py`, `writer.py`, `config.py`, `regress.py`, `tools_imin.py`. |
| `legacy/` | The original scripts that produced the published release, imported verbatim on 2026-08-04. Sources move out of here into `src/gsmp/sources/` task by task; nothing is deleted from `legacy/` until its replacement has a passing regression run (`tools/regress.py`). |
| `archive/` | 19 one-off repair scripts that ran once against already-published data (channel-order fixes, label repairs, budget/dedupe passes, release drivers) and are kept for provenance only — see `archive/README.md`. |
| `tools/` | `audit_sources.py` (tier-1/tier-2 assignment), `recover_imin.py` (empirical i_min cross-check), `regress.py` (compares a migrated source's kept-key set against the published release). |
| `docs/` | This file, plus `source_tiers.md`, `imin_from_code.md`, `imin_recovered.md`. |
| `tests/` | The pytest suite (see the reproducibility checklist below). |

**Target end state for the Hugging Face side (Task 20, not yet executed).**
The dataset repo on Hugging Face currently still carries a `scripts/`
copy: an 8-file snapshot from 2026-05-17 whose `make_parquet_v2.py` is
7,770 bytes behind the code that actually produced the release. A copy
with no mechanism keeping it fresh is worse than a link, so once Task 20
runs, that copy is removed and the dataset README points at this
repository instead. Until then, treat the HF-side `scripts/` copy as
stale — it is not what produced the data you're looking at.

---

## Migration status

Full detail and the tiering rationale: `docs/source_tiers.md`. Summary,
current as of this doc:

- **10 tier-1 sources** — published rows carry a usable
  `(capture, frame_idx)` join key, so a migration can be checked against
  the release with `tools/regress.py`: `gelslam`, `tactile_tracking`,
  `real_tactile_mnist`, `feelanyforce`, `threedcal`, `tacquad`, `unit`,
  `sim_tactile_mnist`, `sim_starstruck`, `sparsh`.
- **3 tier-2 sources** — no join key, so nothing can be proven against the
  release; these are wrapped, not rewritten: `feats`, `fota_labeled`,
  `fota_unlabeled`.

Within tier-1, be precise about what's actually been done — "tier-1" is
eligibility for a regression proof, not evidence that one has been run:

- **Migrated and regression-proven**: `tactile_tracking`
  (`python tools/regress.py --source tactile_tracking` →
  `PASS exact: 2408 keys identical`) and `unit`
  (`python tools/regress.py --source unit` →
  `PASS exact: 387 keys identical`). Their code lives in
  `src/gsmp/sources/tactile_tracking.py` and `src/gsmp/sources/unit.py`.
- **In progress**: `gelslam` — a migration exists but is mid-fix and its
  regression run has not completed; do not treat it as proven. Its
  `legacy/` implementation is still what's authoritative for the
  published data.
- **Not yet migrated** (still `legacy/` only): `real_tactile_mnist`,
  `feelanyforce`, `threedcal`, `tacquad`, `sim_tactile_mnist`,
  `sim_starstruck`, `sparsh`.

Tier-2 sources cannot get a regression proof by construction (no join
key), so "wrapped" (a `SourceSpec` added, filtering logic untouched — see
`src/gsmp/sources/feats.py`, `fota_labeled.py`, `fota_unlabeled.py`) is
the ceiling for them, not a step on the way to "migrated."

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
