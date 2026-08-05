# gelsight-mini-pretrain — preprocessing

Build pipeline for the HuggingFace dataset `yxma/gelsight-mini-pretrain`
(and its non-commercial counterpart, `yxma/gelsight-mini-pretrain-nc`).

The full behavioural reference — filter rule, per-source baseline recipes,
schema, and known defects in the published data — is
[`docs/PIPELINE.md`](docs/PIPELINE.md). This README is about the repo
itself: layout, install, and how to run something.

## Data (read-only, not in this repo)

| What | Default path | Env override |
|---|---|---|
| Raw upstream | `/media/yxma/Disk1/yuxiang/mini_data` | `GSMP_RAW_ROOT` |
| Published (main licence) | `/media/yxma/Disk1/yuxiang/mini_data_parquet` | `GSMP_PARQUET_MAIN` |
| Published (NC licence) | `/media/yxma/Disk1/yuxiang/mini_data_parquet_nc` | `GSMP_PARQUET_NC` |
| Published (video subset) | `/media/yxma/Disk1/yuxiang/mini_data_parquet_video` | `GSMP_PARQUET_VIDEO` |
| Pipeline output | `/media/yxma/Disk1/yuxiang/gsmp_out` | `GSMP_OUT_ROOT` |

Nothing in `src/gsmp/` or `tools/` writes to `RAW_ROOT` or the `PARQUET_*`
paths — see `src/gsmp/config.py`. They are the ground truth this pipeline
is checked against, not build artifacts.

## Repo layout

```
src/gsmp/          the package under active migration
  runner.py          generic frame-decision engine (baseline -> filter ->
                      dedupe -> budget -> parquet), shared by every
                      migrated source, no per-source branches
  sources/<name>.py  one SourceSpec + iterator per source
  spec.py            SourceSpec dataclass + registry
  schema.py          the 30-column pyarrow schema, published_columns(),
                      has_join_key()
  baseline.py        BaselineStrategy implementations (FirstNFrames,
                      PerCaptureMedian, NoBaseline, ...)
  filters.py         area+intensity contact filter, phash dedupe
  encode.py          JPEG encode
  writer.py          sharded parquet writer (2 GB cap, pickle safety net)
  config.py          all filesystem paths, GSMP_* env overrides
  regress.py         compares a migrated source's kept-key set against
                      the published release
  tools_imin.py       shared code for tools/recover_imin.py

legacy/            original scripts, imported verbatim on 2026-08-04.
                    Still authoritative for every not-yet-migrated source.
                    Nothing is deleted from here until its replacement has
                    a passing regression run.

archive/           19 one-off repair scripts that already ran against the
                    published data (channel-order fixes, label repairs,
                    dedupe/budget passes, release drivers). Kept for
                    provenance, not reuse -- see archive/README.md.

tools/             CLI entry points, see "tools/" below.

docs/              PIPELINE.md, source_tiers.md, imin_from_code.md,
                    imin_recovered.md.

tests/             pytest suite.
```

## Install

Python 3.9. From the repo root:

```bash
pip install -e ".[dev]"
```

This installs `gsmp` (from `src/gsmp/`) in editable mode plus `pytest`.
Runtime dependencies: `numpy`, `pyarrow`, `pillow`, `opencv-python`,
`huggingface_hub`, `tqdm` (see `pyproject.toml`).

## Running a source

Every migrated source (`src/gsmp/sources/<name>.py`) declares a
`SourceSpec` and an iterator that yields `(capture, frame_generator)`
units, plus a `dry_run_keys()` helper that runs the filter without
writing anything.

**Check a migration against the published release** (the primary way
this codebase is validated — see "Current state" below for which sources
actually pass this):

```bash
python tools/regress.py --source tactile_tracking
python tools/regress.py --source unit
```

This calls the source module's `dry_run_keys()`, diffs the resulting
`(capture, frame_idx)` key set against the real published parquet
(`gsmp.regress.published_keys()`), and prints a pass/fail summary. It
never writes to the published tree; `--limit N` caps candidate frames for
a fast smoke check.

**Produce actual output** for a migrated source, use `gsmp.runner.run()`
directly with a `ShardWriter`:

```python
from gsmp.sources.tactile_tracking import SPEC, iter_units
from gsmp.runner import run
from gsmp.writer import ShardWriter

writer = ShardWriter(SPEC.out_dir(), prefix=SPEC.name)
result = run(SPEC, iter_units(), writer=writer)
writer.close()  # flush the final shard
print(result)
```

There is no single top-level "build everything" script yet — that
orchestration (looping every tier-1 `SourceSpec` through `run()` and
`ShardWriter`) hasn't been written as part of this migration and isn't
needed until enough sources are migrated to justify a real release run.
For a not-yet-migrated (tier-2, or tier-1-but-still-`legacy/`) source,
run the actual script named in its module's `notes` / `legacy_entrypoint()`
(e.g. `legacy/convert_feats.py` + `archive/reprocess_feats.py` for
`feats`) directly, as documented in `docs/PIPELINE.md`.

## `tools/`

| Script | What it does |
|---|---|
| `tools/audit_sources.py` | Recomputes the tier-1/tier-2 split from `gsmp.schema.has_join_key()` for every published source. Source of truth for `docs/source_tiers.md`. |
| `tools/recover_imin.py` | Empirically estimates `i_min` per source by resampling the published parquet and reconstructing a baseline. A cross-check, not the primary source — see `docs/imin_from_code.md` for why, and its known failure mode (baseline circularity on contact-only published data). |
| `tools/regress.py` | Compares a migrated source's `dry_run_keys()` output against the published release. See "Running a source" above. |

## Tiers: what "migrated" actually means

Full detail: [`docs/source_tiers.md`](docs/source_tiers.md).

- **tier-1** — published rows carry a usable `(capture, frame_idx)` join
  key, so a rewrite can be checked exactly against the release with
  `tools/regress.py`. **10 sources**: `gelslam`, `tactile_tracking`,
  `real_tactile_mnist`, `feelanyforce`, `threedcal`, `tacquad`, `unit`,
  `sim_tactile_mnist`, `sim_starstruck`, `sparsh`.
- **tier-2** — no join key, so no rewrite can be proven correct. These
  sources are wrapped (a `SourceSpec` added for metadata) but their
  original filtering logic is left untouched, permanently. **3 sources**:
  `feats` (force-gated, `frame_idx` is `NULL` in every published row by
  design), `fota_labeled`, `fota_unlabeled` (both published with the
  older 26-column schema, which has no `frame_idx` column at all).

Being tier-1 is eligibility, not completion. Current state:

- **Migrated and regression-proven**: `tactile_tracking`
  (`PASS exact: 2408 keys identical`) and `unit`
  (`PASS exact: 387 keys identical`).
- **In progress**: `gelslam` — migration underway, regression run not
  yet complete. Treat `legacy/` as authoritative for it until that lands.
- **Not yet migrated**: `real_tactile_mnist`, `feelanyforce`,
  `threedcal`, `tacquad`, `sim_tactile_mnist`, `sim_starstruck`,
  `sparsh` — all still `legacy/`-only.
- **Tier-2, wrapped**: `feats`, `fota_labeled`, `fota_unlabeled` — this
  is their permanent state, not a waypoint.

## Not migrated

`legacy/make_parquet_video.py` and the sequence-preserving video subset at
`mini_data_parquet_video/` (gelslam, real_tactile_mnist, tactile_tracking)
are imported verbatim and left alone. That subset's schema carries four
extra columns (sequence_id, frame_in_seq, sequence_length, fps) that the
30-column `gsmp.schema.SCHEMA` does not model. Supporting it needs a
polymorphic schema layer, which is a separate design question, out of
scope for this repo's migration — tracked in the refactor's planning
docs, not duplicated here to avoid yet another copy that can go stale.

## Testing

```bash
python -m pytest -q
```

76 tests, all read-only against the published/raw trees (no test writes
to `PARQUET_MAIN`, `PARQUET_NC`, `PARQUET_VIDEO`, or `RAW_ROOT`; tests
that need the real published data skip when it isn't present on the
machine).

See also [`docs/PIPELINE.md`](docs/PIPELINE.md) for the filter rule,
per-source baseline recipes, and known defects in the published release
(most notably: `fota_labeled` / `fota_unlabeled` shipped with 26 columns,
not the intended 30).
