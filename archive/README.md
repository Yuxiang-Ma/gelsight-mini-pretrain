# Archived one-off scripts

These ran once, their effects are already baked into the published release,
and they are kept verbatim for provenance -- not for reuse. Nothing here is
imported by `src/gsmp/`.

| Script | What it fixed | Affected sources |
|---|---|---|
| `fix_channel_order.py` | Swapped BGR-stored frames to RGB for the 4 subsets that clustered BGR (fota_unlabeled, unit, faf_force_estimation; per-image conditional swap for sparsh) | fota_unlabeled, unit, faf_force_estimation, sparsh |
| `swap_fota_unlabeled.py` | Applied a direct R<->B swap to fota_unlabeled parquet shards | fota_unlabeled |
| `fix_feats_marker_labels.py` | Added the `gel_variant` column (`black_dot` vs `different` sensor/gel) to FEATS shards | feats |
| `fix_fota_marker_labels.py` | Set the `markered` column per capture from dot-density detection (>=10 valid dark blobs = markered) | fota_labeled, fota_unlabeled |
| `redo_fota_unlabeled.py` | Re-extracted fota_unlabeled from raw JPEGs with looser dedupe (cv2-accelerated) to grow the kept set from 66K to a 200K target | fota_unlabeled |
| `subsample_fota_unlabeled.py` | Stride-subsampled fota_unlabeled per capture (800 train / 200 val) down to ~60K, since it had ballooned to ~70% of the real pool | fota_unlabeled |
| `dedupe_cap_fota.py` | Streaming per-shard phash dedupe + budget cap (200K), avoiding pyarrow's 2 GB chunk overflow | fota_labeled, fota_unlabeled |
| `reprocess_feats.py` | Re-extracted FEATS from raw .npy with a relaxed force filter (\|f_z\| >= 0.4 N, was 0.5) plus 1.5% background keep | feats |
| `reprocess_fota.py` | Re-filtered FoTA labeled/unlabeled parquet in place with the unified area+intensity rule | fota_labeled, fota_unlabeled |
| `reprocess_legacy.py` | Re-filtered shards (e.g. real_tactile_mnist, sim_tactile_mnist, sim_starstruck) in place with the legacy area+intensity rule, per-subset I_MIN | multiple |
| `reprocess_upstream.py` | Re-ingested RTM / sim_tactile_mnist / sim_starstruck from raw upstream parquets (rather than the pre-aggregated ones) and re-filtered, to recover frames the earlier single-frame-per-touch extractor discarded | RTM, sim_tactile_mnist, sim_starstruck |
| `reprocess_v7_zscore.py` | Per-pixel z-score validity filter experiment (superseded by the area+intensity rule) | multiple |
| `rebalance_compose.py` | Applied per-object and per-domain caps to compose the final ~1M-row, 60/40 real/sim balanced dataset | all |
| `finalize*.sh`, `touchandgo_retry_loop.sh` | Release drivers per wave | all |

## Corrections made while archiving (Task 17)

The draft table in the task brief mis-described a few scripts; verified
against each script's own docstring/constants before writing the table
above:

- `fix_feats_marker_labels.py` actually adds a `gel_variant` column
  (`black_dot` vs `different`), not the `markered` column.
- `fix_fota_marker_labels.py` is the one that sets `markered`, via
  per-capture dot-density detection -- the two scripts' effects were
  transposed in the original draft.
- `reprocess_feats.py` re-extracts from raw `.npy` with a relaxed force
  threshold; it does not "re-derive force columns".
- `reprocess_legacy.py` only re-filters existing shards with the legacy
  area+intensity rule; it does not touch `domain` or `markered` (grepped
  the file -- neither name appears).
- `redo_fota_unlabeled.py` grows the *fota_unlabeled* kept set from 66K
  (existing strict-dedupe shards) to a 200K target by re-extracting from
  raw JPEGs with looser dedupe; the "516K raw" figure in the draft belongs
  to a different script (`subsample_fota_unlabeled.py`'s description of
  the aggregated real-tactile pool), not to this one.

## Known docstring/constant mismatch

`reprocess_fota.py`'s docstring claims the unified filter is "A=40, I=15",
but the executable constant is `I_MIN = 10` (line 30) -- the constant is
what actually ran. This kind of drift between documentation and executed
code is exactly what this archive exists to preserve, not paper over.

## Why these were not refactored

The dataset is already published. Rewriting a script whose only job was to
repair data that is now correct buys nothing and risks misrepresenting what
actually happened. Deleting them would lose the record of why the release
looks the way it does.
