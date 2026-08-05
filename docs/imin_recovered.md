# Recovered i_min per source

Produced by `python tools/recover_imin.py --all --sample 2000`, run against the
read-only published parquet trees
(`/media/yxma/Disk1/yuxiang/mini_data_parquet{,_nc}`). No shard was written,
moved, or modified.

```
source                      n     min     p01     p05  verdict
gelslam                  2000    0.00   11.05   11.46  likely i_min = 11.1
tactile_tracking         2000    0.00    0.00   10.17  ambiguous
real_tactile_mnist       2000    0.00    0.00    0.00  likely i_min = 0.0 (no bg contamination detected)
feelanyforce             2000    0.00    0.00    0.00  likely i_min = 0.0 (no bg contamination detected)
threedcal                2000    0.00    0.00    0.00  likely i_min = 0.0 (no bg contamination detected)
tacquad                  2000    0.00    0.00    0.00  likely i_min = 0.0 (no bg contamination detected)
unit                      387   18.50   19.12   20.12  likely i_min = 19.1
sim_tactile_mnist        2000    0.00    0.00    0.00  likely i_min = 0.0 (no bg contamination detected)
sim_starstruck           2000    0.00    0.00    0.00  likely i_min = 0.0 (no bg contamination detected)
feats                    1363    0.00    0.00    0.00  likely i_min = 0.0 (no bg contamination detected)
fota_labeled             2000    0.00   10.78   11.39  likely i_min = 10.8
fota_unlabeled           2000   10.39   11.20   11.86  likely i_min = 11.2
sparsh                   2000    0.00    0.00   12.50  ambiguous
```

No source raised an exception; all 13 rows completed (`unit` only has 387
published rows total, so it sampled all of them instead of 2000).

## Human review notes (read before consuming these numbers downstream)

**Anchor check — sparsh vs. the known I_MIN=12 (`legacy/ingest_sparsh.py`):**
sparsh's `p01` came back `0.00`, 12 points off the known value, and the tool's
own verdict logic calls it `ambiguous`. Per the task's ambiguity resolution,
that gap is diagnostic of the *method*, not of sparsh. Investigating: sparsh's
`p05` is `12.50` — within 0.5 of the known 12. Per-capture diagnostics (see
task report) show 2-6% of frames in each capture group read `area=0`
(`intensity=0.0`) against the recomputed per-capture-median baseline, i.e.
the actual background-keep contamination in this reconstruction is higher
than the 1.5% the estimator's percentile choice implicitly assumes. That
pushes the 1st percentile into the contaminated band while the 5th
percentile still lands on the real floor. **Conclusion: the recovery
correctly locates i_min≈12 for sparsh at p05; p01 is not a reliable readout
whenever per-capture contamination exceeds ~1%.** This is a property of the
fixed-percentile estimator applied verbatim per the task brief, not
something to be re-tuned in this task.

**Degenerate `0.00/0.00/0.00` rows are not "i_min = 0":**
`real_tactile_mnist`, `feelanyforce`, `threedcal`, `tacquad`,
`sim_tactile_mnist`, `sim_starstruck`, and `feats` all report a verdict of
`likely i_min = 0.0`, but this is a sampling/baseline artifact, not a
recovered threshold:
- `real_tactile_mnist` and `feats` have exactly one row per distinct
  `capture` value (median group size = 1 for every group), so the
  "per-capture baseline" is the frame itself; every diff is trivially zero.
- `tacquad`, `threedcal`, `sim_tactile_mnist`, `sim_starstruck` have larger
  groups but most rows in the published data are themselves already
  contact-positive (the parquet only contains *kept* frames). Building the
  baseline as the median of an already-mostly-contact group makes the
  baseline itself look like "a typical contact frame," so genuine contact
  frames diff against it as if there were no contact (area=0). This is
  baseline circularity inherent to reconstructing a no-contact reference
  from post-filter (contact-only) data, not evidence that these sources used
  i_min=0.

**Bottom line for Task 12 tiering:** treat the following 9 sources as having
NO usable recovered i_min (same bucket as literal "ambiguous", go to
tier-2/verbatim-wrap, do not guess a value): `tactile_tracking`, `sparsh`,
`real_tactile_mnist`, `feelanyforce`, `threedcal`, `tacquad`,
`sim_tactile_mnist`, `sim_starstruck`, `feats`. Only 4 sources produced a
non-degenerate floor with p01 and p05 close together: `gelslam` (~11.1),
`unit` (~19.1), `fota_labeled` (~10.8), `fota_unlabeled` (~11.2).
