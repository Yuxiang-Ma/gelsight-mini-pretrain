# Recovered i_min per source — empirical cross-check, NOT the primary source

**The authoritative per-source i_min values live in `docs/imin_from_code.md`.**
Every legacy ingest path hardcodes its own explicit `I_MIN` in the code that
actually produced the published files; that table was derived by reading
those scripts directly, with file:line citations. This document is a
*cross-check* of that table, built the other way round: empirically
estimating i_min purely from the published (already-filtered) data, with no
access to the original code's actual baseline.

**Known limitation of this method:** it cannot work for any source whose
published frames are contact-only relative to the grouping key this tool has
available (`capture`). The estimator needs a genuine no-contact reference to
diff against; when it instead builds that reference from the median of
already-kept (already contact-positive) frames — because a capture group is
tiny (down to a single row) or because most rows in a group are legitimately
in contact — the reconstructed "baseline" partially or fully resembles a
contact frame itself, and the recovered intensity is an artifact, not a
threshold. See "Degenerate rows" below for exactly which sources hit this,
and `docs/imin_from_code.md`'s `unit` investigation for a case where the
artifact was non-degenerate (a plausible-looking wrong number, not an
obvious zero) and required a deeper check.

Where this method's precondition holds (capture groups large enough, and
predominantly non-contact), it corroborates the code-derived values closely
— see "Corroboration" below. Produced by
`python tools/recover_imin.py --all --sample 2000`, run against the
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

**Corrected bottom line for Task 12 (superseding this file's original
framing):** i_min is *not* unknown for the 9 "degenerate" or "ambiguous"
sources above — `docs/imin_from_code.md` has a code-derived, file:line-cited
value for every one of them except `feats` (which never used an intensity
filter at all; it is gated on `f_z` force, by design, not something to
recover here). Do not route sources to a "no known i_min" tier just because
this empirical tool returned a degenerate or ambiguous reading — that
reading is a limitation of reconstructing a baseline from contact-only data,
not evidence the value is unknown. Task 12 should consume
`docs/imin_from_code.md` as the source of truth; treat this file purely as
corroborating (or, for `unit`, contradicting-and-investigated) evidence.

Rows where this tool's reading is non-degenerate and corroborates the code
value: `gelslam` (p01 11.05 / p05 11.46 vs. code 10), `fota_labeled` (10.78 /
11.39 vs. 10), `fota_unlabeled` (11.20 / 11.86 vs. 10), and `sparsh` at p05
only (12.50 vs. 12 — p01 is contaminated, see the anchor-check note above).

Rows where this tool's reading is degenerate or contradicts the code value
(all explained above and in `docs/imin_from_code.md`, all with a real
code-derived value to use instead): `tactile_tracking` (code 10; tool
"ambiguous"), `real_tactile_mnist` (code 15; tool degenerate-zero,
one-row-per-capture), `feelanyforce` (code 10; tool degenerate-zero,
baseline circularity), `threedcal` (code 10; tool degenerate-zero, baseline
circularity), `tacquad` (code 12; tool degenerate-zero, baseline
circularity), `sim_tactile_mnist` (code 15; tool degenerate-zero, baseline
circularity), `sim_starstruck` (code 15; tool degenerate-zero, baseline
circularity), `unit` (code 12; tool reads a non-degenerate but wrong ~19.1 —
see the dedicated investigation in `docs/imin_from_code.md`), `feats` (no
code i_min exists; tool degenerate-zero for the same one-row-per-capture
reason as `real_tactile_mnist`, consistent with there being no real
threshold to find).
