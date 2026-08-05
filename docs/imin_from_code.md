# i_min per source, as read from the legacy code that produced the published data

This is the authoritative table. Every legacy ingest path hardcodes its own
explicit `I_MIN` (or, for `feats`, a different filter entirely) — the earlier
belief that i_min was "unknowable from the code" was only true of the single
*global default* (`I_MIN_DEFAULT = 10` at `make_parquet_v2.py:51`, used only
as a fallback for sources not in `VALIDITY_THRESH`). Per-source values were
present all along. `docs/imin_recovered.md` is the empirical cross-check
against this table, not the other way round.

Each row below was verified by reading the actual line, not by grep alone;
where a script's docstring disagreed with its own executable constant, the
constant wins (noted explicitly).

| published source | I_MIN | file : line | quoted line |
|---|---|---|---|
| `gelslam` | 10 | `legacy/make_parquet_v2.py:53` | `"gelslam":           dict(A_min=40, I_min=10),` |
| `tactile_tracking` | 10 | `legacy/make_parquet_v2.py:54` | `"tactile_tracking":  dict(A_min=40, I_min=10),` |
| `real_tactile_mnist` | 15 | `legacy/make_parquet_v2.py:231` | `I_MIN = 15           # RTM uses stricter I (digit imprints are inherently subtle)` |
| `feelanyforce` | 10 | `legacy/make_parquet_v2.py:319` | `A_MIN, I_MIN = 40, 10` |
| `sim_tactile_mnist` | 15 | `legacy/make_parquet_v2.py:379` | `I_MIN = 15           # stricter I to match RTM (sim imprints also vary in strength)` (shared body, called from `iter_sim_tactile_mnist` at :433) |
| `sim_starstruck` | 15 | `legacy/make_parquet_v2.py:379` | same shared body (`_iter_sim_parquet_filtered`), called from `iter_sim_starstruck` at :447 |
| `threedcal` | 10 | `legacy/make_parquet_v2.py:538` | `A_MIN, I_MIN = 40, 10` |
| `tacquad` | 12 | `legacy/ingest_tacquad_full.py:50` | `I_MIN = 12` — see "tacquad conflict" below |
| `unit` | 12 | `legacy/ingest_unit.py:44` | `I_MIN = 12` — see "unit conflict" below |
| `sparsh` | 12 | `legacy/ingest_sparsh.py:46` | `I_MIN = 12` (the calibration anchor used in Task 10) |
| `fota_labeled` | 10 | `legacy/reprocess_fota.py:30` | `I_MIN = 10` (script docstring at line 13 says "unified A=40, I=15" — stale; the executable constant is 10 and is what actually ran) |
| `fota_unlabeled` | 10 | `legacy/reprocess_fota.py:30` (same script, `python reprocess_fota.py fota_unlabeled`); corroborated independently by `legacy/redo_fota_unlabeled.py:63: I_MIN = 10` | `I_MIN = 10` |
| `feats` | **N/A — no intensity filter** | `legacy/reprocess_feats.py:18,113` | `FZ_THRESH = 0.4` ... `passes = abs(fz) >= FZ_THRESH` — FEATS is gated on the physical `f_z` force-sensor reading, not on image contact intensity. There is no `I_MIN` to recover for this source, by design. |

Not part of the 13-source scope for this task, but present in the legacy
tree with their own `I_MIN` (noted for completeness, not verified in depth):
`legacy/ingest_touchandgo.py` and `legacy/ingest_tvl.py` — neither
`touchandgo` nor `tvl` is a published source in either repo today.

## Dead code found while deriving this table

`make_parquet_v2.py:589` defines `iter_faf_force_estimation()` with
`A_MIN, I_MIN = 40, 10`, registered in `SOURCE_ITERS` as
`"faf_force_estimation"`. Its docstring says it reads
`markerless_nc/SparshGelSight` (the same raw tree `ingest_sparsh.py`
consumes) and would write a source called `faf_force_estimation`. **No such
published directory exists** in either `mini_data_parquet/` or
`mini_data_parquet_nc/` — only `sparsh` does, produced by the standalone
`ingest_sparsh.py` (`I_MIN=12`). This iterator was never the path that
produced published data; it reads as an earlier attempt at ingesting the
Sparsh raw data, superseded by `ingest_sparsh.py` before publication. It is
listed here only so a future reader who greps `I_MIN` in `make_parquet_v2.py`
does not mistake it for a live source.

## Conflict: `unit` — code says 12, empirical recovery says ~19.1

`legacy/ingest_unit.py` is unambiguously the script that produced the
published file: its `OUT_BASE`/output path
(`mini_data_parquet/unit/train-00000-of-00001.parquet`) matches the single
published shard exactly, and its `I_MIN = 12` (line 44) is the literal
constant compared against in the filter (line 103:
`passes = (area >= A_MIN) and (inten >= I_MIN)`).

Task 10's empirical tool recovered `p01 ≈ 19.1` for `unit` instead, from
only 387 published rows, all sharing one hardcoded `capture` value
(`"3D_pose_gelsight"`, line 118 — the field is a constant, not a real
per-recording grouping). With every published row in a single capture
group, the recovery tool's per-capture baseline is the median of the 387
*already-kept, already-contact* frames — the same baseline-circularity
failure mode documented for `tacquad`/`threedcal`/etc. in
`docs/imin_recovered.md`, just less severe here (it produced a shifted,
non-degenerate floor rather than collapsing to zero), presumably because the
387 kept poses are spread across varied grip positions.

I did not stop there, since 387 rows makes "it's just sampling noise"
tempting to wave through. I reproduced `ingest_unit.py`'s *actual* baseline
construction directly from the raw zarr
(`/media/yxma/Disk1/yuxiang/mini_data/markerless/UniT/UniT_dataset/3D_pose_gelsight/replay_buffer.zarr`,
120 frames sampled with the same `random.Random(20260520)` seed used in the
script) and applied it to the 387 published frames:

- Of the 120 raw baseline-sample frames, **98.3% have `rgb.mean() < 10`**
  (near-black — the script's own docstring already flags that ~86% of the
  raw recording is LED-off/near-black). The median baseline is therefore
  itself essentially a near-black frame.
- Diffing the 387 published (bright, contact-positive, already
  near-black-excluded) frames against that near-black baseline gives
  `min/p01/p05/median intensity ≈ 112.2 / 112.5 / 113.0 / 115.2` — an order
  of magnitude above both `12` and the tool's `19.1`, and every single row
  has `area == 0` fraction `0.0` (every pixel of every frame "differs" from
  a near-black reference).

**Conclusion:** `I_MIN = 12` is the literal value in the code, and I record
it as the authoritative parameter in the table above because that is what
the running filter compared against. But independent reproduction shows
this comparison was effectively **non-binding** for `unit`: against a
baseline contaminated by ~86-98% near-black samples, almost any adequately
lit frame clears `intensity >= 12` trivially, so the near-black exclusion
(`rgb.mean() < 10`, applied to candidates before the filter) and the
`A_MIN=40` pixel-count gate did the real discriminating work, not `I_MIN`.
Neither my empirical tool's `19.1` nor my zarr reproduction's `~112` should
be read as "the true i_min" — both are baseline-reconstruction artifacts in
different directions (kept-only-median vs. near-black-contaminated-median).
**Recommendation for Task 12: record 12 as the code-stated value for
`unit`, but flag it as weakly-evidenced** — it was not the operative
constraint on this source, unlike e.g. `gelslam` or `sparsh` where the
empirical p01/p05 corroborate the code value directly.

## Conflict: `tacquad` — two legacy code paths, only one published

Two different scripts define a TacQuad ingester with different `I_MIN`:

- `iter_tacquad_mini()` in `legacy/make_parquet_v2.py:457-526`,
  `A_MIN, I_MIN = 40, 10` (line 471), registered in `SOURCE_ITERS` as
  `"tacquad_mini"`, writes `source="tacquad_mini"`, splits by *scene*
  (`data_indoor` / `data_outdoor` / `data_fine_grained`, from directory
  structure).
- `legacy/ingest_tacquad_full.py`, `I_MIN = 12` (line 50), a standalone
  script, writes `source="tacquad"`, one shard per domain named exactly
  `tacquad/{domain}-00000-of-00001.parquet` for
  `domain in ("data_indoor", "data_outdoor", "data_fine")`.

`ingest_tacquad_full.py`'s own docstring settles it:
`"Replaces the existing tacquad_mini/ which only had 4,289 rows."` — i.e.
`tacquad_mini` was the earlier, deprecated ingestion, explicitly superseded.

I confirmed this against the actual published data rather than trusting the
docstring alone:

```
/media/yxma/Disk1/yuxiang/mini_data_parquet/tacquad/data_fine-00000-of-00001.parquet     2,898 rows
/media/yxma/Disk1/yuxiang/mini_data_parquet/tacquad/data_indoor-00000-of-00001.parquet   5,363 rows
/media/yxma/Disk1/yuxiang/mini_data_parquet/tacquad/data_outdoor-00000-of-00001.parquet  3,934 rows
                                                                              TOTAL      12,195 rows
```

- Shard filenames (`data_indoor-00000-of-00001.parquet` etc.) match
  `ingest_tacquad_full.py`'s `out_path` pattern
  (`f"{OUT_BASE}/{domain}-00000-of-00001.parquet"`) exactly; `iter_tacquad_mini`
  does not produce per-domain shards at all — it is a single generator
  consumed by the shared `process()` pipeline elsewhere in
  `make_parquet_v2.py`, and would land under a `tacquad_mini/` output
  directory, which does not exist in the published tree.
- `12,195` exactly matches `mini_data_parquet/README.md:132`
  (`` `tacquad` | 12,195 | indoor/outdoor/fine | markerless | 181 objects ``)
  and `SOURCES.md:557-588`, both of which describe the `tacquad` config with
  the same three splits and the same per-split counts as measured above.
  `181 objects` matches `ingest_tacquad_full.py`'s own header comment
  (101 + 50 + 30 = 181 objects across the three domains).

**Conclusion: `ingest_tacquad_full.py` (`I_MIN = 12`) produced the published
`tacquad` config. `iter_tacquad_mini()` (`I_MIN = 10`) is dead code,
superseded before publication and never reached the released files.**
`I_MIN = 12` is recorded as authoritative for `tacquad` in the table above.

## Corroboration against Task 10's empirical numbers

Where the empirical tool produced a non-degenerate reading (its own
`docs/imin_recovered.md` documents which 9 of 13 sources did not), it lines
up with this table:

| source | code I_MIN | empirical p01 | empirical p05 | agreement |
|---|---|---|---|---|
| `gelslam` | 10 | 11.05 | 11.46 | close |
| `fota_labeled` | 10 | 10.78 | 11.39 | close |
| `fota_unlabeled` | 10 | 11.20 | 11.86 | close |
| `sparsh` | 12 | 0.00 (contaminated) | 12.50 | close at p05 |
| `unit` | 12 | 19.12 | 20.12 | does not agree — see above |

The four sources that agree (`gelslam`, `fota_labeled`, `fota_unlabeled`,
`sparsh` via p05) are exactly the ones where the recovery tool did not hit
either failure mode described in `docs/imin_recovered.md` (single-frame
capture groups, or a capture-group median dominated by already-contact
frames). That the two documents corroborate each other everywhere the
empirical method's preconditions hold is itself evidence the method (and
this table) are both sound; `unit` is the one case where the empirical
preconditions were violated in a way that produced a plausible-looking but
wrong non-degenerate number instead of an obviously-degenerate zero, which
is why it needed the deeper zarr-level check above rather than a one-line
"trust the code" dismissal.
