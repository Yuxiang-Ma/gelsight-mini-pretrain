---
license: cc-by-4.0
task_categories:
  - image-classification
  - feature-extraction
  - image-to-image
tags:
  - tactile
  - gelsight
  - gelsight-mini
  - robotics
  - tactile-sensing
  - pretraining
  - self-supervised
size_categories:
  - 100K<n<1M
pretty_name: GelSight Mini Pretrain
configs:
  - config_name: fota_labeled
    data_files:
      - split: train
        path: fota_labeled/train-*.parquet
      - split: val
        path: fota_labeled/val-*.parquet
  - config_name: fota_unlabeled
    data_files:
      - split: train
        path: fota_unlabeled/train-*.parquet
  - config_name: threedcal
    data_files:
      - split: train
        path: threedcal/train-*.parquet
  - config_name: feats
    data_files:
      - split: train
        path: feats/train-*.parquet
      - split: val
        path: feats/val-*.parquet
      - split: test
        path: feats/test-*.parquet
      - split: test_diff_sensor_new_gel
        path: feats/test_diff_sensor_new_gel-*.parquet
      - split: test_diff_sensor_old_gel
        path: feats/test_diff_sensor_old_gel-*.parquet
      - split: test_unknown_indenters
        path: feats/test_unknown_indenters-*.parquet
  - config_name: gelslam
    data_files:
      - split: train
        path: gelslam/train-*.parquet
      - split: recon
        path: gelslam/recon-*.parquet
  - config_name: tactile_tracking
    data_files:
      - split: train
        path: tactile_tracking/train-*.parquet
  - config_name: real_tactile_mnist
    data_files:
      - split: train
        path: real_tactile_mnist/train-*.parquet
      - split: test
        path: real_tactile_mnist/test-*.parquet
  - config_name: feelanyforce
    data_files:
      - split: train
        path: feelanyforce/train-*.parquet
  - config_name: sim_tactile_mnist
    data_files:
      - split: train
        path: sim_tactile_mnist/train-*.parquet
      - split: test
        path: sim_tactile_mnist/test-*.parquet
  - config_name: sim_starstruck
    data_files:
      - split: train
        path: sim_starstruck/train-*.parquet
      - split: test
        path: sim_starstruck/test-*.parquet
  - config_name: unit
    data_files:
      - split: train
        path: unit/train-*.parquet
  - config_name: tacquad
    data_files:
      - split: data_indoor
        path: tacquad/data_indoor-*.parquet
      - split: data_outdoor
        path: tacquad/data_outdoor-*.parquet
      - split: data_fine
        path: tacquad/data_fine-*.parquet
---

# GelSight Mini Pretrain

**~853K [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile RGB frames, 12 public sources, one parquet schema.** Built for self-supervised representation learning (VAE / MAE / SimCLR / DINO) — every frame contact-filtered, channel-normalized, and re-encoded as JPEG q92.

![overview](assets/summary_pies.png)

| | Frames | Sources |
|---|---:|---|
| Real | 536K | FoTA (labeled+unlabeled), 3DCal, FEATS, GelSLAM, TactileTracking, RTM, FeelAnyForce, UniT, TacQuad |
| Sim  | 317K | sim_tactile_mnist, sim_starstruck (Taxim-rendered, Mini-calibrated) |
| **NC extension** ([repo](https://huggingface.co/datasets/yxma/gelsight-mini-pretrain-nc)) | +66K | Sparsh (CC-BY-NC) |

## Quick start

```python
from datasets import load_dataset, concatenate_datasets

# Single source
ds = load_dataset("yxma/gelsight-mini-pretrain", "fota_unlabeled", split="train")
img = ds[0]["image"]                  # PIL.Image (auto-decoded from JPEG bytes)

# Big real-markerless pretraining pool
pool = concatenate_datasets([
    load_dataset("yxma/gelsight-mini-pretrain", c, split="train")
    for c in ["fota_unlabeled", "gelslam", "feelanyforce",
              "real_tactile_mnist", "tacquad", "threedcal", "tactile_tracking"]
]).filter(lambda r: r["domain"] == "real" and not r["markered"])
```

## Composition

| Subset | Frames | Splits | Gel | Labels |
|---|---:|---|---|---|
| `fota_unlabeled` | **66,761** | train | mixed¹ | object name |
| `gelslam` | 114,019 | train + recon | markerless | episode + object |
| `feelanyforce` | 48,197 | train | markerless | 42 unique objects |
| `real_tactile_mnist` | 30,956 | train + test | markerless | digit + print id |
| `fota_labeled` | 26,394 | train + val | mixed¹ | 6-DoF pose + object |
| `feats` | 16,969 | 6-split OOD bench | **markered** | indenter + 3-axis force |
| `tacquad` | 12,195 | indoor/outdoor/fine | markerless | 181 objects |
| `threedcal` | 6,924 | train | markerless | (x, y) sphere grid |
| `tactile_tracking` | 2,408 | train | markerless | trial + object |
| `unit` | 387 | train | **markered** | 3D-pose target |
| `sim_starstruck` | 166,104 | train + test | markerless | episode (sim) |
| `sim_tactile_mnist` | 150,601 | train + test | markerless | digit + episode (sim) |

¹ FoTA mixes markered + markerless gels per finger; the per-row `markered` column was auto-detected from dot density and is correct.

![composition](assets/composition.png)

## Pipeline (applied to every source)

1. **Unified contact filter** — `area ≥ 40 px ∧ intensity ≥ I_min` on the central-50% greyscale diff vs per-source baseline (I_min = 12 real, 10 sim); 1.5 % background-diversity keep rate.
2. **Channel-order normalization** — Mini's at-rest gel has B > R (3 colored LEDs); subsets where the upstream stored BGR are auto-detected (per-image R-B sign) and swapped to RGB. After this, every frame is guaranteed RGB.
3. **JPEG q=92 re-encode** + chunked-binary parquet writes (handles >2 GB shards safely).
4. **Object diversity preserved** — ~8,500 unique object instances across 13 physical sensor configurations.

## Schema (30 columns, every row identical)

`image` (JPEG bytes), `source`, `domain` (`real`/`sim`), `markered` (bool), `gel_variant` (`markered`/`markerless`), `capture`, `split`, `height`, `width`, `obj_name`, `episode`, `frame_idx`, pose fields (`x_mm`, `y_mm`, `z_mm`, `quat_*`), FEATS fields (`indenter`, `indenter_param`, `f_x`, `f_y`, `f_z`, `grid_z_*`), `digit_class`, etc. — all optional fields are `null` when not applicable.

For per-subset details (paper, license, processing recipe, sample grids, stats), see **[SOURCES.md](SOURCES.md)**.

Full build pipeline — per-source parameters cited to `file:line`, and the
regression harness that checks each source against this release:
<https://github.com/Yuxiang-Ma/gelsight-mini-pretrain>

## Sample images

| | |
|---|---|
| **fota_labeled** (markerless) | **fota_labeled** (markered) |
| ![](assets/samples_40_fota_labeled_markerless.png) | ![](assets/samples_40_fota_labeled_markered.png) |
| **gelslam** | **feats** (markered + force) |
| ![](assets/samples_40_gelslam.png) | ![](assets/samples_40_feats.png) |
| **real_tactile_mnist** | **tacquad** (181 household objects) |
| ![](assets/samples_40_real_tactile_mnist.png) | ![](assets/samples_40_tacquad.png) |
| **sim_tactile_mnist** | **sim_starstruck** |
| ![](assets/samples_40_sim_tactile_mnist.png) | ![](assets/samples_40_sim_starstruck.png) |

## Recommended uses

- **Self-supervised pretraining** (VAE / MAE / SimCLR / DINO) — concat all `markerless` real subsets (~472K frames), then fine-tune.
- **Pose / force regression** — fine-tune on `fota_labeled` (6-DoF), `threedcal` (xy + depth), or `feats` (3-axis force).
- **Sim-to-real transfer** — pretrain on `sim_*`, fine-tune on real.
- **Marker-invariance studies** — train markerless ↔ test on `feats` (markered).

## Citations

Please cite both this aggregation **and** the upstream sources you use:

- **FoTA** ([HF](https://huggingface.co/datasets/alanz-mit/FoundationTactile), [arXiv:2406.13640](https://arxiv.org/abs/2406.13640)) · MIT
- **py3DCal** ([Zenodo](https://zenodo.org/records/18462608)) · CC-BY-4.0
- **FEATS** ([HF](https://huggingface.co/datasets/erikhelmut/FEATS)) · MIT
- **GelSLAM** ([HF](https://huggingface.co/datasets/joehjhuang/GelSLAM_dataset), [arXiv:2508.15990](https://arxiv.org/abs/2508.15990)) · MIT
- **TactileTracking / NormalFlow** ([HF](https://huggingface.co/datasets/joehjhuang/TactileTracking), RA-L 2024) · MIT
- **Real Tactile MNIST** ([HF family](https://huggingface.co/TimSchneider42), [arXiv:2506.06361](https://arxiv.org/abs/2506.06361)) · CC-BY-2.0
- **FeelAnyForce** ([HF](https://huggingface.co/datasets/amirsh1376/FeelAnyForce)) · CC-BY-4.0
- **UniT** ([GitHub](https://github.com/ZeyuYong/UniT)) · BSD-3-Clause-style
- **TacQuad / AnyTouch** ([HF](https://huggingface.co/datasets/xxuan01/TacQuad)) · CC-BY-4.0
- **Taxim** (sim renderer, [GitHub](https://github.com/Robo-Touch/Taxim), [arXiv:2109.04027](https://arxiv.org/abs/2109.04027))

## Investigated but not included

Touch-and-Go, TVL (Touch-Vision-Language), facebook/gelsight-force-estimation, YCB-Sight, TACTO/MidasTouch/DiffTactile — see [SOURCES.md](SOURCES.md#investigated-but-not-included) for reasons (wrong sensor, license, or not Mini-calibrated).

## Changelog

- **2026-08-06** — `fota_labeled` and `fota_unlabeled` were widened from 26 to
  30 columns, so every config now shares one schema and the cross-config
  `concatenate_datasets` example above works. `frame_idx`, `episode` and
  `digit_class` are null for these two subsets: they were never recorded at
  build time and are not recoverable. Image bytes were not re-encoded — every
  original column passes through byte-identical.

## License

**CC-BY-4.0** for this aggregation. Cite the component datasets above. The companion [`yxma/gelsight-mini-pretrain-nc`](https://huggingface.co/datasets/yxma/gelsight-mini-pretrain-nc) repo adds Sparsh (CC-BY-NC-4.0) for non-commercial use.

