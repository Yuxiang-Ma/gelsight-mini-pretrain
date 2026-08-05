---
license: cc-by-nc-4.0
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
  - non-commercial
size_categories:
  - 10K<n<1M
pretty_name: GelSight Mini Pretrain · NC extension
configs:
  - config_name: sparsh
    data_files:
      - split: flat
        path: sparsh/flat-*.parquet
      - split: sharp
        path: sparsh/sharp-*.parquet
      - split: sphere
        path: sparsh/sphere-*.parquet
---

# GelSight Mini Pretrain · Non-Commercial Extension

> ⚠️  **Non-commercial use only.** This repository is licensed
> **CC-BY-NC-4.0** because it includes upstream sources whose licenses
> restrict commercial use. **For commercial-friendly Mini tactile data,
> see the main [yxma/gelsight-mini-pretrain](https://huggingface.co/datasets/yxma/gelsight-mini-pretrain) repo
> (CC-BY-4.0).**

This dataset is the **CC-BY-NC extension** of `yxma/gelsight-mini-pretrain`.
It contains *only* the GelSight Mini sources whose upstream licenses are
**not compatible** with CC-BY-4.0 aggregation. All data is processed
through the same pipeline as the main repo (same schema, same area+intensity
validity filter, same channel-order normalization), so users can simply
load both repos and concatenate them to get a larger pool.

## Sample images

### Sparsh (3 indenter shapes)

| Indenter | Frames | Random samples |
|---|---:|---|
| flat | 8,056 | ![](assets/samples_40_sparsh_flat.png) |
| sharp | 11,844 | ![](assets/samples_40_sparsh_sharp.png) |
| sphere | 46,544 | ![](assets/samples_40_sparsh_sphere.png) |

## When to use this repo

Use the **main repo** if:
- You want commercial use rights
- You only need ~830K frames across 12 sources (already plenty for VAE pretraining)

Use **both** (main + this extension) if:
- Your work is non-commercial (academic, internal research)
- You want maximum coverage of public Mini tactile data
- You're OK respecting the NC clause on the extension subsets only

## How to combine with the main repo

```python
from datasets import concatenate_datasets, load_dataset

# Load all subsets from the main repo (CC-BY-4.0)
main_real = concatenate_datasets([
    load_dataset("yxma/gelsight-mini-pretrain", c, split="train")
    for c in ["fota_labeled", "fota_unlabeled", "threedcal", "feats",
              "gelslam", "tactile_tracking", "real_tactile_mnist",
              "feelanyforce", "unit", "tacquad"]
])

# Add NC extension (sparsh has 3 indenter-named splits)
nc_extras = concatenate_datasets([
    load_dataset("yxma/gelsight-mini-pretrain-nc", "sparsh", split=s)
    for s in ["flat", "sharp", "sphere"]
])

all_markerless = concatenate_datasets([main_real, nc_extras])
print(f"Total: {len(all_markerless):,} frames")
```

## Schema

Identical to the main repo (30 columns, parquet, JPEG q=92 images).
Every row has `domain="real"`, `markered=False`,
`gel_variant="markerless"`, plus source-specific metadata.

## Pipeline parity

Applies the **same unified area+intensity validity filter** as the main
repo:
- `pixel_diff = |frame - baseline|` on central 50% crop, greyscale
- `mask = pixel_diff > 10` (sensor noise floor)
- `area = mask.sum()`, `intensity = pixel_diff[mask].mean()`
- KEEP iff `area >= 40 px AND intensity >= 12 grey-levels`
- ELSE keep with probability `1.5%` (background diversity)

### Channel-order normalization

Sparsh's upstream pickles contain a **mix of RGB and BGR images**
(apparently from heterogeneous data-collection pipelines at Meta). We
normalize all images to RGB by checking each image's per-channel means
and swapping R<->B when `mean(R) > mean(B)` (GelSight Mini's at-rest
gel is illuminated such that B > R is the correct channel order).
After normalization, every image in this repo is guaranteed RGB.

## Sources currently included

| Subset | Upstream | License | Frames | Notes |
|---|---|---|---:|---|
| `sparsh` | [`facebook/SparshGelSight`](https://huggingface.co/datasets/facebook/SparshGelSight) | **CC-BY-NC-4.0** | **66,444** (flat 8K + sharp 12K + sphere 47K) | Part of Meta's Sparsh / TacBench. GelSight Mini, markerless, 3 indenter shapes with paired ATI nano17 force ground truth. Includes channel-order normalization (upstream pkls mix RGB and BGR). |

> **Deprecated:** The earlier `faf_force_estimation/` subset (from
> [`facebook/gelsight-force-estimation`](https://huggingface.co/datasets/facebook/gelsight-force-estimation))
> has been removed -- it was a smaller snapshot of the same collection
> protocol and data as `sparsh`, just released under a different upstream
> repo name. Sparsh is the canonical larger version of the same protocol.

## Citation chain

If you use this extension, please cite **both** the main aggregation and
the upstream NC source:

```bibtex
@dataset{gelsight_mini_pretrain_nc,
  title  = {GelSight Mini Pretrain - NC extension},
  author = {Ma, Yuxiang},
  year   = {2026},
  url    = {https://huggingface.co/datasets/yxma/gelsight-mini-pretrain-nc},
}

@misc{sparsh_2024,
  title  = {Sparsh: Self-supervised touch representations for vision-based tactile sensing},
  author = {Higuera, Carolina and Sharma, Akash and Bodduluri, Chaithanya Krishna and Fan, Taosha and Lancaster, Patrick and Kalakrishnan, Mrinal and Kaess, Michael and Boots, Byron and Lambeta, Mike and Wu, Tingfan and Mukadam, Mustafa},
  year   = {2024},
  url    = {https://huggingface.co/datasets/facebook/SparshGelSight},
}
```

## License

This aggregated release is **CC-BY-NC-4.0** because it inherits the
strictest license among the included sources. Component datasets retain
their own original licenses; cite each upstream source individually.
