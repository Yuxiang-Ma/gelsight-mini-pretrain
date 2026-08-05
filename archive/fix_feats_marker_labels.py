#!/usr/bin/env python3
"""Re-label FEATS parquet rows with accurate gel_variant tagging.

FEATS uses two physical sensor setups:
  - 'black_dot':   standard markered Mini gel with black dots
                   (train, val, test, test_unknown_indenters, test_diff_sensor_old_gel)
  - 'different':   a second sensor with a different gel style (red/orange markers)
                   (test_diff_sensor_new_gel only)

This script adds a gel_variant column and rewrites the FEATS parquet shards.
"""

import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq

FEATS_DIR = "/media/yxma/Disk1/yuxiang/mini_data_parquet/feats"

GEL_BY_SPLIT = {
    "train":                       "black_dot",
    "val":                         "black_dot",
    "test":                        "black_dot",
    "test_unknown_indenters":      "black_dot",
    "test_diff_sensor_old_gel":    "black_dot",
    "test_diff_sensor_new_gel":    "different",
}


def main():
    paths = sorted(glob.glob(os.path.join(FEATS_DIR, "*.parquet")))
    print(f"rewriting {len(paths)} parquet shards in {FEATS_DIR}")
    for p in paths:
        t = pq.read_table(p)
        splits = t.column("split").to_pylist()
        variants = [GEL_BY_SPLIT.get(s, "unknown") for s in splits]
        # add column
        new = t.append_column("gel_variant",
                              pa.array(variants, type=pa.string()))
        # write to temp then rename
        tmp = p + ".tmp"
        pq.write_table(new, tmp, compression="snappy")
        os.replace(tmp, p)
        print(f"  {os.path.basename(p):40s}  {t.num_rows} rows  "
              f"variants: {sorted(set(variants))}")
    print("done")


if __name__ == "__main__":
    main()
