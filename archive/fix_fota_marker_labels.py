#!/usr/bin/env python3
"""Set the `markered` column in FoTA parquet shards based on per-capture
detection. Threshold: a capture is 'markered' if its mean-image yields
≥10 valid dark blobs (markers).

After running:
  fota_labeled and fota_unlabeled shards have correct per-capture markered=True/False
"""

import glob
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
CLASSIF = "/home/yxma/MultimodalData/fota_marker_classification.json"
THRESHOLD = 10


def main():
    results = json.load(open(CLASSIF))
    # capture -> markered bool
    cap_marker = {r["capture"]: bool(r["valid_dots"] >= THRESHOLD) for r in results}
    print(f"loaded {len(cap_marker)} capture classifications")
    print(f"  markered (≥{THRESHOLD} dots): {sum(cap_marker.values())}")
    print(f"  markerless:                   {sum(1 for v in cap_marker.values() if not v)}")

    for sub in ["fota_labeled", "fota_unlabeled"]:
        paths = sorted(glob.glob(os.path.join(PARQUET, sub, "*.parquet")))
        if not paths:
            print(f"skip {sub} (no parquet)")
            continue
        print(f"\n{sub}: rewriting {len(paths)} shards")
        for p in paths:
            t = pq.read_table(p)
            captures = t.column("capture").to_pylist()
            # New per-row markered
            new_markered = pa.array(
                [cap_marker.get(c, False) for c in captures],
                type=pa.bool_(),
            )
            # Replace markered column
            idx = t.column_names.index("markered")
            t2 = t.set_column(idx, "markered", new_markered)
            # Stats
            n_mark = sum(1 for c in captures if cap_marker.get(c, False))
            tmp = p + ".tmp"
            pq.write_table(t2, tmp, compression="snappy")
            os.replace(tmp, p)
            print(f"  {os.path.basename(p):42s}  rows={t.num_rows}  "
                  f"markered={n_mark}  markerless={t.num_rows - n_mark}")


if __name__ == "__main__":
    main()
