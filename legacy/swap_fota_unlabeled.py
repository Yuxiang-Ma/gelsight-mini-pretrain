#!/usr/bin/env python3
"""Swap R<->B in fota_unlabeled parquet shards (one-shot, chunked-binary safe)."""
import glob
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BASE = "/media/yxma/Disk1/yuxiang/mini_data_parquet"
JPEG_Q = 92


def swap_shard(path):
    fname = os.path.basename(path)
    t0 = time.time()
    print(f"  [{fname}] reading...", flush=True)
    t = pq.read_table(path)
    n = t.num_rows
    img_col = t.column("image")
    new_imgs = []
    for i in range(n):
        b = img_col[i].as_py()
        if b is None:
            new_imgs.append(b)
            continue
        try:
            rgb = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
            bgr = rgb[..., ::-1]
            buf = io.BytesIO()
            Image.fromarray(bgr).save(buf, format="JPEG", quality=JPEG_Q)
            new_imgs.append(buf.getvalue())
        except Exception:
            new_imgs.append(b)
        if (i + 1) % 5000 == 0:
            fps = (i + 1) / max(0.01, time.time() - t0)
            print(f"  [{fname}] {i+1:,}/{n:,}  ({fps:.0f} fps)", flush=True)
    # Build chunked binary column to avoid 2GB cast issue
    CHUNK = 5000
    chunks = [pa.array(new_imgs[i:i + CHUNK], type=pa.binary())
              for i in range(0, len(new_imgs), CHUNK)]
    new_col = pa.chunked_array(chunks, type=pa.binary())
    idx = t.column_names.index("image")
    t = t.set_column(idx, "image", new_col)
    tmp = path + ".tmp"
    pq.write_table(t, tmp, compression="snappy")
    os.replace(tmp, path)
    print(f"  [{fname}] DONE in {time.time()-t0:.0f}s", flush=True)
    return path


def main():
    paths = sorted(glob.glob(f"{BASE}/fota_unlabeled/*.parquet"))
    print(f"=== Swapping {len(paths)} fota_unlabeled shards ===", flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(swap_shard, p): p for p in paths}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                import traceback
                traceback.print_exc()
    print("done", flush=True)


if __name__ == "__main__":
    main()
