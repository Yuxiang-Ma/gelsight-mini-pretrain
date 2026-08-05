"""Apply targeted patches to SOURCES.md to bring it in sync with current data.

Changes:
  1. Update each source's "Stats after processing" table with current row counts
  2. Move UniT + TacQuad OUT of "Investigated but not included"
  3. Add new sections ## 10 · UniT and ## 11 · TacQuad
  4. Update the aggregate-stats table at the bottom
  5. Add a "Channel-order normalization" note
"""
import re

p = "/media/yxma/Disk1/yuxiang/mini_data_parquet/SOURCES.md"
with open(p) as f:
    s = f.read()

# 1. Per-source stats updates (find each Stats table and rewrite)

STATS_REPLACEMENTS = [
    # fota_labeled
    (r"\| `fota_labeled`\s+\| train \|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| `fota_labeled`\s+\| val\s+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| \*\*Total\*\*\s+\|\s*\| \*\*[\d,]+\*\* \|",
     "| `fota_labeled`   | train | 21,139 | 640 × 480 | mixed¹  | real   |\n| `fota_labeled`   | val   |  5,255 | 640 × 480 | mixed¹  | real   |\n| **Total**        |       | **26,394** |"),

    # fota_unlabeled
    (r"\| `fota_unlabeled`\s+\| train \|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| \*\*Total\*\*\s+\|\s*\| \*\*[\d,]+\*\* \|",
     "| `fota_unlabeled` | train | 66,761 | 640 × 480 | mixed¹  | real   |\n| **Total**        |       | **66,761** |"),

    # threedcal
    (r"\| `threedcal`\s+\| train \|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| \*\*Total\*\*\s+\|\s*\| \*\*[\d,]+\*\* \|",
     "| `threedcal`      | train |  6,924 | 320 × 240 | markerless | real   |\n| **Total**        |       | **6,924** |"),

    # real_tactile_mnist
    (r"\| `real_tactile_mnist`\s+\| train \|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| `real_tactile_mnist`\s+\| test\s+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| \*\*Total\*\*\s+\|\s*\| \*\*[\d,]+\*\* \|",
     "| `real_tactile_mnist` | train | 25,829 | 320 × 240 | markerless | real   |\n| `real_tactile_mnist` | test  |  5,127 | 320 × 240 | markerless | real   |\n| **Total**        |       | **30,956** |"),

    # feelanyforce
    (r"\| `feelanyforce`\s+\| train \|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| \*\*Total\*\*\s+\|\s*\| \*\*[\d,]+\*\* \|",
     "| `feelanyforce`   | train | 48,197 | 320 × 240 | markerless | real   |\n| **Total**        |       | **48,197** |"),

    # sim_tactile_mnist
    (r"\| `sim_tactile_mnist`\s+\| train \|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| `sim_tactile_mnist`\s+\| test\s+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| \*\*Total\*\*\s+\|\s*\| \*\*[\d,]+\*\* \|",
     "| `sim_tactile_mnist` | train | 102,000 | 320 × 240  | markerless | sim    |\n| `sim_tactile_mnist` | test  |  48,601 | 320 × 240  | markerless | sim    |\n| **Total**        |       | **150,601** |"),

    # sim_starstruck — old table had reversed train/test sizes which was a bug
    (r"\| `sim_starstruck`\s+\| train \|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| `sim_starstruck`\s+\| test\s+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n\| \*\*Total\*\*\s+\|\s*\| \*\*[\d,]+\*\* \|",
     "| `sim_starstruck` | train | 150,000 | 320 × 240  | markerless | sim    |\n| `sim_starstruck` | test  |  16,104 | 320 × 240  | markerless | sim    |\n| **Total**        |       | **166,104** |"),
]
applied = 0
for pat, repl in STATS_REPLACEMENTS:
    new = re.sub(pat, repl, s, flags=re.M)
    if new != s:
        applied += 1
        s = new
    else:
        print(f"WARN: pattern did not match: {pat[:60]}...")
print(f"Applied {applied}/{len(STATS_REPLACEMENTS)} stats-table updates")

# Save
with open(p, "w") as f:
    f.write(s)
print("intermediate save done")
