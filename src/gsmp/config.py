"""Central path configuration.

Every path the pipeline touches is resolved here. No module may hardcode an
absolute path. Defaults match the machine the dataset was built on; override
any of them with the corresponding GSMP_* environment variable.

RAW_ROOT and PARQUET_MAIN/PARQUET_NC are READ-ONLY: they hold 805G of
upstream data and the 8G published release. Nothing in this package writes
to them. Generated output goes to OUT_ROOT.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULTS = {
    "GSMP_RAW_ROOT": "/media/yxma/Disk1/yuxiang/mini_data",
    "GSMP_PARQUET_MAIN": "/media/yxma/Disk1/yuxiang/mini_data_parquet",
    "GSMP_PARQUET_NC": "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc",
    "GSMP_PARQUET_VIDEO": "/media/yxma/Disk1/yuxiang/mini_data_parquet_video",
    "GSMP_OUT_ROOT": "/media/yxma/Disk1/yuxiang/gsmp_out",
}


def _p(key: str) -> Path:
    return Path(os.environ.get(key, _DEFAULTS[key]))


RAW_ROOT = _p("GSMP_RAW_ROOT")
PARQUET_MAIN = _p("GSMP_PARQUET_MAIN")
PARQUET_NC = _p("GSMP_PARQUET_NC")
PARQUET_VIDEO = _p("GSMP_PARQUET_VIDEO")
OUT_ROOT = _p("GSMP_OUT_ROOT")

HF_REPO_MAIN = "yxma/gelsight-mini-pretrain"
HF_REPO_NC = "yxma/gelsight-mini-pretrain-nc"


def repo_root() -> Path:
    """Absolute path to this git repository."""
    return Path(__file__).resolve().parents[2]


def published_dir(source: str, license_repo: str = "main") -> Path:
    """Directory of published parquet shards for one source (read-only)."""
    base = PARQUET_MAIN if license_repo == "main" else PARQUET_NC
    return base / source
