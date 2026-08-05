from __future__ import annotations

import importlib
from pathlib import Path


def test_defaults_point_at_data_disk():
    from gsmp import config

    assert config.RAW_ROOT == Path("/media/yxma/Disk1/yuxiang/mini_data")
    assert config.PARQUET_MAIN == Path("/media/yxma/Disk1/yuxiang/mini_data_parquet")
    assert config.PARQUET_NC == Path("/media/yxma/Disk1/yuxiang/mini_data_parquet_nc")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GSMP_RAW_ROOT", str(tmp_path / "raw"))
    from gsmp import config

    importlib.reload(config)
    assert config.RAW_ROOT == tmp_path / "raw"
    monkeypatch.delenv("GSMP_RAW_ROOT")
    importlib.reload(config)


def test_out_root_is_not_inside_readonly_trees():
    from gsmp import config

    assert config.PARQUET_MAIN not in config.OUT_ROOT.parents
    assert config.RAW_ROOT not in config.OUT_ROOT.parents
