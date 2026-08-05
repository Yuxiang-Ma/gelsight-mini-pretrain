from __future__ import annotations

import dataclasses

import pytest

from gsmp.baseline import FirstNFrames, NoBaseline
from gsmp.spec import SourceSpec


def _spec(**kw):
    base = dict(
        name="demo", domain="real", gel_variant="markerless",
        license_repo="main", baseline=FirstNFrames(10), i_min=10.0,
    )
    base.update(kw)
    return SourceSpec(**base)


def test_i_min_has_no_default():
    fields = {f.name: f for f in dataclasses.fields(SourceSpec)}
    assert fields["i_min"].default is dataclasses.MISSING
    assert fields["i_min"].default_factory is dataclasses.MISSING


def test_spec_is_frozen():
    s = _spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.i_min = 99.0


def test_rejects_unknown_domain():
    with pytest.raises(ValueError, match="domain"):
        _spec(domain="synthetic")


def test_rejects_unknown_channel_mode():
    with pytest.raises(ValueError, match="channel_mode"):
        _spec(channel_mode="rbg")


def test_rejects_nonpositive_a_min():
    with pytest.raises(ValueError, match="a_min"):
        _spec(a_min=0)


def test_defaults_match_documented_pipeline():
    s = _spec()
    assert s.a_min == 40
    assert s.channel_mode == "auto"
    assert s.phash_dist == 4
    assert s.phash_lookback == 30
    assert s.budget == 200_000
    assert s.bg_keep_rate == 0.015


def test_dedupe_disabled_when_phash_dist_is_none():
    assert _spec(phash_dist=None).dedupe_enabled is False
    assert _spec(phash_dist=4).dedupe_enabled is True


def test_no_baseline_sources_are_allowed():
    s = _spec(baseline=NoBaseline())
    assert s.baseline.compute([]) is None
