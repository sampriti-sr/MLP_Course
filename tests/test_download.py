"""Offline tests for fetch(): publisher first, then checksum-verified mirrors. Uses file:// URLs."""
import hashlib
import warnings

import pytest

from mlp_data._download import fetch

PAYLOAD = b"the pinned file"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _file(tmp_path, name, content):
    p = tmp_path / name
    p.write_bytes(content)
    return p.as_uri()


def test_primary_is_used_and_mirror_not_consulted(tmp_path):
    primary = _file(tmp_path, "primary.bin", PAYLOAD)
    dest = fetch(primary, tmp_path / "out" / "f.bin", sha256=SHA, mirrors=[(tmp_path / "nope").as_uri()])
    assert dest.read_bytes() == PAYLOAD


def test_falls_back_to_mirror_when_primary_is_gone(tmp_path, capsys):
    mirror = _file(tmp_path, "mirror.bin", PAYLOAD)
    dest = fetch((tmp_path / "gone").as_uri(), tmp_path / "f.bin", sha256=SHA, mirrors=[mirror])
    assert dest.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("*.part"))
    assert "course mirror" in capsys.readouterr().out


def test_mirror_with_the_wrong_file_is_rejected(tmp_path):
    bad = _file(tmp_path, "bad.bin", b"someone swapped the file")
    dest = tmp_path / "f.bin"
    with pytest.raises(RuntimeError, match="does not match the pinned checksum"):
        fetch((tmp_path / "gone").as_uri(), dest, sha256=SHA, mirrors=[bad])
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def test_second_mirror_is_tried_after_a_bad_first_one(tmp_path):
    bad = _file(tmp_path, "bad.bin", b"wrong")
    good = _file(tmp_path, "good.bin", PAYLOAD)
    dest = fetch((tmp_path / "gone").as_uri(), tmp_path / "f.bin", sha256=SHA, mirrors=[bad, good])
    assert dest.read_bytes() == PAYLOAD


def test_error_names_every_failed_source(tmp_path):
    with pytest.raises(RuntimeError) as err:
        fetch((tmp_path / "gone1").as_uri(), tmp_path / "f.bin", sha256=SHA, mirrors=[(tmp_path / "gone2").as_uri()])
    assert "gone1" in str(err.value) and "gone2" in str(err.value)


def test_mirrors_require_a_checksum(tmp_path):
    with pytest.raises(ValueError, match="pinned sha256"):
        fetch((tmp_path / "x").as_uri(), tmp_path / "f.bin", mirrors=["https://example.invalid/f"])


def test_primary_mismatch_still_only_warns(tmp_path):
    primary = _file(tmp_path, "primary.bin", b"publisher updated the file")
    with pytest.warns(UserWarning, match="checksum differs"):
        dest = fetch(primary, tmp_path / "f.bin", sha256=SHA)
    assert dest.exists()


def test_no_mirrors_keeps_the_original_error(tmp_path):
    with pytest.raises(RuntimeError, match="Could not download"):
        fetch((tmp_path / "gone").as_uri(), tmp_path / "f.bin")


def test_cache_hit_skips_the_network(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"cached")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fetch("https://example.invalid/never", dest, sha256=SHA, mirrors=["https://example.invalid/m"]) == dest


def test_airbnb_loader_passes_its_mirror_and_pinned_checksum(tmp_path, monkeypatch):
    from mlp_data import loaders

    seen = {}

    def fake_fetch(url, dest, **kw):
        seen.update(url=url, **kw)
        raise RuntimeError("stop here")

    monkeypatch.setenv("MLP_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(loaders, "fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="stop here"):
        loaders.load_airbnb()
    assert seen["sha256"] == loaders._AIRBNB_SHA[loaders.AIRBNB_SNAPSHOT]
    assert seen["mirrors"] == [
        f"https://github.com/sampriti-sr/MLP_Course/releases/download/data-v1/"
        f"airbnb_london_{loaders.AIRBNB_SNAPSHOT}_listings.csv.gz"
    ]


def test_movielens_loader_has_a_pinned_mirror_for_the_certificate_chain_problem(tmp_path, monkeypatch):
    from mlp_data import loaders

    seen = {}

    def fake_fetch(url, dest, **kw):
        seen.update(kw)
        raise RuntimeError("stop here")

    monkeypatch.setenv("MLP_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(loaders, "fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="stop here"):
        loaders.load_movielens()
    assert seen["sha256"] == loaders._ML_SHA
    assert seen["mirrors"] == ["https://github.com/sampriti-sr/MLP_Course/releases/download/data-v1/ml-latest-small.zip"]
