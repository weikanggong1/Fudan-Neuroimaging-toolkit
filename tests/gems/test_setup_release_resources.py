"""All GEMS preparation families consume the verified recon Release downloader."""

import hashlib
import io

import numpy as np

from fnit.gems import setup
from fnit.recon_all import assets


def test_all_gems_packs_use_six_release_atlases_and_three_package_luts(tmp_path, monkeypatch):
    class _Response(io.BytesIO):
        status = 200
        headers = {}

    expected = {}
    for family in ("BrainstemSS", "ThalamicNuclei", "HippoSF"):
        for filename in ("AtlasMesh.gz", "AtlasDump.mgz"):
            name = f"average/{family}/atlas/{filename}"
            content = name.encode("utf-8")
            digest = hashlib.sha256(content).hexdigest()
            monkeypatch.setitem(assets.ASSET_FILES, name, (len(content), digest, ".fixture"))
            expected[digest] = content
    monkeypatch.setattr(assets, "release_url_for", lambda digest, size=None:
                        "https://github.com/example/release/" + digest if digest in expected else None)
    downloaded = []
    def request(request, timeout):
        digest = request.full_url.rsplit("/", 1)[1]
        downloaded.append(digest)
        assert request.full_url.startswith("https://github.com/example/release/")
        return _Response(expected[digest])
    monkeypatch.setattr(assets, "urlopen", request)
    monkeypatch.setattr(setup.GEMSAtlas, "from_freesurfer", lambda *args: object())
    monkeypatch.setattr(setup, "smooth_atlas_alphas", lambda *args, **kwargs:
                        np.zeros((1, 21), dtype=np.float32))
    output = setup.prepare_subregion_atlases(tmp_path / "packs", asset_dir=tmp_path / "cache")
    assert set(downloaded) == set(expected)
    assert len(downloaded) == 6
    for pack in ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right"):
        for filename in setup.ATLAS_FILES:
            assert (output / pack / filename).is_file()
        assert (output / pack / "config.json").is_file()
    for filename in setup.ATLAS_FILES:
        assert (output / "hippo-amygdala-left" / filename).read_bytes() == (
            output / "hippo-amygdala-right" / filename).read_bytes()
    setup.prepare_subregion_atlases(output, asset_dir=tmp_path / "cache")
    assert len(downloaded) == 6
