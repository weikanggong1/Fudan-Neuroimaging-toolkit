"""Frozen runtime resources use the mirror without changing template profiles."""
from fnit._release_assets import (RELEASE_BASE, published_release_assets,
                                 release_asset_metadata, release_url_for)
from fnit.mshbm.assets_setup import FILES
from fnit.recon_all.assets import ASSET_FILES, asset_url


def test_vpnl_resource_routes_use_published_exact_bytes():
    paths = [name for name in ASSET_FILES
             if name.endswith('.vpnl.label') or name == 'average/colortable_vpnl.txt']
    assert len(paths) == 17
    for name in paths:
        size, digest, _ = ASSET_FILES[name]
        record = release_asset_metadata(digest, size)
        assert record is not None and record['status'] == 'published'
        assert record['group'] == 'recon'
        assert asset_url(name) == RELEASE_BASE + record['name']
        assert record['publication_basis'] == 'direct_user_confirmation_2026-10-06'


def test_caret_meshes_have_verified_mirror_routes():
    for name in ('left_mni.surf.gii', 'right_mni.surf.gii'):
        _, size, digest = FILES[name]
        record = release_asset_metadata(digest, size)
        assert record is not None and record['group'] == 'mshbm'
        assert release_url_for(digest, size) == RELEASE_BASE + 'mshbm--' + name
        assert record['publication_basis'] == 'direct_user_confirmation_2026-10-06'


def test_oxford_gm_does_not_enter_fsl_standard_profiles():
    digest = 'ab933db7455d7c4b88624d54f41a3065be4ba4289d00b9230daec0cdb1597a77'
    record = release_asset_metadata(digest, 707776)
    assert record is not None and record['group'] == 'oxford'
    assert record['name'] == 'oxford--template_GM.nii.gz'
    assert record['license'].startswith('Apache-2.0')
    assert len(published_release_assets('standard')) == 11
    assert all(item['sha256'] != digest for item in published_release_assets('standard'))
