"""固定 WM 路径搜索的数据布局、顺序和明确失败契约。"""

import numpy as np
import pytest

from fnit.recon_all.wm_remove_paths import (
    _row_views, _search_hemisphere, remove_mtl_paths,
)


def test_byte_write_is_shared_with_typed_row_and_preserves_storage_type():
    for dtype in (np.int32, np.float32):
        raw, typed = _row_views((8, 4, 4), np.dtype(dtype))
        raw[1, 2, 5] = 128
        assert typed[1, 2, 1].view(np.uint32) == 32768
        assert typed[1, 2, 5] == 0
        assert raw[1, 2, 5] == 128
        if dtype == np.int32:
            assert typed[1, 2, 1] == 32768
        else:
            assert 0 < typed[1, 2, 1] < np.finfo(np.float32).tiny


def test_full_scans_and_unreachable_proof_agree_for_both_storage_types():
    wm = np.zeros((40, 24, 24), dtype=np.uint8)
    brain = np.full(wm.shape, 90, dtype=np.uint8)
    aseg = np.zeros(wm.shape, dtype=np.int32)
    aseg[24:28, 12, 6:19] = 17
    wm[24:28, 13, 6:19] = 110
    for dtype in ("int32", "float32"):
        fast, fast_info = remove_mtl_paths(wm=wm, brain=brain, aseg=aseg,
            aseg_storage_dtype=dtype, use_unreachable_proof=True)
        full, full_info = remove_mtl_paths(wm=wm, brain=brain, aseg=aseg,
            aseg_storage_dtype=dtype, use_unreachable_proof=False)
        assert np.array_equal(full, fast) and np.array_equal(full, wm)
        assert fast_info["hemispheres"][0]["proof_unreachable"]
        assert not full_info["hemispheres"][0]["proof_unreachable"]
        assert full_info["filled"] == fast_info["filled"] == 0


def test_active_roi_is_executed_instead_of_assumed_noop():
    wm = np.zeros((24, 24, 24), dtype=np.uint8)
    aseg = np.zeros(wm.shape, dtype=np.int32)
    aseg[7, 12, 12] = 17
    for dtype in ("int32", "float32"):
        result, info = remove_mtl_paths(wm=wm, brain=wm, aseg=aseg,
                                      aseg_storage_dtype=dtype)
        assert not info["hemispheres"][0]["proof_unreachable"]
        assert info["hemispheres"][0]["outer_iterations"] == 1
        assert np.array_equal(result, wm)


def _right_search_fixture():
    shape = (80, 12, 48)
    wm = np.zeros(shape, dtype=np.uint8)
    brain = np.zeros(shape, dtype=np.uint8)
    aseg = np.zeros(shape, dtype=np.int32)
    aseg[40, 4, 10] = aseg[70, 4, 40] = 53
    wm[40, 5, 10] = wm[70, 5, 40] = 110
    aseg[40, 4, 9] = 42
    return wm, brain, aseg


def test_right_byte_scan_finds_cortex_and_fills_source_selected_voxel():
    wm, brain, aseg = _right_search_fixture()
    roi_bytes, roi_values = _row_views(wm.shape, np.dtype(np.int32), np.full(wm.shape, 128))
    hippo_bytes, _ = _row_views(wm.shape, np.dtype(np.int32))
    filled_bytes, filled_values = _row_views(wm.shape, np.dtype(np.int32))
    count, iterations, capped = _search_hemisphere(wm, brain, aseg,
        roi_bytes, roi_values, hippo_bytes, filled_bytes, filled_values,
        np.array([1, 78, 1, 10, 1, 46], dtype=np.int64), False)
    assert count == 1 and iterations == 2 and not capped
    assert wm[40, 4, 9] == 250
    assert np.count_nonzero(wm == 250) == 1


def test_left_nonterminating_typed_byte_fixed_state_fails_explicitly():
    shape = (80, 12, 48)
    wm = np.zeros(shape, dtype=np.uint8)
    aseg = np.zeros(shape, dtype=np.int32)
    aseg[1, 4, 40] = aseg[60, 4, 10] = 17
    wm[1, 5, 40] = wm[60, 5, 10] = 110
    raw_roi, typed_roi = _row_views(shape, np.dtype(np.int32), np.full(shape, 128))
    hippo, _ = _row_views(shape, np.dtype(np.int32))
    filled, typed_filled = _row_views(shape, np.dtype(np.int32))
    with pytest.raises(RuntimeError, match="unchanged byte buffer"):
        _search_hemisphere(wm, np.zeros_like(wm), aseg, raw_roi, typed_roi,
            hippo, filled, typed_filled, np.array([1, 78, 1, 10, 1, 46]), True)
    assert not np.any(wm == 250)


@pytest.mark.parametrize("invalid", ["uint8", "int16", "float64"])
def test_rejects_storage_changes_that_would_change_native_byte_access(invalid):
    wm = np.zeros((8, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        remove_mtl_paths(wm=wm, brain=wm, aseg=np.zeros(wm.shape, np.int32),
                         aseg_storage_dtype=invalid)


def test_rejects_float32_labels_that_cannot_be_stored_exactly():
    wm = np.zeros((8, 8, 8), dtype=np.uint8)
    aseg = np.zeros(wm.shape, np.int32)
    aseg[1, 1, 1] = 16777217
    with pytest.raises(ValueError, match="exactly"):
        remove_mtl_paths(wm=wm, brain=wm, aseg=aseg, aseg_storage_dtype="float32")


@pytest.mark.parametrize("backend", ["cpu", "hybrid"])
def test_file_wrapper_preserves_original_float32_storage(tmp_path, monkeypatch, backend):
    import nibabel as nib
    from fnit.recon_all import edit_wm_aseg_core_python as cpu
    from fnit.recon_all import edit_wm_aseg_torch as hybrid

    shape = (8, 8, 8)
    paths = []
    for name, dtype in (("wm", np.uint8), ("brain", np.uint8),
                        ("aseg", np.float32), ("entowm", np.int32)):
        path = tmp_path / (name + ".mgz")
        nib.save(nib.MGHImage(np.zeros(shape, dtype=dtype), np.eye(4)), path)
        paths.append(path)
    observed = {}

    def fake_profile(wm, brain, aseg, entowm, **kwargs):
        observed.update(kwargs)
        assert aseg.dtype == np.int32
        return wm.copy(), {"total_seconds": 0.0}

    if backend == "cpu":
        monkeypatch.setattr(cpu, "edit_wm_aseg_profile", fake_profile)
        cpu.write_wm_asegedit_profile(*paths, output_file=tmp_path / "output.mgz")
    else:
        monkeypatch.setattr(hybrid, "edit_wm_aseg_hybrid_diagnostic", fake_profile)
        hybrid.write_wm_asegedit_hybrid_diagnostic(*paths, output_file=tmp_path / "output.mgz")
    assert np.dtype(observed["aseg_storage_dtype"]).newbyteorder("=") == np.dtype(np.float32)
