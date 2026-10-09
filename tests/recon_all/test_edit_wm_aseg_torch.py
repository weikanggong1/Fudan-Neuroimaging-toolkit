import numpy as np
import pytest
import torch
import nibabel as nib
from pathlib import Path
from tempfile import TemporaryDirectory

from fnit.recon_all.edit_wm_aseg_core_python import (
    _add_aseg_wm_below_hippocampus, spackle_wm_superior_to_mtl,
    _edit_until_propagation, _propagate_from_filled,
)
from fnit.recon_all.edit_wm_aseg_late_python import (
    apply_late_wm_edits, fill_seg_wm_from_core, fix_subcortical_mass_ha,
)
from fnit.recon_all.edit_wm_aseg_torch import (
    add_aseg_wm_below_hippocampus_torch, apply_late_wm_edits_torch,
    fill_seg_wm_from_core_torch, fix_subcortical_mass_ha_torch,
    spackle_wm_superior_to_mtl_torch,
    _integer_label_array,
    write_late_wm_edits_torch,
    fill_seg_wm_seed_torch, propagate_wm_from_filled_torch,
    _validate_hybrid_aseg_storage, write_wm_asegedit_hybrid_diagnostic,
)


def _fixture():
    rng = np.random.default_rng(9284)
    shape = (24, 19, 27)
    wm = rng.choice(np.array([0, 1, 110, 250, 255], np.uint8), shape)
    aseg = np.zeros(shape, np.int32)
    aseg[3:-3, 3:-3, 3:-3] = rng.choice(
        np.array([0, 2, 3, 4, 5, 17, 18, 28, 41, 42, 44, 53, 54, 60], np.int32),
        (18, 13, 21))
    ento = rng.choice(np.array([0, 3006, 3201, 4006, 4201], np.int32), shape)
    return wm, aseg, ento


def test_late_torch_matches_cpu_with_fill_and_keep_in_precedence():
    wm, aseg, ento = _fixture()
    original = wm.copy()
    core = np.zeros_like(wm)
    for fill in (False, True):
        expected = apply_late_wm_edits(core.copy(), aseg, ento, original,
                                       fill_seg_wm=fill)
        actual = apply_late_wm_edits_torch(*map(torch.from_numpy,
            (core, aseg, ento, original)), fill_seg_wm=fill).numpy()
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(wm, original)
    assert not core.any()


def test_fill_candidate_consumed_at_native_ordered_scan_matches_direct_rule():
    wm, aseg, _ = _fixture()
    # Include additional native labels omitted by the earlier CPU port.
    aseg[4, 5, 6] = 6
    aseg[6, 5, 6] = 45
    aseg[8, 5, 6] = 186
    aseg[9, 5, 6] = 187
    seed = fill_seg_wm_seed_torch(torch.from_numpy(aseg)).numpy()
    direct, precomputed = wm.copy(), wm.copy()
    direct_filled = _edit_until_propagation(direct, aseg, True, None, False)
    precomputed_filled = _edit_until_propagation(precomputed, aseg, True, seed, False)
    np.testing.assert_array_equal(precomputed, direct)
    np.testing.assert_array_equal(precomputed_filled, direct_filled)
    expected = direct.copy()
    _propagate_from_filled(expected, aseg, direct_filled)
    actual = propagate_wm_from_filled_torch(torch.from_numpy(precomputed),
        torch.from_numpy(aseg), torch.from_numpy(precomputed_filled))
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_fixed_filled_propagation_does_not_grow_seed_and_keeps_temporal_labels():
    wm = np.zeros((9, 8, 7), np.uint8)
    aseg = np.zeros(wm.shape, np.int32)
    filled = np.zeros_like(wm)
    filled[3, 3, 3] = 250
    aseg[4, 3, 3] = 186
    aseg[4, 4, 3] = 187
    aseg[5, 3, 3] = 2  # Two steps away: propagation must not become flood-fill.
    actual = propagate_wm_from_filled_torch(torch.from_numpy(wm),
        torch.from_numpy(aseg), torch.from_numpy(filled)).numpy()
    assert actual[4, 3, 3] == actual[4, 4, 3] == 250
    assert actual[5, 3, 3] == 0
    assert np.count_nonzero(filled) == 1
    expected = wm.copy()
    _propagate_from_filled(expected, aseg, filled)
    np.testing.assert_array_equal(actual, expected)


def test_fill_seed_preserves_native_scan_edges_and_cortex_exclusion():
    aseg = np.full((8, 9, 10), 2, np.int32)
    aseg[4, 4, 4] = 3
    seed = fill_seg_wm_seed_torch(torch.from_numpy(aseg)).numpy()
    assert not seed[[0, -1], :, :].any()
    assert not seed[:, [0, -1], :].any()
    assert seed[2, 2, 0] and seed[2, 2, -1]
    assert not seed[3:6, 3:6, 3:6].any()


def test_native_cerebellar_exterior_labels_are_erased_without_cortex():
    wm = np.zeros((12, 13, 14), np.uint8)
    aseg = np.zeros(wm.shape, np.int32)
    wm[4, 5, 6] = wm[7, 5, 6] = 110
    aseg[4, 5, 6] = 6
    aseg[7, 5, 6] = 45
    _edit_until_propagation(wm, aseg)
    assert wm[4, 5, 6] == wm[7, 5, 6] == 0


def test_scm_dilation_radius_and_fill_match_cpu():
    wm, aseg, _ = _fixture()
    for radius in (0, 1, 2, 4):
        actual = fix_subcortical_mass_ha_torch(torch.from_numpy(wm),
                                              torch.from_numpy(aseg), ndilate=radius)
        np.testing.assert_array_equal(actual.numpy(), fix_subcortical_mass_ha(wm, aseg, radius))
    np.testing.assert_array_equal(fill_seg_wm_from_core_torch(
        torch.from_numpy(wm), torch.from_numpy(aseg)).numpy(), fill_seg_wm_from_core(wm, aseg))


def test_mtl_spackle_and_below_hippocampus_match_source_order():
    wm, aseg, _ = _fixture()
    for chunk in (1, 127, 512):
        actual = spackle_wm_superior_to_mtl_torch(torch.from_numpy(wm),
                    torch.from_numpy(aseg), point_chunk=chunk)
        np.testing.assert_array_equal(actual.numpy(), spackle_wm_superior_to_mtl(wm, aseg))
    expected = wm.copy()
    _add_aseg_wm_below_hippocampus(expected, aseg)
    np.testing.assert_array_equal(add_aseg_wm_below_hippocampus_torch(
        torch.from_numpy(wm), torch.from_numpy(aseg)).numpy(), expected)


def test_spackle_handles_continuous_label_ray_and_rejects_undefined_boundary():
    wm = np.zeros((24, 19, 27), np.uint8)
    aseg = np.zeros(wm.shape, np.int32)
    aseg[12, 1:18, 4:20] = 17
    aseg[12, 0, 10] = 3
    actual = spackle_wm_superior_to_mtl_torch(torch.from_numpy(wm), torch.from_numpy(aseg))
    np.testing.assert_array_equal(actual.numpy(), spackle_wm_superior_to_mtl(wm, aseg))
    aseg[12, -1, 10] = 17
    with pytest.raises(ValueError, match="boundary"):
        spackle_wm_superior_to_mtl_torch(torch.from_numpy(wm), torch.from_numpy(aseg))


def test_rejects_invalid_dtype_shape_and_radius():
    wm, aseg, _ = _fixture()
    with pytest.raises(ValueError):
        fix_subcortical_mass_ha_torch(torch.from_numpy(wm).float(), torch.from_numpy(aseg))
    with pytest.raises(ValueError):
        fix_subcortical_mass_ha_torch(torch.from_numpy(wm), torch.from_numpy(aseg), ndilate=-1)
    with pytest.raises(ValueError):
        apply_late_wm_edits_torch(torch.from_numpy(wm), torch.from_numpy(aseg),
                                torch.from_numpy(aseg), torch.from_numpy(wm[:2]))


def test_float_mgh_label_storage_requires_exact_integers():
    labels = np.zeros((5, 6, 7), np.float32)
    labels[2, 3, 4] = 7031
    image = nib.MGHImage(labels, np.eye(4))
    actual = _integer_label_array(image)
    assert actual.dtype == np.int32 and actual[2, 3, 4] == 7031
    labels[2, 3, 4] = 17.5
    with pytest.raises(ValueError, match="noninteger"):
        _integer_label_array(nib.MGHImage(labels, np.eye(4)))


def test_complete_hybrid_checks_original_label_storage_before_integer_conversion():
    shape = (5, 6, 7)
    for dtype in (np.int32, np.float32):
        _validate_hybrid_aseg_storage(nib.MGHImage(np.zeros(shape, dtype), np.eye(4)))
    for dtype in (np.uint8, np.int16):
        with pytest.raises(ValueError, match="original int32 or float32"):
            _validate_hybrid_aseg_storage(nib.MGHImage(np.zeros(shape, dtype), np.eye(4)))
    with TemporaryDirectory() as directory:
        paths = [Path(directory) / f"{name}.mgz" for name in
                 ("wm", "brain", "aseg", "ento", "output")]
        for path, dtype in zip(paths, (np.uint8, np.uint8, np.int16, np.int32)):
            nib.save(nib.MGHImage(np.zeros(shape, dtype), np.eye(4)), path)
        # The storage error must occur before CUDA allocation or geometry proof.
        with pytest.raises(ValueError, match="original int32 or float32"):
            write_wm_asegedit_hybrid_diagnostic(wm_file=paths[0], brain_file=paths[1],
                aseg_file=paths[2], entowm_file=paths[3], output_file=paths[4],
                device="cuda:1", fill_seg_wm=True)


def test_file_api_keeps_grid_dtype_and_matches_late_cpu_with_float_label_storage():
    wm, aseg, ento = _fixture()
    affine = np.diag([1., 1.2, 1.3, 1.])
    affine[:3, 3] = [12., -9., 4.]
    with TemporaryDirectory() as directory:
        paths = [Path(directory) / f"{name}.mgz" for name in
                 ("core", "aseg", "ento", "original", "output")]
        for path, array in zip(paths, (wm, aseg.astype(np.float32), ento, wm)):
            nib.save(nib.MGHImage(array, affine), path)
        report = write_late_wm_edits_torch(*paths, device="cpu", fill_seg_wm=False)
        actual = nib.load(paths[-1])
        expected = apply_late_wm_edits(wm.copy(), aseg, ento, wm, fill_seg_wm=False)
        np.testing.assert_array_equal(np.asarray(actual.dataobj), expected)
        np.testing.assert_array_equal(actual.affine, nib.load(paths[0]).affine)
        assert actual.get_data_dtype() == np.dtype(np.uint8)
        assert report["full_native_core_replaced"] is False
        shifted = affine.copy()
        shifted[0, 3] += 1
        nib.save(nib.MGHImage(aseg, shifted), paths[1])
        with pytest.raises(ValueError, match="grid"):
            write_late_wm_edits_torch(*paths, device="cpu")
