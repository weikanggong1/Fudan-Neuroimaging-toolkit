"""目标搜索并行契约；真实完整四轮基准另存机器收据。"""
import numpy as np
import numba
import pytest

from fnit.recon_all import place_surface_border as border


@pytest.mark.parametrize("surface", ["white", "pial"])
def test_same_body_parallel_keeps_all_six_fields_exact_and_preserves_thread_mask(surface):
    grid = np.indices((24, 12, 12), dtype=np.float32)
    image = np.clip(135 - grid[0] * 5, 0, 255).astype(np.uint8)
    xyz = np.array([[x, y, 5] for x in (8.2, 9.7, 11.1) for y in (3., 5., 7.)], np.float32)
    normals = np.zeros_like(xyz); normals[:, 0] = 1
    normals[-1] = 0
    ripped = np.zeros(len(xyz), np.bool_); ripped[-2] = True
    arguments = dict(volume=image, segmentation=np.zeros_like(image, np.int32),
        surface_xyz=xyz, surface_normals=normals, original_xyz=xyz.copy(),
        ripped=ripped, previous_values=np.full(len(xyz), -1, np.float32),
        sras2vox=np.eye(4, dtype=np.float32),
        thresholds=np.array([125., 100., 40., 5., 70.]),
        hemisphere="lh", surface=surface, sigma=.5, max_thickness=5.)
    previous_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        serial = border.compute_border_values_first_pass(**arguments)
        parallel = border.compute_border_values_first_pass(**arguments, target_search_backend="parallel")
        for left, right in zip(serial, parallel):
            np.testing.assert_array_equal(left, right)
        assert numba.get_num_threads() == min(2, numba.config.NUMBA_NUM_THREADS)
        assert serial[4][:-2].any()  # 非退化搜索，不能只比较全部rip或全零。
        assert not serial[4][-2:].any()
    finally:
        numba.set_num_threads(previous_threads)


def test_dispatchers_share_algorithm_but_use_distinct_numba_cache_identity():
    assert border._search.py_func.__code__.co_code == border._search_parallel.py_func.__code__.co_code
    assert border._search._cache._impl.filename_base != border._search_parallel._cache._impl.filename_base
    assert not border._search.targetoptions.get("parallel", False)
    assert border._search_parallel.targetoptions["parallel"] is True
    assert not border._search_parallel.targetoptions["fastmath"]


@pytest.mark.parametrize("runner_name", ["border", "preaparc", "pial", "final_white"])
def test_invalid_target_backend_fails_before_missing_input_or_compilation(tmp_path, runner_name):
    with pytest.raises(ValueError, match="target_search_backend"):
        if runner_name == "border":
            border.compute_border_values_first_pass(*([None] * 9), hemisphere="lh",
                target_search_backend="unsupported")
        elif runner_name == "preaparc":
            from fnit.recon_all.place_white_preaparc_python import place_white_preaparc
            place_white_preaparc(subject_dir=tmp_path, hemi="lh", output=tmp_path/"out",
                target_search_backend="unsupported")
        elif runner_name == "pial":
            from fnit.recon_all.place_pial_python import place_pial_t1
            place_pial_t1(subject=tmp_path, hemisphere="lh", output=tmp_path/"out",
                target_search_backend="unsupported")
        else:
            from fnit.recon_all.place_final_white_python import place_final_white
            place_final_white(subject_dir=tmp_path, hemi="lh", output=tmp_path/"out",
                target_search_backend="unsupported")
