from types import SimpleNamespace

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pytest
from scipy import ndimage
import torch

from fnit.fast.algorithm import FASTConfig
from fnit.gems.context import SubregionContext
from fnit.gems import pipeline, preprocessing
from fnit.gems.recipes.base import RecipeResult


def _context(*, spacing=1.0, shape=(8, 8, 8)):
    data = (100.25 + np.indices(shape)[0] * .125).astype(np.float32)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = (1.25, -2.5, 3.75)
    image = nib.Nifti1Image(data, affine)
    coarse = np.full(shape, 2, np.int32)
    coarse[0, 0, 0] = 24
    return SubregionContext(image, data, coarse, np.full(shape, 1007, np.int32),
                            np.full(shape, 3007, np.int32), {"coarse_source": "SynthSegPlus"})


def _mock_fast(monkeypatch, context, *, zero_restored=False):
    import fnit.fast
    calls = []

    class Fast:
        def __init__(self, *, device, threads):
            assert str(device) == "cpu" and threads == 3
            self.config = FASTConfig()

        def __call__(self, image, mask):
            calls.append((image, mask))
            assert np.asarray(image.dataobj).dtype == np.float32
            assert np.array_equal(mask.dataobj, context.coarse_segmentation > 0)
            assert np.asarray(mask.dataobj)[0, 0, 0] == 1  # CSF is inside the mask.
            np.testing.assert_array_equal(image.affine, context.image.affine)
            restored = np.zeros(image.shape, np.float32) if zero_restored else context.data / 2
            return SimpleNamespace(restored=nib.Nifti1Image(restored, image.affine),
                                   tissue_means=(20.0, 60.0, 100.0))

    monkeypatch.setattr(fnit.fast, "TorchFAST", Fast)
    return calls


def test_raw_preparation_reuses_default_fast_and_keeps_float_intensity(monkeypatch):
    context = _context()
    original_data = context.data.copy()
    calls = _mock_fast(monkeypatch, context)
    grid_affine = context.image.affine.copy()
    grid_affine[0, 3] += .25
    monkeypatch.setattr(preprocessing, "_coronal_grid", lambda image, data: (image.shape, grid_affine))
    prepared = preprocessing.prepare_automatic_raw_input(context, device="cpu", threads=3)
    assert len(calls) == 1 and prepared.native_image is context.image
    assert np.array_equal(context.data, original_data)
    assert prepared.data.dtype == np.float32 and np.any(prepared.data != np.floor(prepared.data))
    np.testing.assert_array_equal(prepared.image.affine, grid_affine)
    assert prepared.coarse_segmentation.dtype == prepared.cortical_parcellation.dtype == np.int32
    assert set(np.unique(prepared.coarse_segmentation)) <= {0, 2, 24}
    assert set(np.unique(prepared.cortical_parcellation)) <= {0, 1007}
    assert set(np.unique(prepared.wmparc_proxy)) <= {0, 3007}
    record = prepared.metadata["intensity_preprocessing"]
    mask = ndimage.binary_erosion(np.isin(context.coarse_segmentation, (2, 41)))
    expected_median = float(np.median((context.data / 2)[mask]))
    assert record["intensity_scale"] == 110 / expected_median
    assert record["applied"] and record["fast_config"]["execution"] == "tensor"
    assert record["fast_config"]["bias_iterations"] == 4
    assert record["bias_correction_seconds"] >= 0 and record["wm_median_before_scale"] == expected_median
    assert record["processing_geometry"]["affine"] == grid_affine.tolist()
    assert record["original_geometry"]["affine"] == context.image.affine.tolist()


def test_high_resolution_raw_keeps_its_sampling_and_hippocampal_resolution_branch(monkeypatch):
    context = _context(spacing=.8)
    _mock_fast(monkeypatch, context)
    def unexpected_grid(*args):
        raise AssertionError("high-resolution input must keep its grid")
    monkeypatch.setattr(preprocessing, "_coronal_grid", unexpected_grid)
    prepared = preprocessing.prepare_automatic_raw_input(context, device="cpu", threads=3)
    np.testing.assert_array_equal(prepared.image.affine, context.image.affine)
    assert prepared.image.shape == context.image.shape
    assert prepared.coarse_segmentation is context.coarse_segmentation
    assert prepared.wmparc_proxy is context.wmparc_proxy
    assert prepared.metadata["intensity_preprocessing"]["grid_rule"] == "preserve_high_resolution_native_grid"
    mask = ndimage.binary_erosion(np.isin(context.coarse_segmentation, (2, 41)))
    assert float(np.median(prepared.data[mask])) == pytest.approx(110)


def test_cuda_preparation_releases_finished_model_cache_before_fast(monkeypatch):
    import fnit.fast
    context = _context(spacing=.8)
    events = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("synchronize"))
    class Fast:
        def __init__(self, *, device, threads):
            assert str(device) == "cuda:0" and events == ["empty_cache"]
            self.config = FASTConfig()
            events.append("init")
        def __call__(self, image, mask):
            events.append("estimate")
            return SimpleNamespace(restored=image, tissue_means=(20., 60., 100.))
    monkeypatch.setattr(fnit.fast, "TorchFAST", Fast)
    prepared = preprocessing.prepare_automatic_raw_input(context, device="cuda:0", threads=3)
    assert prepared.metadata["intensity_preprocessing"]["applied"]
    assert events == ["empty_cache", "init", "synchronize", "estimate", "synchronize"]


@pytest.mark.parametrize("after_fast", (False, True))
def test_sparse_positive_wm_fallback_preserves_original_input(monkeypatch, after_fast):
    context = _context(shape=(8, 8, 8) if after_fast else (5, 5, 5))
    calls = _mock_fast(monkeypatch, context, zero_restored=after_fast)
    prepared = preprocessing.prepare_automatic_raw_input(context, device="cpu", threads=3)
    assert len(calls) == int(after_fast)
    assert prepared.image is context.image and prepared.data is context.data
    assert prepared.coarse_segmentation is context.coarse_segmentation
    assert prepared.native_image is context.image
    record = prepared.metadata["intensity_preprocessing"]
    assert not record["applied"] and record["intensity_scale"] == 1
    assert "fewer_than_100" in record["fallback_reason"]
    assert record["grid_rule"] == "preserve_original_input"


@pytest.mark.parametrize("shape,spacing,width", (((32, 33, 34), 1.0, 256),
                                                  ((129, 130, 132), 2.0, 256),
                                                  ((150, 150, 150), 2.0, 300)))
def test_coronal_grid_uses_only_original_header_center_and_width_rule(shape, spacing, width):
    context = _context(shape=shape, spacing=spacing)
    target_shape, target_affine = preprocessing._coronal_grid(context.image, context.data)
    assert target_shape == (width, width, width)
    np.testing.assert_array_equal(target_affine[:3, :3], [[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    original_center = nib.affines.apply_affine(context.image.affine, np.asarray(shape) / 2)
    target_center = nib.affines.apply_affine(target_affine, np.full(3, width / 2))
    np.testing.assert_allclose(target_center, original_center, atol=2e-5, rtol=0)


def test_processing_winners_confidence_and_support_share_nearest_native_sampling():
    native_affine = np.diag([2., 2., 2., 1.]);native_affine[:3, 3] = .15
    processing_affine = np.eye(4);processing_affine[:3, 3] = .25
    native = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), native_affine)
    processing = nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), processing_affine)
    context = SubregionContext(processing, np.asarray(processing.dataobj), None, None, None,
                               {}, native)
    processing_labels = np.zeros(processing.shape, np.int32)
    processing_labels[2, 2, 2] = processing_labels[4, 4, 4] = 174
    processing_confidence = np.zeros(processing.shape, np.float32)
    processing_confidence[2, 2, 2], processing_confidence[4, 4, 4] = .95, 1.
    support = np.zeros(processing.shape, bool)
    support[2, 2, 2] = True
    interpolated = resample_from_to(nib.Nifti1Image(processing_confidence, processing.affine),
                                   (native.shape, native.affine), order=1)
    assert float(np.asarray(interpolated.dataobj)[1, 1, 1]) < .8
    # All three products must select processing voxel (2,2,2). Interpolating
    # confidence would incorrectly prevent this winner from replacing label175.
    outcome = SimpleNamespace(native_labels=processing_labels,
                              native_confidence=processing_confidence,
                              native_support=support)
    labels, confidence, sampled_support = pipeline._outcome_on_native_grid(outcome, context, native)
    combined, best = np.zeros(native.shape, np.int32), np.zeros(native.shape, np.float32)
    combined[1, 1, 1], best[1, 1, 1] = 175, .8
    pipeline._merge_native(combined, best, labels, confidence, sampled_support)
    assert labels[1, 1, 1] == labels[2, 2, 2] == 174
    assert confidence[1, 1, 1] == pytest.approx(.95) and confidence[2, 2, 2] == 1
    assert sampled_support[1, 1, 1] and not sampled_support[2, 2, 2]
    assert combined[1, 1, 1] == 174 and combined[2, 2, 2] == 0
    assert np.count_nonzero(combined) == 1 and best[1, 1, 1] == pytest.approx(.95)


@pytest.mark.parametrize("automatic", (False, True))
def test_pipeline_preparation_scope_native_output_and_volume_contract(tmp_path, monkeypatch, automatic):
    import fnit.gems.recipes as recipes
    native = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.diag([2., 2., 2., 1.]))
    image = nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)) if automatic else native
    context = SubregionContext(native, np.asarray(native.dataobj), np.full(native.shape, 2, np.int32),
                               None, None, {"coarse_source": "provided"})
    directory = tmp_path / "brainstem";directory.mkdir();(directory / "AtlasMesh.gz").touch()
    monkeypatch.setattr(SubregionContext, "prepare", lambda *args, **kwargs: context)
    prepared_contexts = []
    def prepare(current, *, device, threads):
        assert automatic and current is context and str(device) == "cpu" and threads == 4
        result = SubregionContext(image, np.asarray(image.dataobj), np.full(image.shape, 2, np.int32),
                                  None, None, {"intensity_preprocessing": {"intensity_scale": 2.,
                                  "bias_correction_seconds": .25, "fast_config": {"execution": "tensor"}}}, native)
        prepared_contexts.append(result)
        return result
    monkeypatch.setattr(preprocessing, "prepare_automatic_raw_input", prepare)
    labels = np.zeros((16, 16, 16), np.int32);labels[4, 4, 4] = 173
    fine_affine = np.diag([.5, .5, .5, 1.])
    highres = nib.Nifti1Image(labels, fine_affine)
    class Recipe:
        def __init__(self):self.directory = directory
        def run(self, current, device):
            assert current.image is image
            output = np.zeros(image.shape, np.int32)
            output[2 if automatic else 1, 2 if automatic else 1, 2 if automatic else 1] = 173
            return RecipeResult(SimpleNamespace(labels=torch.as_tensor(labels), affine=fine_affine),
                                highres, output, np.full(image.shape, .8, np.float32),
                                np.ones(image.shape, bool), {173: 4.5}, {})
    monkeypatch.setattr(recipes, "make_recipe", lambda name, root: Recipe())
    monkeypatch.setattr(pipeline.GEMSAtlas, "from_freesurfer", lambda *args: SimpleNamespace(
        label_ids=np.asarray([0, 173]), label_names=("Unknown", "Midbrain")))
    result = pipeline.segment_4_subregions(native, tmp_path, structures="brainstem", device="cpu",
                                          coarse_segmentation=None if automatic else context.coarse_segmentation)
    assert len(prepared_contexts) == int(automatic)
    assert result.labels.shape == result.confidence.shape == native.shape
    np.testing.assert_array_equal(result.labels.affine, native.affine)
    assert np.asarray(result.labels.dataobj)[1, 1, 1] == 173
    assert result.volumes[173]["soft_volume_mm3"] == 4.5
    assert result.volumes[173]["hard_volume_mm3"] == pytest.approx(8.0)
    assert result.structure_results["brainstem"].highres_labels is highres
    if automatic:
        shared = result.initialization["shared_preprocessing"]["intensity_preprocessing"]
        assert shared["intensity_scale"] == 2 and shared["bias_correction_seconds"] == .25
        assert shared["fast_config"]["execution"] == "tensor"
    assert result.timings["compute_seconds"] > 0
