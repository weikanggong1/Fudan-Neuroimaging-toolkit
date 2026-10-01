import numpy as np
import nibabel as nib
import torch
import pytest
import json
from types import SimpleNamespace

from fnit.gems.atlas import GEMSAtlas, read_compression_lut
from fnit.gems.context import SubregionContext, build_wmparc_proxy
from fnit.gems.pipeline import _expand_structures, _merge_native
from fnit.gems.recipes import HippoAmygdalaRecipe, ThalamusRecipe
from fnit.gems.recipes.base import GEMSRecipe


def _atlas(ids, names):
    vertices = np.asarray([[1, 1, 1], [5, 1, 1], [1, 5, 1], [1, 1, 5]], float)
    alpha = np.full((4, len(ids)), 1 / len(ids), np.float32)
    return GEMSAtlas(vertices, vertices, np.asarray([[0, 1, 2, 3]]), alpha, 0.05,
                     np.ones((4, 3), bool), ids, names)


def test_structure_alias_and_support_first_merge():
    assert _expand_structures("hippo-amygdala") == ["hippo-amygdala-left", "hippo-amygdala-right"]
    assert _expand_structures(["brainstem", "thalamus", "brainstem"]) == ["brainstem", "thalamus"]
    labels = np.zeros((2, 2, 2), np.int32)
    confidence = np.zeros(labels.shape, np.float32)
    first = np.full(labels.shape, 173, np.int32)
    _merge_native(labels, confidence, first, np.full(labels.shape, 0.9),
                  np.asarray([[[True, False], [False, False]],
                              [[False, False], [False, False]]]))
    second = np.full(labels.shape, 8109, np.int32)
    _merge_native(labels, confidence, second, np.full(labels.shape, 0.8),
                  np.asarray([[[True, True], [False, False]],
                              [[False, False], [False, False]]]))
    assert labels[0, 0, 0] == 173
    assert labels[0, 0, 1] == 8109
    assert np.count_nonzero(labels) == 2


def test_wmparc_proxy_stays_in_own_hemisphere_and_white_matter():
    coarse = np.zeros((12, 6, 4), np.int32)
    coarse[:6, 1:5] = 2
    coarse[6:, 1:5] = 41
    cortex = np.zeros_like(coarse)
    cortex[0, 2, 1] = 1006
    cortex[0, 3, 1] = 1007
    cortex[0, 4, 1] = 1016
    cortex[11, 2, 1] = 2006
    cortex[11, 3, 1] = 2007
    cortex[11, 4, 1] = 2016
    proxy = build_wmparc_proxy(coarse, cortex)
    assert proxy.shape == coarse.shape
    assert set(np.unique(proxy[coarse == 2])) <= {3006, 3007, 3016}
    assert set(np.unique(proxy[coarse == 41])) <= {4006, 4007, 4016}
    assert set(np.unique(proxy[coarse == 0])) == {0}
    assert all(np.any(proxy == value) for value in (3006, 3007, 3016,
                                                   4006, 4007, 4016))
    distant = np.zeros((40, 2, 2), np.int32)
    distant[1:39] = 2
    parc = np.zeros_like(distant)
    parc[0] = 1006
    local = build_wmparc_proxy(distant, parc, max_distance_mm=10)
    assert local[5, 0, 0] == 3006 and local[30, 0, 0] == 2


def test_native_geometry_rejects_mismatched_auxiliary_image():
    from fnit.gems.context import _native_labels
    source = nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4))
    shifted = nib.Nifti1Image(np.ones((8, 8, 8), np.int16),
                             np.diag([1, 1, 1, 1]).astype(float))
    shifted.affine[0, 3] = 2
    try:
        _native_labels(shifted, source, "wmparc")
    except ValueError as error:
        assert "shape and affine" in str(error)
    else:
        raise AssertionError("mismatched native affine must be rejected")


def test_official_label_groups_include_two_thalamic_components(tmp_path):
    from importlib.resources import files
    thal = files("fnit").joinpath("gems/data/thalamus_compressionLookupTable.txt")
    hippo = files("fnit").joinpath("gems/data/hippo_compressionLookupTable.txt")
    thal_atlas = _atlas(*read_compression_lut(thal))
    hippo_atlas = _atlas(*read_compression_lut(hippo))
    recipe = ThalamusRecipe("thalamus", tmp_path)
    assert recipe.segmentation_groups(thal_atlas).max() == 13
    first = recipe.intensity_groups(thal_atlas, 0)
    second = recipe.intensity_groups(thal_atlas, 1)
    assert first.max() == 13 and second.max() == 14
    name_index = {name: index for index, name in enumerate(thal_atlas.label_names)}
    assert second[name_index["Left-PuA"]] != second[name_index["Left-CL"]]
    assert second[name_index["Left-CL"]] == 13
    assert second[name_index["Left-PuA"]] == 14
    left = HippoAmygdalaRecipe("left", tmp_path)
    right = HippoAmygdalaRecipe("right", tmp_path)
    synthetic = left.segmentation_groups(hippo_atlas)
    ids = {name: index for index, name in enumerate(hippo_atlas.label_names)}
    assert synthetic[ids["fimbria"]] == synthetic[ids["Left-Cerebral-Cortex"]]
    assert synthetic[ids["fimbria"]] != synthetic[ids["Left-Cerebral-White-Matter"]]
    assert left.intensity_groups(hippo_atlas, 0).max() == 12
    left.high_res_input = True
    highres = left.intensity_groups(hippo_atlas, 0)
    assert highres.max() == 13
    assert highres[ids["molecular_layer_HP-head"]] != highres[ids["CA1-head"]]
    assert left.alignment_ids == (17, 18) and right.alignment_ids == (53, 54)
    assert left.resolution_mm == right.resolution_mm == 0.33333


def test_thalamic_hyperparameters_split_brighter_and_darker(tmp_path):
    ids, names = read_compression_lut(__import__("importlib.resources", fromlist=["files"]).files("fnit").joinpath(
        "gems/data/thalamus_compressionLookupTable.txt"))
    atlas = _atlas(ids, names)
    data = np.full((10, 10, 10), 80, np.float32)
    coarse = np.full(data.shape, 10, np.int32)
    context = SubregionContext(nib.Nifti1Image(data, np.eye(4)), data, coarse, None, None)
    recipe = ThalamusRecipe("thalamus", tmp_path)
    groups = recipe.intensity_groups(atlas, 1)
    means, counts = recipe.gaussian_hyperparameters(context, atlas, groups)
    assert means[13] == 85 and means[14] == 75
    assert counts[13] == counts[14] == 25


def test_thalamic_reticular_group_samples_recoded_bilateral_white_matter(tmp_path):
    atlas = _atlas(np.asarray([0, 2, 8125, 8225, 8109, 8226, 28, 60]),
                   ("Unknown", "WM", "Left-R", "Right-R", "Left-CL", "Right-MDm", "LDC", "RDC"))
    coarse = np.zeros((30, 30, 30), np.int32)
    data = np.full(coarse.shape, 40, np.float32)
    coarse[3:10, 3:10, 3:10] = 2
    coarse[13:20, 3:10, 3:10] = 41
    coarse[3:10, 15:22, 3:10] = 10
    coarse[13:20, 15:22, 3:10] = 49
    data[coarse == 2] = 100
    data[coarse == 41] = 120
    data[np.isin(coarse, (10, 49))] = 80
    original = coarse.copy()
    context = SubregionContext(nib.Nifti1Image(data, np.eye(4)), data, coarse, None, None)
    means, counts = ThalamusRecipe("thalamus", tmp_path).gaussian_hyperparameters(
        context, atlas, np.asarray([0, 1, 1, 1, 2, 2, 3, 3]))
    assert means[1] == 110 and means[2] == 80
    assert counts[1] == counts[2] and counts[1] > 10
    assert np.array_equal(coarse, original)


def test_hippocampal_background_hyperprior_uses_only_local_roi(tmp_path, monkeypatch):
    atlas = _atlas(np.asarray([0, 2, 3]), ("Unknown", "WM", "GM"))
    coarse = np.zeros((31, 31, 31), np.int32)
    coarse[15, 15, 15] = 17
    data = np.full(coarse.shape, 200, np.float32)
    data[10:21, 10:21, 10:21] = 60
    wm = coarse.copy()
    context = SubregionContext(nib.Nifti1Image(data, np.eye(4)), data, coarse, None, wm)
    recipe = HippoAmygdalaRecipe("left", tmp_path)
    monkeypatch.setattr(recipe, "_partial_volume_hyperparameters",
                         lambda atlas, classes, means, counts, context: (means, counts))
    means, counts = recipe.gaussian_hyperparameters(context, atlas, np.arange(3))
    assert means[0] == 60
    assert 1000 < counts[0] < 30000
    assert np.array_equal(wm, coarse)


def test_hippocampal_image_mask_dilates_three_mm_after_linear_resampling(tmp_path):
    coarse = np.full((31, 31, 31), 3, np.int32)
    coarse[15, 15, 15] = 17
    data = np.full(coarse.shape, 100, np.float32)
    context = SubregionContext(nib.Nifti1Image(data, np.eye(4)), data, coarse, None, None)
    image, _, _ = HippoAmygdalaRecipe("left", tmp_path).prepare_working_image(context)
    def sample(point):
        index = np.rint((np.linalg.inv(image.affine) @ [*point, 1])[:3]).astype(int)
        return np.asarray(image.dataobj)[tuple(index)]
    assert sample((15, 15, 15)) == 100
    assert sample((18, 18, 18)) == 100
    assert sample((20, 15, 15)) == 0


def test_recipe_smooths_transformed_mesh_and_ignores_population_cache(tmp_path, monkeypatch):
    import json
    import fnit.gems.smoothing as smoothing
    atlas = _atlas(np.asarray([0, 10]), ("Unknown", "ROI"))
    classes = np.asarray([0, 1])
    cached_name = "seg-sigma1-first.npy"
    np.save(tmp_path / cached_name, atlas.alphas)
    (tmp_path / "config.json").write_text(json.dumps({
        "alpha_label_classes": {cached_name: classes.tolist()}}))
    references = []
    def smooth(current_atlas, current_classes, sigma, *, device, cache=None):
        references.append(current_atlas.reference_vertices.copy())
        return current_atlas.alphas
    monkeypatch.setattr(smoothing, "smooth_atlas_alphas", smooth)
    class Recipe(GEMSRecipe):
        def synthetic_means(self, atlas, classes):
            return np.asarray([1., 2.], np.float32)
    recipe = Recipe("test", tmp_path)
    recipe._reference_vertices = atlas.reference_vertices
    transform = np.diag([2., 2., 2., 1.])
    transform[:3, 3] = 5
    recipe._fit(atlas.transformed(transform), np.ones((25, 25, 25), np.float32),
                np.eye(4), classes, ((1., 0),), synthetic=True, device=torch.device("cpu"))
    assert len(references) == 1
    np.testing.assert_allclose(np.ptp(references[0], axis=0),
                               2 * np.ptp(atlas.reference_vertices, axis=0))


def test_gpu_cpu_fixed_gaussian_consistency_when_cuda_available():
    if not torch.cuda.is_available():
        return
    from fnit.gems.core import TorchGEMS
    from fnit.gems.gaussian import GaussianParameters
    atlas = _atlas(np.asarray([0, 10]), ("Unknown", "ROI"))
    image = np.ones((8, 8, 8), np.float32)
    params_cpu = GaussianParameters(torch.tensor([[1.], [2.]]),
                                    torch.ones((2, 1, 1)))
    cpu = TorchGEMS(atlas, device="cpu")(image, fixed_gaussians=params_cpu,
                                          em_iterations=1)
    params_gpu = GaussianParameters(params_cpu.means.cuda(), params_cpu.covariances.cuda())
    gpu = TorchGEMS(atlas, device="cuda:0")(image, fixed_gaussians=params_gpu,
                                             em_iterations=1)
    np.testing.assert_array_equal(cpu.labels.numpy(), gpu.labels.cpu().numpy())


def test_posterior_soft_volume_uses_working_voxel_volume(tmp_path):
    class TinyRecipe(GEMSRecipe):
        support_ids = (10,)

        def foreground_ids(self, atlas):
            return {10}

    atlas = _atlas(np.asarray([0, 10]), ("Unknown", "ROI"))
    image = nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4))
    context = SubregionContext(image, np.ones(image.shape, np.float32),
                               np.full(image.shape, 10, np.int32), None, None)
    posterior = torch.stack((torch.full(image.shape, 0.75),
                             torch.full(image.shape, 0.25)))
    affine = np.diag([0.5, 0.5, 0.5, 1])
    fit = SimpleNamespace(labels=torch.full(image.shape, 10), posterior=posterior,
                          affine=affine)
    result = TinyRecipe("tiny", tmp_path).postprocess(fit, context, atlas)
    assert result.soft_volumes_mm3[10] == 4 * 4 * 4 * 0.25 * 0.5**3


@pytest.mark.parametrize("preprocessing", ("provided", "SynthSeg", "SynthSegPlus", "provided-coarse-plus"))
def test_all_recipes_merge_on_native_grid_with_metadata(tmp_path, monkeypatch, preprocessing):
    from fnit.gems import pipeline
    import fnit.gems.recipes as recipes
    from fnit.gems.recipes.base import RecipeResult
    import fnit.synthseg_parc as synthseg

    image = nib.Nifti1Image(np.ones((5, 5, 5), np.float32),
                            np.diag([-1, 1, 1, 1]))
    atlas_root = tmp_path / "atlases"
    names = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")
    atlas_labels = (173, 8109, 238, 7001)
    calls = {"SynthSeg": 0, "SynthSegPlus": 0}
    contexts = []

    class FakeSeg:
        kind = "SynthSeg"

        def __init__(self, **kwargs):
            pass

        def __call__(self, source, *, keep_geometry):
            calls[self.kind] += 1
            return SimpleNamespace(
                segmentation=nib.Nifti1Image(np.full(image.shape, 2, np.int32), image.affine),
                cortical_parcellation=nib.Nifti1Image(np.zeros(image.shape, np.int32), image.affine))

    class FakePlus(FakeSeg):
        kind = "SynthSegPlus"

    monkeypatch.setattr(synthseg, "SynthSeg", FakeSeg)
    monkeypatch.setattr(synthseg, "SynthSegPlus", FakePlus)
    for name in names:
        folder = atlas_root / name
        folder.mkdir(parents=True)
        (folder / "AtlasMesh.gz").touch()
        (folder / "compressionLookupTable.txt").touch()

    def fake_atlas(path, lut):
        index = names.index(path.parent.name)
        identifier = atlas_labels[index]
        return _atlas(np.asarray([0, identifier]), ("Unknown", ("Midbrain", "Left-LGN",
                      "CA1-body", "Lateral-nucleus")[index]))

    class FakeRecipe:
        def __init__(self, name):
            self.name = name
            self.directory = atlas_root / name

        def run(self, context, device):
            contexts.append(context)
            index = names.index(self.name)
            label = atlas_labels[index] + (10000 if index == 3 else 0)
            output = np.zeros(image.shape, np.int32)
            output[index, 1, 1] = label
            conf = np.full(image.shape, 0.8, np.float32)
            fit = SimpleNamespace(labels=torch.as_tensor(output), affine=image.affine)
            return RecipeResult(fit, image, output, conf, output != 0,
                                {label: 1.25}, {"seconds": 0.0})

    monkeypatch.setattr(pipeline.GEMSAtlas, "from_freesurfer", fake_atlas)
    monkeypatch.setattr(recipes, "make_recipe", lambda name, root: FakeRecipe(name))
    coarse = np.ones(image.shape, np.int32)
    provided_coarse = preprocessing in ("provided", "provided-coarse-plus")
    provided_wm = preprocessing in ("provided", "SynthSeg")
    result = pipeline.segment_subregions(image, atlas_root,
                                        coarse_segmentation=coarse if provided_coarse else None,
                                        wmparc=coarse if provided_wm else None,
                                        synthseg_weights="unused-mocked-weights",
                                        device="cpu", output_dir=tmp_path / "output")
    assert len(contexts) == 4 and all(context is contexts[0] for context in contexts)
    np.testing.assert_array_equal(contexts[0].coarse_segmentation,
                                  coarse if provided_coarse else np.full(image.shape, 2, np.int32))
    expected_calls = {"SynthSeg": int(preprocessing == "SynthSeg"),
                      "SynthSegPlus": int(not provided_wm)}
    shared = result.initialization["shared_preprocessing"]
    assert calls == shared["model_calls"] == expected_calls
    assert shared["coarse_source"] == ("provided" if provided_coarse else preprocessing)
    assert shared["cortical_parcellation_source"] == (None if provided_wm else "SynthSegPlus")
    assert result.labels.shape == image.shape
    np.testing.assert_array_equal(result.labels.affine, image.affine)
    assert set(result.structure_results) == set(names)
    assert result.mask("Midbrain").sum() == 1
    assert result.mask("Left-LGN").sum() == 1
    assert result.mask("Left-CA1-body").sum() == 1
    assert result.mask("Right-Lateral-nucleus").sum() == 1
    assert result.label_metadata[17001].parent == "amygdala"
    assert result.volumes[17001]["soft_volume_mm3"] == 1.25
    saved = nib.load(result.output_files["labels"])
    np.testing.assert_array_equal(saved.dataobj, result.labels.dataobj)
    np.testing.assert_allclose(saved.affine, image.affine)
    assert len(result.output_files) == 8
    assert (tmp_path / "output" / "labels.tsv").is_file()
    assert "17001\tRight-Lateral-nucleus" in (tmp_path / "output" / "volumes.tsv").read_text()
    assert result.timings["compute_seconds"] > 0 and result.timings["save_seconds"] > 0
    report = json.loads(result.output_files["report"].read_text())
    assert report["initialization"]["shared_preprocessing"] == shared
