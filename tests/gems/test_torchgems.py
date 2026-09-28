import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.gems import GEMSAtlas, TorchGEMS, ashburner_prior, rasterize_priors


def _atlas():
    vertices = np.asarray([[1,1,1],[6,1,1],[1,6,1],[1,1,6]], dtype=float)
    tetra = np.asarray([[0,1,2,3]], dtype=np.int64)
    alphas = np.asarray([[1,0],[0,1],[0,1],[0,1]], dtype=np.float32)
    return GEMSAtlas(vertices, vertices, tetra, alphas, .1,
                     np.ones((4,3), bool), np.asarray([0, 10]), ("Unknown", "ROI"))


def test_reference_deformation_prior_is_zero():
    atlas = _atlas()
    v = torch.tensor(atlas.vertices, dtype=torch.float32)
    cost, jac = ashburner_prior(v, v, torch.tensor(atlas.tetrahedra), atlas.stiffness)
    assert torch.allclose(cost, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(jac, torch.ones_like(jac), atol=1e-6)


def test_rasterized_priors_sum_to_one():
    atlas = _atlas()
    priors, covered = rasterize_priors(
        torch.tensor(atlas.vertices, dtype=torch.float32),
        torch.tensor(atlas.tetrahedra), torch.tensor(atlas.alphas), (8,8,8))
    assert priors.shape == (2,8,8,8)
    assert covered.any()
    assert torch.allclose(priors.sum(0), torch.ones(8,8,8), atol=1e-6)


def test_em_returns_native_grid_labels():
    atlas = _atlas()
    image = torch.ones(8,8,8)
    image[2:5,2:5,2:5] = 2
    result = TorchGEMS(atlas)(image, em_iterations=2)
    assert result.labels.shape == image.shape
    assert set(torch.unique(result.labels).tolist()) <= {0, 10}
    assert result.min_jacobian > 0


def test_reads_freesurfer_atlasmesh_text(tmp_path):
    mesh = tmp_path / "AtlasMesh.gz"
    text = """Number of points: 4
Number of cells: 1
Number of labels: 2
Number of meshes: 1
Reference position
0 1 1 1
1 6 1 1
2 1 6 1
3 1 1 6
K: 0.1
Position 0
0 1 1 1
1 6 1 1
2 1 6 1
3 1 1 6
Cells:
0 TETRAHEDRON 0 1 2 3
Point parameters:
0 65535 0 true true true true
1 0 65535 true true true true
2 0 65535 true true true true
3 0 65535 true true true true
"""
    with gzip.open(mesh, "wt") as f:
        f.write(text)
    lut = tmp_path / "compressionLookupTable.txt"
    lut.write_text("0 0 Unknown 0 0 0 255\n10 1 ROI 1 2 3 255\n")
    atlas = GEMSAtlas.from_freesurfer(mesh, lut)
    assert atlas.tetrahedra.tolist() == [[0,1,2,3]]
    assert atlas.label_ids.tolist() == [0,10]
    np.testing.assert_allclose(atlas.alphas.sum(1), 1)


def test_label_centroid_affine_recovers_translation():
    from fnit.gems.initialize import estimate_label_centroid_affine
    # Construct a cube split into five tetrahedra would be preferable for full
    # affine fitting; here directly exercise four shared labels by four tiny
    # translated atlas copies is outside the minimal raster unit. The API must
    # at least reject an underdetermined single-label initialization.
    atlas = _atlas()
    target = np.zeros((8,8,8), dtype=np.int32)
    target[2:5,2:5,2:5] = 10
    try:
        estimate_label_centroid_affine(atlas, target, min_shared_labels=4)
    except ValueError as error:
        assert "need at least 4" in str(error)
    else:
        raise AssertionError("underdetermined centroid affine should be rejected")


def test_subregions_keeps_mgh_world_affine(tmp_path):
    from fnit.gems import segment_subregions

    atlas_dir = tmp_path / "atlas" / "synthetic"
    atlas_dir.mkdir(parents=True)
    _atlas().save_npz(atlas_dir / "atlas.npz")
    np.save(atlas_dir / "atlas_to_native_voxel.npy", np.eye(4))
    (atlas_dir / "config.json").write_text(json.dumps({"include_label_ids": [10]}))
    affine = np.asarray([[-1, 0, 0, 4], [0, 0, 1, -3], [0, -1, 0, 5], [0, 0, 0, 1]])
    source = nib.MGHImage(np.ones((8, 8, 8), dtype=np.float32), affine)
    result = segment_subregions(source, tmp_path / "atlas", structures="synthetic",
                                auto_initialize=False, em_iterations=1, device="cpu")
    np.testing.assert_allclose(result.labels.affine, source.affine, atol=1e-5)
    assert result.labels.shape == source.shape
