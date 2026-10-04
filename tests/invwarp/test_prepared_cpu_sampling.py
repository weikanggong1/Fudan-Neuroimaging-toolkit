"""Prepared CPU sources keep default sampling and final validity contracts."""
import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.applywarp.core import _sample_linear, _spatial_grid, _fsl_voxel_matrix
from fnit.convertwarp.core import _PullField
from fnit.fnirt.io import make_fsl_coefficient_image
from fnit.fnirt.spline import fsl_control_shape


@pytest.mark.parametrize("kind", ["relative", "absolute", "coefficient"])
@pytest.mark.parametrize("threads", [1, 8])
def test_prepared_source_retains_mapped_bits_and_final_validity(kind, threads):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        shape = (11, 13, 17)
        geometry = np.diag([-2.0, 2.5, 3.0, 1.0])
        reference = nib.Nifti1Image(np.zeros(shape, np.float32), geometry)
        if kind == "coefficient":
            controls = fsl_control_shape(shape, (3, 3, 3))
            values = np.random.default_rng(97).normal(0, 0.1, (*controls, 3)).astype(np.float32)
            affine = np.diag([1.03, 0.98, 1.02, 1.0])
            affine[:3, 3] = [0.2, -0.1, 0.05]
            image = make_fsl_coefficient_image(values, shape, (2, 2.5, 3), (3, 3, 3), affine)
            convention = "auto"
        else:
            values = np.random.default_rng(97).normal(0, 0.1, (*shape, 3)).astype(np.float32)
            if kind == "absolute":
                values += np.moveaxis(_spatial_grid(shape, _fsl_voxel_matrix(reference), torch.device("cpu")).reshape(3, *shape).numpy(), 0, -1)
            image = nib.Nifti1Image(values, geometry)
            convention = kind
        field = _PullField(image, torch.device("cpu"), convention)
        # More than 32,768 queries exercises the threaded sampler batching;
        # include points on and outside all field boundaries.
        axes = [torch.linspace(-3.0, (n + 2) * step, count, dtype=torch.float64)
                for n, step, count in zip(shape, (2, 2.5, 3), (33, 35, 37))]
        query = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        old_mapped, old_valid = field.sample(query)
        source = field.values[None].contiguous(memory_format=torch.channels_last_3d)
        new_mapped, new_valid = field.sample(query, prepared_source=source)
        without_flag, absent_flag = field.sample(query, prepared_source=source, calculate_valid=False)
        assert torch.equal(old_mapped.view(torch.int64), new_mapped.view(torch.int64))
        assert torch.equal(old_mapped.view(torch.int64), without_flag.view(torch.int64))
        assert torch.equal(old_valid, new_valid)
        assert not old_valid.all() and old_valid.any()
        assert absent_flag is None
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "layout"])
def test_prepared_source_rejects_mismatched_image_contract(invalid):
    data = torch.ones((3, 4, 5, 6), dtype=torch.float32)
    query = torch.zeros((3, 2, 2, 2), dtype=torch.float64)
    source = data[None].contiguous(memory_format=torch.channels_last_3d)
    if invalid == "shape":
        source = source[:, :, :-1]
    elif invalid == "dtype":
        source = source.double()
    else:
        source = source.contiguous()
    with pytest.raises(ValueError, match="prepared_source must match"):
        _sample_linear(data, query, prepared_source=source)
