"""覆盖顺序、舍入、标签掩膜和颜色表的有意义回归。"""
import gzip
from pathlib import Path
import tempfile
import unittest
import nibabel as nib
import numpy as np
import torch

from fnit.recon_all.defects_label_volume_torch import (
    _color_tag, _ctab, _footer_and_colors, defects_to_volume, project_defects,
)


def test_last_vertex_wins_even_when_label_is_smaller():
    # 两个重叠面，后遍历顶点标签1，不能错误地选择最大缺陷号9。
    vertices = torch.tensor([[1., 1., 1.], [1., 2., 1.], [2., 1., 1.]] * 2)
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    labels = torch.tensor([9, 0, 0, 1, 0, 0])
    result = project_defects(vertices, faces, labels, torch.zeros((4, 4, 4)),
                             torch.eye(4), offset=1000, voxel_size_x=1.)
    assert result.dtype == torch.int32
    assert int(result[1, 1, 1]) == 1001
    assert int(torch.count_nonzero(result)) == 1


def test_merge_cortex_and_zero_face():
    vertices = torch.tensor([[1., 1., 1.]] * 3)
    faces = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([1, 0, 2])
    template = torch.full((3, 3, 3), 1001, dtype=torch.int32)
    cortex = torch.tensor([True, False, False])
    result = project_defects(vertices, faces, labels, template, torch.eye(4),
                             offset=2000, voxel_size_x=1., merge=True, cortex=cortex)
    assert int(result[1, 1, 1]) == 2001
    assert int(result[0, 0, 0]) == 1001
    assert int(template[1, 1, 1]) == 1001  # 不修改调用方模板


def test_half_integer_rounding_and_out_of_bounds():
    vertices = torch.tensor([[.5, .5, .5]] * 3 + [[-.5, -.5, -.5]] * 3)
    result = project_defects(vertices, torch.tensor([[0, 1, 2], [3, 4, 5]]),
                             torch.tensor([1, 0, 0, 2, 0, 0]), torch.zeros((3, 3, 3)),
                             torch.eye(4), offset=1000, voxel_size_x=1.)
    assert int(result[1, 1, 1]) == 1001
    assert int(torch.count_nonzero(result)) == 1


def test_color_table_serialization_and_invalid_input():
    entries = {0: ("unknown", 0, 0, 0, 255), 1001: ("Defect-1001", 3, 7, 9, 0)}
    encoded = _color_tag(entries)
    end, decoded = _ctab(encoded, 0)
    assert end == len(encoded) and decoded == entries
    with unittest.TestCase().assertRaises(ValueError):
        project_defects(torch.zeros((3, 3)), torch.tensor([[0, 1, 5]]),
                        torch.ones(3, dtype=torch.int64), torch.zeros((2, 2, 2)),
                        torch.eye(4), offset=1000, voxel_size_x=1.)


def test_fortran_order_template_is_written_without_copy_loss():
    vertices = torch.tensor([[1., 1., 1.]] * 3)
    template = torch.from_numpy(np.zeros((3, 3, 3), order="F", dtype=np.int32))
    assert not template.is_contiguous()
    result = project_defects(vertices, torch.tensor([[0, 1, 2]]),
                             torch.tensor([1, 0, 0]), template, torch.eye(4),
                             offset=1000, voxel_size_x=1.)
    assert int(result[1, 1, 1]) == 1001


def test_mgh_header_padding_and_second_hemisphere_merge():
    with tempfile.TemporaryDirectory() as directory:
        _check_mgh_merge(Path(directory))


def _check_mgh_merge(tmp_path):
    # nibabel的字段块只有90字节，数据偏移284；完整写出必须可再次读取合并。
    template = tmp_path / "orig.mgz"
    nib.save(nib.MGHImage(np.zeros((4, 4, 4), np.uint8), np.eye(4)), template)
    surface, overlay = tmp_path / "orig.nofix", tmp_path / "defect_labels"
    geometry = dict(head=np.array([2, 0, 20]), valid="1", filename="orig.mgz",
                    volume=np.array([4, 4, 4]), voxelsize=np.ones(3),
                    xras=np.array([1., 0., 0.]), yras=np.array([0., 1., 0.]),
                    zras=np.array([0., 0., 1.]), cras=np.zeros(3))
    nib.freesurfer.io.write_geometry(surface, np.zeros((3, 3)),
                                     np.array([[0, 1, 2]]), volume_info=geometry)
    nib.freesurfer.io.write_morph_data(overlay, np.array([1, 0, 0], np.float32))
    output = tmp_path / "surface.defects.mgz"
    defects_to_volume(surface_file=surface, defect_file=overlay, template_file=template,
                      output_file=output, offset=1000, device="cpu")
    first = nib.load(output)
    assert first.header.get_data_offset() == 284
    assert first.get_data_dtype() == np.dtype(">i4")
    assert np.count_nonzero(np.asarray(first.dataobj) == 1001) == 1
    defects_to_volume(surface_file=surface, defect_file=overlay, template_file=output,
                      output_file=output, offset=2000, merge=True, device="cpu")
    second = nib.load(output)
    assert np.count_nonzero(np.asarray(second.dataobj) == 2001) == 1
    _, colors = _footer_and_colors(gzip.decompress(output.read_bytes()), 284 + 4**3 * 4)
    assert colors[1001][0] == "Defect-1001" and colors[2001][0] == "Defect-2001"
    np.testing.assert_array_equal(second.affine, first.affine)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
def test_cuda_label_priority_matches_cpu():
    args = (torch.tensor([[1., 1., 1.], [1., 2., 1.], [2., 1., 1.]] * 2),
            torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.tensor([9, 0, 0, 1, 0, 0]),
            torch.zeros((4, 4, 4)), torch.eye(4))
    expected = project_defects(*args, offset=1000, voxel_size_x=1.)
    actual = project_defects(*(value.cuda() for value in args), offset=1000, voxel_size_x=1.)
    assert torch.equal(expected, actual.cpu())
