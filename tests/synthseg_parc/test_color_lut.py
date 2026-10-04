"""The explicit LUT survives a real NIfTI save/load and is FS-tag readable."""

import io
import struct

import nibabel as nib
import numpy as np
import pytest

from fnit.synthseg_parc.color_lut import attach_color_lut, color_lut_extension


def read_color_table(content):
    stream = io.BytesIO(content)
    assert stream.read(4) == b">\x00\x00\x01"
    tag, length = struct.unpack(">iq", stream.read(12))
    assert tag == 1
    table = io.BytesIO(stream.read(length))
    version, maximum, filename_length, count = struct.unpack(">iiii", table.read(16))
    assert version == -2 and filename_length == 0
    entries = {}
    for _ in range(count):
        label, name_length = struct.unpack(">ii", table.read(8))
        name = table.read(name_length).decode("utf-8").rstrip("\x00")
        color = struct.unpack(">iiii", table.read(16))
        entries[label] = (name, color)
    assert not table.read()
    assert struct.unpack(">iqi", stream.read(16)) == (7, 4, 1)
    assert struct.unpack(">iq", stream.read(12)) == (-1, 1)
    assert stream.read(1) == b"*"
    return entries


def test_explicit_lut_roundtrip_keeps_sparse_ids_rgba_and_geometry(tmp_path):
    lut = tmp_path / "labels.txt"
    lut.write_text("# sparse labels\n0 Unknown 0 0 0 255\n77 WMH 12 34 56 0\n")
    image = nib.Nifti1Image(np.full((3, 4, 5), 77, dtype=np.int32),
                           np.diag([1.5, 2.0, 3.0, 1.0]))
    image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"retained"))
    attach_color_lut(image, lut)
    attach_color_lut(image, lut)
    path = tmp_path / "segmentation.nii.gz"
    nib.save(image, path)
    actual = nib.load(path)
    assert actual.header.extensions.get_codes() == [6, 14]
    assert read_color_table(actual.header.extensions[1].get_content()) == {
        0: ("Unknown", (0, 0, 0, 255)), 77: ("WMH", (12, 34, 56, 0))}
    np.testing.assert_array_equal(actual.affine, image.affine)
    np.testing.assert_array_equal(actual.dataobj, image.dataobj)


@pytest.mark.parametrize("row", ["", "2 Label 3 4 5", "2 Label -1 0 0 0",
                                  "2 Label 3 4 5 0\n2 Duplicate 0 0 0 0",
                                  "-1 Label 0 0 0 0"])
def test_invalid_lut_is_rejected(tmp_path, row):
    path = tmp_path / "invalid.txt"
    path.write_text(row)
    with pytest.raises(ValueError):
        color_lut_extension(path)
