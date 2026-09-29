"""Focused Talairach affine conversion and standalone-stage tests."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fnit.synthmorph.affine_no_surfa import AffineGeometry, AffineTransform


MODULE = Path(os.environ.get(
    "TALAIRACH_MODULE_PATH",
    Path(__file__).resolve().parents[2] / "src/fnit/recon_all/talairach_synthmorph.py"))
spec = importlib.util.spec_from_file_location("talairach_synthmorph", MODULE)
talairach = importlib.util.module_from_spec(spec)
spec.loader.exec_module(talairach)


def _affine():
    source = np.eye(4)
    source[:3, :3] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    source[:3, 3] = [4, -6, 2]
    target = np.eye(4)
    target[:3, :3] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    shape = (16, 18, 20)
    moving = AffineGeometry(shape, source, np.ones(3), source[:3, :3],
                            (source @ np.array([8, 9, 10, 1]))[:3])
    fixed = AffineGeometry(shape, target, np.ones(3), target[:3, :3],
                           (target @ np.array([8, 9, 10, 1]))[:3])
    matrix = np.array([[1.1, 0.03, 0.01, 3.5],
                       [-0.02, 0.95, 0.04, -5.25],
                       [0.01, -0.01, 1.2, 2.75],
                       [0, 0, 0, 1]], np.float32)
    return AffineTransform(matrix, source=moving, target=fixed)


def test_talairach_conversion_preserves_world_coordinate_mapping():
    affine = _affine()
    matrix = talairach.talairach_matrix(affine)
    np.testing.assert_allclose(matrix[:3], affine.matrix[:3], atol=1e-5, rtol=0)


def test_register_talairach_writes_readable_xfm_without_native_program(tmp_path, monkeypatch):
    affine = _affine()
    called = {}

    class FakeModel:
        def __init__(self, **kwargs):
            called.update(kwargs)

        def __call__(self, moving, template, *, header_only):
            called["input"] = (moving, template, header_only)
            return SimpleNamespace(transform=affine)

    monkeypatch.setattr(talairach, "SynthMorph", FakeModel)
    path = tmp_path / "transforms/talairach.xfm"
    matrix = talairach.register_talairach(
        "synthstrip.mgz", "mni305.cor.stripped.mgz", "weights", path, threads=1)
    lines = path.read_text().splitlines()
    assert lines[0] == "MNI Transform File"
    assert lines[4] == "Linear_Transform ="
    assert lines[7].endswith(";")
    parsed = np.array([[float(value) for value in line.rstrip(";").split()]
                       for line in lines[5:8]])
    np.testing.assert_allclose(parsed, matrix[:3], atol=5e-9, rtol=0)
    assert called == {"weights": "weights", "device": "cpu", "model": "affine",
                      "extent": 256,
                      "input": ("synthstrip.mgz", "mni305.cor.stripped.mgz", True)}


def test_register_talairach_disables_tf32_only_for_affine_inference(tmp_path, monkeypatch):
    affine = _affine()
    flags = talairach.torch.backends
    monkeypatch.setattr(flags.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(flags.cudnn, "allow_tf32", True)
    observed = []

    class FakeModel:
        def __init__(self, **kwargs):
            # The real SynthMorph constructor enables both flags.
            flags.cuda.matmul.allow_tf32 = True
            flags.cudnn.allow_tf32 = True

        def __call__(self, moving, template, *, header_only):
            assert header_only
            observed.append((flags.cuda.matmul.allow_tf32, flags.cudnn.allow_tf32))
            return SimpleNamespace(transform=affine)

    monkeypatch.setattr(talairach, "SynthMorph", FakeModel)
    talairach.register_talairach("moving", "template", "weights",
                                 tmp_path / "talairach.xfm", threads=1)
    assert observed == [(False, False)]
    assert (flags.cuda.matmul.allow_tf32, flags.cudnn.allow_tf32) == (False, True)


def test_register_talairach_restores_tf32_when_inference_fails(tmp_path, monkeypatch):
    flags = talairach.torch.backends
    monkeypatch.setattr(flags.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(flags.cudnn, "allow_tf32", False)

    class FailingModel:
        def __init__(self, **kwargs):
            flags.cuda.matmul.allow_tf32 = True
            flags.cudnn.allow_tf32 = True

        def __call__(self, moving, template, *, header_only):
            assert header_only
            assert (flags.cuda.matmul.allow_tf32, flags.cudnn.allow_tf32) == (False, False)
            raise RuntimeError("inference failed")

    monkeypatch.setattr(talairach, "SynthMorph", FailingModel)
    with pytest.raises(RuntimeError, match="inference failed"):
        talairach.register_talairach("moving", "template", "weights",
                                     tmp_path / "talairach.xfm", threads=1)
    assert (flags.cuda.matmul.allow_tf32, flags.cudnn.allow_tf32) == (True, False)
