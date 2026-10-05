"""Finite state contracts only; the fake CUDA chain computes tensors on CPU."""

from dataclasses import fields
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from fnit.synthseg_parc.pipeline import SynthSegParc
from fnit.synthseg_parc.precision import cuda_precision_state, cuda_tf32_scope
from fnit.synthseg_parc.segment import SynthSegParcResult, run_synthseg_parc_t1
from fnit.synthseg_parc.synthseg_plus import SynthSegPlus, SynthSegPlusResult


@pytest.fixture(autouse=True)
def preserve_state():
    flags = cuda_precision_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = flags["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
        torch.set_num_threads(threads)


def set_flags(matmul, cudnn):
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    return cuda_precision_state()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("policy", [True, False, None])
@pytest.mark.parametrize("caller", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_scope_restores_normal_and_exception_and_does_not_touch_autocast(device, policy, caller, fail):
    expected = set_flags(*caller)
    backend = (torch.backends.mkldnn.enabled, torch.backends.cudnn.benchmark,
               torch.backends.cudnn.deterministic, torch.is_autocast_enabled("cpu"),
               torch.is_autocast_enabled("cuda"))
    trace = {}
    try:
        with cuda_tf32_scope(device, policy, trace):
            active = expected if device == "cpu" else {
                "matmul_tf32": True, "cudnn_tf32": caller[1] if policy is None else policy}
            assert cuda_precision_state() == active
            # A nested scope restores the outer active policy before caller restore.
            with cuda_tf32_scope(device, not caller[1]):
                pass
            assert cuda_precision_state() == active
            if fail:
                raise RuntimeError("declared scope failure")
    except RuntimeError as error:
        assert fail and str(error) == "declared scope failure"
    assert cuda_precision_state() == expected
    assert trace["cuda_precision_before"] == trace["cuda_precision_after"] == expected
    assert trace["cuda_precision_restored"]
    assert backend == (torch.backends.mkldnn.enabled, torch.backends.cudnn.benchmark,
                       torch.backends.cudnn.deterministic, torch.is_autocast_enabled("cpu"),
                       torch.is_autocast_enabled("cuda"))


@pytest.mark.parametrize("policy", [0, 1, "false", np.bool_(False)])
def test_bad_policy_precedes_resource_loading(policy):
    with pytest.raises(ValueError, match="cudnn_tf32"):
        SynthSegParc("not-read.h5", "not-read.npy", cudnn_tf32=policy)
    with pytest.raises(ValueError, match="cudnn_tf32"):
        SynthSegPlus(weights="not-read", cudnn_tf32=policy)
    with pytest.raises(ValueError, match="cudnn_tf32"):
        run_synthseg_parc_t1(*(["not-read"] * 5), cudnn_tf32=policy)


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("policy", [True, False, None])
def test_none_device_resolves_scope_without_overwriting_flags(available, policy):
    expected = set_flags(False, False)
    with mock.patch("torch.cuda.is_available", return_value=available):
        with cuda_tf32_scope(None, policy):
            assert cuda_precision_state() == ({"matmul_tf32": True,
                "cudnn_tf32": False if policy is None else policy} if available else expected)
    assert cuda_precision_state() == expected


class ConstantParc(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.fail = fail

    def load_h5(self, path):
        return self

    def forward(self, image):
        if self.fail:
            raise RuntimeError("parcel forward failed")
        return torch.full((1, 69, *image.shape[2:]), 1 / 69, dtype=image.dtype)


def parcellator(policy, fail=False):
    with mock.patch("fnit.synthseg_parc.pipeline.ParcUNet", return_value=ConstantParc(fail)):
        return SynthSegParc("unused.h5", np.arange(69), "cpu", cudnn_tf32=policy)


def test_cuda_requested_constructor_does_not_reset_flags():
    expected = set_flags(False, False)
    selected = []

    def choose(device, *, configure_precision):
        selected.append((device, configure_precision))
        return torch.device("cpu")  # State contract, not a CUDA execution claim.

    with mock.patch("fnit.synthseg_parc.pipeline.configure_device", side_effect=choose), \
            mock.patch("fnit.synthseg_parc.pipeline.ParcUNet", return_value=ConstantParc()):
        model = SynthSegParc("unused.h5", np.arange(69), "cuda:0", cudnn_tf32=False)
    assert selected == [("cuda:0", False)]
    assert model.cudnn_tf32 is False
    assert cuda_precision_state() == expected


class ConstantSeg:
    def __init__(self, policy, fail=False):
        self.cudnn_tf32 = policy
        self.device = torch.device("cpu")
        self.labels = torch.arange(33)
        self.fail = fail
        self.precision = None

    def posterior(self, image, *, flip, smooth):
        self.precision = {"forwards": [{**cuda_precision_state(), "flip": flip, "smooth": smooth}]}
        if self.fail:
            raise RuntimeError("segmentation forward failed")
        return torch.full((33, *image.shape), 1 / 33, dtype=torch.float32)


def chain_fixtures(tmp_path, monkeypatch, policy, *, fail=None):
    from fnit.synthseg_parc import segment
    labels = tmp_path / "labels.npy"
    topology = tmp_path / "topology.npy"
    np.save(labels, np.arange(33))
    np.save(topology, np.zeros(33, dtype=np.int64))
    prepared = SimpleNamespace(image=torch.ones((2, 2, 2)), content_slices=(slice(0, 2),) * 3,
        aligned_affine=np.eye(4), input_affine=np.eye(4), original_shape=(2, 2, 2), voxel_volume_mm3=1.)
    monkeypatch.setattr(segment, "preprocess_t1", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(segment, "postprocess_segmentation", lambda posterior, *args, **kwargs:
                        (torch.zeros((2, 2, 2), dtype=torch.int64), posterior))
    seg = ConstantSeg(policy, fail=fail == "segment")
    parc = parcellator(policy, fail=fail == "parcel")
    if fail == "fast_blur":
        monkeypatch.setattr(segment, "_blur", mock.Mock(side_effect=RuntimeError("fast blur failed")))
    return labels, topology, seg, parc


@pytest.mark.parametrize("policy", [True, False, None])
@pytest.mark.parametrize("caller_cudnn", [False, True])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("device", ["cuda:0", None])
def test_full_scope_covers_seg_parc_and_fast_blur_and_restores(tmp_path, monkeypatch, policy, caller_cudnn, fast, device):
    expected = set_flags(False, caller_cudnn)
    labels, topology, seg, parc = chain_fixtures(tmp_path, monkeypatch, policy)
    with mock.patch("torch.cuda.is_available", return_value=True):
        result = run_synthseg_parc_t1("not-read", "unused.h5", labels, "unused.h5", np.arange(69),
            device=device, topology_classes=topology, fast=fast,
            segmenter=seg, parcellator=parc, cudnn_tf32=policy)
    trace = result.precision
    effective = caller_cudnn if policy is None else policy
    rows = trace["segmentation"]["forwards"] + trace["parcellation"]["forwards"] + \
           trace["parcellation"]["operations"] + trace["operations"]
    assert all(row["matmul_tf32"] is True and row["cudnn_tf32"] is effective for row in rows)
    assert len(trace["operations"]) == int(fast)
    assert trace["cache_precision_matches"] and trace["cuda_precision_restored"]
    assert cuda_precision_state() == expected


@pytest.mark.parametrize("failure", ["segment", "parcel", "fast_blur"])
def test_failed_chain_restores_caller_flags(tmp_path, monkeypatch, failure):
    expected = set_flags(False, True)
    labels, topology, seg, parc = chain_fixtures(tmp_path, monkeypatch, False, fail=failure)
    with pytest.raises(RuntimeError, match="failed"):
        run_synthseg_parc_t1("not-read", "unused.h5", labels, "unused.h5", np.arange(69),
            device="cuda:0", topology_classes=topology, fast=failure == "fast_blur",
            segmenter=seg, parcellator=parc, cudnn_tf32=False)
    assert cuda_precision_state() == expected


@pytest.mark.parametrize("which", ["segmenter", "parcellator"])
@pytest.mark.parametrize("cached,requested", [(True, False), (False, True), (None, True), (True, None)])
def test_cached_policy_mismatch_precedes_t1_loading(monkeypatch, which, cached, requested):
    from fnit.synthseg_parc import segment
    preprocess = mock.Mock(side_effect=AssertionError("must not preprocess"))
    monkeypatch.setattr(segment, "preprocess_t1", preprocess)
    with pytest.raises(ValueError, match="cached cudnn_tf32"):
        run_synthseg_parc_t1(*(["not-read"] * 5),
            cudnn_tf32=requested, **{which: SimpleNamespace(cudnn_tf32=cached)})
    preprocess.assert_not_called()


def test_none_cache_inherits_new_caller_cudnn_each_call(tmp_path, monkeypatch):
    labels, topology, seg, parc = chain_fixtures(tmp_path, monkeypatch, None)
    for inherited in (False, True):
        expected = set_flags(False, inherited)
        result = run_synthseg_parc_t1("not-read", "unused.h5", labels, "unused.h5", np.arange(69),
            device="cuda:0", topology_classes=topology, fast=True,
            segmenter=seg, parcellator=parc, cudnn_tf32=None)
        assert result.precision["operations"][0]["cudnn_tf32"] is inherited
        assert cuda_precision_state() == expected
        assert seg.cudnn_tf32 is parc.cudnn_tf32 is None


def test_public_plus_rejects_policy_change_without_io_or_stale_precision(monkeypatch):
    model = object.__new__(SynthSegPlus)
    model.cudnn_tf32 = False
    model.precision = {"old_completed_call": True}
    model._segmenter = SimpleNamespace(cudnn_tf32=True)
    model._parcellator = SimpleNamespace(cudnn_tf32=False)
    loader = mock.Mock(side_effect=AssertionError("must not read T1"))
    monkeypatch.setattr("fnit.synthseg_parc.synthseg_plus.nib.load", loader)
    expected = set_flags(False, True)
    with pytest.raises(ValueError, match="segmenter cached cudnn_tf32"):
        model("not-read.nii.gz")
    loader.assert_not_called()
    assert model.precision is None
    assert model._segmenter.cudnn_tf32 is True
    assert cuda_precision_state() == expected


def test_parcel_blur_failure_restores_nested_scope(tmp_path, monkeypatch):
    expected = set_flags(False, True)
    labels, topology, seg, parc = chain_fixtures(tmp_path, monkeypatch, False)
    with torch.backends.mkldnn.flags(enabled=True), \
            mock.patch("fnit.synthseg_parc.pipeline.F.conv3d",
                       side_effect=RuntimeError("parcel blur failed")):
        with pytest.raises(RuntimeError, match="parcel blur failed"):
            run_synthseg_parc_t1("not-read", "unused.h5", labels, "unused.h5", np.arange(69),
                device="cuda:0", topology_classes=topology, fast=False,
                segmenter=seg, parcellator=parc, cudnn_tf32=False)
    assert cuda_precision_state() == expected
    assert parc.precision["cuda_precision_restored"]
    row = parc.precision["operations"][0]
    assert row["cudnn_tf32"] is False and "output_dtype" not in row


def test_result_field_is_appended_and_optional():
    for result_type in (SynthSegParcResult, SynthSegPlusResult):
        assert fields(result_type)[-1].name == "precision"
        assert fields(result_type)[-1].default is None
