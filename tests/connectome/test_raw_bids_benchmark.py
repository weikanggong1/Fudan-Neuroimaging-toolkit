"""Benchmark contracts: real functions, stable seeds, honest timing and reports."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch


def tool():
    path = Path(__file__).parents[2] / "tools/benchmark_connectome_raw_bids.py"
    spec = importlib.util.spec_from_file_location("raw_bids_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("allocated,reserved,expected", (
    (1_000_000_000, 23_000_000_000, False),
    (1_000_000_000, 20_000_000_000, False),
    (1_000_000_000, 19_999_999_999, True),
    (None, 23_000_000_000, False),
    (1_000_000_000, None, None),
    (None, None, None),
))
def test_memory_budget_counts_reserved_peak_and_missing_evidence(allocated, reserved, expected):
    summary = tool().memory_budget_summary(
        {"allocated_bytes": allocated, "reserved_bytes": reserved},
        {"peak_process_tree_bytes": 20_000_000_000})
    assert summary["allocator_observed_below_limit"] is expected
    assert summary["process_tree_sampled_below_limit"] is False
    assert summary["continuous_process_tree_bound_verified"] is False


def options(module, tmp_path, mode="wall", device="cpu", extra=()):
    return module.parse_options([
        "--mode", mode, "--report", str(tmp_path / "report.json"), *extra,
        "--", "fnit", "UKBConnectome_pipeline", "--bids-root", str(tmp_path / "bids"),
        "--subject", "01", "--n-seeds", "100", "--device", device,
        "--output-dir", str(tmp_path / "outputs"),
    ])


class CUDA:
    def __init__(self):
        self.allocated, self.reserved = 0, 0
        self.syncs, self.resets = [], []
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def current_device(self):
        return 0

    def reset_peak_memory_stats(self, *args, **kwargs):
        self.resets.append((args, kwargs))
        self.allocated, self.reserved = 0, 0

    def max_memory_allocated(self, device):
        return self.allocated

    def max_memory_reserved(self, device):
        return self.reserved

    def synchronize(self, device):
        self.syncs.append(device)


def fake_modules():
    calls = []

    class Eddy:
        def run(self, *args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(qc={"actual_seed": kwargs.get("gp_seed")})

    def prepare(*args, **kwargs):
        Eddy().run(out="data", overwrite=True)
        return SimpleNamespace(dwi="corrected.nii.gz", bvals="raw.bval",
                               bvecs="rotated.bvec", freesurfer_subject_dir="subject",
                               stages={"topup": "completed", "eddy": "completed",
                                       "recon_all": "supplied"})

    class Pipeline:
        def __call__(self, value):
            return value

    noop = lambda *args, **kwargs: None
    bids = SimpleNamespace(TorchEDDY=Eddy, prepare_bids_connectome=prepare,
                           locate_bids_dwi=noop, stage_bids_dwi=noop,
                           prepare_ukb_eddy=noop, _prepare_ap_only=noop,
                           run_ukb_topup=lambda *args: (SimpleNamespace(qc={}), {}))
    pipeline = SimpleNamespace(UKBConnectome_pipeline=Pipeline,
                               fit_mrtrix_msmt_csd=lambda *args: "real-result")
    return bids, pipeline, calls


def test_arguments_preserve_cli_and_forbid_wall_exports(tmp_path):
    module = tool()
    parsed = options(module, tmp_path)
    assert parsed.cli_arguments[0] == "UKBConnectome_pipeline"
    assert parsed.eddy_gp_seed == 12345
    assert parsed.cli_arguments[-1] == str(tmp_path / "outputs")
    with pytest.raises(SystemExit):
        options(module, tmp_path, extra=("--checkpoint-dir", str(tmp_path / "checkpoints")))
    with pytest.raises(SystemExit):
        options(module, tmp_path, extra=("--memory-sample-interval", "nan"))
    for seed in ("0", "-1", "4294967296"):
        with pytest.raises(SystemExit):
            options(module, tmp_path, extra=("--eddy-gp-seed", seed))


def test_wall_keeps_function_outputs_and_has_no_new_cuda_sync(tmp_path):
    module = tool()
    bids, pipeline, calls = fake_modules()
    cuda = CUDA()
    original_prepare, original_eddy = bids.prepare_bids_connectome, bids.TorchEDDY.run
    with module.Measure(SimpleNamespace(cuda=cuda), bids, pipeline,
                        options(module, tmp_path, device="cuda:0")) as measure:
        selected = bids.prepare_bids_connectome("bids", "output")
        explicit = bids.TorchEDDY().run(gp_seed=77)
        missing = bids.TorchEDDY().run(gp_seed=None)
    assert selected.dwi == "corrected.nii.gz"
    assert explicit.qc["actual_seed"] == 77
    assert missing.qc["actual_seed"] == 12345
    assert [call["gp_seed"] for call in calls] == [12345, 77, 12345]
    assert measure.eddy_seeds == [12345, 77, 12345]
    assert measure.preprocessing[0]["eddy"] == "completed"
    assert measure.stages == {} and cuda.syncs == []
    assert bids.prepare_bids_connectome is original_prepare
    assert bids.TorchEDDY.run is original_eddy


def test_wall_result_capture_keeps_original_object_without_export_or_sync(tmp_path):
    module = tool()
    bids, pipeline, _ = fake_modules()
    cuda = CUDA()
    export_path = tmp_path / "post_timing_result"
    parsed = options(module, tmp_path, device="cuda:0",
                     extra=("--result-export-dir", str(export_path)))
    original = pipeline.UKBConnectome_pipeline.__call__
    result = object()
    with module.Measure(SimpleNamespace(cuda=cuda), bids, pipeline, parsed) as measure:
        assert pipeline.UKBConnectome_pipeline()(result) is result
        assert measure.returned_result is result
        assert not export_path.exists()
    assert pipeline.UKBConnectome_pipeline.__call__ is original
    assert cuda.syncs == []
    with pytest.raises(SystemExit):
        options(module, tmp_path, mode="diagnostic",
                extra=("--result-export-dir", str(export_path)))


def test_diagnostic_runs_original_function_and_restores_after_exception(tmp_path):
    module = tool()
    bids, pipeline, _ = fake_modules()
    cuda = CUDA()
    original = pipeline.fit_mrtrix_msmt_csd
    with pytest.raises(RuntimeError, match="abort"):
        with module.Measure(SimpleNamespace(cuda=cuda), bids, pipeline,
                            options(module, tmp_path, mode="diagnostic", device="cuda:0")) as measure:
            assert pipeline.fit_mrtrix_msmt_csd("raw") == "real-result"
            bids.prepare_bids_connectome("raw", "output")
            raise RuntimeError("abort")
    assert pipeline.fit_mrtrix_msmt_csd is original
    assert measure.stages["fit_mrtrix_msmt_csd"][0]["status"] == "completed"
    assert measure.stages["eddy_run_with_save"][0]["seconds_inclusive"] >= 0
    assert cuda.syncs and set(cuda.syncs) == {"cuda:0"}


def test_peak_ledger_keeps_pre_reset_max_and_ignores_other_device():
    module = tool()
    cuda = CUDA()
    ledger = module.PeakLedger(SimpleNamespace(cuda=cuda), "cuda:0")
    cuda.allocated, cuda.reserved = 14, 19
    ledger.reset()
    cuda.allocated, cuda.reserved = 5, 8
    ledger.capture("final")
    cuda.allocated, cuda.reserved = 900, 1000
    ledger.capture("other_gpu", "cuda:1")
    assert cuda.resets == [((), {})]
    record = ledger.report()
    assert record["allocated_bytes"] == 14
    assert record["reserved_bytes"] == 19
    assert len(record["intervals"]) == 2


def test_smi_parser_and_unavailable_memory_are_not_zero():
    module = tool()
    assert module.parse_smi_processes("GPU-A, 20, 7\nGPU-A, 21, [N/A]\nnoise") == [
        ("GPU-A", 20, 7 * 2**20)]
    monitor = module.GPUProcessMonitor(SimpleNamespace(), "cpu")
    monitor.start()
    report = monitor.finish()
    assert report["status"] == "not_measured"
    assert report["peak_process_tree_bytes"] is None


def test_monitor_aggregates_only_target_gpu_process_tree(monkeypatch):
    module = tool()
    monitor = module.GPUProcessMonitor(SimpleNamespace(), "cuda:0", gpu_uuid="GPU-A")
    monkeypatch.setattr(monitor, "_processes", lambda: [
        ("GPU-A", 101, 30), ("GPU-A", 102, 20),
        ("GPU-A", 999, 9999), ("GPU-B", 101, 8888)])
    monkeypatch.setattr(module, "descendant_pid", lambda pid, root: pid in (101, 102))
    monitor._sample()
    assert monitor.finish()["peak_process_tree_bytes"] == 50


def test_json_is_strict_for_nonfinite_python_and_numpy_values(tmp_path):
    module = tool()
    path = tmp_path / "strict.json"
    module.atomic_json(path, {"nan": float("nan"), "inf": np.float64(np.inf),
                              "size": np.int64(7), "ok": np.bool_(True)})
    assert json.loads(path.read_text()) == {"nan": None, "inf": None, "size": 7, "ok": True}
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_checkpoints_stream_real_paths_and_preserve_dtype_affine(tmp_path):
    module = tool()
    affine = torch.eye(4, dtype=torch.float64)
    paths = (torch.tensor([[0.0, 1, 2], [1, 2, 3]], dtype=torch.float32),
             torch.tensor([[2.0, 1, 0], [3, 2, 1], [4, 3, 2]], dtype=torch.float32))
    result = SimpleNamespace(
        dwi_affine=affine, five_tissue_affine=affine,
        wm_sh=torch.ones(2, 2, 2, 45), fa=torch.ones(2, 2, 2),
        brain_mask=torch.ones(2, 2, 2, dtype=torch.bool),
        five_tissue=torch.ones(2, 2, 2, 5), gmwmi=torch.ones(2, 2, 2),
        dwi_to_t1_world=affine, sift2_weights=torch.tensor([0.123456789012, 2.0], dtype=torch.float64),
        tractogram=SimpleNamespace(paths=paths, lengths_mm=torch.tensor([1.0, 2.0]),
                                  mean_fa=torch.tensor([0.3, 0.4]), endpoints=torch.ones(2, 2, 3)),
    )
    exporter = module.Checkpoints(tmp_path / "checkpoints")
    exporter.core(result)
    loaded = nib.streamlines.load(exporter.directory / "tracks.tck").tractogram.streamlines
    assert len(loaded) == 2
    for actual, expected in zip(loaded, paths):
        np.testing.assert_array_equal(actual, expected.numpy())
    weights = np.load(exporter.directory / "track_metrics.npz")["weights"]
    assert weights.dtype == np.float64
    np.testing.assert_array_equal(weights, result.sift2_weights.numpy())
    geometry = np.load(exporter.directory / "geometry.npz")
    assert geometry["five_tissue_affine"].dtype == np.float64
    np.testing.assert_array_equal(geometry["five_tissue_affine"], result.five_tissue_affine.numpy())
    image = nib.load(exporter.directory / "wm_fod_normalized.nii.gz")
    assert image.shape == (2, 2, 2, 45)
    np.testing.assert_array_equal(image.affine, affine.numpy())
    assert nib.load(exporter.directory / "brain_mask.nii.gz").get_data_dtype() == np.uint8
    assert exporter.seconds > 0 and not any(hasattr(value, "detach") for value in vars(exporter).values())
    with pytest.raises(FileExistsError):
        module.Checkpoints(exporter.directory)


@pytest.mark.parametrize("mode", ("wall", "diagnostic"))
def test_tracking_checkpoint_uses_actual_bound_inputs_before_stage(tmp_path, monkeypatch, mode):
    module = tool()
    bids, pipeline, _ = fake_modules()
    returned = SimpleNamespace(actual="original tracking result")
    calls = []

    def tracking(wm_sh, fod_affine, five_tissue, five_tissue_affine, gmwmi, *,
                 n_seeds, lmax=8, five_tissue_spacing_mm=None, fa=None,
                 seed=0, batch_size=8192, arc_proposals=16, max_length_mm=250.,
                 min_length_mm=None, step_mm=None, max_angle_degrees=45.,
                 cutoff=0.1, power=0.5, compile_arc=False):
        assert (tmp_path / "checkpoints" / "tracking_inputs.pt").exists() == (mode == "diagnostic")
        assert not (tmp_path / "checkpoints" / "tracking_inputs.pt.tmp").exists()
        calls.append(n_seeds)
        return returned

    pipeline.probabilistic_tractography = tracking
    wm = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    affine = torch.eye(4, dtype=torch.float64)
    affine[0, 1], affine[2, 3] = 0.123456789123456, 12.987654321987654
    five = torch.ones(2, 2, 2, 5, dtype=torch.float64)
    anatomy_affine = affine.clone()
    anatomy_affine[0, 0] = 1.7
    gmwmi = torch.ones(2, 2, 2, dtype=torch.float32)
    fa = torch.ones(2, 2, 2, dtype=torch.float64)
    spacing = tuple(np.float32(value) for value in (0.9, 1.1, 1.3))
    checkpoint_dir = tmp_path / "checkpoints"
    extra = ("--checkpoint-dir", str(checkpoint_dir)) if mode == "diagnostic" else ()
    # Checkpoint first, then stage start/stop: exporting is not tracking time.
    clocks = iter((20., 25., 10., 12.))
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: next(clocks)))
    with module.Measure(SimpleNamespace(cuda=CUDA()), bids, pipeline,
                        options(module, tmp_path, mode=mode, extra=extra)) as measure:
        result = pipeline.probabilistic_tractography(
            wm, affine, five, five_tissue_affine=anatomy_affine, gmwmi=gmwmi,
            n_seeds=np.int64(123), seed=np.int64(7), fa=fa,
            five_tissue_spacing_mm=spacing, compile_arc=np.bool_(False))
    assert result is returned and calls == [123]
    assert pipeline.probabilistic_tractography is tracking
    if mode == "wall":
        assert measure.stages == {} and measure.checkpoints.seconds == 0
        assert not checkpoint_dir.exists()
        return
    assert measure.stages["probabilistic_tractography"][0]["seconds_inclusive"] == 2.
    assert measure.checkpoints.seconds == 5.
    path = checkpoint_dir / "tracking_inputs.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in (("wm_sh", wm), ("fod_affine", affine),
                          ("five_tissue", five), ("five_tissue_affine", anatomy_affine),
                          ("gmwmi", gmwmi), ("fa", fa)):
        assert checkpoint[key].dtype == expected.dtype
        assert checkpoint[key].device.type == "cpu"
        assert torch.equal(checkpoint[key], expected)
    assert checkpoint["five_tissue_spacing_mm"] == tuple(float(value) for value in spacing)
    assert all(type(value) is float for value in checkpoint["five_tissue_spacing_mm"])
    parameters = checkpoint["tracking_kwargs"]
    assert parameters["n_seeds"] == 123 and type(parameters["n_seeds"]) is int
    assert parameters["seed"] == 7 and parameters["compile_arc"] is False
    assert parameters["lmax"] == 8 and parameters["batch_size"] == 8192
    assert parameters["five_tissue_spacing_mm"] == checkpoint["five_tissue_spacing_mm"]
    assert "lmax" not in checkpoint["explicit_arguments"]
    assert "five_tissue_affine" in checkpoint["explicit_arguments"]
    assert measure.checkpoints.paths == [path]
    assert not any(isinstance(value, torch.Tensor) for value in vars(measure.checkpoints).values())


def test_tracking_checkpoint_preserves_default_spacing_none(tmp_path):
    module = tool()
    exporter = module.Checkpoints(tmp_path / "checkpoints")
    def tracking(wm_sh, fod_affine, five_tissue, five_tissue_affine, gmwmi, *,
                 n_seeds, five_tissue_spacing_mm=None, fa=None):
        return None
    image = torch.ones(1, 1, 1, 1)
    affine = torch.eye(4, dtype=torch.float32)
    exporter.tracking_inputs(tracking, (image, affine, image, affine, image), {"n_seeds": 1})
    checkpoint = torch.load(exporter.directory / "tracking_inputs.pt", weights_only=True)
    assert checkpoint["five_tissue_spacing_mm"] is None
    assert checkpoint["tracking_kwargs"]["five_tissue_spacing_mm"] is None
    assert checkpoint["fa"] is None
    assert checkpoint["fod_affine"].dtype == torch.float64
    assert checkpoint["five_tissue_affine"].dtype == torch.float64
    assert torch.equal(checkpoint["fod_affine"], affine.double())


def test_failed_cli_writes_report_and_returns_failure(tmp_path, monkeypatch):
    module = tool()
    bids, pipeline, _ = fake_modules()
    fake_torch = SimpleNamespace(cuda=CUDA())
    def failed_cli(arguments):
        raise ValueError("incomplete raw metadata")
    modules = {"torch": fake_torch, "fnit.cli": SimpleNamespace(main=failed_cli),
               "fnit.connectome.bids": bids, "fnit.connectome.pipeline": pipeline}
    monkeypatch.setattr(module.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(module, "source_provenance", lambda torch: {"test_stub": True})
    monkeypatch.setattr(module, "collect_inputs", lambda *args: {})
    monkeypatch.setattr(module, "inspect_outputs", lambda *args: {"status": "incomplete"})
    selected_options = options(module, tmp_path)
    assert module.run(selected_options) == 1
    report = json.loads(selected_options.report.read_text())
    assert report["status"] == "failed" and report["exit_code"] == 1
    assert report["exception"]["type"] == "ValueError"
    assert report["outputs"]["status"] == "incomplete"
    assert report["total_runtime_seconds"] >= 0


def test_output_completeness_rejects_missing_or_nonfinite_matrix(tmp_path):
    module = tool()
    opts = options(module, tmp_path)
    root = tmp_path / "outputs"
    directory = root / "atlases/fs-aparc"
    directory.mkdir(parents=True)
    for name in ("five_tissue_dwi_world.nii.gz", "gmwmi_dwi_world.nii.gz", "fa_dwi.nii.gz",
                 "brain_mask_dwi.nii.gz", "dwi_to_t1_world.csv", "dataset_description.json", "run_state.json"):
        (root / name).write_text("fixture")
    (directory / "nodes.tsv").write_text("index\toriginal_label\themisphere\tname\n1\t1\tL\ta\n2\t2\tR\tb\n")
    np.savetxt(directory / "region_labels.csv", [1, 2], delimiter=",")
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.int32), np.eye(4)), directory / "atlas_dwi.nii.gz")
    for name in ("count", "sift2_fbc", "mean_length", "mean_fa"):
        np.savetxt(directory / f"connectome_{name}.csv", [[0, 2], [2, 0]], delimiter=",")
    assert module.inspect_outputs(opts)["status"] == "complete"
    np.savetxt(directory / "connectome_mean_fa.csv", [[0, np.nan], [np.nan, 0]], delimiter=",")
    assert module.inspect_outputs(opts)["status"] == "incomplete"
    (directory / "connectome_mean_fa.csv").unlink()
    assert module.inspect_outputs(opts)["status"] == "incomplete"
