"""CLI modes must preserve requested EDDY seeds and reference volumes."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import fnit.eddy.cli as cli
import fnit.eddy.ukb as ukb


def _invoke(entrypoint, arguments):
    if entrypoint == "standalone":
        cli.main(arguments)
    else:
        parser = argparse.ArgumentParser()
        cli.add_parser(parser.add_subparsers())
        args = parser.parse_args(["eddy", *arguments])
        args._fnit_handler(args)


@pytest.mark.parametrize("entrypoint", ["standalone", "subcommand"])
@pytest.mark.parametrize("options,reference,seed", [
    ([], None, None),
    (["--ref-scan-no", "2", "--gp-seed", "73491"], 2, 73491),
])
def test_raw_cli_forwards_reference_and_seed(monkeypatch, entrypoint, options,
                                            reference, seed):
    calls = []

    def run_ukb(*inputs, **kwargs):
        calls.append((inputs, kwargs))
        return SimpleNamespace(qc={}), {}

    monkeypatch.setattr(cli, "run_ukb_eddy", run_ukb)
    _invoke(entrypoint, [
        "--raw-dir", "raw", "--topup-dir", "topup", "--output-dir", "out",
        "--device", "cpu", "--synthstrip-weights", "weights", "--overwrite",
        *options,
    ])
    assert calls == [(("raw", "topup", "out"), {
        "device": "cpu", "overwrite": True, "synthstrip_weights": "weights",
        "ref_scan_no": reference, "gp_seed": seed,
    })]


@pytest.mark.parametrize("options,reference,seed", [
    ([], 0, None),
    (["--ref-scan-no", "2", "--gp-seed", "73491"], 2, 73491),
])
def test_direct_cli_preserves_reference_default_and_explicit_options(monkeypatch,
                                                                    options, reference, seed):
    calls = []

    class Eddy:
        def __init__(self, device):
            assert device == "cpu"

        def run(self, *inputs, **kwargs):
            calls.append((inputs, kwargs))
            return SimpleNamespace(qc={})

    monkeypatch.setattr(cli, "TorchEDDY", Eddy)
    cli.main([
        "--imain", "ap", "--mask", "mask", "--acqp", "acq",
        "--index", "index", "--bvecs", "bvecs", "--bvals", "bvals",
        "--out", "out/data", "--device", "cpu", *options,
    ])
    assert calls == [(("ap", "mask", "acq", "index", "bvecs", "bvals"), {
        "topup": None, "out": "out/data", "ref_scan_no": reference,
        "gp_seed": seed, "overwrite": False,
    })]


@pytest.mark.parametrize("reference,seed,selected_reference", [
    (None, None, 5),
    (2, 73491, 2),
])
def test_ukb_wrapper_routes_reference_to_preparation_and_seed_to_eddy(
        monkeypatch, reference, seed, selected_reference):
    preparation_calls, eddy_calls = [], []
    inputs = {"imain": Path("raw/AP.nii.gz"), "ref_scan_no": selected_reference}
    result = SimpleNamespace(qc={})

    def prepare(*paths, **kwargs):
        preparation_calls.append((paths, kwargs))
        return inputs

    class Eddy:
        def __init__(self, *, device):
            assert device == "cpu"

        def run(self, **kwargs):
            eddy_calls.append(kwargs)
            return result

    monkeypatch.setattr(ukb, "prepare_ukb_eddy", prepare)
    monkeypatch.setattr(ukb, "TorchEDDY", Eddy)
    actual = ukb.run_ukb_eddy(
        "raw", "topup", "out", device="cpu", overwrite=True,
        synthstrip_weights="weights", ref_scan_no=reference, gp_seed=seed,
    )
    assert actual == (result, inputs)
    assert preparation_calls == [(("raw", "topup", "out"), {
        "device": "cpu", "overwrite": True, "synthstrip_weights": "weights",
        "ref_scan_no": reference,
    })]
    assert eddy_calls == [{**inputs, "out": Path("out/data"),
                           "overwrite": True, "gp_seed": seed}]
