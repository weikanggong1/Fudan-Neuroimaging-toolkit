"""Private registration manifests and the single-subject CLI contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fnit.msm import MSMAllInputs
from fnit.msm.cli import load_inputs


def manifest(tmp_path):
    values = {"source_sphere": "source.surf.gii", "source_features": "source.func.gii",
              "reference_sphere": "/templates/reference.surf.gii",
              "reference_features": "/templates/reference.func.gii",
              "initial_sphere": None}
    path = tmp_path / "inputs.private.json"
    path.write_text(json.dumps({"L": values, "R": values}))
    return path


def test_manifest_resolves_local_paths_and_optional_unset_fields(tmp_path):
    result = load_inputs(manifest(tmp_path), MSMAllInputs)
    assert result["L"].source_sphere == tmp_path / "source.surf.gii"
    assert result["R"].reference_sphere == Path("/templates/reference.surf.gii")
    assert result["L"].initial_sphere is None
    assert result["L"].source_weights is None


@pytest.mark.parametrize("change", ["hemisphere", "field", "value"])
def test_manifest_rejects_ambiguous_or_invalid_inputs(tmp_path, change):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    if change == "hemisphere":
        del data["R"]
    elif change == "field":
        data["L"]["moving_volume"] = "wrong.nii.gz"
    else:
        data["L"]["source_features"] = 123
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_inputs(path, MSMAllInputs)


def test_msmall_cli_forwards_explicit_configuration_and_execution(tmp_path, monkeypatch):
    import fnit.msm as msm
    from fnit.msm import cli

    observed = {}
    def register(inputs, output_dir, **options):
        observed.update(inputs=inputs, output_dir=output_dir, **options)
        return {"L": Path("/results/L.surf.gii"), "R": Path("/results/R.surf.gii")}
    monkeypatch.setattr(msm, "run_msmall", register)
    assert cli.main(["msmall", "--inputs-json", str(manifest(tmp_path)),
                     "--output-dir", "/results", "--config", "/configs/full.conf",
                     "--device", "cpu", "--execution", "reference"]) == 0
    assert observed["config"] == "/configs/full.conf"
    assert observed["execution"] == "reference" and observed["device"] == "cpu"
    assert isinstance(observed["inputs"]["L"], MSMAllInputs)


def test_surface_cli_forwards_optional_msmall_manifest(monkeypatch):
    from fnit.fmri import cli
    observed = {}
    def surface(**options):
        observed.update(options)
        return SimpleNamespace(dtseries="/results/data.dtseries.nii")
    monkeypatch.setattr(cli, "fMRISurface_pipeline", surface)
    cli.main(["surface", "--bids-root", "/bids", "--derivatives-root", "/derivatives",
              "--subject", "0001", "--recon-all", "/recon-all",
              "--surface-assets-dir", "/templates", "--msmall-inputs-json", "/inputs.json",
              "--msmall-config", "/configs/refine.conf"])
    assert observed["msmall_inputs"] == "/inputs.json"
    assert observed["msmall_config"] == "/configs/refine.conf"


@pytest.mark.parametrize("options", [
    {"msmall_config": "/configs/refine.conf"},
    {"msmall_inputs": {"L": None, "R": None}},
    {"msmall_inputs": "/inputs.json", "registered_spheres": ("/L.surf.gii", "/R.surf.gii")},
])
def test_surface_rejects_inapplicable_msmall_inputs_before_loading_bids(options):
    from fnit.fmri import fMRISurface_pipeline
    with pytest.raises(ValueError, match="msmall"):
        fMRISurface_pipeline("/missing/bids", "/missing/derivatives", subject="0001",
                            recon_all="/missing/recon-all", hcp_assets_dir="/missing/templates",
                            **options)


def test_feature_cli_resolves_files_but_preserves_classification_indices(tmp_path, monkeypatch):
    from fnit.msm import feature_cli, features
    observed = {}
    def variance_normalization(**options):
        observed.update(options)
        return Path(options["output_file"])
    monkeypatch.setattr(features, "compute_msmall_variance_normalization", variance_normalization)
    path = tmp_path / "vn.json"
    path.write_text(json.dumps({"clean_dtseries": "clean.dtseries.nii",
                                "ica_timecourses": "mixing.txt", "noise_components": [1, 3, 5],
                                "output_file": "VN.dscalar.nii", "device": "cpu"}))
    assert feature_cli.main(["vn", "--inputs-json", str(path)]) == 0
    assert observed["clean_dtseries"] == str(tmp_path / "clean.dtseries.nii")
    assert observed["noise_components"] == [1, 3, 5]
    assert observed["device"] == "cpu"


def test_feature_cli_preserves_wrn_template_order_and_workbench_command(tmp_path, monkeypatch):
    from fnit.msm import feature_cli, features
    observed = {}
    def regression(**options):
        observed.update(options)
        return features.MSMAllRegressionResult(*(Path("/output") / name for name in
                                                ("maps.nii", "weights.nii", "nodes.tsv", "report.json")))
    monkeypatch.setattr(features, "run_msmall_regression", regression)
    path = tmp_path / "regression.json"
    path.write_text(json.dumps({"low_dimensional_maps": ["d7.nii", "d8.nii"],
                                "method": "WRN", "wb_command": "wb_command"}))
    assert feature_cli.main(["regression", "--inputs-json", str(path)]) == 0
    assert observed["low_dimensional_maps"] == [str(tmp_path / "d7.nii"), str(tmp_path / "d8.nii")]
    assert observed["method"] == "WRN" and observed["wb_command"] == "wb_command"


def test_msmsulc_cli_forwards_qc_policy(tmp_path, monkeypatch):
    import fnit.msm as msm
    from fnit.msm import cli
    observed = {}
    def register(inputs, output_dir, **options):
        observed.update(options)
        return {"L": Path("/results/L.surf.gii"), "R": Path("/results/R.surf.gii")}
    monkeypatch.setattr(msm, "run_msmsulc", register)
    values = {"native_sphere": "native.surf.gii", "rotated_sphere": "rotated.surf.gii",
              "native_sulc": "native.shape.gii", "reference_sphere": "reference.surf.gii",
              "reference_sulc": "reference.shape.gii", "affine": "affine.txt"}
    path = tmp_path / "msmsulc.inputs.json"
    path.write_text(json.dumps({"L": values, "R": values}))
    assert cli.main(["msmsulc", "--inputs-json", str(path),
                     "--output-dir", "/results", "--device", "cpu",
                     "--qc-policy", "repair"]) == 0
    assert observed["qc_policy"] == "repair"
