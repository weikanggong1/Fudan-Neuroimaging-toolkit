"""重复性工具回归；小数组仅检验工具契约，不作为科学 benchmark。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from tools.connectome_repeat_common import (
    NAMES, check_metadata, envelope, load_matrices, load_profiles, profile_metrics, seed_labels, sha256,
)
from tools import benchmark_connectome_seven_atlas_envelope as multi
from tools import benchmark_connectome_tracking_100k_envelope as single


def matrices(factor=1):
    upper = np.triu_indices(4, 1)
    values = np.zeros((4, 4))
    values[upper] = np.arange(1, 7) * factor
    values[(upper[1], upper[0])] = values[upper]
    return {name: values * scale for name, scale in zip(NAMES, (1, 2, 10, .03))}


def write_csv(directory, factor=1, prefix="connectome_", metadata=False):
    directory.mkdir(parents=True)
    for name, value in matrices(factor).items():
        np.savetxt(directory / f"{prefix}{name}.csv", value, delimiter=",")
    if metadata:
        (directory / "nodes.tsv").write_text(
            "index\toriginal_label\themisphere\tname\n" +
            "".join(f"{i}\t{1000+i}\tlh\tregion{i}\n" for i in range(1, 5)))
        np.savetxt(directory / "region_labels.csv", np.arange(1, 5), fmt="%d")
        (directory / "atlas.sha256").write_text("a" * 64 + "\n")


def reference_tool():
    path = Path(__file__).parents[2] / "tools/reference/benchmark_connectome_repeats_official.py"
    spec = importlib.util.spec_from_file_location("repeat_official_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload():
    affine = torch.diag(torch.tensor([2.5, 2.5, 2.5, 1.], dtype=torch.float64))
    affine[0, 3] = 0.123456789012345
    five_affine = torch.eye(4, dtype=torch.float64)
    five_affine[0, 0] = 1.000000040000001
    return {"wm_sh": torch.zeros((4, 4, 4, 45)), "fod_affine": affine,
            "five_tissue": torch.ones((4, 4, 4, 5)), "five_tissue_affine": five_affine,
            "five_tissue_spacing_mm": (1., 1., 1.), "gmwmi": torch.ones((4, 4, 4)), "fa": None,
            "tracking_kwargs": {"n_seeds": 100000, "lmax": 8, "step_mm": None,
                "min_length_mm": None, "max_length_mm": 250., "max_angle_degrees": 45.,
                "cutoff": .1, "power": .5, "seed": 0, "batch_size": 8192,
                "arc_proposals": 16, "compile_arc": False, "five_tissue_spacing_mm": (1., 1., 1.)}}


def test_one_sided_gate_keeps_range_and_accepts_smaller_error_and_higher_similarity():
    errors = envelope([.2, .3, .4], [.1, .4, .5])
    assert errors["official_min_max"] == [.2, .4]
    assert errors["inside_count"] == 1
    assert errors["comparison_accepted"] == [True, True, False]
    assert errors["status"] == "failed"
    similar = envelope([.8, .9], [1., .8], similarity=True)
    assert similar["inside_count"] == 1
    assert similar["comparison_accepted"] == [True, True]
    assert similar["status"] == "passed"


def test_undefined_is_null_not_a_pass_and_empty_support_dice_is_one():
    assert envelope([None, .2], [.1])["status"] == "not_assessed"
    assert envelope([.1, .2], [])["status"] == "not_assessed"
    zeros = {name: np.zeros((3, 3)) for name in NAMES}
    result = profile_metrics(zeros, zeros)
    assert result["count_support_dice"] == 1
    assert result["count_pearson"] is None
    assert result["count_relative_l1"] is None
    assert result["mean_fa_common_normalized_mae"] is None
    json.dumps(result, allow_nan=False)


def test_common_count_policy_and_diagonal_report():
    a, b = matrices(), matrices()
    a["count"][0, 1] = b["count"][0, 1] = 0
    b["mean_fa"][0, 1] = 99
    b["count"][1, 1] = 6
    assert profile_metrics(a, b)["mean_fa_common_normalized_mae"] == 0
    assert profile_metrics(a, b)["diagonal"]["count"]["mae"] == 1.5


def test_five_repeats_current_multi_csv_format_two_candidates(tmp_path):
    official = [tmp_path / f"mrtrix{i}" for i in range(5)]
    candidates = [tmp_path / f"fnit{i}" for i in range(2)]
    for root, factor in zip(official + candidates, (.9, .95, 1., 1.05, 1.1, 1., 1.01)):
        for profile in ("fs-aparc", "schaefer200+tian-s1"):
            write_csv(root / "atlases" / profile, factor, "" if root in official else "connectome_", True)
    result = multi.compare(official, candidates, dataset="public fixture", official_seeds=list(range(5)), fnit_seeds=[0, 1])
    assert result["dataset"] == "public fixture"
    assert result["matrix_envelope_status"] == "passed"
    assert result["fnit_reproducibility_status"] == "passed"
    for info in result["profiles"].values():
        assert len(info["pairwise"]["official"]) == 10
        assert len(info["pairwise"]["cross"]) == 10
        assert len(info["pairwise"]["fnit"]) == 1
        assert info["comparison_counts"] == {"official": 10, "fnit": 1, "cross": 10}
        assert len(info["fnit_reproducibility_ranges"]["count_relative_l1"]["fnit_within"]) == 1
        assert info["input_identity"]["node_rows"]["status"] == "verified_equal"


def test_legacy_npz_and_no_torch_needed_by_comparison_loader(tmp_path):
    np.savez(tmp_path / "legacy.npz", **matrices())
    (tmp_path / "report.json").write_text(json.dumps({"profiles": {"legacy": {
        "nodes": 4, "atlas_sha256": "b" * 64}}}))
    result = load_profiles(tmp_path)["legacy"]
    assert np.array_equal(result[0]["count"], matrices()["count"])
    assert result[1]["matrix_npz_sha256"] and result[1]["node_rows"] is None


def test_metadata_node_order_and_atlas_identity_failures(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_csv(first, metadata=True)
    write_csv(second, metadata=True)
    (second / "atlas.sha256").write_text("b" * 64)
    with pytest.raises(ValueError, match="atlas_sha256"):
        check_metadata([load_matrices(first), load_matrices(second)], "fixture")
    (second / "atlas.sha256").write_text("a" * 64)
    (second / "nodes.tsv").write_text((second / "nodes.tsv").read_text().replace("region4", "changed"))
    with pytest.raises(ValueError, match="node_rows"):
        check_metadata([load_matrices(first), load_matrices(second)], "fixture")


def test_duplicate_resolved_directories_or_seed_labels_cannot_claim_repeats(tmp_path):
    root = tmp_path / "same"
    write_csv(root)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="duplicate resolved"):
        single.compare([root, alias, root], [root])
    with pytest.raises(ValueError, match="duplicate seed"):
        seed_labels([0, 0, 1], 3, "official")


def test_profile_set_and_matrix_shape_must_match(tmp_path):
    official = [tmp_path / f"ref{i}" for i in range(3)]
    for root in official:
        write_csv(root / "atlases/fs-aparc")
    candidate = tmp_path / "candidate"
    write_csv(candidate / "atlases/different")
    with pytest.raises(ValueError, match="profiles differ"):
        multi.compare(official, [candidate])
    np.savetxt(official[0] / "atlases/fs-aparc/connectome_count.csv", np.zeros((3, 2)), delimiter=",")
    with pytest.raises(ValueError, match="finite square"):
        load_profiles(official[0])


def test_single_cli_five_repeats_dynamic_labels(tmp_path, capsys):
    roots = [tmp_path / f"run{i}" for i in range(6)]
    for root, factor in zip(roots, (.9, .95, 1., 1.05, 1.1, 1.)):
        write_csv(root, factor)
    output = tmp_path / "result.json"
    single.main(["--official", *map(str, roots[:5]), "--fnit", str(roots[5]),
                 "--dataset", "OpenNeuro new subject", "--official-seeds", "0", "1", "2", "3", "4",
                 "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["dataset"] == "OpenNeuro new subject" and "ds004666" not in output.read_text()
    assert result["matrix_envelope_status"] == "passed" and len(result["pairwise"]["official"]) == 10
    assert result["fnit_reproducibility_status"] == "not_assessed"
    assert json.loads(capsys.readouterr().out)["matrix_envelope_status"] == "passed"
    with pytest.raises(SystemExit):
        single.main(["--official", *map(str, roots[:2]), "--fnit", str(roots[5]), "--output", str(output)])


def test_optional_figures_handle_five_official_and_two_fnit_runs(tmp_path):
    pytest.importorskip("matplotlib")
    roots = [tmp_path / f"run{i}" for i in range(7)]
    for root, factor in zip(roots, (.9, .95, 1., 1.05, 1.1, 1., 1.01)):
        write_csv(root / "atlases/fs-aparc", factor)
    report = multi.compare(roots[:5], roots[5:], dataset="new fixture")
    multi.figure(tmp_path / "multi.png", report)
    single_report = single.compare([root / "atlases/fs-aparc" for root in roots[:5]],
                                   [root / "atlases/fs-aparc" for root in roots[5:]])
    single.figure(tmp_path / "single.png", roots[0] / "atlases/fs-aparc",
                  roots[5] / "atlases/fs-aparc", single_report)
    assert (tmp_path / "multi.png").stat().st_size and (tmp_path / "single.png").stat().st_size


def test_actual_checkpoint_defaults_and_attempt_count_guard():
    module = reference_tool()
    result = module.effective_parameters(payload(), 100000)
    assert result["step_mm"] == 1.25 and result["min_length_mm"] == 5
    assert result["samples"] == 3 and result["power"] == .5 and result["cutoff"] == .1
    with pytest.raises(ValueError, match="differs"):
        module.effective_parameters(payload(), 1000000)
    for key, invalid in (("min_length_mm", 251.), ("step_mm", 250.), ("min_length_mm", float("nan"))):
        values = payload()
        values["tracking_kwargs"][key] = invalid
        with pytest.raises(ValueError, match="actual"):
            module.effective_parameters(values, 100000)


def test_official_plan_reuses_sift2_preserves_definition_and_thread_modes(tmp_path):
    module = reference_tool()
    profiles = {f"atlas{i}": {} for i in range(8)}
    plan = module.command_plan(tmp_path / "bin", tmp_path / "results", profiles, list(range(5)),
                               module.effective_parameters(payload(), 100000), 8)
    assert len(plan) == 198  # 13 header readbacks + five sets of 37 actual analysis commands
    assert all(item["stage"] == "input_readback" for item in plan[:13])
    assert not any(".mif" in argument for item in plan for argument in item["argv"])
    assert sum(item["stage"] == "sift2" for item in plan) == 5
    assert sum(item["stage"] == "matrix_count" for item in plan) == 40
    for record in plan:
        argv = record["argv"]
        if record["stage"] == "tracking":
            for option, value in (("-nthreads", "0"), ("-seeds", "100000"), ("-select", "0"),
                                  ("-samples", "3"), ("-step", "1.25"), ("-minlength", "5.0"),
                                  ("-trials", "1000"), ("-max_attempts_per_seed", "1000"),
                                  ("-downsample", "2")):
                assert argv[argv.index(option) + 1] == value
        elif record["stage"].startswith("matrix_"):
            assert argv[argv.index("-nthreads") + 1] == "8"
            assert "-zero_diagonal" not in argv
            if record["stage"] == "matrix_count":
                assert "-tck_weights_in" not in argv
            else:
                assert "-tck_weights_in" in argv
            if record["stage"] in ("matrix_mean_fa", "matrix_mean_length"):
                assert argv[argv.index("-stat_edge") + 1] == "mean"


def test_reference_nifti2_export_preserves_geometry_and_values(tmp_path):
    module = reference_tool()
    values = payload()
    values["five_tissue_affine"][3] = torch.tensor(
        [5.287e-19, 4.686e-18, -4.632e-19, 0.9999999999999994], dtype=torch.float64)
    checkpoint, native = tmp_path / "checkpoints", tmp_path / "native"
    checkpoint.mkdir()
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), values["fod_affine"].numpy()), checkpoint / "fa.nii.gz")
    profile = native / "atlases/fs-aparc"
    write_csv(profile, metadata=True)
    labels = np.tile(np.arange(1, 5, dtype=np.int32), (4, 4, 1))
    nib.save(nib.Nifti1Image(labels, values["fod_affine"].numpy()), profile / "atlas_dwi.nii.gz")
    (profile / "atlas.sha256").unlink()  # synthetic metadata SHA is not a real image hash
    exported = module.export_inputs(tmp_path / "reference", values, checkpoint, load_profiles(native))
    image = nib.load(exported["wm_fod"]["path"])
    assert isinstance(image, nib.Nifti2Image)
    assert np.array_equal(image.affine, values["fod_affine"].numpy())
    assert np.array_equal(np.asarray(image.dataobj), values["wm_sh"].numpy())
    assert exported["five_tissue_act"]["header_spacing"] == [1., 1., 1.]
    act = exported["five_tissue_act"]
    assert act["sform_3x4_bits_verified_equal"]
    assert act["affine"][3] == values["five_tissue_affine"][3].tolist()
    assert act["file_affine"][3] == [0., 0., 0., 1.]
    assert np.array_equal(nib.load(act["path"]).affine[:3], values["five_tissue_affine"].numpy()[:3])
    assert exported["five_tissue_sift2"]["header_spacing"][0] == values["five_tissue_affine"][0, 0].item()
    assert exported["atlases"]["fs-aparc"]["original_affine_max_abs_difference"] > 0


def reference_cli_fixture(module, tmp_path):
    checkpoint, native, binary = tmp_path / "checkpoints", tmp_path / "native", tmp_path / "bin"
    checkpoint.mkdir()
    binary.mkdir()
    values = payload()
    torch.save(values, checkpoint / "tracking_inputs.pt")
    np.savez(checkpoint / "geometry.npz", dwi_affine=values["fod_affine"].numpy(),
             five_tissue_affine=values["five_tissue_affine"].numpy())
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), values["fod_affine"].numpy()), checkpoint / "fa.nii.gz")
    tracks = nib.streamlines.Tractogram([np.array([[0., 0., 0.], [1., 1., 1.]], np.float32)], affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tracks, checkpoint / "tracks.tck")
    np.savez(checkpoint / "track_metrics.npz", weights=np.ones(1), lengths=np.ones(1),
             mean_fa=np.ones(1), endpoints=np.ones((1, 2, 3)))
    profile = native / "atlases/fs-aparc"
    write_csv(profile, metadata=True)
    nib.save(nib.Nifti1Image(np.tile(np.arange(1, 5, dtype=np.int32), (4, 4, 1)),
                           values["fod_affine"].numpy()), profile / "atlas_dwi.nii.gz")
    (profile / "atlas.sha256").unlink()
    for name in module.PROGRAMS:
        (binary / name).write_text("#!/bin/sh\nexit 0\n")
        (binary / name).chmod(0o755)
    return ["--mrtrix-bin", str(binary), "--checkpoint-dir", str(checkpoint),
            "--fnit-dir", str(native), "--output-dir", str(tmp_path / "reference"),
            "--dataset", "CPU tool regression fixture"]


def test_reference_dry_run_does_not_create_outputs_or_execute(tmp_path, monkeypatch, capsys):
    module = reference_tool()
    arguments = reference_cli_fixture(module, tmp_path)
    monkeypatch.setattr(module, "_run", lambda *args: pytest.fail("dry-run executed an official command"))
    module.main([*arguments, "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["seeds"] == [0, 1, 2, 3, 4]
    assert result["execution_completed"] is False
    assert result["source_readiness"]["status"] == "computed_outputs_ready"
    assert not (tmp_path / "reference").exists()


def test_pretracking_snapshot_alone_is_not_completed_core_and_geometry_must_match(tmp_path):
    module = reference_tool()
    arguments = reference_cli_fixture(module, tmp_path)
    checkpoint = tmp_path / "checkpoints"
    source = module.source_readiness(checkpoint, payload())
    assert source["accepted_tracks"] == 1
    np.savez(checkpoint / "geometry.npz", dwi_affine=np.eye(4),
             five_tissue_affine=payload()["five_tissue_affine"].numpy())
    with pytest.raises(ValueError, match="completed core geometry differs"):
        module.source_readiness(checkpoint, payload())
    (checkpoint / "tracks.tck").unlink()
    with pytest.raises(FileNotFoundError, match="not ready"):
        module.main([*arguments, "--dry-run"])


def test_nifti2_preserves_original_nan_payload_without_finite_filter(tmp_path):
    module = reference_tool()
    values = np.array([0x7FC00042], dtype=np.uint32).view(np.float32).reshape(1, 1, 1)
    result = module._save_image(tmp_path / "nan_fa.nii.gz", values, np.eye(4))
    assert result["voxel_bits_verified_equal"]
    assert result["nonfinite_count"] == 1 and result["nonfinite_coordinates"] == [[0, 0, 0]]
    decoded = np.asarray(nib.load(result["path"]).dataobj)
    assert np.array_equal(decoded.view(np.uint32), values.view(np.uint32))


@pytest.mark.parametrize("affine_dtype", [np.float32, np.float64])
def test_sform_implicit_bottom_accepts_only_dtype_machine_roundoff(tmp_path, affine_dtype):
    module = reference_tool()
    affine = np.eye(4, dtype=affine_dtype)
    epsilon = np.finfo(affine_dtype).eps
    affine[3] = [epsilon / 4, -epsilon / 8, epsilon / 16, 1 - 2 * epsilon]
    result = module._save_image(tmp_path / "roundoff.nii.gz", np.ones((2, 3, 4), np.float32), affine)
    contract = result["affine_serialization"]
    assert contract["machine_epsilon"] == epsilon
    assert contract["machine_bound_gamma4"] == 4 * epsilon / (1 - 4 * epsilon)
    assert contract["source_bottom_row"] == affine[3].tolist()
    assert result["file_affine"][3] == [0., 0., 0., 1.]
    assert result["affine"] == affine.tolist()
    bound = contract["machine_bound_gamma4"]
    affine[3] = [np.nextafter(affine_dtype(bound), affine_dtype(np.inf)), 0, 0, 1]
    with pytest.raises(ValueError, match="projective/non-affine"):
        module._save_image(tmp_path / "beyond_machine_bound.nii.gz", np.ones((2, 3, 4), np.float32), affine)
    assert not (tmp_path / "beyond_machine_bound.nii.gz").exists()


def test_projective_bottom_rejected_and_first_3x4_is_bit_checked(tmp_path, monkeypatch):
    module = reference_tool()
    affine = np.eye(4)
    affine[3, 0] = 1e-8
    with pytest.raises(ValueError, match="projective/non-affine"):
        module._save_image(tmp_path / "projective.nii.gz", np.ones((2, 3, 4), np.float32), affine)
    affine = np.eye(4)
    original_save = nib.save
    def corrupt_sform(image, path):
        wrong = image.affine.copy()
        wrong[0, 0] = np.nextafter(wrong[0, 0], np.inf)
        image.set_sform(wrong)
        original_save(image, path)
    monkeypatch.setattr(module.nib, "save", corrupt_sform)
    with pytest.raises(ValueError, match="representable 3x4 sform"):
        module._save_image(tmp_path / "one_bit_change.nii.gz", np.ones((2, 3, 4), np.float32), affine)


def test_actual_mrinfo_contract_records_geometry_without_posthoc_tolerance(tmp_path):
    module = reference_tool()
    path = tmp_path / "image.nii.gz"
    affine = np.diag([-2.5, 2.5, 2.5, 1.])
    affine[3, 0] = np.finfo(np.float64).eps / 4
    exported = module._save_image(path, np.ones((4, 5, 6), np.float32), affine)
    decoded_affine = np.diag([2.5, 2.5, 2.5, 1.])
    decoded_affine[0, 3] = -7.5
    transform = decoded_affine.copy()
    transform[:3, :3] /= 2.5
    actual = {"name": str(path), "format": "NIfTI-2 (GZip compressed)", "datatype": "Float32LE",
              "size": [4, 5, 6], "strides": [-1, 2, 3], "spacing": [2.5, 2.5, 2.5],
              "transform": transform.tolist(), "intensity_offset": 0, "intensity_scale": 1}
    record = {"json": str(tmp_path / "mrinfo.json")}
    Path(record["json"]).write_text(json.dumps(actual))
    result = module.input_readback(record, exported)
    assert result["maximum_corner_difference_from_source_affine_mm"] == 0
    assert result["axis_mapping_to_source"][0] == [-1, 0, 0, 3]
    assert result["source_voxel_to_world_affine"][3] == affine[3].tolist()
    assert result["file_voxel_to_world_affine"][3] == [0., 0., 0., 1.]
    assert result["maximum_corner_difference_from_file_affine_mm"] == 0
    assert "tolerance" not in result and "scientific_parity" not in result
    actual["intensity_scale"] = 2
    Path(record["json"]).write_text(json.dumps(actual))
    with pytest.raises(ValueError, match="altered intensity scaling"):
        module.input_readback(record, exported)


def test_reference_failure_persists_failed_command_and_leaves_parity_unassessed(tmp_path, monkeypatch):
    module = reference_tool()
    arguments = reference_cli_fixture(module, tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="CPU fixture version", stderr="", returncode=0))
    monkeypatch.setattr(module, "_run", lambda record, environment:
                        {**record, "seconds_inclusive": .001, "returncode": 7})
    with pytest.raises(RuntimeError, match="official command failed"):
        module.main(arguments)
    result = json.loads((tmp_path / "reference/reference_manifest.json").read_text())
    assert result["completed_commands"][-1]["returncode"] == 7
    assert result["execution_completed"] is False
    assert result["scientific_parity"] == "not_assessed"
    assert result["error"]["type"] == "RuntimeError"


def test_absent_trailing_nodes_align_by_zero_without_changing_raw_official_csv(tmp_path):
    module = reference_tool()
    directory = tmp_path / "reference"
    write_csv(directory, metadata=True)
    for name, values in matrices().items():
        (directory / f"connectome_{name}.csv").unlink()
        np.savetxt(directory / f"{name}.csv", values[:3, :3], delimiter=",")
    raw_hashes = {name: sha256(directory / f"{name}.csv") for name in NAMES}
    result = module.align_official_matrices(directory, canonical_nodes=4, maximum_present_label=3)
    assert result["absent_trailing_nodes"] == [4]
    assert result["raw_matrix_sha256"] == raw_hashes
    arrays, metadata = load_matrices(directory)
    for name, values in arrays.items():
        assert np.array_equal(values[:3, :3], matrices()[name][:3, :3])
        assert not values[-1].any() and not values[:, -1].any()
        assert sha256(directory / f"{name}.csv") == raw_hashes[name]
    assert metadata["canonical_matrix_alignment"]["original_nodes"] == 3
    with pytest.raises(ValueError, match="truncates present"):
        module.align_official_matrices(directory, canonical_nodes=4, maximum_present_label=4)


def test_population_five_official_one_fnit_preserves_point_visit_definition(tmp_path):
    from tools import benchmark_connectome_tracking_population as population
    affine = np.diag([2.5, 2.5, 2.5, 1.])
    grid = tmp_path / 'grid.nii.gz'
    nib.save(nib.Nifti2Image(np.zeros((8, 8, 8), dtype=np.float32), affine), grid)
    tracks = [np.array([[0., 0., 0.], [2.5, 0., 0.], [2.5, 0., 0.]], dtype=np.float32),
              np.array([[5., 5., 5.], [7.5, 5., 5.]], dtype=np.float32)]
    paths = []
    for index in range(6):
        path = tmp_path / f'tracks_{index}.tck'
        nib.streamlines.save(nib.streamlines.Tractogram(tracks, affine_to_rasmm=np.eye(4)), path)
        paths.append(path)
    report, data = population.compare(paths[:5], paths[5:], grid, dataset='tool fixture only',
                                      n_seeds=10, official_seeds=list(range(5)), fnit_seeds=[0],
                                      return_data=True)
    assert report['comparison_counts'] == {'official': 10, 'fnit': 0, 'cross': 5}
    assert report['fnit_reproducibility_status'] == 'not_assessed'
    assert report['four_voxel_block_axes_mm'] == [10., 10., 10.]
    assert report['accepted_fractions']['fnit_0'] == .2
    assert len(report['pairs']) == 15
    assert data['fnit_0'][2].sum() == 5
    assert data['fnit_0'][2][1, 0, 0] == 2  # repeated stored points count twice
    assert report['ranges']['length_ks']['comparison_accepted'] == [True] * 5
    assert 'tdi_8mm_block_pearson' not in report['ranges']
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match='duplicate resolved'):
        population.compare(paths[:5], [paths[0]], grid)
    with pytest.raises(ValueError, match='duplicate seed'):
        population.compare(paths[:5], paths[5:], grid, official_seeds=[0] * 5)


def test_population_undefined_correlation_is_not_a_pass():
    from tools import benchmark_connectome_tracking_population as population
    assert population._correlation(np.zeros(4), np.zeros(4)) is None
    assert population._correlation(np.ones(4), np.ones(4)) is None
    result = envelope([None] * 10, [None] * 5, similarity=True)
    assert result['status'] == 'not_assessed'
    assert result['comparison_accepted'] == [None] * 5


@pytest.mark.parametrize('completed', [[], [{'stage': 'tracking', 'argv': ['tckgen'], 'returncode': 1}]])
def test_readonly_official_audit_rejects_incomplete_or_failed_commands(tmp_path, completed):
    path = Path(__file__).parents[2] / 'tools/reference/audit_connectome_repeats.py'
    spec = importlib.util.spec_from_file_location('readonly_audit_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'execution_completed': True, 'seeds': [0, 1, 2],
        'commands': [{'stage': 'tracking', 'argv': ['tckgen']}], 'completed_commands': completed}))
    with pytest.raises(ValueError, match='command'):
        module.audit(manifest, tmp_path / 'missing_core', tmp_path / 'missing_fnit')
