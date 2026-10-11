"""只读比较的完成边界、对应关系及真实nibabel格式合同，非MRI benchmark。"""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import nibabel.freesurfer.io as fsio
import numpy as np
import pytest

import compare_recon_genetic_mesh as tool


@pytest.fixture
def mesh(tmp_path):
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    geometry = dict(head=np.array([2, 0, 20]), valid="1  # volume info valid",
        filename="fixture", volume=np.array([256, 256, 256]),
        voxelsize=np.ones(3), xras=np.array([-1., 0, 0]),
        yras=np.array([0, 0, -1.]), zras=np.array([0, 1., 0]), cras=np.zeros(3))
    module_path = Path(__file__).parents[2] / "validation/recon_all/accuracy_20261003/task_04/reference_surface_distance.py"
    spec = importlib.util.spec_from_file_location("_actual_triangle_diagnostic", module_path)
    distance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distance)
    return xyz, faces, geometry, distance


def test_same_ordered_mesh_has_exact_full_point_proof(tmp_path, mesh):
    xyz, faces, geometry, distance = mesh
    a, b, out = tmp_path / "candidate", tmp_path / "reference", tmp_path / "distances.npz"
    for p in (a, b):
        fsio.write_geometry(str(p), xyz, faces, volume_info=geometry)
    result = tool.compare_mesh(candidate=a, reference=b, distance_module=distance,
                               output_checkpoint=out)
    assert result["vertex_correspondence"] and result["decoded_coordinates_equal"]
    assert result["indexed_distance_mm"]["max_mm"] == 0
    assert result["forward_proof"]["all_vertices_evaluated"]
    assert result["reverse_proof"]["total_vertices"] == 4
    assert result["candidate_topology"]["valid"]
    with np.load(out) as data:
        assert data["candidate_to_reference"].shape == (4,)
        assert data["reference_to_candidate"].tobytes() == np.zeros(4, np.float64).tobytes()


def test_changed_ordered_faces_never_get_indexed_comparison(tmp_path, mesh):
    xyz, faces, geometry, distance = mesh
    a, b, out = tmp_path / "candidate", tmp_path / "reference", tmp_path / "distances.npz"
    fsio.write_geometry(str(a), xyz, np.roll(faces, 1, axis=1), volume_info=geometry)
    fsio.write_geometry(str(b), xyz, faces, volume_info=geometry)
    result = tool.compare_mesh(candidate=a, reference=b, distance_module=distance,
                               output_checkpoint=out)
    assert not result["ordered_faces_equal"] and not result["vertex_correspondence"]
    assert result["indexed_distance_mm"] is None
    assert result["coordinate_elements_different"] is None
    assert result["candidate_to_reference_triangle"]["max_mm"] == 0


def test_changed_ras_geometry_fails_before_distance(tmp_path, mesh):
    xyz, faces, geometry, distance = mesh
    a, b, out = tmp_path / "candidate", tmp_path / "reference", tmp_path / "distances.npz"
    fsio.write_geometry(str(a), xyz, faces, volume_info=geometry)
    fsio.write_geometry(str(b), xyz, faces, volume_info={**geometry, "cras": np.ones(3)})
    with pytest.raises(ValueError, match="same surface RAS geometry"):
        tool.compare_mesh(candidate=a, reference=b, distance_module=distance,
                          output_checkpoint=out)
    assert not out.exists()


def test_incomplete_own_stage_never_opens_reference(tmp_path):
    out = tmp_path / "diagnostic"
    unavailable = tmp_path / "reference_must_not_be_opened"
    with patch.object(tool, "verified_premesh", side_effect=ValueError("incomplete own stage")):
        with pytest.raises(ValueError, match="incomplete own stage"):
            tool.compare(premesh_directory=tmp_path / "own", orig_directory=tmp_path / "orig",
                strengthened_report=tmp_path / "stronger", reference_subject=unavailable,
                reference_benchmark=unavailable, reference_runtime=unavailable,
                reference_provenance=unavailable, case_name="case", hemisphere="lh",
                surface_distance_module=unavailable, output_directory=out, threads=4)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["message"] == "incomplete own stage"
    assert not (out / "COMPLETE.json").exists()


@pytest.fixture
def complete_contract(tmp_path, mesh):
    """小网格与假影像字节只检验完成/身份合同，不作为真实MRI验证。"""
    xyz, faces, geometry, _ = mesh
    premesh_dir, orig_dir, ref = (tmp_path / "premesh", tmp_path / "orig", tmp_path / "ref")
    for directory in (premesh_dir, orig_dir, ref / "surf", ref / "mri"):
        directory.mkdir(parents=True)
    premesh, orig = premesh_dir / "lh.orig.premesh", orig_dir / "lh.orig"
    for path in (premesh, orig, ref / "surf/lh.orig.premesh", ref / "surf/lh.orig"):
        fsio.write_geometry(str(path), xyz, faces, volume_info=geometry)
    inputs = {"nofix": ref / "surf/lh.orig.nofix", "qsphere": ref / "surf/lh.qsphere.nofix",
              "brain": ref / "mri/brain.mgz"}
    for key, path in inputs.items():
        path.write_bytes(key.encode())
    parent = {"input_sha256": {key: tool.sha(path) for key, path in inputs.items()}, "threads": 4}
    (premesh_dir / "report.json").write_text(json.dumps(parent))
    (premesh_dir / "COMPLETE.json").write_text("{}")
    own = dict(status="complete_orig_stage", native_program_called=False,
        reference_inputs_used=False, all_algorithm_inputs_self_generated=True,
        input_sha256={"premesh": tool.sha(premesh), "premesh_report": tool.sha(premesh_dir / "report.json"),
                      "premesh_complete": tool.sha(premesh_dir / "COMPLETE.json")},
        parent_input_sha256=parent["input_sha256"], output_sha256={orig.name: tool.sha(orig)}, cleanup={})
    (orig_dir / "report.json").write_text(json.dumps(own))
    (orig_dir / "COMPLETE.json").write_text(json.dumps(dict(status=own["status"],
        report_sha256=tool.sha(orig_dir / "report.json"))))
    stronger = tmp_path / "stronger.json"
    stronger.write_text(json.dumps(dict(status="passed_strengthened_orig_completion_validation",
        parent_report_sha256=tool.sha(orig_dir / "report.json"),
        parent_complete_sha256=tool.sha(orig_dir / "COMPLETE.json"),
        parent_premesh_report_sha256=tool.sha(premesh_dir / "report.json"),
        actual_orig_sha256=tool.sha(orig), quality={"geometry": {"valid": True}})))
    benchmark, runtime, provenance = (tmp_path / name for name in ("benchmark.json", "runtime.json", "provenance.json"))
    program_sha = "a" * 64
    benchmark.write_text(json.dumps(dict(execution="complete", threads=4,
        native_program_sha256={"mris_fix_topology_fnit": program_sha})))
    runtime.write_text(json.dumps({"topology_repair": {"sha256": program_sha}}))
    provenance.write_text(json.dumps(dict(frozen_original_source_commit="fixture_only", cases=[dict(case="case",
        original_benchmark_receipt_sha256=tool.sha(benchmark), original_runtime_receipt_sha256=tool.sha(runtime),
        recorded_original_program_sha256=program_sha, recorded_code_version="fixture_only",
        preflight_reference_surface_sha256={stage: tool.sha(ref / "surf" / f"lh.{stage}")
            for stage in ("orig.premesh", "orig")})])))
    args = dict(premesh_directory=premesh_dir, orig_directory=orig_dir, strengthened_report=stronger,
        reference_subject=ref, reference_benchmark=benchmark, reference_runtime=runtime,
        reference_provenance=provenance, case_name="case", hemisphere="lh", threads=4,
        surface_distance_module=Path(__file__).parents[2] / "validation/recon_all/accuracy_20261003/task_04/reference_surface_distance.py",
        output_directory=tmp_path / "comparison")
    return args, premesh, parent


def test_completed_readonly_comparison_binds_actual_artifacts(complete_contract):
    args, premesh, parent = complete_contract
    with patch.object(tool, "verified_premesh", return_value=(premesh, parent)), \
            patch.object(tool, "final_quality", return_value={"status": "passed"}):
        report = tool.compare(**args)
    assert report["strict_geometry_reproduction"] == "passed"
    assert report["reference_full_stage_repeatability"] == "not_assessed"
    assert report["overall_metric_equivalence"] == "not_assessed"
    assert not report["native_program_called"]
    complete = json.loads((args["output_directory"] / "COMPLETE.json").read_text())
    assert complete["report_sha256"] == tool.sha(args["output_directory"] / "report.json")


def test_reference_mutation_rejected_before_distance(complete_contract):
    args, premesh, parent = complete_contract
    changed = args["reference_subject"] / "surf/lh.orig"
    with changed.open("ab") as stream:
        stream.write(b"changed-after-preflight")
    with patch.object(tool, "verified_premesh", return_value=(premesh, parent)), \
            patch.object(tool, "final_quality", return_value={"status": "passed"}), \
            patch.object(tool, "compare_mesh") as mesh_compare:
        with pytest.raises(ValueError, match="frozen preflight observation"):
            tool.compare(**args)
        mesh_compare.assert_not_called()
    assert not (args["output_directory"] / "COMPLETE.json").exists()
