"""Publication boundary tests; these fixtures are not performance benchmarks."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load_tool(name):
    path = Path(__file__).resolve().parents[2] / "tools" / (name + ".py")
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_public_figure_rejects_other_input_before_reading_or_writing(tmp_path, monkeypatch):
    tool = load_tool("plot_fnirt_invwarp_cpu_public")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("This is not the authorized public T1 input")
    destination = tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["plot", "--public-run", str(tmp_path / "absent-run"),
                                    "--public-input", str(unrelated), "--output-dir", str(destination)])
    with pytest.raises(ValueError, match="Only the verified public"):
        tool.main()
    assert not destination.exists()


@pytest.mark.parametrize("single_observation", [False, True])
def test_numeric_public_summary_omits_private_paths(tmp_path, single_observation):
    tool = load_tool("benchmark_fnirt_invwarp_cpu_public")
    private_marker = "/private/input/output/tree"
    worker = {"median_seconds": 3.0, "maximum_rss_kib": 1200,
              "outputs": {"iout": private_marker}, "fnit_source": private_marker,
              "output_metadata": {"iout": {"sha256": "ef", "bytes": 100}}}
    record = {"case_id": "public_cc0_t1_default", "threads": 1, "affinity": [0],
              "warmup_full_process": {"official": 4.0, "candidate": 3.0},
              "full_process": {"official": [4.0, 3.9, 3.8], "candidate": [3.0, 2.9, 2.8]},
              "median_full_process_seconds": {"official": 3.85, "candidate": 2.85},
              "process_cpu_usage": {"candidate": [{"user_seconds": 2.0}]},
              "worker_results": {"candidate": [worker]},
              "accuracy": {"candidate_vs_official": {"iout": {"mean_absolute_error": 1.0}}},
              "adapter_accuracy": {"region_voxels": 100},
              "official_program_metadata": [{"name": "fnirt", "sha256": "ab", "bytes": 100,
                                               "path_private": private_marker}],
              "official_output_metadata": {"iout": {"sha256": "fg", "bytes": 100}},
              "input_metadata_before": {private_marker: {"bytes": 100}}}
    if single_observation:
        record.update({"timing_protocol": "single_observation",
                       "single_full_process_seconds": {"official": 4.0, "candidate": 3.0}})
    suite = {"status": "executed_with_numeric_comparisons", "timing_policy": "paired",
             "records": [record], "candidate_root": private_marker,
             "source_metadata": {"candidate": {"tree_sha256": "cd", "snapshot": {"source": private_marker}}}}
    path = tmp_path / "suite.private.json"
    path.write_text(json.dumps(suite))
    report = tool.summarize_suite(path)
    assert private_marker not in json.dumps(report)
    if single_observation:
        assert report["records"][0]["single_full_process_seconds"]["candidate"] == 3.0
        assert "median_full_process_seconds" not in report["records"][0]
        assert "warmup_seconds" not in report["records"][0]
    else:
        assert report["records"][0]["paired_full_process_seconds"]["candidate"] == [2.9, 2.8]


def test_figure_rejects_completed_suite_without_public_input_binding(tmp_path):
    suite = {"status": "executed_with_numeric_comparisons", "records": [
        {"threads": 8, "input_metadata_before": {"/private/case": {"sha256": "different"}}}]}
    path = tmp_path / "suite.private.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="not bound"):
        load_tool("plot_fnirt_invwarp_cpu_public").select_outputs(path, 8)


def test_incomplete_suite_cannot_supply_public_figures_or_summary(tmp_path):
    path = tmp_path / "suite.private.json"
    path.write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="unfinished"):
        load_tool("benchmark_fnirt_invwarp_cpu_public").summarize_suite(path)
    with pytest.raises(ValueError, match="completed"):
        load_tool("plot_fnirt_invwarp_cpu_public").select_outputs(path, 8)


def test_saved_public_figures_resolve_api_subdirectory_and_verify_output_hash(tmp_path):
    tool = load_tool("plot_fnirt_invwarp_cpu_public")
    candidate = tmp_path / "pair_0" / "candidate" / "api_0" / "iout.nii.gz"
    official = tmp_path / "pair_0" / "official" / "iout.nii.gz"
    candidate.parent.mkdir(parents=True)
    official.parent.mkdir()
    candidate.write_bytes(b"candidate")
    official.write_bytes(b"official")
    record = {"threads": 8, "input_metadata_before": {"input": {"sha256": tool.PUBLIC_T1_SHA256}},
              "worker_results": {"candidate": [{"outputs": {"iout": str(candidate)},
                  "output_metadata": {"iout": {"sha256": tool.sha256(candidate)}}}]},
              "official_output_metadata": {"iout": {"sha256": tool.sha256(official)}}}
    suite = tmp_path / "suite.private.json"
    suite.write_text(json.dumps({"status": "executed_with_numeric_comparisons", "records": [record]}))
    assert tool.select_outputs(suite, 8) == ({"iout": str(official)}, {"iout": str(candidate)})
    official.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after"):
        tool.select_outputs(suite, 8)
