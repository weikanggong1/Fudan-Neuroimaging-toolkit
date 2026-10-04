"""Oracle identity tests; no fixture is used as a speed benchmark."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


def tool():
    path = Path(__file__).resolve().parents[2] / "tools/benchmark_invwarp_cpu_reused_reference.py"
    spec = importlib.util.spec_from_file_location("inverse_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_output_names_and_their_linked_uses_are_normalized():
    module = tool()
    def commands(directory):
        return [["invwarp", "--warp=/fixed/forward.nii.gz", f"--out={directory}/native.nii.gz", "--rel"],
                ["convertwarp", f"--warp1={directory}/native.nii.gz", f"--out={directory}/final.nii.gz", "--absout"]]
    first = module.normalized_commands(commands("/first"))
    assert first == module.normalized_commands(commands("/second"))
    assert first[0][1] == "--warp=/fixed/forward.nii.gz"
    changed = commands("/second")
    changed[0][1] = "--warp=/other/forward.nii.gz"
    assert first != module.normalized_commands(changed)


@pytest.mark.parametrize("field", ["argv", "input_sha", "program_sha", "dependency_sha", "threads", "affinity", "environment", "format"])
def test_any_contract_difference_forbids_reuse(field):
    module = tool()
    before = {"argv": ["--rel"], "input_sha": "a", "program_sha": "b", "dependency_sha": "c",
              "threads": 1, "affinity": [34], "environment": "d", "format": ".nii.gz"}
    after = deepcopy(before)
    after[field] = "changed"
    assert module.contract_digest(before) != module.contract_digest(after)
    assert module.contract_digest(before) == module.contract_digest(dict(reversed(list(before.items()))))


def test_public_reused_oracle_has_no_native_clock_or_private_paths():
    import json
    module = tool()
    private_path = "/private/subject/input.nii.gz"
    record = {"case_id": "generic_coefficient_relative", "threads": 1, "affinity": [34],
              "reference_reused": True, "reference_contract_sha256": "contract", "timing_protocol": "single_observation",
              "native_timing_scope": "reused precision oracle; native clock omitted",
              "single_full_process_seconds": {"candidate": 1.0}, "process_cpu_usage": {"official": [], "candidate": []},
              "accuracy": {}, "adapter_accuracy": {}, "official_output_metadata": {},
              "input_metadata_before": {private_path: {"sha256": "input", "bytes": 20}},
              "official_program_metadata": [{"name": "invwarp", "sha256": "program", "bytes": 50, "path_private": private_path}],
              "worker_results": {"candidate": [{"median_seconds": 0.8, "maximum_rss_kib": 100, "output_metadata": {}, "outputs": {"inverse": private_path}}]}}
    suite = {"status": "running", "timing_policy": "strict reuse", "expected_complete_observations": 28,
             "source_metadata": {"candidate": {"tree_sha256": "source", "private": private_path}},
             "benchmark_script_sha256": "script", "records": [record]}
    result = module.public_report(suite)
    assert private_path not in json.dumps(result)
    assert "official" not in result["records"][0]["single_full_process_seconds"]
    assert result["records"][0]["reference_reused"] is True
