"""Prevent an older derivatives directory from passing as a frozen benchmark."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


driver_path = Path(__file__).resolve().parents[2] / "validation/fmri/compare_mcflirt_optimization.py"
spec = importlib.util.spec_from_file_location("mcflirt_optimization_source_evidence", driver_path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def captured_derivative(tmp_path):
    source_root = tmp_path / "frozen_source"
    source_file = source_root / "src/fnit/mcflirt/core.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# frozen implementation\n")
    hashes = {"mcflirt/core.py": file_sha256(source_file)}
    manifest = hashlib.sha256(json.dumps(
        hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    image_path = tmp_path / "saved_clean.nii.gz"
    image_path.with_name("saved_clean.json").write_text(json.dumps({
        "FNIT": {"Source": {"SourceSHA256": hashes, "SourceManifestSHA256": manifest}}
    }))
    return image_path, source_root, source_file


def test_accepts_captured_runtime_matching_frozen_tree(tmp_path):
    image_path, source_root, _ = captured_derivative(tmp_path)
    result = driver.verify_executed_sources(image_path, source_root, file_sha256)
    assert result["all_match"] is True
    assert result["files_checked"] == 1


def test_rejects_older_runtime_despite_well_formed_manifest(tmp_path):
    image_path, source_root, source_file = captured_derivative(tmp_path)
    source_file.write_text("# a different revision with the same module name\n")
    with pytest.raises(ValueError, match="differs from supplied frozen source tree"):
        driver.verify_executed_sources(image_path, source_root, file_sha256)
