"""Removed preparation helpers can only be replayed from an explicit snapshot."""
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_PATH = ROOT / "validation/torchgems/cpu_initial_alignment_20261007/alignment_experiment.py"
SNAPSHOT = ROOT / "validation/robust_register/rigid_affine_20261006/candidate_source/robust_register/preparation.py"


def bridge():
    name = "_fnit_archived_preparation_contract"
    specification = importlib.util.spec_from_file_location(name, BRIDGE_PATH)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_archived_helper_source_must_be_explicit():
    with pytest.raises(ValueError, match="preparation_source is required"):
        bridge()._load_preparation_source(None)


def test_archived_helper_source_requires_pinned_checksum(tmp_path):
    source = tmp_path / "preparation.py"
    source.write_text("raise AssertionError('must never execute')\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        bridge()._load_preparation_source(source)


def test_archived_helper_source_loads_declared_snapshot_only():
    module = bridge()._load_preparation_source(SNAPSHOT)
    assert Path(module.__file__).resolve() == SNAPSHOT.resolve()
    assert callable(module.prepare_subregion_alignment_target)
    assert callable(module.reflect_atlas_header)
    production = __import__("fnit.robust_register", fromlist=["FNIT_mri_robust_register"])
    assert not hasattr(production, "prepare_subregion_alignment_target")
