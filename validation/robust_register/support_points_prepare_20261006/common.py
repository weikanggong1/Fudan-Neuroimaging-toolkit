"""Finite saved-geometry diagnostic helpers; metadata only at import time."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def check_bindings(bindings):
    actual = {}
    for name, expected in bindings.items():
        path = Path(expected["path"])
        value = {"bytes": int(path.stat().st_size), "sha256": digest(path)}
        if value != {key: expected[key] for key in ("bytes", "sha256")}:
            raise ValueError("frozen source/input/reference changed: " + name)
        actual[name] = value
    return actual


def save_exclusive(path, value):
    """Serialize first, then atomically publish a new mode-600 JSON file."""
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n").encode()
    path = Path(path)
    temporary = path.with_name(path.name + ".writing." + str(os.getpid()))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def owned_regular_lock(path):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid():
        os.close(descriptor)
        raise ValueError("common CPU lock must be an owned regular file")
    return descriptor


def validate_plan(plan):
    if plan["schema"] != 1 or plan["status"] != "prepared_not_run":
        raise ValueError("only the explicit prepared plan is supported")
    if len(plan["physical_cores"]) != 8 or len(set(plan["physical_cores"])) != 8:
        raise ValueError("exactly eight distinct physical CPU cores required")
    if plan["parameters"] != {"spatial_chunk_size": 131072}:
        raise ValueError("the original sampler chunk size is fixed")
    if plan["limits"] != {
        "lock_wait_seconds": 120, "child_seconds": 120,
        "outer_seconds": 300, "terminate_wait_seconds": 2,
        "kill_wait_seconds": 2, "address_space_bytes": 20000000000,
        "scientific_attempts": 1,
    }:
        raise ValueError("the predeclared finite resource limits changed")
    if plan["expected_target_shape"] != [39, 45, 56]:
        raise ValueError("the same saved target grid is required")
    if plan["expected_baseline"]["rigid"] != {
        "warp_relative_L2": 2.5840940291760602e-06,
        "warp_nonzero_support_difference_voxels": 1,
    } or plan["expected_baseline"]["affine"] != {
        "warp_relative_L2": 2.556642902132947e-05,
        "warp_nonzero_support_difference_voxels": 1,
    }:
        raise ValueError("the persisted baseline scalar values changed")
    if plan["unchanged_formal_gates"] != {
        "mapped_geometry_consistency_max_mm": 1e-05,
        "physical_displacement_RMS_mm_max": 0.001,
        "physical_displacement_max_mm_max": 0.01,
        "warped_same_target_grid_relative_L2_max": 1e-05,
        "warped_nonzero_support_difference_voxels": 0,
        "total": 20, "persisted_passed": 17, "persisted_failed": 3,
    }:
        raise ValueError("the original acceptance definition changed")
    root = Path(plan["canonical_root"])
    workspace = Path(plan["workspace_directory"])
    run = Path(plan["run_directory"])
    for path, kind in ((workspace, "workspaces"), (run, "runs")):
        relative = path.relative_to(root)
        if ".." in path.parts or relative.parts[:3] != (
                kind, "smri_cpu_20261004", "remaining_20261006"):
            raise ValueError("new canonical task directory required")
        if relative.parts[-1] != "robust-support-points-20261006-v1":
            raise ValueError("an independently named first diagnostic directory is required")
    if run == Path(plan["saved_output_directory"]).parent:
        raise ValueError("never write into the original failed or score directory")
    if Path(plan["worker"]) != workspace / "inspect_saved_points.py":
        raise ValueError("the diagnostic worker must be the separately frozen file")
    if set(plan["harness_bindings"]) != {"common.py", "inspect_saved_points.py", "run_prepared.py"}:
        raise ValueError("all three diagnostic source files must be bound")
    for name, binding in plan["harness_bindings"].items():
        if Path(binding["path"]) != workspace / name:
            raise ValueError("diagnostic source is not in the separately frozen directory")
