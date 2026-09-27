"""Multi-subject scheduler for the Conda recon-all pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import subprocess
import sys


def run_recon_all_python_batch(
    jobs: list[dict], weights_dir: str | Path, assets_dir: str | Path,
    *, devices: tuple[str, ...] = ("cuda:0",), threads: int = 4,
    n4_python: str | Path | None = None,
    native_bin_dir: str | Path | None = None,
    native_topology: bool = False,
    native_surface_metrics: bool = False,
    native_registration: bool = False,
    native_sphere: bool = False,
    native_white_preaparc: bool = False,
) -> list[dict]:
    """Run one subject per device; each reconstruction uses a separate Python process.

    Each job maps ``t1`` to an image path and ``subject_dir`` to an empty
    output folder. Return one single-subject run-report dict per input job,
    preserving job order; failures raise RuntimeError.
    """
    if not devices or len(set(devices)) != len(devices) or any(
        device != "cpu" and re.fullmatch(r"cuda:\d+", device) is None for device in devices
    ):
        raise ValueError("devices must be distinct CPU/CUDA device names")
    if threads < 1:
        raise ValueError("threads must be positive")
    if native_bin_dir is None:
        raise ValueError("native stages require native_bin_dir")
    if native_registration and not native_topology:
        raise ValueError("native_registration requires native_topology")
    if native_sphere and not native_topology:
        raise ValueError("native_sphere requires native_topology")
    if native_white_preaparc and not native_topology:
        raise ValueError("native_white_preaparc requires native_topology")
    weights, assets = Path(weights_dir).resolve(), Path(assets_dir).resolve()
    if not weights.is_dir() or not assets.is_dir():
        raise FileNotFoundError("weights_dir and assets_dir must exist")
    prepared = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) != {"t1", "subject_dir"}:
            raise ValueError(f"job {index} needs t1 and subject_dir")
        t1, subject = Path(job["t1"]).resolve(), Path(job["subject_dir"]).resolve()
        if not t1.is_file():
            raise FileNotFoundError(t1)
        if subject.exists() and (not subject.is_dir() or any(subject.iterdir())):
            raise ValueError(f"job {index} subject_dir must be empty")
        if any(subject == previous or subject.is_relative_to(previous)
               or previous.is_relative_to(subject) for _, previous in prepared):
            raise ValueError(f"job {index} subject_dir overlaps another job")
        prepared.append((t1, subject))

    def run_device(device: str, assignments: list[tuple[int, Path, Path]]):
        results = []
        for index, t1, subject in assignments:
            command = [sys.executable, "-m", "fnit.recon_all.native_free",
                       str(t1), str(subject), "--weights-dir", str(weights),
                       "--assets-dir", str(assets), "--device", device,
                       "--threads", str(threads)]
            if n4_python is not None:
                command += ["--n4-python", str(n4_python)]
            if native_bin_dir is not None:
                command += ["--native-bin-dir", str(Path(native_bin_dir).resolve())]
            if native_topology:
                command.append("--native-topology")
            if native_surface_metrics:
                command.append("--native-surface-metrics")
            if native_registration:
                command.append("--native-registration")
            if native_sphere:
                command.append("--native-sphere")
            if native_white_preaparc:
                command.append("--native-white-preaparc")
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                results.append((index, None, completed.stderr.strip()))
            else:
                results.append((index, json.loads(
                    (subject / "fnit-native-free-run.json").read_text()), None))
        return results

    assignments = [[] for _ in devices]
    for index, (t1, subject) in enumerate(prepared):
        assignments[index % len(devices)].append((index, t1, subject))
    reports, errors = [None] * len(prepared), []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(run_device, device, assigned)
                   for device, assigned in zip(devices, assignments) if assigned]
        for future in futures:
            for index, report, error in future.result():
                reports[index] = report
                if error:
                    errors.append(f"job {index}: {error}")
    if errors:
        raise RuntimeError("batch reconstruction failed: " + "; ".join(errors))
    return reports
