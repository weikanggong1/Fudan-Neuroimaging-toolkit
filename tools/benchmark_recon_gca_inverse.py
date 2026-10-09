"""真实nu/掩膜/GCA的完整首轮候选评分：仅测评分迁移，不冒充完整注册。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

import numpy as np
import torch

from fnit.recon_all import mri_em_register as source_module
from fnit.recon_all import mri_em_register_score_gpu as score_module
from fnit.recon_all.mri_em_register import (
    _vnl_affine_inverse, atlas_label_peak, estimate_image_white_matter_peak,
    find_stable_samples, gca_centroid, gca_mean_volume, read_gca,
    read_masked_input, scale_input_intensity,
)
from fnit.recon_all.mri_em_register_score_gpu import GCASearchScorer, vnl_affine_inverse_tensor
from fnit.recon_all.mri_em_register_search_source import search_linear_iteration_source
from fnit.recon_all.profiling import ProcessTreeDeviceSampler


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, report):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)


class RecordingScorer:
    """只记录原搜索生成的矩阵；返回零，不把该运行称为真实优化轨迹。"""
    candidate_chunk = 1_000_000

    def __init__(self):
        self.blocks = []

    def __call__(self, samples, source, matrix):
        return np.float32(0)

    def score_many(self, matrices):
        self.blocks.append(matrices.copy())
        return np.zeros(len(matrices), np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report_path = args.output / "report.json"
    report = {"status": "running", "scope": "complete first linear grid scoring, fixed real inputs; not full GCA or recon-all",
              "host": platform.node(), "code_version": args.code_version,
              "inputs_sha256": {name: sha(getattr(args, name)) for name in ("nu", "mask", "atlas")},
              "source_sha256": {"score": sha(score_module.__file__), "source": sha(source_module.__file__), "script": sha(__file__)},
              "device": args.device, "threads": args.threads, "torch": torch.__version__,
              "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "PYTORCH_NO_CUDA_MEMORY_CACHING")},
              "matrix_grid": "source search grid from identity at real atlas centroid; not a production optimization trajectory",
              "precision": "FP32 source arithmetic; source reciprocal and likelihood use FP64; TF32 enabled; no half precision",
              "acceptance_before_test": {"inverse": "float32 bits exact", "score_strict_reproduction": "float32 bits exact (diagnostic)",
                                         "optimization_regression": "existing scorer abs tolerance 1e-2 and first argmax unchanged; full registration still required",
                                         "existing_score_abs": 1e-2},
              "whole_metric_equivalence": "not_assessed", "trials": []}
    save(report_path, report)
    try:
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(args.device)
        started = time.perf_counter()
        torch.cuda.synchronize(args.device)
        report["cuda_init_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        atlas = read_gca(args.atlas)
        masked = read_masked_input(args.nu, args.mask)
        peak, _, _ = estimate_image_white_matter_peak(atlas, masked)
        source = scale_input_intensity(masked, atlas_label_peak(atlas, 2), peak)
        samples = find_stable_samples(atlas)
        center = gca_centroid(gca_mean_volume(atlas))
        report["preparation_seconds"] = time.perf_counter() - started
        report["sample_count"] = len(samples.means)
        started = time.perf_counter()
        score_module._density_log_terms(samples.variances, samples.priors)
        report["shared_numba_log_terms_warmup_seconds"] = time.perf_counter() - started
        recorder = RecordingScorer()
        started = time.perf_counter()
        search_linear_iteration_source(samples, source, np.eye(4, dtype=np.float32), center,
                                       1, .85, 1.15, scorer=recorder)
        matrices = recorder.blocks[0]
        report["matrix_generation_seconds"] = time.perf_counter() - started
        report["candidate_count"] = len(matrices)
        matrix_path = args.output / "first_grid.npy"
        np.save(matrix_path, matrices)
        report["matrix_sha256"] = sha(matrix_path)
        expected = np.stack([_vnl_affine_inverse(matrix) for matrix in matrices])
        # 与标量公式比较完整真实网格；不只检查随机小矩阵。
        inverse = vnl_affine_inverse_tensor(torch.tensor(matrices, device=args.device)).cpu().numpy()
        report["inverse_comparison"] = {
            "different_float32_bits": int(np.count_nonzero(expected.view(np.uint32) != inverse.view(np.uint32))),
            "max_abs": float(np.abs(expected.astype(float) - inverse.astype(float)).max(initial=0))}
        del expected, inverse
        save(report_path, report)
        reference = None
        for backend, chunk in (("cpu", 64), ("torch", 64), ("torch", 256),
                               ("torch", 1024), ("torch", 256), ("cpu", 64)):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            stop = threading.Event()
            sampler = ProcessTreeDeviceSampler(device=args.device, parent_pid=os.getpid(), interval=.2)

            def sample():
                while not stop.is_set():
                    sampler.sample_if_due()
                    stop.wait(.05)

            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()
            started = time.perf_counter()
            scorer = GCASearchScorer(samples, source, device=args.device, candidate_chunk=chunk,
                                     sample_chunk=8192, reduce_on_device=True, inverse_backend=backend)
            setup = time.perf_counter() - started
            scores = scorer.score_many(matrices)
            torch.cuda.synchronize(args.device)
            seconds = time.perf_counter() - started
            stop.set()
            monitor.join()
            np.save(args.output / f"scores_{len(report['trials'])}.npy", scores)
            if reference is None:
                reference = scores.copy()
            delta = np.abs(reference.astype(float) - scores.astype(float))
            trial = {"inverse_backend": backend, "candidate_chunk": chunk,
                     "setup_and_transfer_seconds": setup, "complete_scoring_seconds": seconds,
                     "different_score_bits": int(np.count_nonzero(scores.view(np.uint32) != reference.view(np.uint32))),
                     "max_abs": float(delta.max(initial=0)), "p99_abs": float(np.percentile(delta, 99)),
                     "first_argmax": int(np.argmax(scores)), "first_argmax_same": bool(np.argmax(scores) == np.argmax(reference)),
                     "allocated_peak_bytes": torch.cuda.max_memory_allocated(args.device),
                     "reserved_peak_bytes": torch.cuda.max_memory_reserved(args.device),
                     "process_memory": sampler.report()}
            report["trials"].append(trial)
            save(report_path, report)
            del scorer
        report["status"] = "complete"
        report["strict_reproduction"] = ("exact_for_fixed_grid" if
            report["inverse_comparison"]["different_float32_bits"] == 0 and
            all(trial["different_score_bits"] == 0 for trial in report["trials"]) else "not_passed")
        report["optimization_regression"] = ("fixed_grid_only_selection_unchanged" if
            report["inverse_comparison"]["different_float32_bits"] == 0 and
            all(trial["max_abs"] <= 1e-2 and trial["first_argmax_same"] for trial in report["trials"])
            else "not_passed")
        save(report_path, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save(report_path, report)
        raise


if __name__ == "__main__":
    main()
