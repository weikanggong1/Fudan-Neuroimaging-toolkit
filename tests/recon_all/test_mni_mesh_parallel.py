"""标准库单测：并行生命周期、线程/缓存契约和失败报告；不替代真实回归。"""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from fnit.recon_all import mni_mesh_parallel as api


class Sampler:
    def __init__(self, **kwargs): self.kwargs = kwargs
    def add_worker(self, pid): self.pid = pid
    def sample_if_due(self, **kwargs): pass
    def report(self): return {"status": "ownership_unresolved", "peak_tree_total_bytes": None}


class MNIParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "surf").mkdir()
        for hemi in ("lh", "rh"):
            for name in ("orig", "white", "pial", "sphere.reg"):
                (self.root / "surf" / f"{hemi}.{name}").write_bytes(b"frozen mesh hash input")
        self.kwargs = dict(subject=self.root, weights=self.root, assets=self.root,
                           warp_binaries=("one", "two", "three"), device="cuda:1", threads=4,
                           report_path=self.root / "report.json", execution="parallel")

    def tearDown(self): self.temporary.cleanup()

    def invoke(self, *, mesh_status="passed", returncode=0):
        root, records = self.root, {}
        class Child:
            pid = 987654
            def __init__(self, command, **kwargs):
                records.update(command=command, kwargs=kwargs)
                self.returncode = returncode
                self.worker = Path(command[-1])
            def wait(self):
                self.worker.write_text(json.dumps({"status": "complete", "value": {"precision": "FP32"},
                    "operation_started_monotonic": time.monotonic()-1,
                    "operation_finished_monotonic": time.monotonic()}))
                return self.returncode
        with patch("fnit.recon_all.native_free._validate_meshes", return_value={"status": mesh_status}), \
             patch("fnit.recon_all.profiling.ProcessTreeDeviceSampler", Sampler), \
             patch("fnit.recon_all.hemisphere_parallel._cancel") as cancel, \
             patch.object(torch.cuda, "is_initialized", return_value=False), \
             patch.object(api.subprocess, "Popen", Child):
            if returncode or mesh_status != "passed":
                with self.assertRaises(RuntimeError): api.run_mni_and_validate(**self.kwargs)
                self.assertTrue(cancel.called)
                result = json.loads((root / "report.json").read_text())
                self.assertEqual(result["status"], "failed")
            else:
                result = api.run_mni_and_validate(**self.kwargs)
        return result, records

    def test_split_and_fresh_exec_preserve_parent(self):
        before = (torch.get_num_threads(), os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
                  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        result, records = self.invoke()
        after = (torch.get_num_threads(), os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
                 torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        self.assertEqual(before, after)
        self.assertEqual((result["parent_threads"], result["child_threads"]), (2, 2))
        self.assertTrue(records["kwargs"]["start_new_session"])
        self.assertIn("hemisphere_worker", records["command"][-3])
        self.assertEqual(records["kwargs"]["env"]["NUMBA_NUM_THREADS"], "2")
        self.assertEqual(result["device_process_tree"]["peak_tree_total_bytes"], None)
        self.assertEqual(result["status"], "complete")

    def test_child_failure_report(self): self.invoke(returncode=1)
    def test_mesh_failure_report(self): self.invoke(mesh_status="failed")

    def test_argument_guards(self):
        for value in ("cpu", "cuda"):
            with self.assertRaises(ValueError): api.run_mni_and_validate(**dict(self.kwargs, device=value))
        for value in (True, 1, 2.1):
            with self.assertRaises(ValueError): api.run_mni_and_validate(**dict(self.kwargs, threads=value))
        (self.root / "report.json").write_text("existing result")
        with self.assertRaises(FileExistsError): api.run_mni_and_validate(**self.kwargs)

    def test_mni_target_uses_restoring_device_scope(self):
        from fnit.recon_all.mni_aux_chain import TEMPLATE_DIR
        names = ("mri/orig.mgz", "mri/transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz",
                 "mri/transforms/synthmorph.1.0mm.1.0mm/aff.lta", "synthmorph.deform.3.h5",
                 str(TEMPLATE_DIR / "mni152.1.0mm.cropped.nii.gz"),
                 str(TEMPLATE_DIR / "mni152.1.0mm.nii.gz"), "one", "two", "three", "result")
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"contract fixture")
        result = {name: str(self.root / "result") for name in ("forward", "inverse", "check")}
        with patch("fnit.recon_all.mni_nonlinear_chain.run_mni_nonlinear_chain", return_value=result), \
             patch.object(torch.cuda, "device") as scope:
            api.run_mni_stage(subject=str(self.root), weights=str(self.root), assets=str(self.root),
                warp_binaries=[str(self.root / name) for name in ("one", "two", "three")],
                device="cuda:1", threads=2)
            scope.assert_called_once_with("cuda:1")
            self.assertTrue(scope.return_value.__enter__.called)
            self.assertTrue(scope.return_value.__exit__.called)


if __name__ == "__main__": unittest.main()
