"""共享卡启动显存门：mock 验证调度边界，不代替真实 GPU benchmark。"""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


RUNNER = Path(__file__).resolve().parents[2] / "tools" / "run_recon_single_gpu_job.py"
SPEC = importlib.util.spec_from_file_location("recon_single_gpu_admission_test", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


class SharedGpuAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.receipt_dir = self.root / "receipt"
        self.lock = self.root / "gpu.lock"
        self.unresolved = self.root / "gpu.lock.unresolved.json"
        self.job = {
            "run_id": "mock-shared-job", "threads": 4,
            "cpu_affinity": [0, 1, 2, 3], "expected_host": "mock-host",
            "cuda_visible_devices": "0", "expected_gpu_uuid": GPU_UUID,
            "require_idle_gpu": False, "minimum_free_gpu_bytes": 22_000_000_000,
            "argv": ["python", "science.py"], "cwd": str(self.root),
            "receipt_dir": str(self.receipt_dir),
        }
        self.job_file = self.root / "job.json"
        self.child = Mock(pid=1234567)
        self.child.poll.return_value = 0
        self.child.wait.return_value = 0
        self.popen = self._patch("subprocess.Popen", return_value=self.child)
        self.snapshot = self._patch(
            "gpu_snapshot", return_value=({0: GPU_UUID}, "999, external, 20000", 0))
        self.free_memory = self._patch("gpu_free_memory", return_value={
            "physical_gpu_index": 0, "gpu_uuid": GPU_UUID,
            "memory_free_mib": 24_000, "memory_free_bytes": 24_000 * 1048576,
            "query_started_unix": 1.0, "query_completed_unix": 2.0,
            "memory_unit": "MiB", "resource_reservation": False,
        })
        self._patch("socket.gethostname", return_value="mock-host")
        self._patch("os.sched_getaffinity", return_value={0, 1, 2, 3})
        self._patch("os.sched_setaffinity")
        self._patch("enable_child_subreaper")
        self._patch("kernel_descendants", return_value={})
        self._patch("reap_adopted_zombies", return_value=[])
        self._patch("processes", return_value={os.getpid(): {
            "ppid": 0, "start_ticks": 1234, "state": "S"}})

    def _patch(self, name, **kwargs):
        pieces = name.split(".")
        owner = runner
        for piece in pieces[:-1]:
            owner = getattr(owner, piece)
        patcher = patch.object(owner, pieces[-1], **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def _execute(self):
        self.job_file.write_text(json.dumps(self.job))
        return runner.execute(self.job_file, self.lock)

    def _assert_lock_released(self):
        self.assertFalse(self.unresolved.exists())
        with self.lock.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_insufficient_free_memory_blocks_science_spawn_and_releases_lock(self):
        self.free_memory.return_value["memory_free_bytes"] = 21_999_999_999
        with self.assertRaisesRegex(RuntimeError, "启动前空闲显存不足"):
            self._execute()
        self.popen.assert_not_called()
        self.snapshot.assert_called_once_with(0, GPU_UUID)
        self.free_memory.assert_called_once_with(0, GPU_UUID)
        receipt = json.loads((self.receipt_dir / "receipt.private.json").read_text())
        self.assertEqual(receipt["minimum_free_gpu_bytes"], 22_000_000_000)
        self.assertEqual(receipt["gpu_memory_before_spawn"]["memory_free_bytes"],
                         21_999_999_999)
        self.assertFalse(receipt.get("science_spawn_attempted", False))
        self.assertEqual(receipt["status"], "dispatch_or_exit_audit_failed")
        self._assert_lock_released()

    def test_sufficient_memory_allows_shared_job_using_verified_uuid(self):
        self.assertEqual(self._execute(), 0)
        self.popen.assert_called_once()
        self.assertEqual(self.popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"],
                         GPU_UUID)
        self.assertEqual(self.snapshot.call_count, 1)
        self.free_memory.assert_called_once_with(0, GPU_UUID)
        receipt = json.loads((self.receipt_dir / "receipt.private.json").read_text())
        self.assertEqual(receipt["gpu_processes_before"], "999, external, 20000")
        self.assertEqual(receipt["exit_code"], 0)
        self.assertFalse(receipt["gpu_memory_before_spawn"]["resource_reservation"])
        self.assertEqual((self.receipt_dir / "EXIT").read_text(), "0\n")
        self._assert_lock_released()

    def test_invalid_threshold_fails_before_any_science_or_gpu_query(self):
        for value in (True, False, 0, -1, 22e9, "22000000000", None):
            with self.subTest(value=value):
                self.job["minimum_free_gpu_bytes"] = value
                with self.assertRaisesRegex(ValueError, "正整数字节数"):
                    self._execute()
        self.popen.assert_not_called()
        self.snapshot.assert_not_called()
        self.free_memory.assert_not_called()
        self.assertFalse(self.receipt_dir.exists())

    def test_strict_idle_gpu_still_rejects_existing_process(self):
        self.job["require_idle_gpu"] = True
        del self.job["minimum_free_gpu_bytes"]
        with self.assertRaisesRegex(RuntimeError, "目标卡已有计算进程"):
            self._execute()
        self.popen.assert_not_called()
        self.free_memory.assert_not_called()
        self._assert_lock_released()

    def test_legacy_shared_job_without_threshold_keeps_existing_behavior(self):
        del self.job["minimum_free_gpu_bytes"]
        self.assertEqual(self._execute(), 0)
        self.snapshot.assert_not_called()
        self.free_memory.assert_not_called()
        self.assertEqual(self.popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")


class FreeMemoryQueryTests(unittest.TestCase):
    def test_query_checks_target_identity_and_converts_mib_to_bytes(self):
        with patch.object(runner.subprocess, "run", return_value=Mock(
                stdout=f"1, {GPU_UUID}, 22000\n")) as query:
            result = runner.gpu_free_memory(1, GPU_UUID)
        self.assertEqual(result["memory_free_bytes"], 22_000 * 1048576)
        self.assertFalse(result["resource_reservation"])
        self.assertIn(f"--id={GPU_UUID}", query.call_args.args[0])
        self.assertGreaterEqual(result["query_completed_unix"], result["query_started_unix"])

    def test_unknown_memory_or_drifted_index_or_uuid_fails(self):
        for output in (f"0, {GPU_UUID}, N/A\n", f"1, {GPU_UUID}, 22000\n",
                       "0, GPU-other, 22000\n", "", f"0, {GPU_UUID}, 22.5\n"):
            with self.subTest(output=output), patch.object(
                    runner.subprocess, "run", return_value=Mock(stdout=output)):
                with self.assertRaises(RuntimeError):
                    runner.gpu_free_memory(0, GPU_UUID)


if __name__ == "__main__":
    unittest.main()
