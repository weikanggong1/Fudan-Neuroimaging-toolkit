"""CPU-local transport protocol checks; no official/MRI/GPU execution."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.connectome.test_raw_anatomy_prep_tool import PreparationFixture
from tools import benchmark_connectome_anatomy_prep as prep
from tools import benchmark_connectome_raw_cohort as cohort
from tools import benchmark_connectome_staged_gpu as staged


class LocalTransportTests(PreparationFixture):
    def test_frozen_staged_bundle_imports_in_an_isolated_directory(self):
        scripts = Path(staged.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for name in ("benchmark_connectome_raw_cohort.py","benchmark_connectome_raw_recovery.py",
                         "benchmark_connectome_raw_rerun.py","benchmark_connectome_staged_gpu.py"):
                shutil.copyfile(scripts / name,bundle / name)
            result = subprocess.run([sys.executable,str(bundle / "benchmark_connectome_staged_gpu.py"),"--help"],
                                    capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("--prep-bindings",result.stdout)

    def local(self):
        self.config["cpu_transport"] = "local"
        return self.config

    def test_unknown_transport_and_missing_or_wrong_declared_host_rejected(self):
        for change in ({"cpu_transport":"other"}, {"cpu_transport":"local","cpu_host":None},
                       {"cpu_transport":"local","cpu_host":"localhost"}, {"cpu_transport":"local","cpu_host":"gpucw1"}):
            config = {**self.config, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                prep.validate_config(config)

    def test_wrong_actual_host_cannot_launch_local_worker(self):
        with patch.object(cohort.socket,"gethostname",return_value="gpucw1"), patch.object(prep.subprocess,"run") as launch:
            with self.assertRaises(ValueError):
                prep.remote(self.local(),"recon",self.cases[0],self.root/"stderr.log")
            launch.assert_not_called()

    def test_worker_also_refuses_wrong_actual_host_before_execution(self):
        with patch.object(cohort.socket,"gethostname",return_value="headcw"), patch.object(cohort,"worker") as official:
            with self.assertRaises(ValueError):
                prep.worker({"config":self.local(),"action":"recon","case":self.cases[0],"version":"candidate"})
            official.assert_not_called()

    def test_correct_local_host_uses_subprocess_list_and_json_not_ssh(self):
        result = SimpleNamespace(returncode=0,stdout='{"status":"protocol_only"}',stderr="diagnostic stderr")
        with patch.object(cohort.socket,"gethostname",return_value="nodecw10.cluster"), patch.object(prep.subprocess,"run",return_value=result) as launch:
            reported = prep.remote(self.local(),"recon",self.cases[0],self.root/"stderr.log")
        self.assertEqual(reported["status"],"protocol_only")
        self.assertEqual(launch.call_args.args[0],[self.config["cpu_python"],self.config["anatomy_prep_script"],"_worker"])
        self.assertEqual(json.loads(launch.call_args.kwargs["input"])["case"],self.cases[0])
        self.assertNotIn("shell",launch.call_args.kwargs)
        self.assertEqual((self.root/"stderr.log").read_text(),"diagnostic stderr")

    def test_local_worker_failure_preserves_stderr(self):
        result = SimpleNamespace(returncode=5,stdout="",stderr="actual protocol failure")
        with patch.object(cohort.socket,"gethostname",return_value="nodecw10"), patch.object(prep.subprocess,"run",return_value=result):
            with self.assertRaises(RuntimeError):
                prep.remote(self.local(),"recon",self.cases[0],self.root/"stderr.log")
        self.assertEqual((self.root/"stderr.log").read_text(),"actual protocol failure")

    def test_passive_frozen_runtime_verification_does_not_execute_local_cpu(self):
        self.local()
        with patch.object(cohort.socket,"gethostname",return_value="gpucw1"), patch.object(prep.subprocess,"run") as launch:
            prep.verify_runtime_files(self.config)
            launch.assert_not_called()

    def test_ordinary_ssh_transport_uses_existing_cohort_remote(self):
        with patch.object(cohort,"remote",return_value={"status":"ssh_protocol_only"}) as launch:
            result = prep.remote(self.config,"recon",self.cases[0],self.root/"stderr.log")
        self.assertEqual(result["status"],"ssh_protocol_only")
        self.assertEqual(launch.call_args.args[1],"cpu")

    def test_local_duration_has_correct_coordinator_scope(self):
        result = prep.case_timing(10.,30.,5.,15.,cpu_local=True)
        self.assertEqual(result["preparation_coordinator_case_wall_seconds"],20.)
        self.assertNotIn("head_case_wall_seconds",result)
        ordinary = prep.case_timing(10.,30.,5.,15.)
        self.assertEqual(ordinary["head_case_wall_seconds"],20.)

    def test_parent_connection_record_is_hash_bound_and_never_rewritten(self):
        path = self.root/"head_connection_start.json"
        raw = b'{"scope":"actual head connection start; protocol fixture"}'
        path.write_bytes(raw)
        self.local()["parent_connection"] = {"path":str(path),"sha256":cohort.sha256(path)}
        prep.verify_runtime_files(self.config)
        self.assertEqual(path.read_bytes(),raw)
        path.write_bytes(raw+b"\n")
        with self.assertRaises(ValueError):
            prep.verify_runtime_files(self.config)

    def local_staged_timing_fixture(self):
        path = self.root / "actual_head_start.json"
        cohort.atomic_json(path,{"start_utc":"2026-10-03T00:00:00+00:00","coordinator_identity":{"hostname":"headcw"}})
        prepared = {"coordinator_transport":"local", "coordinator_identity":{"hostname":"nodecw10"},
            "parent_connection":{"path":str(path),"sha256":cohort.sha256(path)},
            "start_utc":"2026-10-03T00:00:10+00:00","end_utc":"2026-10-03T00:00:30+00:00",
            "preparation_report":{"recon_command_seconds":10.},"preparation_worker_wall_seconds":18.,
            "timing":{"preparation_coordinator_case_wall_seconds":20.,"cpu_driver_queue_seconds":1.}}
        gpu = {"gpu_lock_queue_seconds":5.,"raw_dwi_cli_total_runtime_seconds":10.}
        return path,prepared,gpu

    def test_local_staged_timer_never_uses_cross_host_clock_skew_as_wall(self):
        _,prepared,gpu = self.local_staged_timing_fixture()
        with patch.object(cohort.socket,"gethostname",return_value="headcw"):
            result = staged.staged_timing(prepared,gpu,"2026-10-03T00:00:20+00:00","2026-10-03T00:00:40+00:00",20.)
        self.assertEqual(result["cross_host_prep_to_gpu_utc_difference_seconds"],-10.)
        self.assertFalse(result["cross_host_utc_differences_are_wall_timers"])
        self.assertEqual(result["head_C_batch_start_to_case_gpu_end_observed_utc_seconds"],40.)
        self.assertNotIn("preparation_head_wall_seconds",result)
        self.assertNotIn("staged_observed_utc_elapsed_seconds",result)
        self.assertEqual(result["raw_dwi_head_wall_excluding_gpu_queue_seconds"],15.)

    def test_local_staged_timer_rejects_missing_changed_or_other_head_origin(self):
        path,prepared,gpu = self.local_staged_timing_fixture()
        with patch.object(cohort.socket,"gethostname",return_value="other-head"), self.assertRaises(ValueError):
            staged.staged_timing(prepared,gpu,"2026-10-03T00:00:20+00:00","2026-10-03T00:00:40+00:00",20.)
        path.write_bytes(path.read_bytes()+b"\n")
        with patch.object(cohort.socket,"gethostname",return_value="headcw"), self.assertRaises(ValueError):
            staged.staged_timing(prepared,gpu,"2026-10-03T00:00:20+00:00","2026-10-03T00:00:40+00:00",20.)
        prepared["parent_connection"] = None
        with self.assertRaises(ValueError):
            staged.staged_timing(prepared,gpu,"2026-10-03T00:00:20+00:00","2026-10-03T00:00:40+00:00",20.)


if __name__ == "__main__":
    unittest.main()
