"""退出审计的异 UID 归属门；这些进程夹具不代替真实 GPU 验证。"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[2] / "tools" / "run_recon_single_gpu_job.py"
SPEC = importlib.util.spec_from_file_location("recon_single_gpu_job_test", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def identity(pid, ppid, uid, start_ticks, comm="python"):
    return {"pid": pid, "ppid": ppid, "start_ticks": start_ticks,
            "uids": [uid] * 4, "namespace_pids": [pid],
            "pid_namespace": runner.INITIAL_PID_NAMESPACE,
            "pid_namespace_source": "direct_ns_symlink",
            "procfs_view_namespace": runner.INITIAL_PID_NAMESPACE,
            "comm": comm}


EXTERNAL_CHAIN = [identity(40255, 40254, 1997, 2000),
                  identity(40254, 1, 1997, 1999),
                  identity(1, 0, 0, 1)]


def audit(chain, *, second_chain=None, observed=None, adopted=None, lease=None,
          snapshot="40255, python, 100 MiB", second_snapshot=None):
    lease = lease if lease is not None else identity(10, 1, 2056, 1000)
    second_chain = chain if second_chain is None else second_chain
    second_snapshot = snapshot if second_snapshot is None else second_snapshot
    with patch.object(runner, "process_chain", return_value=None), \
         patch.object(runner, "process_identity", return_value=lease), \
         patch.object(runner, "local_procfs_namespace_anchor",
                      return_value=runner.INITIAL_PID_NAMESPACE), \
         patch.object(runner, "process_ownership_chain",
                      side_effect=[chain, second_chain]):
        return runner.audit_gpu_process_ownership(
            snapshot, second_snapshot,
            observed=observed or {}, adopted=adopted or {},
            lease_pid=10, lease_start_ticks=1000,
            child_pid=11, child_start_ticks=1001)


class OwnershipAuditTests(unittest.TestCase):
    def test_other_uid_with_complete_stable_kernel_chain_is_external(self):
        result = audit(EXTERNAL_CHAIN)
        self.assertEqual(result["unresolved"], "")
        self.assertEqual(result["external"], "40255, python, 100 MiB")
        self.assertEqual(result["audit"][0]["classification"], "external")
        self.assertEqual(result["audit"][0]["kernel_ancestry"], EXTERNAL_CHAIN)

    def test_redacted_target_namespace_is_unknown_but_host_procfs_view_is_checked(self):
        redacted = [dict(row, pid_namespace=None,
                         pid_namespace_source="permission_denied")
                    for row in EXTERNAL_CHAIN[:-1]] + [EXTERNAL_CHAIN[-1]]
        self.assertEqual(audit(redacted)["external"], "40255, python, 100 MiB")
        drifted = [dict(redacted[0], procfs_view_namespace="pid:[123]"), *redacted[1:]]
        self.assertTrue(audit(redacted, second_chain=drifted)["unresolved"])
        wrong_view = [dict(redacted[0], procfs_view_namespace="pid:[123]"), *redacted[1:]]
        self.assertTrue(audit(wrong_view)["unresolved"])

    def test_absolute_gpu_process_path_matches_kernel_comm(self):
        row = "40255, /a/private/venv/bin/python, 100 MiB"
        result = audit(EXTERNAL_CHAIN, snapshot=row)
        self.assertEqual(result["external"], row)
        self.assertEqual(result["unresolved"], "")

    def test_same_uid_missing_ancestry_or_owned_process_remains_unresolved(self):
        cases = [
            ([identity(40255, 40254, 2056, 2000),
              identity(40254, 1, 1997, 1999), identity(1, 0, 0, 1)], {}, {}),
            (None, {}, {}),
            (EXTERNAL_CHAIN, {40255: 2000}, {}),
            (EXTERNAL_CHAIN, {}, {40254: {"start_ticks": 1999}}),
            ([identity(40255, 11, 1997, 2000),
              identity(11, 1, 1997, 1001), identity(1, 0, 0, 1)], {}, {}),
        ]
        for chain, observed, adopted in cases:
            with self.subTest(chain=chain, observed=observed, adopted=adopted):
                result = audit(chain, observed=observed, adopted=adopted)
                self.assertEqual(result["external"], "")
                self.assertEqual(result["unresolved"], "40255, python, 100 MiB")

    def test_identity_change_or_unverifiable_lease_remains_unresolved(self):
        drifted = [dict(EXTERNAL_CHAIN[0], start_ticks=2001), *EXTERNAL_CHAIN[1:]]
        self.assertTrue(audit(EXTERNAL_CHAIN, second_chain=drifted)["unresolved"])
        self.assertTrue(audit(EXTERNAL_CHAIN, lease=identity(10, 1, 2056, 999))["unresolved"])
        wrong_lease_namespace = dict(identity(10, 1, 2056, 1000),
                                     pid_namespace="pid:[123]")
        self.assertTrue(audit(EXTERNAL_CHAIN, lease=wrong_lease_namespace)["unresolved"])

    def test_gpu_name_mismatch_is_not_exempted(self):
        chain = [dict(EXTERNAL_CHAIN[0], comm="other-program"), *EXTERNAL_CHAIN[1:]]
        result = audit(chain)
        self.assertEqual(result["external"], "")
        self.assertIn("40255, python, 100 MiB", result["unresolved"])

    def test_noninitial_pid_namespace_does_not_map_driver_pid(self):
        with patch.object(runner, "local_procfs_namespace_anchor",
                          return_value="pid:[4026532213]"):
            self.assertIsNone(runner.distinct_uid_external_chain(
                40255, observed={}, adopted={}, lease_pid=10,
                lease_start_ticks=1000, child_pid=11, child_start_ticks=1001))

    def test_noninitial_namespace_blocks_full_command_chain_too(self):
        full_chain = [dict(row, state="S", argv0="python", exe="/usr/bin/python")
                      for row in EXTERNAL_CHAIN]
        with patch.object(runner, "process_chain", return_value=full_chain), \
             patch.object(runner, "process_name_matches", return_value=True), \
             patch.object(runner, "local_procfs_namespace_anchor",
                          return_value="pid:[4026532213]"):
            result = runner.audit_gpu_process_ownership(
                "40255, python, 100 MiB", "40255, python, 100 MiB",
                observed={}, adopted={}, lease_pid=10, lease_start_ticks=1000,
                child_pid=11, child_start_ticks=1001)
        self.assertEqual(result["external"], "")
        self.assertTrue(result["unresolved"])

    def test_full_command_chain_containing_lease_is_never_external(self):
        own_chain = [dict(identity(40255, 10, 2056, 2000), state="S",
                          argv0="python", exe="/usr/bin/python"),
                     dict(identity(10, 1, 2056, 1000), state="S",
                          argv0="python", exe="/usr/bin/python"),
                     dict(identity(1, 0, 0, 1), state="S",
                          argv0="init", exe="/sbin/init")]
        with patch.object(runner, "process_chain", return_value=own_chain), \
             patch.object(runner, "process_name_matches", return_value=True), \
             patch.object(runner, "local_procfs_namespace_anchor",
                          return_value=runner.INITIAL_PID_NAMESPACE):
            result = runner.audit_gpu_process_ownership(
                "40255, python, 100 MiB", "40255, python, 100 MiB",
                observed={}, adopted={}, lease_pid=10, lease_start_ticks=1000,
                child_pid=40255, child_start_ticks=2000)
        self.assertEqual(result["external"], "")
        self.assertEqual(result["unresolved"], "40255, python, 100 MiB")

    def test_new_pid_in_second_gpu_snapshot_is_not_exempted(self):
        with patch.object(runner, "process_chain", return_value=None), \
             patch.object(runner, "process_identity", return_value=identity(10, 1, 2056, 1000)), \
             patch.object(runner, "process_ownership_chain", return_value=EXTERNAL_CHAIN):
            result = runner.audit_gpu_process_ownership(
                "", "40255, python, 100 MiB", observed={}, adopted={},
                lease_pid=10, lease_start_ticks=1000,
                child_pid=11, child_start_ticks=1001)
        self.assertEqual(result["external"], "")
        self.assertEqual(result["unresolved"], "40255, python, 100 MiB")

    def test_disappeared_pid_or_changed_gpu_name_is_not_exempted(self):
        for second_snapshot in ("", "40255, other-program, 100 MiB"):
            with self.subTest(second_snapshot=second_snapshot):
                result = audit(EXTERNAL_CHAIN, second_snapshot=second_snapshot)
                self.assertEqual(result["external"], "")
                self.assertIn("40255, python, 100 MiB", result["unresolved"])

    @unittest.skipUnless(sys.platform == "linux", "Linux /proc identity required")
    def test_kernel_identity_reads_current_process_without_command_access(self):
        status = Path("/proc/self/status").read_text()
        if f"NSpid:\t{os.getpid()}\n" not in status:
            self.skipTest("single visible PID namespace is required")
        with patch.object(Path, "read_bytes", side_effect=PermissionError):
            current = runner.process_ownership_identity(os.getpid())
        self.assertIsNotNone(current)
        self.assertEqual(current["pid"], os.getpid())
        self.assertEqual(current["uids"][:3], list(os.getresuid()))
        self.assertEqual(current["namespace_pids"], [os.getpid()])

    @unittest.skipUnless(sys.platform == "linux", "Linux /proc identity required")
    def test_old_kernel_without_nspid_and_cross_uid_namespace_permission(self):
        pid = os.getpid()
        original_read_text = Path.read_text
        original_readlink = os.readlink
        def read_text(path, *args, **kwargs):
            value = original_read_text(path, *args, **kwargs)
            if str(path) == f"/proc/{pid}/status":
                value = "\n".join(row for row in value.splitlines()
                                  if not row.startswith("NSpid:")) + "\n"
            return value
        def readlink(path, *args, **kwargs):
            if str(path) == f"/proc/{pid}/ns/pid":
                raise PermissionError("simulated cross-UID procfs restriction")
            return original_readlink(path, *args, **kwargs)
        with patch.object(Path, "read_text", read_text), \
             patch.object(runner.os, "readlink", readlink):
            current = runner.process_ownership_identity(pid)
        self.assertIsNotNone(current)
        self.assertEqual(current["pid_namespace_source"],
                         "permission_denied")
        self.assertIsNone(current["pid_namespace"])
        self.assertEqual(current["procfs_view_namespace"],
                         runner.local_procfs_namespace_anchor())
        self.assertEqual(current["uids"][:3], list(os.getresuid()))

    def test_unverified_procfs_anchor_rejects_identity(self):
        with patch.object(runner, "local_procfs_namespace_anchor", return_value=None):
            self.assertIsNone(runner.process_ownership_identity(os.getpid()))

    def test_undecodable_procfs_evidence_remains_unresolved(self):
        decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        with patch.object(Path, "read_text", side_effect=decode_error):
            self.assertIsNone(runner.local_procfs_namespace_anchor())
            self.assertIsNone(runner.process_ownership_identity(os.getpid()))


if __name__ == "__main__":
    unittest.main()
