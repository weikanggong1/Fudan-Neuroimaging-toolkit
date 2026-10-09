"""Linux x86-64只跟踪新启动的具名参考进程，捕获致命信号/PC/库映射。"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import struct
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not command or args.output.exists() or args.log.exists():
        raise ValueError("仅支持Linux x86-64、新报告/日志路径与非空命令")
    args.working_directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    library = ctypes.CDLL(None, use_errno=True)
    library.ptrace.restype = ctypes.c_long
    library.ptrace.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]

    def ptrace(request, pid, address=0, data=0):
        ctypes.set_errno(0)
        result = library.ptrace(request, pid, ctypes.c_void_p(address), ctypes.c_void_p(data))
        if result == -1 and ctypes.get_errno():
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        return result

    report = {"scope": "own_fresh_native_reference_fatal_signal_diagnostic_not_performance_gate",
              "hostname": platform.node(), "architecture": platform.machine(), "command": command,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "native_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
              "thread_environment": {key: os.environ.get(key) for key in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
              "cpu_affinity_count": len(os.sched_getaffinity(0)), "fatal_signals": [],
              "backtrace_scope": "fault PC only; no raw MRI/license/core memory exported",
              "native_core_dump_limit_bytes": 0,
              "memory_samples": [], "memory_scope": "whole cgroup plus own native RSS; other jobs may contribute"}
    inputs = {flag: Path(command[index + 1]) for index, flag in enumerate(command[:-1])
              if flag in ("--adgws-in", "--wm", "--invol", "--i", "--seg")}
    report["input_sha256_before"] = {flag: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for flag, path in inputs.items()}
    started = time.perf_counter()
    log_stream = args.log.open("x")
    faulthandler.enable(file=log_stream, all_threads=True)
    child = os.fork()
    if child == 0:
        # 不产生可能含影像/许可证内存的完整core，信号和故障PC仍原样捕获。
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.chdir(args.working_directory)
        descriptor = log_stream.fileno()
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        if descriptor > 2:
            os.close(descriptor)
        ptrace(0, 0)  # PTRACE_TRACEME: this new child only.
        os.execv(command[0], command)
        os._exit(127)
    report["native_pid"] = child
    stopped = threading.Event()

    def sample_memory():
        while not stopped.is_set():
            sample = {"seconds": time.perf_counter() - started}
            for name in ("usage_in_bytes", "max_usage_in_bytes", "limit_in_bytes", "failcnt"):
                path = Path("/sys/fs/cgroup/memory") / f"memory.{name}"
                try:
                    sample[name] = int(path.read_text().strip())
                except (FileNotFoundError, PermissionError, ValueError):
                    sample[name] = None
            try:
                for line in Path(f"/proc/{child}/status").read_text().splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                        sample[line.split(":", 1)[0] + "_bytes"] = int(line.split()[1]) * 1024
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
            report["memory_samples"].append(sample)
            stopped.wait(0.25)

    memory_thread = threading.Thread(target=sample_memory, daemon=True)
    memory_thread.start()
    args.output.write_text(json.dumps(report, indent=2))
    active, initial = {child}, True
    exit_code = None
    while active:
        pid, status = os.waitpid(-1, 0x40000000)  # __WALL includes traced pthreads.
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            active.discard(pid)
            if pid == child:
                exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
            continue
        if not os.WIFSTOPPED(status):
            continue
        stopped_signal, event = os.WSTOPSIG(status), status >> 16
        if initial and pid == child:
            ptrace(0x4200, pid, 0, 8)  # PTRACE_O_TRACECLONE; no syscall-by-syscall tracing.
            initial = False
        if event == 3:  # PTRACE_EVENT_CLONE.
            new_pid = ctypes.c_ulong()
            ptrace(0x4201, pid, 0, ctypes.addressof(new_pid))
            active.add(int(new_pid.value))
        if stopped_signal in (signal.SIGBUS, signal.SIGSEGV, signal.SIGILL, signal.SIGFPE, signal.SIGABRT):
            registers = (ctypes.c_ulonglong * 27)()
            information = (ctypes.c_ubyte * 128)()
            ptrace(12, pid, 0, ctypes.addressof(registers))
            ptrace(0x4202, pid, 0, ctypes.addressof(information))
            signo, error, code = struct.unpack_from("iii", bytes(information))
            fault_address = struct.unpack_from("Q", bytes(information), 16)[0] if code > 0 else None
            sender = struct.unpack_from("ii", bytes(information), 16) if code <= 0 else (None, None)
            pc = int(registers[16])
            maps = Path(f"/proc/{pid}/maps").read_text()
            containing = []
            for line in maps.splitlines():
                fields = line.split(maxsplit=5)
                low, high = (int(value, 16) for value in fields[0].split("-"))
                if low <= pc < high or fault_address is not None and low <= fault_address < high:
                    containing.append(line)
            resolved = []
            for line in containing:
                fields = line.split(maxsplit=5)
                low, high = (int(value, 16) for value in fields[0].split("-"))
                if low <= pc < high and len(fields) == 6 and Path(fields[5]).is_file():
                    object_path = fields[5]
                    elf = Path(object_path).open("rb")
                    header = elf.read(24)
                    elf.close()
                    object_type = struct.unpack_from("H", header, 16)[0]
                    object_pc = pc if object_type == 2 else pc - low + int(fields[2], 16)
                    symbol = subprocess.run(["addr2line", "-f", "-C", "-i", "-e", object_path, hex(object_pc)],
                                            capture_output=True, text=True)
                    resolved.append({"object": object_path, "object_address": hex(object_pc),
                                     "symbol": symbol.stdout, "symbol_error": symbol.stderr})
            report["fatal_signals"].append({"thread_pid": pid, "signal": signo,
                "signal_name": signal.Signals(signo).name, "si_code": code, "si_errno": error,
                "signal_sender_pid": sender[0], "signal_sender_uid": sender[1],
                "fault_address": hex(fault_address) if fault_address is not None else None,
                "instruction_pointer": hex(pc),
                "containing_mappings": containing, "resolved_instruction": resolved,
                "loaded_library_mappings": maps.splitlines()})
            args.output.write_text(json.dumps(report, indent=2))
        # Preserve the native fatal signal/exit; don't hide it or kill other tasks.
        deliver = 0 if stopped_signal in (signal.SIGTRAP, signal.SIGSTOP) else stopped_signal
        ptrace(7, pid, 0, deliver)
    stopped.set()
    memory_thread.join(timeout=2)
    samples = report["memory_samples"]
    valid = [sample["failcnt"] for sample in samples if sample["failcnt"] is not None]
    report["whole_cgroup_failcnt_delta_during_reference"] = valid[-1] - valid[0] if valid else None
    report.update(returncode=exit_code, traced_wall_seconds=time.perf_counter() - started,
                  wrapper_completed=True, native_execution_started=not initial)
    report["input_sha256_after"] = {flag: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for flag, path in inputs.items()}
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_suffix(".exit.json").write_text(json.dumps({"native_returncode": exit_code,
                                                               "wrapper_exit_sentinel": True}))
    raise SystemExit(0 if exit_code == 0 else 1)


if __name__ == "__main__":
    main()
