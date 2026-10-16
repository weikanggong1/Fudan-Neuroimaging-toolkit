"""单卡共享验证调度：本作业租约、冻结哈希、GPU归属与退出回执。

调度器不实现任何科学算法。job JSON 的 argv 必须是已审查的完整命令。
shared_gpu_explicit=True 允许已有外部计算进程，但始终核验 GPU UUID，
记录进入时进程及每两秒总卡/本作业显存，退出时复核进程归属。
它不终止外部进程，也不声称共享环境时间可与独占运行直接配对。
CUDA 在 fresh exec 之前映射；等待租约的时间不计入验证墙钟。
"""

import argparse
import csv
import ctypes
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processes():
    rows = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = path.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            rows[int(path.name)] = {
                "ppid": int(stat[1]), "start_ticks": int(stat[19]), "state": stat[0]
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def descendants(parent, rows):
    found = set()
    pending = [parent]
    while pending:
        pid = pending.pop()
        children = {n for n, row in rows.items() if row["ppid"] == pid}
        new = children - found
        found.update(new)
        pending.extend(new)
    return found


def kernel_descendants(parent):
    """读取内核每个线程的children；权限失败保留阻断，退出竞态允许重查。"""
    found = {}
    pending = [parent]
    while pending:
        pid = pending.pop()
        try:
            tasks = list(Path(f"/proc/{pid}/task").iterdir())
        except FileNotFoundError:
            continue
        children = set()
        for task in tasks:
            try:
                children.update(int(x) for x in task.joinpath("children").read_text().split())
            except FileNotFoundError:
                continue  # 线程已经退出；没有把权限或解析失败当作空树。
        for child in children:
            if child in found:
                continue
            try:
                stat = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                continue
            found[child] = {"ppid": int(stat[1]), "start_ticks": int(stat[19]), "state": stat[0]}
            pending.append(child)
    return found


def enable_child_subreaper():
    """Linux内核收养脱离父进程的后代；不依赖采样发现完整进程树。"""
    if kernel_descendants(os.getpid()):
        raise RuntimeError("租约调度器已有子进程，不能声明独立作业树")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "无法启用子进程收养；不启动科学作业")
    enabled = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
        raise RuntimeError("无法确认PR_GET_CHILD_SUBREAPER；不启动科学作业")


def child_subreaper_enabled():
    enabled = ctypes.c_int()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "无法复核PR_GET_CHILD_SUBREAPER")
    return enabled.value == 1


def reap_adopted_zombies():
    """直接科学子进程已经wait后，回收内核收养的已退出后代。"""
    reaped = []
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        reaped.append({"pid": pid, "wait_status": status})
    return reaped


def physical_gpu_index(job):
    """将job中的单一物理nvidia-smi索引解析为整数；拒绝列表和非规范写法。"""
    value = job.get("cuda_visible_devices")
    if (not isinstance(value, str) or not value.isascii() or not value.isdecimal()
            or str(int(value)) != value):
        raise ValueError("cuda_visible_devices必须是单一非负物理GPU索引字符串")
    return int(value)


def gpu_snapshot(index, expected_uuid):
    """核对物理索引/完整UUID，仅返回目标GPU的实时计算进程快照。

    nvidia-smi任何查询失败、设备身份漂移或进程行无法归属时抛异常；
    其他物理GPU上可识别的计算进程不占用本作业的目标卡。
    """
    inventory_output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15, check=True
    ).stdout
    inventory = {}
    uuids = set()
    for row in csv.reader(io.StringIO(inventory_output), skipinitialspace=True):
        if (len(row) != 2 or not row[0].strip().isascii()
                or not row[0].strip().isdecimal()):
            raise RuntimeError("GPU设备清单格式无法核验")
        ordinal, uuid = int(row[0].strip()), row[1].strip()
        if (str(ordinal) != row[0].strip() or ordinal in inventory or uuid in uuids
                or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                                uuid) is None):
            raise RuntimeError("GPU索引或UUID不唯一/不完整")
        inventory[ordinal] = uuid
        uuids.add(uuid)
    if inventory.get(index) != expected_uuid:
        raise RuntimeError("目标物理GPU索引/UUID与冻结作业不符")
    apps_output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True
    ).stdout
    selected = []
    other_count = 0
    for row in csv.reader(io.StringIO(apps_output), skipinitialspace=True):
        if len(row) != 4 or not row[1].strip().isdecimal():
            raise RuntimeError("GPU计算进程清单无法核验")
        uuid = row[0].strip()
        if uuid not in uuids:
            raise RuntimeError("GPU计算进程的设备UUID不在已核验清单中")
        if uuid == expected_uuid:
            selected.append(", ".join(cell.strip() for cell in row[1:]))
        else:
            other_count += 1
    return inventory, "\n".join(selected), other_count


def gpu_process_entries(snapshot):
    """解析已按目标 GPU UUID 筛选的计算进程；保留原始行供私有审计。"""
    entries = []
    for line in snapshot.splitlines():
        pid_text, separator, remainder = line.partition(", ")
        name, last_separator, memory = remainder.rpartition(", ")
        if (not separator or not last_separator or not pid_text.isascii()
                or not pid_text.isdecimal() or str(int(pid_text)) != pid_text
                or not name or not memory):
            raise RuntimeError("目标GPU计算进程PID/名称无法核验")
        entries.append({"pid": int(pid_text), "process_name": name, "raw": line})
    if len({entry["pid"] for entry in entries}) != len(entries):
        raise RuntimeError("目标GPU计算进程PID重复")
    return entries


def shared_memory_sample(*, gpu_index, expected_uuid, lease_pid, child_pid,
                         rows):
    """一次同时采样整卡与计算进程显存；不能归属时返回空的自身峰值。"""
    inventory, snapshot, _ = gpu_snapshot(gpu_index, expected_uuid)
    card = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=15, check=True).stdout
    card_rows = list(csv.reader(io.StringIO(card), skipinitialspace=True))
    selected = [row for row in card_rows if len(row) == 3
                and row[0].strip() == expected_uuid]
    if len(selected) != 1 or any(not value.strip().isdecimal()
                                  for value in selected[0][1:]):
        raise RuntimeError("目标GPU整卡显存无法核验")
    owned_pids = descendants(lease_pid, rows) | {child_pid}
    own = 0
    unknown = []
    external = []
    entries = []
    for entry in gpu_process_entries(snapshot):
        memory_text = entry["raw"].rpartition(", ")[2]
        memory = int(memory_text) if memory_text.isdecimal() else None
        identity = rows.get(entry["pid"])
        owned = (entry["pid"] in owned_pids and identity is not None
                 and identity["state"] != "Z")
        classification = ("owned" if owned else "external"
                          if identity is not None else "unattributed")
        if owned and memory is not None:
            own += memory
        elif owned or identity is None:
            unknown.append(entry["pid"])
        if classification == "external":
            external.append(entry["pid"])
        entries.append({"pid": entry["pid"], "class": classification,
                        "used_mib": memory, "process": entry["process_name"]})
    return {"unix": time.time(), "gpu_uuid": expected_uuid,
            "physical_gpu_index": gpu_index,
            "card_used_mib": int(selected[0][1].strip()),
            "card_total_mib": int(selected[0][2].strip()),
            "owned_compute_mib": own if not unknown else None,
            "owned_compute_mib_lower_bound": own,
            "unattributed_pids": unknown, "external_pids": external,
            "processes": entries}


def process_identity(pid):
    """读取 PID、启动时钟、UID、可执行文件与命名空间；证据不足返回 None。

    旧内核的 status 可能没有 NSpid。此时仅在 stat/status 的 PID 与父 PID
    一致、exe 可读、且目标与本进程的 PID 命名空间相同时推得单一命名空间 PID。
    完整父链仍由 process_chain 再核验，无法读取他人 /proc 时保持未决。
    """
    path = Path("/proc") / str(pid)
    try:
        stat = path.joinpath("stat").read_text()
        if not stat.startswith(f"{pid} ("):
            return None
        fields = stat.rsplit(")", 1)[1].split()
        status = path.joinpath("status").read_text().splitlines()
        uid_rows = [row.split()[1:] for row in status if row.startswith("Uid:")]
        namespace_rows = [row.split()[1:] for row in status if row.startswith("NSpid:")]
        if len(uid_rows) != 1 or len(uid_rows[0]) != 4 or len(namespace_rows) > 1:
            return None
        parent_pid = int(fields[1])
        start_ticks = int(fields[19])
        if parent_pid < 0 or start_ticks <= 0:
            return None
        namespace = os.readlink(path / "ns" / "pid")
        command = path.joinpath("cmdline").read_bytes().split(b"\0", 1)[0]
        argv0 = os.fsdecode(command) if command else None
        try:
            exe = os.readlink(path / "exe")
        except OSError:
            exe = None
        if namespace_rows:
            if not namespace_rows[0]:
                return None
            namespace_pids = [int(value) for value in namespace_rows[0]]
            namespace_pid_source = "status_nspid"
        else:
            pid_rows = [row.split()[1:] for row in status if row.startswith("Pid:")]
            parent_rows = [row.split()[1:] for row in status if row.startswith("PPid:")]
            if (len(pid_rows) != 1 or pid_rows[0] != [str(pid)]
                    or len(parent_rows) != 1 or parent_rows[0] != [str(parent_pid)]
                    or namespace != os.readlink("/proc/self/ns/pid")
                    or not argv0 or not exe):
                return None
            namespace_pids = [pid]
            namespace_pid_source = "stat_status_local_namespace"
        identity = {
            "pid": pid, "ppid": parent_pid, "state": fields[0],
            "start_ticks": start_ticks,
            "uids": [int(value) for value in uid_rows[0]],
            "namespace_pids": namespace_pids,
            "namespace_pid_source": namespace_pid_source,
            "pid_namespace": namespace,
            "argv0": argv0, "exe": exe,
        }
        return identity
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def process_chain(pid):
    """一直读取到 PID 1；缺失、循环或混合 PID 命名空间均不能证明外部归属。"""
    chain = []
    seen = set()
    try:
        local_namespace = os.readlink("/proc/self/ns/pid")
    except OSError:
        return None
    if re.fullmatch(r"pid:\[[0-9]+\]", local_namespace) is None:
        return None
    while pid != 0 and pid not in seen:
        seen.add(pid)
        identity = process_identity(pid)
        if (identity is None or identity["start_ticks"] <= 0
                or identity["pid_namespace"] != local_namespace
                or identity["namespace_pids"] != [pid]):
            return None
        chain.append(identity)
        if pid == 1:
            return chain if identity["ppid"] == 0 else None
        pid = identity["ppid"]
    return None


def process_name_matches(entry, identity):
    """GPU 进程名称至少要和 /proc 的 argv0 或 exe 路径精确相符。"""
    name = entry["process_name"]
    return any(name == value or os.path.realpath(name) == os.path.realpath(value)
               for value in (identity["argv0"], identity["exe"]) if value)


def audit_gpu_process_ownership(snapshot, second_snapshot, *, observed, adopted,
                                lease_pid, lease_start_ticks, child_pid,
                                child_start_ticks, preexisting=None):
    """只豁免两次 GPU 快照间身份、UID、完整父链均稳定的外部进程。

    已观测/收养的同启动时钟后代一律保留阻断。其他 PID 必须可映射到
    同一 /proc 命名空间，且父链稳定到 PID 1。subreaper 能收养作业产生的
    孤儿；父链证据不足时不推断它来自别的任务。
    """
    before = gpu_process_entries(snapshot)
    after = gpu_process_entries(second_snapshot)
    after_by_pid = {entry["pid"]: entry for entry in after}
    audit = []
    unresolved = []
    external = []
    preexisting = preexisting or {}
    for entry in before:
        pid = entry["pid"]
        evidence = {"pid": pid, "gpu_process_name": entry["process_name"],
                    "raw": entry["raw"], "classification": "unresolved"}
        first_chain = process_chain(pid)
        second_entry = after_by_pid.get(pid)
        second_chain = process_chain(pid) if second_entry is not None else None
        evidence["ancestry_first"] = first_chain
        evidence["ancestry_second"] = second_chain
        prior = preexisting.get(pid)
        current = processes().get(pid)
        if (prior is not None and second_entry is not None and current is not None
                and current["start_ticks"] == prior["start_ticks"]
                and entry["process_name"] == prior["process_name"]
                and second_entry["process_name"] == prior["process_name"]
                and observed.get(pid) != current["start_ticks"]):
            evidence["classification"] = "external"
            evidence["reason"] = "运行前已在目标卡，双快照及PID启动时钟一致"
            external.append(entry["raw"])
        elif first_chain is None or second_chain is None:
            evidence["reason"] = "PID/UID/父链或命名空间不可核验"
        elif (first_chain[0]["state"] == "Z" or second_chain[0]["state"] == "Z"
              or second_entry["process_name"] != entry["process_name"]
              or [row["pid"] for row in first_chain] != [row["pid"] for row in second_chain]
              or any((a["start_ticks"], a["ppid"], a["uids"], a["namespace_pids"],
                      a.get("namespace_pid_source"), a["pid_namespace"])
                     != (b["start_ticks"], b["ppid"], b["uids"], b["namespace_pids"],
                         b.get("namespace_pid_source"), b["pid_namespace"])
                     for a, b in zip(first_chain, second_chain))
              or (first_chain[0]["argv0"], first_chain[0]["exe"])
                 != (second_chain[0]["argv0"], second_chain[0]["exe"])):
            evidence["reason"] = "GPU进程或父链身份在双重快照间变化"
        elif not process_name_matches(entry, first_chain[0]):
            evidence["reason"] = "GPU进程名称与/proc可执行路径不符"
        else:
            identity = first_chain[0]
            owned = (observed.get(pid) == identity["start_ticks"]
                     or (pid == child_pid and
                         (child_start_ticks is None or child_start_ticks == identity["start_ticks"]))
                     or any(row["pid"] == lease_pid and
                            row["start_ticks"] == lease_start_ticks for row in first_chain)
                     or any(row["pid"] in adopted and
                            adopted[row["pid"]]["start_ticks"] == row["start_ticks"]
                            for row in first_chain))
            if owned:
                evidence["reason"] = "作业自有PID或subreaper后代仍存活"
            else:
                evidence["classification"] = "external"
                evidence["reason"] = "稳定且不属于作业树的完整父链"
                external.append(entry["raw"])
        if evidence["classification"] == "unresolved":
            unresolved.append(entry["raw"])
        audit.append(evidence)
    # 第二次快照中新出现的 PID 未经两次身份核验，保持阻断。
    for entry in after:
        if entry["pid"] not in {row["pid"] for row in before}:
            unresolved.append(entry["raw"])
            audit.append({"pid": entry["pid"], "raw": entry["raw"],
                          "classification": "unresolved",
                          "reason": "第二次GPU快照新出现的PID"})
    return {"external": "\n".join(external), "unresolved": "\n".join(unresolved),
            "audit": audit}


def execute(job_file, lock_path):
    job_file = Path(job_file).resolve()
    job = json.loads(job_file.read_text())
    shared_gpu = job.get("shared_gpu_explicit") is True
    if shared_gpu and job.get("require_idle_gpu") is not False:
        raise ValueError("显式共享GPU作业必须单独声明require_idle_gpu=False")
    audit_target_gpu = job.get("require_idle_gpu", True) or shared_gpu
    if job.get("threads") != 4:
        raise ValueError("单卡队列总线程必须为4")
    gpu_index = physical_gpu_index(job)
    cpus = job["cpu_affinity"]
    if len(set(cpus)) != 4 or not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("CPU 亲和性必须是目标节点允许的四个不同核心")
    if job["expected_host"] != socket.gethostname():
        raise ValueError("当前主机与已核验目标节点不同")
    if not isinstance(job["argv"], list) or not all(isinstance(x, str) for x in job["argv"]):
        raise ValueError("argv 必须是字符串数组；不执行 shell 拼接命令")
    lock_path = Path(lock_path)
    unresolved = lock_path.with_suffix(lock_path.suffix + ".unresolved.json")
    receipt_dir = Path(job["receipt_dir"])
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if unresolved.exists():
            raise RuntimeError(f"前一租约未核实退出，保留阻断回执：{unresolved}")
        receipt_dir.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        report = {
            "run_id": job["run_id"], "job_sha256": sha256(job_file),
            "status": "validating", "host": socket.gethostname(),
            "pid": os.getpid(), "start_ticks": processes()[os.getpid()]["start_ticks"],
            "started_unix": time.time(), "cpu_affinity": cpus, "threads": 4,
            "requested_physical_gpu_index": gpu_index,
            "cuda_visible_devices": job["cuda_visible_devices"], "lock": str(lock_path),
            "cross_host_speed_comparison": "prohibited", "verified_files": [],
            "shared_gpu_explicit": shared_gpu,
            "shared_environment_speed_claim": "unpaired" if shared_gpu else "idle_gpu",
            "memory_sample_interval_seconds": 2.0,
            "memory_sample_count": 0,
            "max_card_used_mib": None,
            "max_owned_compute_mib": None,
            "max_owned_memory_bytes": None,
        }
        save(unresolved, report)
        try:
            for expected in job.get("expected_files", []):
                actual = sha256(expected["path"])
                report["verified_files"].append({**expected, "actual": actual})
                if actual != expected["sha256"]:
                    raise ValueError(f"冻结输入或源码哈希不符：{expected['path']}")
            for output in job.get("new_output_paths", []):
                if Path(output).exists():
                    raise FileExistsError(f"拒绝覆盖旧输出：{output}")
            preexisting_gpu = {}
            if audit_target_gpu:
                inventory, snapshot, other_count = gpu_snapshot(
                    gpu_index, job["expected_gpu_uuid"])
                report["actual_gpu_inventory"] = inventory
                report["actual_gpu_uuids"] = [inventory[key] for key in sorted(inventory)]
                report["selected_gpu_uuid"] = job["expected_gpu_uuid"]
                report["other_gpu_process_count_before"] = other_count
                report["gpu_processes_before"] = snapshot
                snapshot_rows = processes()
                for entry in gpu_process_entries(snapshot):
                    identity = snapshot_rows.get(entry["pid"])
                    if identity is None:
                        raise RuntimeError("共享GPU现有进程PID启动时钟无法核验")
                    preexisting_gpu[entry["pid"]] = {
                        "start_ticks": identity["start_ticks"],
                        "process_name": entry["process_name"]}
                report["preexisting_gpu_process_identities"] = preexisting_gpu
                if snapshot and not shared_gpu:
                    raise RuntimeError("目标卡已有计算进程；保留回执，不抢占或终止")
            env = os.environ.copy()
            env.update(job.get("env", {}))
            for name in job.get("unset_env", []):
                env.pop(name, None)
            # 完整UUID避免物理索引与CUDA运行时枚举顺序不同；进程内仍是cuda:0。
            env["CUDA_VISIBLE_DEVICES"] = (
                job["expected_gpu_uuid"] if audit_target_gpu
                else job["cuda_visible_devices"])
            report["cuda_visible_devices"] = env["CUDA_VISIBLE_DEVICES"]
            report["logical_cuda_device"] = "cuda:0"
            for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]:
                value = job.get("env", {}).get(name, "4")
                if not 1 <= int(value) <= 4:
                    raise ValueError("库线程预算必须在1到4之间：" + name)
                env[name] = value
            os.sched_setaffinity(0, cpus)
            report["preparation_seconds"] = time.monotonic() - started
            enable_child_subreaper()
            report["descendant_audit_method"] = "Linux child subreaper plus kernel-adopted descendants"
            observed = {}
            with (receipt_dir / "stdout.private.log").open("wb") as log, (
                    receipt_dir / "gpu_memory_samples.private.jsonl").open("w") as samples:
                report["science_spawn_attempted"] = True
                save(unresolved, report)
                child_started = time.monotonic()
                child = subprocess.Popen(job["argv"], cwd=job["cwd"], env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock.fileno(),))
                child_identity = processes().get(child.pid)
                report.update(status="running", child_pid=child.pid,
                              child_start_ticks=(child_identity["start_ticks"]
                                                 if child_identity is not None else None))
                save(unresolved, report)
                save(receipt_dir / "receipt.private.json", report)
                next_sample = time.monotonic()
                while child.poll() is None:
                    rows = processes()
                    for pid in descendants(os.getpid(), rows):
                        observed[pid] = rows[pid]["start_ticks"]
                    if audit_target_gpu and time.monotonic() >= next_sample:
                        try:
                            sample = shared_memory_sample(
                                gpu_index=gpu_index,
                                expected_uuid=job["expected_gpu_uuid"],
                                lease_pid=os.getpid(), child_pid=child.pid,
                                rows=rows)
                            samples.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            samples.flush()
                            report["memory_sample_count"] += 1
                            card_mib = sample["card_used_mib"]
                            report["max_card_used_mib"] = max(
                                report["max_card_used_mib"] or 0, card_mib)
                            own_mib = sample["owned_compute_mib"]
                            if own_mib is not None:
                                report["max_owned_compute_mib"] = max(
                                    report["max_owned_compute_mib"] or 0, own_mib)
                                report["max_owned_memory_bytes"] = (
                                    report["max_owned_compute_mib"] * 1048576)
                        except Exception as exc:
                            report["memory_sampling_error"] = repr(exc)
                        next_sample = time.monotonic() + 2.0
                    time.sleep(0.25)
                code = child.wait()
            report.update(exit_code=code, command_wall_seconds=time.monotonic() - child_started,
                          status="command_exited" if code == 0 else "command_failed")
            audit_started = time.monotonic()
            report["reaped_adopted_descendants"] = reap_adopted_zombies()
            rows = processes()
            adopted = kernel_descendants(os.getpid())
            live = {pid: row for pid, row in rows.items()
                    if observed.get(pid) == row["start_ticks"] and row["state"] != "Z"}
            live.update({pid: row for pid, row in adopted.items() if row["state"] != "Z"})
            report["kernel_adopted_descendants_after_exit"] = sorted(adopted)
            report["live_observed_descendants_after_exit"] = live
            if audit_target_gpu:
                inventory, snapshot, other_count = gpu_snapshot(
                    gpu_index, job["expected_gpu_uuid"])
                report["actual_gpu_inventory_after"] = inventory
                report["other_gpu_process_count_after"] = other_count
                report["gpu_processes_after_raw"] = snapshot
            report["reaped_adopted_descendants"].extend(reap_adopted_zombies())
            final_adopted = kernel_descendants(os.getpid())
            # 收养审计之后仍可能发生父子退出；再核验活后代，不仅依赖历史采样。
            live.update({pid: row for pid, row in final_adopted.items()
                         if row["state"] != "Z"})
            report["live_observed_descendants_after_exit"] = live
            for _ in range(5):
                if not final_adopted or any(row["state"] != "Z" for row in final_adopted.values()):
                    break
                report["reaped_adopted_descendants"].extend(reap_adopted_zombies())
                time.sleep(0.01)
                final_adopted = kernel_descendants(os.getpid())
            report["kernel_adopted_descendants_final"] = sorted(final_adopted)
            if audit_target_gpu:
                report["child_subreaper_enabled_at_exit"] = child_subreaper_enabled()
                if not report["child_subreaper_enabled_at_exit"]:
                    raise RuntimeError("退出时subreaper状态丢失，GPU进程归属无法核验")
                if snapshot:
                    second_inventory, second_snapshot, second_other_count = gpu_snapshot(
                        gpu_index, job["expected_gpu_uuid"])
                    report["actual_gpu_inventory_after_second"] = second_inventory
                    report["other_gpu_process_count_after_second"] = second_other_count
                    report["gpu_processes_after_second_raw"] = second_snapshot
                    ownership = audit_gpu_process_ownership(
                        snapshot, second_snapshot, observed=observed,
                        adopted=final_adopted, lease_pid=os.getpid(),
                        lease_start_ticks=report["start_ticks"],
                        child_pid=child.pid,
                        child_start_ticks=report["child_start_ticks"],
                        preexisting=preexisting_gpu)
                    report["external_gpu_processes_after"] = ownership["external"]
                    report["gpu_processes_after"] = ownership["unresolved"]
                    report["gpu_process_ownership_audit"] = ownership["audit"]
                else:
                    report["external_gpu_processes_after"] = ""
                    report["gpu_processes_after"] = ""
                    report["gpu_process_ownership_audit"] = []
                final_adopted = kernel_descendants(os.getpid())
                report["kernel_adopted_descendants_final"] = sorted(final_adopted)
            report["exit_audit_seconds"] = time.monotonic() - audit_started
            report["inclusive_wall_seconds"] = time.monotonic() - started
            save(receipt_dir / "receipt.private.json", report)
            if live or final_adopted or report.get("gpu_processes_after"):
                save(unresolved, report)
                raise RuntimeError("父命令已退出但子进程或GPU计算仍存活；租约保持未解决")
            unresolved.unlink()
            (receipt_dir / "EXIT").write_text(str(code) + "\n")
            return code
        except BaseException as exc:
            report.update(status="dispatch_or_exit_audit_failed", error=repr(exc),
                          inclusive_wall_seconds=time.monotonic() - started)
            save(receipt_dir / "receipt.private.json", report)
            # 无科学子进程的前置校验失败可安全释放，运行中断必须人工审计。
            if not report.get("science_spawn_attempted", False):
                unresolved.unlink()
            else:
                save(unresolved, report)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--lock", required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.job, args.lock))
