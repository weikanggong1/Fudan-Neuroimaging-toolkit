"""单卡验证调度：标准库、独占租约、冻结文件哈希和退出回执。

调度器不实现任何科学算法。job JSON 的 argv 必须是已审查的完整命令。
CUDA 在 fresh exec 之前映射；等待租约的时间不计入验证墙钟。
共享卡作业可设 minimum_free_gpu_bytes（正整数字节数），在启动前核对
目标 UUID 和空闲显存；这次查询不预留资源，启动后仍可能有其他作业进入。
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


# include/uapi/linux/nsfs.h: PID_NS_INIT_INO = 0xEFFFFFFC.
INITIAL_PID_NAMESPACE = f"pid:[{0xEFFFFFFC}]"


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


def minimum_free_gpu_bytes(job):
    """读取可选的启动显存门槛（字节）；不接受 bool、零或浮点数。"""
    if "minimum_free_gpu_bytes" not in job:
        return None
    value = job["minimum_free_gpu_bytes"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("minimum_free_gpu_bytes必须是正整数字节数")
    return value


def gpu_free_memory(index, expected_uuid):
    """查询目标物理卡当前空闲字节和 UUID；失败或身份不符时禁止启动。

    nvidia-smi 的 memory.free 以整数 MiB 返回，按 1,048,576 换算字节；
    报告保留原始值与查询时间。此快照不构成显存预留。
    """
    query_started_unix = time.time()
    output = subprocess.run(
        ["nvidia-smi", f"--id={expected_uuid}",
         "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True
    ).stdout
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError("目标GPU空闲显存查询格式无法核验")
    ordinal, uuid, free_mib = (cell.strip() for cell in rows[0])
    if (ordinal != str(index) or uuid != expected_uuid or not free_mib.isascii()
            or not free_mib.isdecimal() or str(int(free_mib)) != free_mib):
        raise RuntimeError("目标GPU空闲显存或物理索引/UUID无法核验")
    return {"physical_gpu_index": index, "gpu_uuid": uuid,
            "memory_free_mib": int(free_mib),
            "memory_free_bytes": int(free_mib) * 1048576,
            "query_started_unix": query_started_unix,
            "query_completed_unix": time.time(),
            "memory_unit": "MiB", "resource_reservation": False}


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


def local_procfs_namespace_anchor():
    """证明 /proc 的 PID 编号与本进程所在命名空间一致，供旧内核权限受限场景使用。"""
    try:
        own_pid = os.getpid()
        own_stat = Path("/proc/self/stat").read_text()
        own_status = Path("/proc/self/status").read_text().splitlines()
        init_stat = Path("/proc/1/stat").read_text()
        init_status = Path("/proc/1/status").read_text().splitlines()
        mountinfo = Path("/proc/self/mountinfo").read_text().splitlines()
        def field(lines, name):
            rows = [row.split()[1:] for row in lines if row.startswith(name + ":")]
            return rows[0] if len(rows) == 1 else None
        namespace = os.readlink("/proc/self/ns/pid")
        proc_mounts = []
        for row in mountinfo:
            mount, separator, filesystem = row.partition(" - ")
            fields = mount.split()
            if separator and len(fields) >= 5 and fields[4] == "/proc":
                proc_mounts.append((fields[3], filesystem.split()[0]))
        if (not own_stat.startswith(f"{own_pid} (")
                or field(own_status, "Pid") != [str(own_pid)]
                or field(own_status, "Tgid") != [str(own_pid)]
                or not init_stat.startswith("1 (")
                or int(init_stat.rsplit(")", 1)[1].split()[1]) != 0
                or field(init_status, "Pid") != ["1"]
                or field(init_status, "Tgid") != ["1"]
                or field(init_status, "PPid") != ["0"]
                or proc_mounts != [("/", "proc")]
                or re.fullmatch(r"pid:\[[0-9]+\]", namespace) is None):
            return None
        return namespace
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def process_ownership_identity(pid):
    """只用内核 stat/status 与本地 procfs 锚点读取父链，不要求跨 UID 的 exe/cmdline。"""
    path = Path("/proc") / str(pid)
    try:
        stat = path.joinpath("stat").read_text()
        if not stat.startswith(f"{pid} ("):
            return None
        comm = stat[len(f"{pid} ("):].rsplit(")", 1)[0]
        fields = stat.rsplit(")", 1)[1].split()
        status = path.joinpath("status").read_text().splitlines()
        values = {}
        for key in ("Pid", "Tgid", "PPid", "Uid"):
            rows = [row.split()[1:] for row in status if row.startswith(key + ":")]
            if len(rows) != 1:
                return None
            values[key] = [int(value) for value in rows[0]]
        namespace_rows = [row.split()[1:] for row in status if row.startswith("NSpid:")]
        if len(namespace_rows) > 1:
            return None
        namespace_pids = ([int(value) for value in namespace_rows[0]]
                          if namespace_rows else [pid])
        parent_pid = int(fields[1])
        start_ticks = int(fields[19])
        if (not comm or fields[0] == "Z" or parent_pid < 0 or start_ticks <= 0
                or values["Pid"] != [pid] or values["Tgid"] != [pid]
                or values["PPid"] != [parent_pid]
                or namespace_pids != [pid] or len(values["Uid"]) != 4
                or any(uid < 0 for uid in values["Uid"])):
            return None
        local_namespace = local_procfs_namespace_anchor()
        if local_namespace is None:
            return None
        try:
            namespace = os.readlink(path / "ns" / "pid")
            namespace_source = "direct_ns_symlink"
        except PermissionError:
            # 旧内核无 NSpid，跨 UID 的 ns 链接不可读；同一 procfs 的
            # stat/status PID 与 PPID 链仍可由本进程和 PID 1 锚点核验。
            # 目标自身的 PID 命名空间未知，不能记作已经直接核验。
            namespace = None
            namespace_source = "permission_denied"
        if namespace is not None and namespace != local_namespace:
            return None
        repeated = path.joinpath("stat").read_text()
        if not repeated.startswith(f"{pid} ("):
            return None
        repeated_comm = repeated[len(f"{pid} ("):].rsplit(")", 1)[0]
        repeated_fields = repeated.rsplit(")", 1)[1].split()
        if (repeated_comm != comm or repeated_fields[0] == "Z"
                or int(repeated_fields[1]) != parent_pid
                or int(repeated_fields[19]) != start_ticks):
            return None
        return {"pid": pid, "ppid": parent_pid, "start_ticks": start_ticks,
                "uids": values["Uid"], "namespace_pids": namespace_pids,
                "pid_namespace": namespace, "pid_namespace_source": namespace_source,
                "procfs_view_namespace": local_namespace,
                "comm": comm}
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def process_ownership_chain(pid):
    """权限受限时仍核对到 PID 1；缺失任何祖先都不能推断外部归属。"""
    chain = []
    seen = set()
    while pid != 0 and pid not in seen:
        seen.add(pid)
        identity = process_ownership_identity(pid)
        if identity is None:
            return None
        chain.append(identity)
        if pid == 1:
            return chain if identity["ppid"] == 0 else None
        pid = identity["ppid"]
    return None


def distinct_uid_external_chain(pid, *, observed, adopted, lease_pid,
                                lease_start_ticks, child_pid, child_start_ticks):
    """双重最小父链证明异 UID 外部进程；同 UID 或任何证据缺失均返回 None。"""
    if local_procfs_namespace_anchor() != INITIAL_PID_NAMESPACE:
        return None
    lease = process_identity(lease_pid)
    if lease is None or lease["start_ticks"] != lease_start_ticks:
        return None
    first = process_ownership_chain(pid)
    second = process_ownership_chain(pid)
    if first is None or second is None or len(first) != len(second):
        return None
    fields = ("pid", "ppid", "start_ticks", "uids", "namespace_pids",
              "pid_namespace", "pid_namespace_source", "procfs_view_namespace", "comm")
    if any(tuple(a[key] for key in fields) != tuple(b[key] for key in fields)
           for a, b in zip(first, second)):
        return None
    lease_uids = set(lease["uids"])
    if (first[0]["procfs_view_namespace"] != lease["pid_namespace"]
            or not set(first[0]["uids"]).isdisjoint(lease_uids)
            or not all(set(row["uids"]).isdisjoint(lease_uids) for row in first[:-1])):
        return None
    for row in first:
        row_pid, start_ticks = row["pid"], row["start_ticks"]
        if (row_pid == lease_pid and start_ticks == lease_start_ticks
                or row_pid == child_pid and
                   (child_start_ticks is None or start_ticks == child_start_ticks)
                or observed.get(row_pid) == start_ticks
                or (row_pid in adopted and
                    adopted[row_pid]["start_ticks"] == start_ticks)):
            return None
    return first


def process_name_matches(entry, identity):
    """GPU 进程名称至少要和 /proc 的 argv0 或 exe 路径精确相符。"""
    name = entry["process_name"]
    return any(name == value or os.path.realpath(name) == os.path.realpath(value)
               for value in (identity["argv0"], identity["exe"]) if value)


def audit_gpu_process_ownership(snapshot, second_snapshot, *, observed, adopted,
                                lease_pid, lease_start_ticks, child_pid,
                                child_start_ticks):
    """只豁免初始 PID 命名空间中身份、UID、完整父链均稳定的外部进程。

    已观测/收养的同启动时钟后代一律保留阻断。其他 PID 必须可映射到
    同一 /proc 命名空间，且父链稳定到 PID 1。subreaper 能收养作业产生的
    孤儿；父链证据不足时不推断它来自别的任务。
    """
    before = gpu_process_entries(snapshot)
    after = gpu_process_entries(second_snapshot)
    after_by_pid = {entry["pid"]: entry for entry in after}
    host_pid_namespace_verified = local_procfs_namespace_anchor() == INITIAL_PID_NAMESPACE
    audit = []
    unresolved = []
    external = []
    for entry in before:
        pid = entry["pid"]
        evidence = {"pid": pid, "gpu_process_name": entry["process_name"],
                    "raw": entry["raw"], "classification": "unresolved",
                    "host_pid_namespace_verified": host_pid_namespace_verified}
        first_chain = process_chain(pid)
        second_entry = after_by_pid.get(pid)
        second_chain = process_chain(pid) if second_entry is not None else None
        evidence["ancestry_first"] = first_chain
        evidence["ancestry_second"] = second_chain
        if first_chain is None or second_chain is None:
            redacted_chain = None
            if (host_pid_namespace_verified and second_entry is not None
                    and second_entry["process_name"] == entry["process_name"]):
                redacted_chain = distinct_uid_external_chain(
                    pid, observed=observed, adopted=adopted,
                    lease_pid=lease_pid, lease_start_ticks=lease_start_ticks,
                    child_pid=child_pid, child_start_ticks=child_start_ticks)
            if (redacted_chain is not None and
                    os.path.basename(entry["process_name"]) != redacted_chain[0]["comm"]):
                evidence["reason"] = "GPU进程名称与内核comm不符"
                redacted_chain = None
            if redacted_chain is None:
                if "reason" not in evidence:
                    evidence["reason"] = "PID/UID/父链或命名空间不可核验"
            else:
                evidence["classification"] = "external"
                evidence["reason"] = "异UID且双重稳定的完整内核父链；无需跨UID读取命令行"
                evidence["kernel_ancestry"] = redacted_chain
                external.append(entry["raw"])
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
            elif not host_pid_namespace_verified:
                evidence["reason"] = "无法证明GPU驱动PID与本地/proc同属初始PID命名空间"
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
    if job.get("threads") != 4:
        raise ValueError("单卡队列总线程必须为4")
    gpu_index = physical_gpu_index(job)
    free_memory_threshold = minimum_free_gpu_bytes(job)
    require_idle_gpu = job.get("require_idle_gpu", True)
    verify_gpu_identity = require_idle_gpu or free_memory_threshold is not None
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
            "cross_host_speed_comparison": "prohibited", "verified_files": []
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
            if verify_gpu_identity:
                inventory, snapshot, other_count = gpu_snapshot(
                    gpu_index, job["expected_gpu_uuid"])
                report["actual_gpu_inventory"] = inventory
                report["actual_gpu_uuids"] = [inventory[key] for key in sorted(inventory)]
                report["selected_gpu_uuid"] = job["expected_gpu_uuid"]
                report["other_gpu_process_count_before"] = other_count
                report["gpu_processes_before"] = snapshot
                if require_idle_gpu and snapshot:
                    raise RuntimeError("目标卡已有计算进程；保留回执，不抢占或终止")
            env = os.environ.copy()
            env.update(job.get("env", {}))
            for name in job.get("unset_env", []):
                env.pop(name, None)
            # 完整UUID避免物理索引与CUDA运行时枚举顺序不同；进程内仍是cuda:0。
            env["CUDA_VISIBLE_DEVICES"] = (
                job["expected_gpu_uuid"] if verify_gpu_identity
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
            with (receipt_dir / "stdout.private.log").open("wb") as log:
                if free_memory_threshold is not None:
                    report["minimum_free_gpu_bytes"] = free_memory_threshold
                    memory = gpu_free_memory(gpu_index, job["expected_gpu_uuid"])
                    report["gpu_memory_before_spawn"] = memory
                    if memory["memory_free_bytes"] < free_memory_threshold:
                        raise RuntimeError(
                            "目标GPU启动前空闲显存不足："
                            f"{memory['memory_free_bytes']} < {free_memory_threshold}字节")
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
                while child.poll() is None:
                    rows = processes()
                    for pid in descendants(os.getpid(), rows):
                        observed[pid] = rows[pid]["start_ticks"]
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
            if job.get("require_idle_gpu", True):
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
            if job.get("require_idle_gpu", True):
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
                        child_start_ticks=report["child_start_ticks"])
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
