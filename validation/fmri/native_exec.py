"""原软件 benchmark 的进程证据；不由 FNIT 运行接口调用。"""

from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import subprocess


LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _verified_command_trace_provenance(trace_path, root_event, provenance):
    """仅凭冻结构造、私有命令记录与实际 root exec 验证 strace -f 来源。"""
    source_path = Path(provenance["frozen_source_path"])
    record_path = Path(provenance["command_record_path"])
    source = source_path.read_bytes()
    record = record_path.read_bytes()
    trace = Path(trace_path).read_bytes()
    identities = {"frozen_source_sha256": hashlib.sha256(source).hexdigest(),
                  "command_record_sha256": hashlib.sha256(record).hexdigest(),
                  "saved_trace_sha256": hashlib.sha256(trace).hexdigest()}
    if any(provenance.get(key) != value for key, value in identities.items()):
        raise ValueError("source_record_or_trace_hash_mismatch")
    functions = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef) and node.name == "run_traced"]
    if len(functions) != 1:
        raise ValueError("frozen_run_traced_function_missing")
    function = functions[0]
    allowed_filters = ["trace=execve,exit_group",
                       "trace=execve,exit_group,clone,fork,vfork,wait4,waitid"]
    expected = {
        "original": '["/bin/sh", "-c", command] if shell else list(map(str, command))',
    }
    invoked = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "invoked" for target in node.targets)]
    matching_filters = [value for value in allowed_filters if len(invoked) == 1 and
        ast.dump(invoked[0]) == ast.dump(ast.parse(
            '[executable, "-f", "-s", "4096", "-e", ' + repr(value) +
            ', "-o", str(trace_path), *original]', mode="eval").body)]
    if len(matching_filters) != 1:
        raise ValueError("frozen_strace_invocation_shape_mismatch")
    trace_filter = matching_filters[0]
    for name, expression in expected.items():
        assignments = [node.value for node in ast.walk(function)
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == name
                           for target in node.targets)]
        if len(assignments) != 1 or ast.dump(assignments[0]) != ast.dump(
                ast.parse(expression, mode="eval").body):
            raise ValueError("frozen_strace_invocation_shape_mismatch")
    runs = [node for node in ast.walk(function) if isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and node.func.attr == "run" and
            isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"]
    if len(runs) != 1 or not runs[0].args or ast.dump(runs[0].args[0]) != "Name(id='invoked', ctx=Load())":
        raise ValueError("frozen_subprocess_invocation_mismatch")
    mutations = [node for node in ast.walk(function) if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Attribute) and
                 isinstance(node.func.value, ast.Name) and node.func.value.id == "invoked"]
    if mutations:
        raise ValueError("frozen_invoked_argv_mutated")
    command_record = json.loads(record)
    command = command_record
    pointer = provenance["command_record_pointer"]
    if not isinstance(pointer, list) or not pointer:
        raise ValueError("private_command_pointer_missing")
    for key in pointer:
        if type(key) not in (str, int):
            raise ValueError("private_command_pointer_invalid")
        command = command[key]
    if provenance["command_kind"] == "shell_pipeline" and isinstance(command, str):
        original = ["/bin/sh", "-c", command]
    elif provenance["command_kind"] == "argv" and isinstance(command, list) and command and all(
            isinstance(value, str) for value in command):
        original = command
    else:
        raise ValueError("private_command_format_invalid")
    decoder = json.JSONDecoder()
    filename, offset = decoder.raw_decode(root_event[len("execve("):])
    remaining = root_event[len("execve(") + offset:].lstrip()
    if not remaining.startswith(","):
        raise ValueError("root_exec_argv_unparsed")
    argv, _ = decoder.raw_decode(remaining[1:].lstrip())
    if filename != original[0] or argv != original:
        raise ValueError("actual_root_exec_does_not_match_private_command")
    # 当时只记录了原命令；strace 路径未保存，以下明确为规范化重建，不能当 full argv 原日志。
    normalized = ["strace", "-f", "-s", "4096", "-e", trace_filter,
                  "-o", "<private-trace>", *original]
    actual_invoked_sha = None
    record_kind = "Reconstructed normalized argv from the verified frozen invocation and recorded private command; the actual root exec filename and every argument match the saved trace. The original strace executable path was not recorded."
    actual_invoked = command_record.get("invoked_argv") if isinstance(command_record, dict) else None
    if actual_invoked is not None:
        if (not isinstance(actual_invoked, list) or not actual_invoked or
                not all(isinstance(value, str) for value in actual_invoked) or
                Path(actual_invoked[0]).name != "strace" or
                actual_invoked[1:] != ["-f", "-s", "4096", "-e", trace_filter,
                                      "-o", str(trace_path), *original]):
            raise ValueError("recorded_actual_strace_argv_mismatch")
        actual_invoked_sha = hashlib.sha256(json.dumps(
            actual_invoked, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        record_kind = "Actual invoked argv recorded privately before subprocess.run; the frozen invocation shape and actual root exec filename and every argument match. Only hashes and a redacted template are public."
    return {**identities,
            "frozen_invocation_function_ast_sha256": hashlib.sha256(
                ast.dump(function).encode()).hexdigest(),
            "normalized_reconstructed_invoked_argv_sha256": hashlib.sha256(
                json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            "recorded_actual_invoked_argv_sha256": actual_invoked_sha,
            "argv_template": f"strace -f -s 4096 -e {trace_filter} -o <private-trace> <recorded-command-root>",
            "invocation_record_kind": record_kind,
            "follow_spawned_command_root_verified": True}


def trace_exit_evidence(trace_path, returncode, *, command_trace_provenance=None):
    """核对首个命令 PID 及完整 exec/exit/SIGCHLD 链，不靠输出文件推断成功。"""
    trace = Path(trace_path).read_text(errors="replace")
    executed = set()
    exec_counts = {}
    pending_exec = set()
    pending_exit = {}
    exits = {}
    signals = []
    spawned = []
    pending_spawn = {}
    root_pid = None
    root_event = None
    first_event_is_exec = False
    errors = []

    def error(kind, **details):
        errors.append({"kind": kind, **details})

    for line in trace.splitlines():
        record = re.match(r"^(\d+)\s+(.*)$", line)
        if record is None:
            continue
        pid, event = int(record[1]), record[2]
        if root_pid is None:
            root_pid = pid
            root_event = event
            first_event_is_exec = event.startswith("execve(")
        if event.startswith("execve("):
            if pid in pending_exec:
                error("second_exec_before_resume", pid=pid)
            if event.endswith("<unfinished ...>"):
                pending_exec.add(pid)
            elif re.search(r"\)\s+= 0$", event):
                executed.add(pid)
                exec_counts[pid] = exec_counts.get(pid, 0) + 1
        elif event.startswith("<... execve resumed>"):
            if pid not in pending_exec:
                error("exec_resume_without_start", pid=pid)
            else:
                pending_exec.remove(pid)
                if re.search(r"\)\s+= 0$", event):
                    executed.add(pid)
                    exec_counts[pid] = exec_counts.get(pid, 0) + 1
        elif event.startswith("exit_group("):
            match = re.match(r"exit_group\((-?\d+)(?:\)|\s+<unfinished \.\.\.>)", event)
            if match is None or (not event.endswith("<unfinished ...>") and
                                 not re.fullmatch(r"exit_group\(-?\d+\)\s+= \?", event)):
                error("unparsed_exit_group", pid=pid)
            elif pid in exits or pid in pending_exit:
                error("duplicate_exit_group", pid=pid)
            elif event.endswith("<unfinished ...>"):
                pending_exit[pid] = int(match[1]) % 256
            else:
                exits[pid] = int(match[1]) % 256
        elif event.startswith("<... exit_group resumed>"):
            code = pending_exit.pop(pid, None)
            if code is None:
                error("exit_resume_without_start", pid=pid)
            elif not re.fullmatch(r"<\.\.\. exit_group resumed>\)\s+= \?", event):
                error("unparsed_exit_group_resume", pid=pid)
            elif pid in exits:
                error("duplicate_exit_group", pid=pid)
            else:
                exits[pid] = code
        elif re.match(r"(?:clone|fork|vfork)\(", event):
            call = event.split("(", 1)[0]
            if event.endswith("<unfinished ...>"):
                pending_spawn[pid, call] = event
            else:
                child = re.search(r"\)\s+= (\d+)$", event)
                if child and int(child[1]) > 0:
                    spawned.append((pid, int(child[1]), "CLONE_THREAD" in event))
        elif re.match(r"<\.\.\. (?:clone|fork|vfork) resumed>", event):
            call = event.split()[1]
            start = pending_spawn.pop((pid, call), None)
            child = re.search(r"\)\s+= (\d+)$", event)
            if start is None:
                error("spawn_resume_without_start", pid=pid)
            elif child and int(child[1]) > 0:
                spawned.append((pid, int(child[1]), "CLONE_THREAD" in start + event))
        elif event.startswith("--- SIGCHLD "):
            fields = dict(re.findall(r"\b(si_\w+)=([^,}]+)", event))
            child_text = fields.get("si_pid", "").strip()
            status_text = fields.get("si_status", "").strip()
            child = int(child_text) if child_text.isdigit() else None
            status = int(status_text) if status_text.isdigit() else None
            signals.append({"parent_pid": pid, "child_pid": child,
                            "si_code": fields.get("si_code", "").strip() or None,
                            "si_status": status})
        elif event.startswith("+++ killed by "):
            error("process_killed_by_signal", pid=pid)
        elif event.startswith("+++ exited with "):
            terminal = re.match(r"\+\+\+ exited with (\d+) \+\+\+", event)
            if terminal and int(terminal[1]) != 0 and pid not in exits:
                exits[pid] = int(terminal[1]) % 256

    verified_provenance = None
    if command_trace_provenance is not None:
        try:
            verified_provenance = _verified_command_trace_provenance(
                trace_path, root_event or "", command_trace_provenance)
        except (KeyError, TypeError, ValueError, OSError, SyntaxError, IndexError):
            error("command_trace_provenance_not_verified")

    for pid in sorted(pending_exec):
        error("exec_start_without_resume", pid=pid)
    for pid in sorted(pending_exit):
        error("exit_start_without_resume", pid=pid)
    for pid, _ in sorted(pending_spawn):
        error("spawn_start_without_resume", pid=pid)
    if not first_event_is_exec:
        error("root_record_does_not_begin_with_exec")
    if root_pid not in executed:
        error("root_exec_not_completed")
    if root_pid not in exits:
        error("root_exit_missing")
    elif exits[root_pid] != returncode:
        error("root_exit_disagrees_with_returncode", trace_exit_code=exits[root_pid],
              observed_returncode=returncode)

    children_by_parent = {}
    observed_parent = {}
    thread_parents = {child: parent for parent, child, is_thread in spawned if is_thread}

    def process_group(pid):
        visited = set()
        while pid in thread_parents:
            if pid in visited:
                error("cyclic_thread_genealogy", pid=pid)
                break
            visited.add(pid)
            pid = thread_parents[pid]
        return pid

    threaded_signal_receivers = 0
    for signal in signals:
        receiver = signal["parent_pid"]
        signal["parent_pid"] = process_group(receiver)
        if receiver != signal["parent_pid"]:
            threaded_signal_receivers += 1
        children_by_parent.setdefault(signal["parent_pid"], []).append(signal)
        parent, child, status = (signal[key] for key in
                                 ("parent_pid", "child_pid", "si_status"))
        if child == parent or child == root_pid:
            error("invalid_child_parent_relation", parent_pid=parent, child_pid=child)
        if child is not None:
            if child in observed_parent:
                error("repeated_child_termination", parent_pid=parent, child_pid=child)
            observed_parent[child] = parent
        if parent not in executed:
            error("sigchld_parent_exec_missing", parent_pid=parent)
        if child not in executed:
            error("sigchld_child_exec_missing", parent_pid=parent, child_pid=child)
        if child not in exits:
            error("sigchld_child_exit_missing", parent_pid=parent, child_pid=child)
        if signal["si_code"] != "CLD_EXITED":
            error("sigchld_is_not_normal_exit", parent_pid=parent, child_pid=child,
                  si_code=signal["si_code"])
        if child in exits and exits[child] != status:
            error("sigchld_exit_status_mismatch", parent_pid=parent, child_pid=child)

    # 未知、被信号杀死或退出非零的孩子保留在图中，不能先过滤再证明。
    proven = {}

    def child_chain_succeeded(pid, visiting):
        if pid in proven:
            return proven[pid]
        if pid in visiting or pid not in executed or exits.get(pid) != 255:
            return False
        children = children_by_parent.get(pid, [])
        proof = bool(children) and all(
            signal["si_code"] == "CLD_EXITED" and
            signal["child_pid"] in executed and signal["child_pid"] in exits and
            exits[signal["child_pid"]] == signal["si_status"] and
            (signal["si_status"] == 0 or
             (signal["si_status"] == 255 and
              child_chain_succeeded(signal["child_pid"], visiting | {pid})))
            for signal in children)
        proven[pid] = proof
        return proof

    abnormal = []
    for pid, code in exits.items():
        if code != 255:
            continue
        children = children_by_parent.get(pid, [])
        proof = child_chain_succeeded(pid, set())
        abnormal.append({"launcher_pid": pid, "launcher_exit_code": code,
                         "launcher_exec_completed": pid in executed,
                         "observed_direct_child_count": len(children),
                         "successful_direct_child_exec_count": sum(
                             signal["child_pid"] in executed for signal in children),
                         "direct_child_exit_codes": [exits.get(signal["child_pid"])
                                                     for signal in children],
                         "direct_child_sigchld_statuses": [signal["si_status"]
                                                           for signal in children],
                         "direct_child_sigchld_codes": [signal["si_code"]
                                                        for signal in children],
                         "proven_nested_launcher_count": sum(
                             signal["si_status"] == 255 and proven.get(signal["child_pid"], False)
                             for signal in children),
                         "inner_success_proven": proof})

    # 正常退出的 shell 也不能掩盖管道中的真实非零退出。所有进程须属于命令根的证据链。
    reachable = set()

    def visit(pid):
        if pid in reachable:
            return
        reachable.add(pid)
        for signal in children_by_parent.get(pid, []):
            if signal["child_pid"] is not None:
                visit(signal["child_pid"])
        for parent, child, is_thread in spawned:
            if not is_thread and process_group(parent) == pid:
                visit(child)

    visit(root_pid)
    coalesced_normal_zero = []
    for pid in sorted(executed | set(exits)):
        if pid not in executed:
            error("exit_without_successful_exec", pid=pid)
        if pid not in exits:
            error("executed_process_exit_missing", pid=pid)
        if pid not in reachable:
            if verified_provenance and pid in executed and exits.get(pid) == 0:
                coalesced_normal_zero.append(pid)
            else:
                error("process_not_in_root_child_chain", pid=pid)
        if pid in exits and exits[pid] not in (0, 255):
            error("nonzero_native_exit", pid=pid, exit_code=exits[pid])
    all_inner_success = all(row["inner_success_proven"] for row in abnormal)
    root_success = (root_pid in executed and exits.get(root_pid) == returncode and
                    (returncode == 0 or
                     (returncode == 255 and child_chain_succeeded(root_pid, set()))))
    accepted = root_success and all_inner_success and not errors
    return {"schema_version": 2, "launcher_exit_code": returncode,
            "evidence_parser_loaded_sha256": LOADED_SOURCE_SHA256,
            "trace_enabled": True,
            "command_root_pid": root_pid,
            "root_exec_completed": root_pid in executed,
            "root_trace_exit_code": exits.get(root_pid),
            "root_success_proven": root_success,
            "successful_exec_process_count": len(executed),
            "successful_exec_event_count": sum(exec_counts.values()),
            "observed_sigchld_count": len(signals),
            "observed_spawn_event_count": len(spawned),
            "verified_thread_count": len(thread_parents),
            "verified_thread_process_groups": [
                {"thread_pid": pid, "process_group_pid": process_group(pid)}
                for pid in sorted(thread_parents)],
            "sigchld_received_by_verified_threads_count": threaded_signal_receivers,
            "command_trace_provenance": verified_provenance,
            "normal_zero_descendants_without_individual_sigchld_count": len(coalesced_normal_zero),
            "trace_integrity_errors": errors,
            "exit255_records": abnormal,
            "all_exit255_have_inner_success": all_inner_success,
            "original_process_accepted": accepted,
            "exit_code_was_rewritten": False}


def run_traced(command, *, trace_path, env=None, stdout=None, stderr=None,
               capture_output=False, text=False, shell=False):
    """记录 exec、exit 和 SIGCHLD；不更改子进程退出码或参数。"""
    executable = shutil.which("strace")
    if not executable:
        raise FileNotFoundError("This reference requires strace for launcher evidence")
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    original = ["/bin/sh", "-c", command] if shell else list(map(str, command))
    invoked = [executable, "-f", "-s", "4096", "-e", "trace=execve,exit_group,clone,fork,vfork,wait4,waitid",
               "-o", str(trace_path), *original]
    command_record = trace_path.with_name(trace_path.name + ".command.private.json")
    command_record.write_text(json.dumps({"schema_version": 1, "original_argv": original,
                                         "invoked_argv": invoked,
                                         "loaded_source_sha256": LOADED_SOURCE_SHA256},
                                        ensure_ascii=False, indent=2) + "\n")
    result = subprocess.run(invoked, env=env, stdout=stdout, stderr=stderr,
                            capture_output=capture_output, text=text)
    evidence = trace_exit_evidence(trace_path, result.returncode, command_trace_provenance={
        "frozen_source_path": Path(__file__), "frozen_source_sha256": LOADED_SOURCE_SHA256,
        "command_record_path": command_record,
        "command_record_sha256": hashlib.sha256(command_record.read_bytes()).hexdigest(),
        "command_record_pointer": ["original_argv"], "command_kind": "argv",
        "saved_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    })
    if not evidence["original_process_accepted"]:
        raise RuntimeError("Original process trace does not prove a complete successful command")
    return result, evidence
