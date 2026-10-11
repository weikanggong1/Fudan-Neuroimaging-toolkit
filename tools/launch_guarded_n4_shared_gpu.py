"""仅替换启动资源策略：共享低利用率GPU，同UUID双采样、锁及启动复查。"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from fnit.recon_all.nvml_process_memory import NvmlProcessQuery


def observation(query, index, uuid):
    rows, used, error = query.snapshot(uuid)
    handle = query.handles[uuid]
    memory = query.module.nvmlDeviceGetMemoryInfo(handle)
    utilization = query.module.nvmlDeviceGetUtilizationRates(handle)
    graphics = query.module.nvmlDeviceGetGraphicsRunningProcesses(handle)
    return {"monotonic": time.monotonic(), "physical_index": int(index), "uuid": uuid,
        "compute_processes": rows, "graphics_processes": [{"pid": int(row.pid)} for row in graphics],
        "device_used_bytes": used, "device_free_bytes": int(memory.free),
        "device_total_bytes": int(memory.total), "utilization_percent": int(utilization.gpu),
        "device_error_type": error.split("(",1)[0] if error else None,
        "qualifies": int(index) != 4 and not error and utilization.gpu <= 10 and memory.free >= 32000000000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    root, task = args.fnit_root, args.task
    source, run = root/"workspaces"/task/"shared_v2", root/"runs"/task
    protocol = run/"resource_protocol_v2.private.json"
    assert json.loads(protocol.read_text())["policy"] == "shared_low_utilization_v2"
    query = NvmlProcessQuery()
    devices = query.devices()
    status = run/"launch_shared_v2.private.json"
    try:
        while True:
            observations = []
            for index in sorted(devices, key=lambda x:(x!="6",int(x))):
                if index == "4":
                    continue
                try:
                    first = observation(query,index,devices[index])
                    observations.append(first)
                    if not first["qualifies"]:
                        continue
                    time.sleep(1)
                    second = observation(query,index,devices[index])
                    assert second["monotonic"]-first["monotonic"] >= 1
                    observations.append(second)
                    if not second["qualifies"]:
                        continue
                    with contextlib.ExitStack() as stack:
                        names = set(Path("/tmp").glob("*gpu"+index+"*.lock"))
                        names.add(Path("/tmp", "fnit-recon-gpu"+index+".lock"))
                        for name in sorted(names):
                            lock = stack.enter_context(name.open("a"))
                            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        final = observation(query,index,devices[index])
                        observations.append(final)
                        if not final["qualifies"]:
                            continue
                        receipt = {"status":"resource_gate_passed", "policy":"shared_low_utilization_v2",
                            "nonexclusive":True, "same_uuid_observations":[first,second,final],
                            "locks":[str(name) for name in sorted(names)]}
                        (run/"resource_gate_v2.private.json").write_text(json.dumps(receipt,indent=2)+"\n")
                        status.write_text(json.dumps({"status":"dispatching_shared_v2","MRI_started":None,
                            "resource_gate":receipt},indent=2)+"\n")
                        env = dict(os.environ,CUDA_VISIBLE_DEVICES=devices[index],
                            FNIT_RECON_FORBID_ORIGINAL_STAGES="1",PYTORCH_NO_CUDA_MEMORY_CACHING="1")
                        command=[sys.executable,str(source/"tools/benchmark_guarded_n4_worker.py"),
                            "--inputs",str(run/"inputs.private.json"),"--output",str(run/"execution_shared_v2"),
                            "--diagnostic",str(source/"tools/diagnose_recon_nvml_namespace.py"),
                            "--wheel",str(root/"archive/transfers/fudan_neuroimaging_toolkit-0.16.0-py3-none-any.whl"),
                            "--install-report",str(root/"archive/transfers/recon-fnit5-root-install-v3.public.json"),
                            "--installed-target",str(run/"install_v1"),
                            "--audit-helper",str(source/"tools/recon_python_execution_audit"),
                            "--gpu-uuid",devices[index],"--idle-baseline-bytes",str(final["device_used_bytes"]),
                            "--resource-protocol",str(protocol)]
                        with (run/"execution_shared_v2.private.log").open("w") as log:
                            done=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                        (run/"exit_execution_shared_v2").write_text(str(done.returncode)+"\n")
                        status.write_text(json.dumps({"status":"completed" if done.returncode==0 else "failed",
                            "exit_code":done.returncode,"nonexclusive":True,"resource_gate":receipt},indent=2)+"\n")
                        return done.returncode
                except BlockingIOError:
                    observations.append({"physical_index":int(index),"lock_available":False})
                except Exception as error:
                    observations.append({"physical_index":int(index),"error_type":type(error).__name__})
            state={"status":"waiting_shared_resource_gate","MRI_started":False,
                "time_unix":time.time(),"nonexclusive":True,"observations":observations}
            status.write_text(json.dumps(state,indent=2)+"\n")
            with (run/"shared_resource_history_v2.private.jsonl").open("a") as stream:
                stream.write(json.dumps(state)+"\n")
            time.sleep(30)
    finally:
        query.close()


if __name__=="__main__":
    raise SystemExit(main())
