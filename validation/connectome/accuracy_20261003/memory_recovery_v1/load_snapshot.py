"""Read-only NVML shared-load snapshot; no CUDA context or scientific work."""
import json
from pathlib import Path
import sys
import time
import pynvml

root=Path('/cwStorage/home/gongwk/Notebook_code/FNIT/runs/connectome-accuracy-memory-recovery-20261003-v1')
controller_pids=[json.loads((root/name).read_bytes())['pid'] for name in ('launch_receipt.json','launch_receipt_v2.json') if (root/name).is_file()]
def own(pid):
    seen=set()
    while pid>1 and pid not in seen:
        if pid in controller_pids:return True
        seen.add(pid)
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except (FileNotFoundError,ProcessLookupError,PermissionError):return False
    return False
pynvml.nvmlInit()
try:
    uuid='GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba';handle=pynvml.nvmlDeviceGetHandleByUUID(uuid)
    memory=pynvml.nvmlDeviceGetMemoryInfo(handle);util=pynvml.nvmlDeviceGetUtilizationRates(handle)
    rows=[{'pid':int(row.pid),'bytes':int(row.usedGpuMemory),'own_recovery_tree':own(int(row.pid))}
          for row in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
          if row.usedGpuMemory!=getattr(pynvml,'NVML_VALUE_NOT_AVAILABLE',(1<<64)-1)]
    record={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu_uuid':uuid,'known_controller_pids':controller_pids,'total_device_used_bytes':int(memory.used),
            'total_device_memory_bytes':int(memory.total),'GPU_utilization_percent':int(util.gpu),'memory_utilization_percent':int(util.memory),
            'compute_processes':rows,'foreign_compute_bytes':sum(row['bytes'] for row in rows if not row['own_recovery_tree']),
            'own_compute_bytes':sum(row['bytes'] for row in rows if row['own_recovery_tree']),
            'scope':'periodic shared load observation; distinct from the frozen wall monitor memory gate'}
    with (root/'load_snapshots.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    print(json.dumps(record))
finally:pynvml.nvmlShutdown()
