"""隔离表面benchmark的NVML采样；缺少可证明PID归属时返回None。"""
from __future__ import annotations
import os
from pathlib import Path
import threading
import time

from .nvml_process_memory import NvmlProcessQuery
from .profiling import ProcessTreeDeviceSampler


class PlacementMemorySampler:
    """uuid指定物理卡，interval_seconds默认0.5；start/finish只查询NVML。

    仅记录本benchmark无GPU子进程的当前进程；原版参考为CPU单独子进程。
    单NSpid或NVML数字PID相同均不能证明归属。trusted_host_pid_map默认
    None，沿用ProcessTreeDeviceSampler可信宿主映射合同，逐样本重核
    当前进程starttime、namespace和存活身份；没有可信映射则本进程
    own_peak_bytes为None。不使用sched猜测、不依赖tools源码。
    整卡峰值单列且绝不替代本任务。后台线程仅等待/查询，不运行计算。
    NVML缺失和查询失败保留状态，不影响算法；finish停止并释放本次句柄。
    """
    def __init__(self, *, uuid: str, interval_seconds: float = .5,
                 trusted_host_pid_map: dict | None = None):
        if interval_seconds <= 0: raise ValueError("positive sampling interval required")
        self.uuid, self.interval = uuid, float(interval_seconds)
        self.stop = threading.Event()
        self.samples = []
        self.query = None
        self.failure = None
        self.thread = None
        # 只复用成熟身份合同；此实例不调用其CUDA/命令行采样接口。
        self.identity_sampler = ProcessTreeDeviceSampler(
            device="cuda:0", parent_pid=os.getpid(), query_backend="nvml",
            trusted_host_pid_map=trusted_host_pid_map)
        lines = Path("/proc/self/status").read_text().splitlines()
        ids = next((line.split()[1:] for line in lines if line.startswith("NSpid:")), [])
        self.pid_namespace_single = len(ids) == 1 and int(ids[0]) == os.getpid()

    def start(self):
        try:
            self.query = NvmlProcessQuery()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except Exception as error: self.failure = type(error).__name__
        return self

    def _run(self):
        while not self.stop.is_set():
            row = dict(time_monotonic=time.perf_counter(), own_bytes=None, device_bytes=None)
            try:
                processes, total, error = self.query.snapshot(self.uuid)
                row['device_bytes'] = total
                mapping = self.identity_sampler._pid_mapping({os.getpid()})
                row['pid_mapping_status'] = mapping['status']
                row['pid_mapping_unavailable_reasons'] = mapping['unavailable_reasons']
                aliases = mapping['aliases']
                found = [entry['bytes'] for entry in processes
                         if aliases.get(entry['pid']) == os.getpid()]
                if mapping['status'] == 'verified' and len(found) == 1:
                    row['own_bytes'] = found[0]
                row['query_error'] = error
            except Exception as error: row['query_error'] = type(error).__name__
            self.samples.append(row)
            self.stop.wait(self.interval)

    def finish(self):
        self.stop.set()
        if self.thread is not None: self.thread.join()
        if self.query is not None: self.query.close()
        own = [row['own_bytes'] for row in self.samples if row['own_bytes'] is not None]
        total = [row['device_bytes'] for row in self.samples if row['device_bytes'] is not None]
        times = [row['time_monotonic'] for row in self.samples]
        return dict(uuid=self.uuid, interval_seconds=self.interval, samples=len(times),
            own_observed_samples=len(own), own_peak_bytes=max(own) if own else None,
            device_total_peak_bytes=max(total) if total else None,
            own_attribution='verified_trusted_host_mapping_rechecked_each_sample' if own else 'unknown_no_verified_ownership_observation',
            pid_namespace_single=self.pid_namespace_single,
            maximum_gap_seconds=max((b-a for a,b in zip(times,times[1:])), default=None),
            query_failed_samples=sum(bool(row.get('query_error')) for row in self.samples),
            initialization_error_type=self.failure, raw_samples=self.samples)
