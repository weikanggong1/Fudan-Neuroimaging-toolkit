"""以外层单调时钟记录冻结Python阶段冷CLI；不依赖系统time程序。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time


def time_command(*, command: list[str], log_path: Path, report_path: Path) -> dict:
    """执行显式argv的冻结Python benchmark，含启动、导入、计算及读写。

    command为非空字符串列表，无shell展开；log_path/report_path是新
    文件，父目录须存在。stdout/stderr写同一日志。返回JSON兼容dict：
    子进程退出码、完整CLI墙钟、子进程最大RSS（非同时父子合计）和
    argv哈希；不公开argv或私人路径。没有GPU采样，显存为None。
    非法argv/已有文件直接失败；OS启动错误记录失败收据后再次抛出。
    不把退出0解释为质量通过，由阶段独立质量报告验收。
    """
    if not command or any(not isinstance(x, str) or '\0' in x for x in command): raise ValueError('nonempty string argv required')
    if log_path.exists() or report_path.exists(): raise FileExistsError('fresh timing log/report required')
    report = {'schema_version': 1, 'scope': 'single frozen Python benchmark cold CLI; not end-to-end recon-all',
              'command_sha256': hashlib.sha256(json.dumps(command, ensure_ascii=False).encode()).hexdigest(),
              'timer_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu_memory_bytes': None, 'simultaneous_parent_child_peak_cpu_memory_bytes': None}
    tick = time.perf_counter()
    try:
        with log_path.open('x') as stream:
            child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False)
        report.update(execution_status='complete' if child.returncode == 0 else 'failed', child_exit_code=child.returncode)
    except OSError as error:
        report.update(execution_status='failed_to_start', child_exit_code=None, error_type=type(error).__name__)
        raise
    finally:
        report['cold_cli_wall_seconds'] = time.perf_counter()-tick
        report['max_child_cpu_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log-path', type=Path, required=True); p.add_argument('--report-path', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args(); command = a.command[1:] if a.command[:1] == ['--'] else a.command
    report = time_command(command=command, log_path=a.log_path, report_path=a.report_path)
    raise SystemExit(report['child_exit_code'])


if __name__ == '__main__': main()
