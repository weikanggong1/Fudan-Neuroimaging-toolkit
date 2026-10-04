"""暂缓本轮整例队列父调度器，当前子计算继续；阶段退出后自动恢复。

只向launch receipt中command完全匹配、独立session且PID/starttime一致的
本轮run_whole_queue或run_resource_replay_queue父进程发送STOP/CONT；不操作子进程、共享锁或他人任务。
输入两个私有launch JSON、待观察诊断PID、新输出目录、最长等待秒数。
输出launch/status/completion JSON；退出诊断不表示通过，结果另行检查。
只影响排队顺序，当前整例外部monitor墙钟与算法计时不变。父队列的wrapper
elapsed会含调度暂缓，不能作为算法墙钟。无官方等价命令，不是生产算法。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    """返回Linux进程启动tick，消失时None；用于避免PID复用。"""
    try:
        stat = (Path('/proc') / str(pid) / 'stat').read_text()
        return stat.rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def write(path, value):
    """原子替换诊断状态文件，读者不会看到半份JSON。"""
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def prioritize(*, launches, watch_pids, output_directory, timeout_seconds=21600):
    """最多等待6小时；优先诊断，不推断其正确性；总返回/文件为状态字典。"""
    if timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be positive')
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    drivers = []
    for path in launches:
        config = json.loads(Path(path).read_text())
        pid = int(config['pid'])
        tick = identity(pid)
        if tick is None:
            raise RuntimeError('queue driver no longer present')
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        args = [a.decode() for a in args if a]
        if args != config['command'] or os.getsid(pid) != pid:
            raise ValueError('queue driver identity differs')
        if not any('accuracy_20261003' in a and a.endswith(('run_whole_queue.py', 'run_resource_replay_queue.py')) for a in args):
            raise ValueError('only this round whole-queue drivers can be gated')
        drivers.append({'pid': pid, 'starttime': tick, 'launch': str(path)})
    watched = {str(pid): identity(pid) for pid in set(watch_pids)}
    stopped, started = [], time.monotonic()
    report = {'status': 'starting', 'drivers': drivers, 'diagnostics': watched,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'queue parents only; existing child commands continue',
              'diagnostic_success': 'not_inferred', 'timeout_seconds': timeout_seconds}
    write(output / 'launch.json', report)
    def interrupted(signum, frame):
        raise SystemExit('priority guard interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for driver in drivers:
            if identity(driver['pid']) != driver['starttime']:
                raise RuntimeError('queue exited before gating')
            os.kill(driver['pid'], signal.SIGSTOP)
            stopped.append(driver)
        while True:
            alive = [pid for pid, tick in watched.items()
                     if tick is not None and identity(int(pid)) == tick]
            report.update(status='diagnostics_priority', remaining_pids=alive,
                          elapsed_seconds=time.monotonic() - started)
            write(output / 'status.json', report)
            if not alive:
                report['resume_reason'] = 'observed diagnostic processes exited; check results separately'
                break
            if time.monotonic() - started >= timeout_seconds:
                report['resume_reason'] = 'priority timeout; diagnostics may remain incomplete'
                break
            time.sleep(5)
    finally:
        for driver in stopped:
            if identity(driver['pid']) == driver['starttime']:
                os.kill(driver['pid'], signal.SIGCONT)
        report.update(status='queue_drivers_resumed', finished_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started)
        write(output / 'completion.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launches', type=Path, nargs='+', required=True)
    parser.add_argument('--watch-pids', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=21600)
    args = parser.parse_args()
    print(json.dumps(prioritize(launches=args.launches, watch_pids=args.watch_pids,
                               output_directory=args.output, timeout_seconds=args.timeout_seconds)))
