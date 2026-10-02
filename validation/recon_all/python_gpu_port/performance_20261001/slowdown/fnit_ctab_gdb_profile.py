# coding: utf-8
"""仅对本次自建 WM 子进程计数 CTABunique 与重复颜色检查；不修改寄存器或输入。"""
import gdb
import json
import os
import time

clock = getattr(time, 'monotonic', time.time)
report = {'clock': 'monotonic' if hasattr(time, 'monotonic') else 'realtime', 'calls': []}
active = []


class EndUnique(gdb.FinishBreakpoint):
    def __init__(self, row):
        gdb.FinishBreakpoint.__init__(self, gdb.newest_frame(), internal=True)
        self.silent = True
        self.row = row

    def stop(self):
        self.row['seconds'] = clock() - self.row.pop('started')
        self.row['return_value'] = str(self.return_value)
        active.pop()
        return False


class StartUnique(gdb.Breakpoint):
    def stop(self):
        row = {'started': clock(), 'repeat_checks': 0,
               'caller': gdb.newest_frame().older().name()}
        report['calls'].append(row)
        active.append(row)
        EndUnique(row)
        return False


class CountRepeats(gdb.Breakpoint):
    def stop(self):
        if active:
            active[-1]['repeat_checks'] += 1
        return False


def save(event):
    report['inferior_exit_code'] = getattr(event, 'exit_code', None)
    with open(os.environ['FNIT_CTAB_PROFILE'], 'w') as stream:
        json.dump(report, stream, indent=2)


gdb.execute('set pagination off')
gdb.execute('set confirm off')
StartUnique('CTABunique', internal=True).silent = True
CountRepeats('CTABcountRepeats', internal=True).silent = True
gdb.events.exited.connect(save)
