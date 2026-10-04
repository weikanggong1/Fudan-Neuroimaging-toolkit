"""无 GPU 的取消窗口回归：信号发生在 Popen 内，新 controller 必须清理后退出。"""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('metrics_pair',Path(__file__).with_name('run_metrics_roi_pair.py'))
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)


class CancellationTests(unittest.TestCase):
    def test_signal_inside_popen_forwards_after_assignment_and_skips_b(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);handlers={};spawned=[]
            class Child:
                pid=12345
                def __init__(self):self.signals=[];self.alive=True
                def poll(self):return None if self.alive else 143
                def send_signal(self,value):self.signals.append(value)
                def wait(self):
                    # Simulate controller's owned-tree cleanup before returning.
                    if self.signals!=[signal.SIGTERM]:raise AssertionError('new child not cancelled')
                    self.alive=False
                    return 143
            child=Child()
            def register(value,handler):handlers[value]=handler
            def spawn(*args,**kwargs):
                spawned.append(args[0]);handlers[signal.SIGTERM](signal.SIGTERM,None)
                return child
            argv=['run_metrics_roi_pair.py','--run',str(root),'--controller',str(root/'controller.py')]
            with patch.object(sys,'argv',argv),patch.object(pair.signal,'signal',register),patch.object(pair.subprocess,'Popen',spawn):
                result=pair.main()
            report=json.loads((root/'queue.json').read_text())
            self.assertEqual(result,1)
            self.assertEqual(child.signals,[signal.SIGTERM])
            self.assertEqual(len(spawned),1)
            self.assertEqual(report['status'],'cancelled')
            self.assertEqual(report['arms'],[{'name':'A8f','controller_exit_code':143}])
            self.assertEqual(report['cancellation_signals'],[signal.SIGTERM])
            self.assertFalse((root/'B3a.controller.log').exists())
            self.assertFalse(child.alive)


if __name__=='__main__':unittest.main()
