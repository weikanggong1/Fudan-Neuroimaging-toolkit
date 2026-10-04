"""调度单测；模拟数字不作为真实影像 benchmark。"""
import importlib.util
from pathlib import Path
import tempfile
import json
import sys
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('admission',Path(__file__).with_name('resource_admission.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class AdmissionTests(unittest.TestCase):
    def test_legacy_tool_command_and_explicit_external_source_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config={'code_root':str(root/'source'), 'python':sys.executable}
            argv=module.benchmark_command(config,root/'attempt')
            self.assertEqual(argv[1],str(root/'source/validation/recon_all/python_gpu_port/run_monitored.py'))
            self.assertEqual(argv[8],str(root/'source/validation/recon_all/optimizations/20261002_parallel/execute_whole_case.py'))
            for key in ('code_root','weights','assets','native_bin_dir'):
                config[key]=str(root/key);Path(config[key]).mkdir()
            raw=root/'raw.nii';raw.write_bytes(b'fixture');config['input']=str(raw)
            tools={}
            for role in ('monitor','whole_case_driver'):
                path=root/(role+'.py');path.write_text('# verified tool '+role)
                tools[role]={'path':str(path),'sha256':module.digest(path)}
            config['benchmark_tools']=tools
            frozen=module.inventory(config)
            for row in tools.values():self.assertEqual(frozen[row['path']],row['sha256'])
            argv=module.benchmark_command(config,root/'attempt')
            self.assertEqual(argv[1],tools['monitor']['path'])
            self.assertEqual(argv[8],tools['whole_case_driver']['path'])
            self.assertEqual(argv[3],module.GPU_UUID)
            self.assertEqual(argv[-1],str(root/'attempt/retry_config.json'))
            Path(tools['monitor']['path']).write_text('# changed tool')
            with self.assertRaisesRegex(ValueError,'SHA changed'):module.inventory(config)
            with self.assertRaisesRegex(ValueError,'SHA changed'):module.benchmark_command(config,root/'attempt')

    def test_external_tools_reject_incomplete_relative_or_unfrozen_declarations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'tool.py';p.write_text('# fixture')
            row={'path':str(p),'sha256':module.digest(p)}
            config={'code_root':str(root/'source'),'benchmark_tools':{'monitor':row}}
            with self.assertRaisesRegex(ValueError,'exactly'):module.benchmark_tool_paths(config)
            config['benchmark_tools']={'monitor':row,'whole_case_driver':{'path':'relative.py','sha256':row['sha256']}}
            with self.assertRaisesRegex(ValueError,'absolute'):module.benchmark_tool_paths(config)
            config['benchmark_tools']['whole_case_driver']={'path':str(p),'sha256':'unfrozen'}
            with self.assertRaisesRegex(ValueError,'frozen SHA'):module.benchmark_tool_paths(config)
            config['benchmark_tools']=None
            with self.assertRaisesRegex(ValueError,'exactly'):module.benchmark_tool_paths(config)

    def test_exact_decimal_boundary(self):
        self.assertFalse(module.admitted(dict(gpu_uuid=module.GPU_UUID,free_bytes=19_999_999_999)))
        self.assertTrue(module.admitted(dict(gpu_uuid=module.GPU_UUID,free_bytes=20_000_000_000)))
        self.assertFalse(module.admitted(dict(gpu_uuid='other',free_bytes=80_000_000_000)))

    def test_query_units_and_wrong_uuid(self):
        class Result:
            stdout=module.GPU_UUID+', 81000, 80500, 500\n'
        with patch.object(module.subprocess,'run',return_value=Result()):
            sample=module.query_gpu()
        self.assertEqual(sample['free_bytes'],500*1048576)
        Result.stdout='other, 81000, 0, 81000\n'
        with patch.object(module.subprocess,'run',return_value=Result()):
            with self.assertRaises(ValueError): module.query_gpu()

    def test_retry_preserves_entry_and_protected_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=dict(gpu_uuid=module.GPU_UUID,invocation='initialized_cuda_api',output=str(root/'old'),
                        diagnostic_root=str(root/'old_diag'),input_sha256='same',pipeline_kwargs={'hemisphere_workers':2})
            retry=module.retry_config(config,root/'retry')
            self.assertEqual({k:v for k,v in config.items() if k not in ('output','diagnostic_root')},
                             {k:v for k,v in retry.items() if k not in ('output','diagnostic_root')})
            with self.assertRaises(ValueError):module.retry_config(config,root/'old'/'nested')
            (root/'existing').mkdir()
            with self.assertRaises(FileExistsError):module.retry_config(config,root/'existing')

    def test_under_lock_recheck_releases_without_launch(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'old_diag').mkdir()
            config=dict(gpu_uuid=module.GPU_UUID,invocation='cli',output=str(root/'old'),
                        diagnostic_root=str(root/'old_diag'),input='unused',input_sha256='same',
                        code_root=str(root/'source'))
            (root/'config.json').write_text(json.dumps(config))
            (root/'old_diag'/'launch.json').write_text(json.dumps(dict(config,candidate_native_free_sha256='same')))
            sufficient=dict(gpu_uuid=module.GPU_UUID,free_bytes=20_000_000_000)
            insufficient=dict(gpu_uuid=module.GPU_UUID,free_bytes=1)
            for name, second, expected in [('low',insufficient,'wait_timeout'),
                                            ('error',RuntimeError('GPU query failed'),'query_or_validation_failed'),
                                            ('signal',module.Interrupted('signal 15'),'interrupted')]:
                argv=['admission','--config',str(root/'config.json'),'--retry-root',str(root/name),
                      '--report',str(root/(name+'.json')),'--lock',str(root/'lock'), '--maximum-wait-seconds','0']
                with patch.object(sys,'argv',argv), patch.object(module,'digest',return_value='same'), \
                     patch.object(module,'inventory',return_value={'program':'same'}), \
                     patch.object(module,'query_gpu',side_effect=[sufficient,second]), \
                     patch.object(module.subprocess,'Popen') as popen:
                    module.main()
                    popen.assert_not_called()
                self.assertEqual(json.loads((root/(name+'.json')).read_text())['status'],expected)
                with (root/'lock').open('a+') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fcntl.flock(lock,fcntl.LOCK_UN)


class ProcessTreeTests(unittest.TestCase):
    def test_cleanup_independent_session_before_unlock(self):
        import subprocess
        import time
        import fcntl
        import os
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            # Grandchild and worker ignore TERM to exercise bounded KILL.
            grandchild="import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"
            worker=("import subprocess,signal,time,pathlib,os; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    "p=subprocess.Popen(["+repr(sys.executable)+",'-c',"+repr(grandchild)+"]); "
                    "pathlib.Path("+repr(str(root/'pids'))+").write_text(str(os.getpid())+' '+str(p.pid)); "
                    "time.sleep(120)")
            monitor=("import subprocess,time; subprocess.Popen(["+repr(sys.executable)+",'-c',"+repr(worker)+"],start_new_session=True); time.sleep(120)")
            outsider=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True)
            child=subprocess.Popen([sys.executable,'-c',monitor],start_new_session=True)
            owned={}
            try:
                deadline=time.monotonic()+5
                while not (root/'pids').exists() and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue((root/'pids').exists())
                worker_pid,grandchild_pid=map(int,(root/'pids').read_text().split())
                # Let grandchild install TERM handler before sending signals.
                time.sleep(.1)
                module.snapshot_descendants([child.pid],owned)
                self.assertIn(worker_pid,owned)
                self.assertIn(grandchild_pid,owned)
                self.assertNotEqual(owned[worker_pid]['pgid'],owned[child.pid]['pgid'])
                report={}
                checks=[]
                with (root/'lock').open('a+') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    def checkpoint():
                        # Independent subprocess must not acquire the lock while
                        # cleanup is in progress, including after monitor exits.
                        command="import fcntl; f=open("+repr(str(root/'lock'))+",'a+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
                        result=subprocess.run([sys.executable,'-c',command],capture_output=True)
                        checks.append(result.returncode)
                        self.assertNotEqual(result.returncode,0)
                    # Exercise gpucw1's legacy-kernel path without pidfd.
                    with patch.object(module.os,'pidfd_open',None):
                        module.cleanup_owned_tree(child,owned,report,checkpoint,term_seconds=.2)
                    self.assertEqual(module.active_owned(owned),[])
                    self.assertTrue(report['cleanup_kill_required'])
                    fcntl.flock(lock,fcntl.LOCK_UN)
                self.assertGreaterEqual(len(checks),2)
                child.wait(timeout=5)
                self.assertEqual(report['cleanup_status'],'all_owned_computation_exited')
                self.assertIsNone(outsider.poll())  # Unrelated session remains running.
            finally:
                module.snapshot_descendants([child.pid,*owned],owned)
                for identity in module.active_owned(owned):
                    module.signal_owned(identity,9)
                child.wait(timeout=5)
                outsider.terminate()
                outsider.wait(timeout=5)

    def test_reused_pid_is_not_signaled(self):
        import os
        identity=module.process_identity(os.getpid())
        identity=dict(identity,starttime=identity['starttime']+1)
        with patch.object(module.os,'kill') as kill:
            module.signal_owned(identity,15)
            kill.assert_not_called()

if __name__=='__main__':unittest.main()
