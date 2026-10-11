"""API作用域合同；只替换内部算法入口，不模拟MRI benchmark。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import importlib
import numba
import torch


def test_prefix_api_restores_threads_and_backend_on_success_and_failure():
    module=importlib.import_module('fnit.recon_all.volume_prefix_python')
    original=module._caller_backend_state()
    original_numba=numba.get_num_threads();original_torch=torch.get_num_threads()
    try:
        numba.set_num_threads(2);torch.set_num_threads(2)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic=False
        expected=module._caller_backend_state()
        for failed in (False,True):
            def internal(*args,**kwargs):
                assert torch.get_num_threads()==4 and numba.get_num_threads()==4
                assert module._caller_backend_state()==expected
                torch.set_float32_matmul_precision('highest')
                torch.backends.cudnn.allow_tf32=True
                torch.backends.cudnn.benchmark=False
                torch.backends.cudnn.deterministic=True
                if failed:raise RuntimeError('injected algorithm failure')
                return {'status':'completed','precision':{},'total_seconds_including_IO':0.,'stages':[]}
            with TemporaryDirectory() as directory,patch.object(module,'_run_volume_prefix',side_effect=internal):
                if failed:
                    try:module.run_volume_prefix('unused',directory,'unused','unused')
                    except RuntimeError as error:assert str(error)=='injected algorithm failure'
                    else:raise AssertionError('algorithm failure suppressed')
                    assert not (Path(directory)/'volume-prefix.private.json').exists()
                else:
                    report=module.run_volume_prefix('unused',directory,'unused','unused')
                    assert report['thread_budget']['restoration_complete']
                    assert report['precision']['caller_backend_restored']
                    assert report['compute_seconds_after_resource_verification']==0.
                    assert report['total_seconds_including_IO']>0.
            assert torch.get_num_threads()==2 and numba.get_num_threads()==2
            assert module._caller_backend_state()==expected
    finally:
        numba.set_num_threads(original_numba);torch.set_num_threads(original_torch)
        torch.set_float32_matmul_precision(original['matmul_precision'])
        torch.backends.cudnn.allow_tf32=original['cudnn_tf32']
        torch.backends.cudnn.enabled=original['cudnn_enabled']
        torch.backends.cudnn.benchmark=original['cudnn_benchmark']
        torch.backends.cudnn.deterministic=original['cudnn_deterministic']


def test_prefix_failure_marks_only_a_new_receipt_and_preserves_old_receipt():
    import json
    module=importlib.import_module('fnit.recon_all.volume_prefix_python')
    for existing in (False,True):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'volume-prefix.private.json'
            old='{"status":"older_completed_receipt"}\n'
            if existing:path.write_text(old)
            def internal(*args,**kwargs):
                if not existing:path.write_text(json.dumps({'status':'completed','precision':{},'total_seconds_including_IO':0.}))
                raise RuntimeError('injected boundary failure')
            with patch.object(module,'_run_volume_prefix',side_effect=internal):
                try:module.run_volume_prefix('unused',directory,'unused','unused')
                except RuntimeError:pass
                else:raise AssertionError('failure suppressed')
            if existing:assert path.read_text()==old
            else:
                report=json.loads(path.read_text())
                assert report['status']=='failed' and report['failed_stage']=='api_boundary'
                assert report['precision']['caller_backend_restored']
