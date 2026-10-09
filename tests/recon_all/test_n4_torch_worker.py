"""完整N4 exec契约：父缓存/精度保留和文件哈希；不替代真实N4回归。"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from fnit.recon_all import n4_torch_worker as worker


@pytest.mark.parametrize('device,threads',[('cpu',4),('cuda',4),('cuda:0',False)])
def test_invalid_execution_parameters(tmp_path,device,threads):
    source=tmp_path/'orig.mgz';source.write_bytes(b'only contract placeholder')
    with pytest.raises(ValueError):
        worker.run_isolated_n4(input_path=source,output_path=tmp_path/'nu0.mgz',
            report_path=tmp_path/'n4.json',device=device,threads=threads)


def test_cached_exec_preserves_parent_policy_and_binds_hashes(monkeypatch,tmp_path):
    source=tmp_path/'orig.mgz';source.write_bytes(b'input contract only')
    output=tmp_path/'nu0.mgz';report=tmp_path/'n4.json'
    monkeypatch.setenv('PYTORCH_NO_CUDA_MEMORY_CACHING','1')
    policy=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    seen={}
    def fake_run(command,env,check):
        seen.update(command=command,env=env,check=check)
        output.write_bytes(b'output contract only')
        report.write_text(json.dumps({'input_sha256':worker._sha(source),'output_sha256':worker._sha(output),
                                     'api':{'iterations':200}}))
    monkeypatch.setattr(worker.subprocess,'run',fake_run)
    result=worker.run_isolated_n4(input_path=source,output_path=output,report_path=report,
        device='cuda:0',threads=4,profile=True)
    assert seen['check'] is True and seen['command'][1]=='-c'
    assert 'PYTORCH_NO_CUDA_MEMORY_CACHING' not in seen['env']
    assert os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']=='1'
    assert policy==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    assert '--profile' in seen['command'] and result['api']['iterations']==200
    assert result['isolated_cli_wall_seconds']>=0


def test_worker_cannot_change_initialized_cuda(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    with pytest.raises(ValueError,match='before CUDA'):
        worker.run_worker(input_path=tmp_path/'orig',output_path=tmp_path/'nu0',report_path=tmp_path/'report',device='cuda:0')


def test_output_hash_mismatch_is_not_a_success(monkeypatch,tmp_path):
    source=tmp_path/'orig';source.write_bytes(b'input contract only')
    output=tmp_path/'output';report=tmp_path/'report'
    def fake_run(*args,**kwargs):
        output.write_bytes(b'changed output')
        report.write_text(json.dumps({'input_sha256':worker._sha(source),'output_sha256':'incorrect'}))
    monkeypatch.setattr(worker.subprocess,'run',fake_run)
    with pytest.raises(ValueError,match='hash mismatch'):
        worker.run_isolated_n4(input_path=source,output_path=output,report_path=report,device='cuda:0')
