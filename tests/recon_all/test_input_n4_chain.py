"""输入N4链只复用完整实现；路由契约不代替真实影像回归。"""
from pathlib import Path

import pytest

from fnit.recon_all import input_n4_chain as chain
from fnit.recon_all import n4_itk_torch_experimental as complete


def fake_upstream(monkeypatch, directory):
    calls=[]
    def upstream(*args,**kwargs):
        calls.append((args,kwargs));(directory/'mri').mkdir(parents=True)
        return {'talairach_xfm':'self-generated.xfm','upstream':'self'}
    monkeypatch.setattr(chain,'run_input_talairach_chain',upstream)
    monkeypatch.setattr(chain,'make_nu',lambda *args:(1.2,(3,42)))
    return calls


@pytest.mark.parametrize('backend,binary', [('wrong',None),('native',None)])
def test_invalid_backend_fails_before_model_or_output(monkeypatch,tmp_path,backend,binary):
    def should_not_run(*args,**kwargs):raise AssertionError('model loaded before validation')
    monkeypatch.setattr(chain,'run_input_talairach_chain',should_not_run)
    output=tmp_path/'unused'
    with pytest.raises(ValueError):
        chain.run_input_n4_chain(t1='T1.nii.gz',subject_dir=output,weights_dir='weights',
                                assets_dir='assets',n4_backend=backend,n4_binary=binary)
    assert not output.exists()


def test_torch_branch_uses_complete_volume_api(monkeypatch,tmp_path):
    output=tmp_path/'subject';fake_upstream(monkeypatch,output);seen={}
    def full(**kwargs):
        seen.update(kwargs);return {'iterations':200,'backend':'full-fixed-recipe','dtype':'uint8'}
    monkeypatch.setattr(complete,'correct_volume',full)
    result=chain.run_input_n4_chain(t1='raw.nii.gz',subject_dir=output,weights_dir='weights',
                                  assets_dir='assets',device='cuda:0',threads=4,n4_backend='torch',profile=True)
    assert seen=={'input_path':output/'mri/orig.mgz','output_path':output/'mri/tmp/nu0.mgz',
                  'device':'cuda:0','profile':True}
    assert result['n4_details']['iterations']==200
    assert result['total_seconds']>=result['n4_seconds']+result['n4_wrapper_seconds']
    assert result['production_default_changed'] is False


def test_native_default_still_fixed_single_thread(monkeypatch,tmp_path):
    output=tmp_path/'subject';fake_upstream(monkeypatch,output);seen={}
    monkeypatch.setattr(chain,'correct_volume',lambda **kwargs:seen.update(kwargs))
    result=chain.run_input_n4_chain(t1='raw.nii.gz',subject_dir=output,weights_dir='weights',
                                  assets_dir='assets',n4_binary='own-conda-n4',threads=4)
    assert seen['binary']=='own-conda-n4'
    assert seen['reconstruction_threads']==1 and seen['profile_path'] is None
    assert result['n4_backend']=='native'
    assert result['n4_details']['backend']=='native_conda_itk'
