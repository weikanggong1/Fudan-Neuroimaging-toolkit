"""验证并行输入路由的依赖屏障、单次完整分割和创建输出前参数检查。"""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fnit.recon_all import native_free
from fnit.recon_all.batch import run_recon_all_python_batch


@pytest.mark.parametrize('options', [
    {'execution': 'invalid'}, {'n4_backend': 'native'}, {'n4_backend': 'torch'},
    {'n4_execution': 'isolated'}, {'device': 'cpu'}, {'device': 'cuda'},
    {'threads': 3}, {'threads': 8}, {'threads': True},
])
def test_invalid_parallel_request_fails_before_any_output(options, tmp_path):
    defaults = dict(execution='parallel', n4_backend='simpleitk', n4_execution='in-process',
                    device='cuda:1', threads=4)
    defaults.update(options)
    with pytest.raises(ValueError):
        native_free.run_recon_all_python(t1='input', subject_dir=tmp_path/'absent',
            weights_dir='weights', assets_dir='assets', input_execution=defaults.pop('execution'),
            **defaults)
    assert not (tmp_path/'absent').exists()


def test_parallel_rejects_inherited_autocast_before_dispatch(monkeypatch):
    monkeypatch.setattr('fnit.recon_all.profiling.autocast_state', lambda kind: {'enabled':True})
    with pytest.raises(ValueError, match='autocast disabled'):
        native_free._validate_input_execution('parallel',n4_backend='simpleitk',
            n4_execution='in-process',device='cuda:1',threads=4)


def test_public_api_and_cli_forward_parallel_without_changing_serial_default(monkeypatch,tmp_path):
    @contextlib.contextmanager
    def budget(**kwargs):
        yield {'requested_threads':kwargs['threads']}
    monkeypatch.setattr('fnit.recon_all.thread_budget.thread_budget',budget)
    internal = Mock(return_value={'status':'complete','total_seconds':0.})
    monkeypatch.setattr(native_free,'_run_recon_all_python',internal)
    (tmp_path/'subject').mkdir()
    native_free.run_recon_all_python(t1='input',subject_dir=tmp_path/'subject',weights_dir='weights',
        assets_dir='assets',device='cuda:1',threads=4,backend='python',input_execution='parallel')
    assert internal.call_args.kwargs['input_execution']=='parallel'
    assert internal.call_args.kwargs['n4_backend']=='simpleitk'
    api = Mock(return_value={'status':'complete'})
    monkeypatch.setattr(native_free,'run_recon_all_python',api)
    with contextlib.redirect_stdout(io.StringIO()):
        native_free.main(['input','output','--weights-dir','weights','--assets-dir','assets',
                          '--backend','python','--input-execution','parallel'])
    assert api.call_args.kwargs['input_execution']=='parallel'
    with contextlib.redirect_stdout(io.StringIO()):
        native_free.main(['input','output','--weights-dir','weights','--assets-dir','assets'])
    assert api.call_args.kwargs['input_execution']=='serial'


def test_batch_python_effective_profile_reaches_each_cli(monkeypatch,tmp_path):
    raw=tmp_path/'raw.nii.gz';raw.write_bytes(b'unit contract, not real benchmark')
    weights=tmp_path/'weights';assets=tmp_path/'assets';weights.mkdir();assets.mkdir()
    subject=tmp_path/'subject';commands=[]
    def run(command,**kwargs):
        commands.append(command);subject.mkdir()
        (subject/'fnit-native-free-run.json').write_text(json.dumps({'status':'complete'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('fnit.recon_all.batch.subprocess.run',run)
    run_recon_all_python_batch([{'t1':raw,'subject_dir':subject}],weights,assets,
        devices=('cuda:1',),threads=4,backend='python',input_execution='parallel')
    command=commands[0]
    assert command[command.index('--input-execution')+1]=='parallel'
    with pytest.raises(ValueError,match='simpleitk N4'):
        run_recon_all_python_batch([],weights,assets,input_execution='parallel')
    assert len(commands)==1


def run_production_input_slice(tmp_path, execution, monkeypatch, *, fail_group=False):
    """编译生产函数真实输入段，截在GCA前；只mock算子，不编造影像精度。"""
    source=Path(native_free.__file__).read_text();module=ast.parse(source)
    original=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_run_recon_all_python')
    begin=next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign) and
               isinstance(n.targets[0],ast.Name) and n.targets[0].id=='initial')
    end=next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign) and
             isinstance(n.targets[0],ast.Name) and n.targets[0].id=='lta')
    function=ast.FunctionDef(name='input_slice',args=ast.arguments(posonlyargs=[],args=[],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(original.body[begin:end])+
        [ast.parse('return stiv_mm3').body[0]],decorator_list=[])
    compiled=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    subject=tmp_path/'subject';(subject/'mri').mkdir(parents=True)
    (subject/'mri/orig.mgz').write_bytes(b'own-conformed-input')
    stages=[];models=[];report={'precision':{}}
    def n4(**kwargs):
        Path(kwargs['output_path']).write_bytes(b'own-n4');return {'backend':'simpleitk_python'}
    def group(**kwargs):
        assert kwargs['orig_path']==subject/'mri/orig.mgz'
        if fail_group:raise RuntimeError('own group failed')
        n4(input_path=kwargs['orig_path'],output_path=kwargs['n4_output'])
        Path(kwargs['segmentation_output']).write_bytes(b'complete-labels')
        Path(kwargs['volumes_output']).write_bytes(b'complete-volumes')
        return {'workers':{'n4':{'value':{'backend':'simpleitk_python'}},
            'synthseg':{'value':{'total_intracranial_mm3':1234.,'precision':{'forward':'FP32'}}}}}
    def nu(*args,**kwargs):assert (subject/'mri/tmp/nu0.mgz').is_file()
    def model(**kwargs):
        models.append(kwargs)
        def infer(*args,**kwargs):
            return SimpleNamespace(total_intracranial_mm3=1234.,precision={'forward':'FP32'},
                segmentation=SimpleNamespace(save=lambda p:Path(p).write_bytes(b'complete-labels')),
                write_volumes_csv=lambda original,p:Path(p).write_bytes(b'complete-volumes'))
        class Segmenter:
            segmenter=SimpleNamespace(precision={'forward':'FP32'})
            def __call__(self,*args,**kwargs):return infer(*args,**kwargs)
        return Segmenter()
    def stage(name,fn,*args,**kwargs):stages.append(name);return fn(*args,**kwargs)
    monkeypatch.setattr('fnit.recon_all.input_n4_chain.run_n4_stage',n4)
    monkeypatch.setattr('fnit.recon_all.input_stage_parallel.run_input_stage_group',group)
    env=dict(__name__=native_free.__name__,__package__=native_free.__package__,
        stage=stage,run_input_talairach_chain=lambda *a,**k:{'talairach_xfm':'own.xfm','synthstrip':'own-strip'},
        t1='raw',subject=subject,weights=tmp_path/'weights',assets=tmp_path/'assets',device='cuda:1',
        threads=4,profile_stages=False,n4_backend='simpleitk',n4_execution='in-process',
        input_execution=execution,n4_binary=None,report=report,make_nu=nu,
        normalize_t1=lambda *a,**k:None,mask_volume=lambda *a,**k:None,
        normalization_options={},SynthSeg=model,torch=SimpleNamespace(
            device=lambda x:SimpleNamespace(type='cuda'),cuda=SimpleNamespace(empty_cache=Mock())))
    exec(compile(compiled,native_free.__file__,'exec'),env)
    return env['input_slice'],stages,models,report,subject


@pytest.mark.parametrize('execution', ['serial','parallel'])
def test_production_slice_has_one_segmentation_and_correct_join(tmp_path,monkeypatch,execution):
    function,stages,models,report,subject=run_production_input_slice(tmp_path,execution,monkeypatch)
    assert function()==1234.
    assert (subject/'mri/synthseg.rca.mgz').read_bytes()==b'complete-labels'
    assert (subject/'stats/synthseg.vol.csv').read_bytes()==b'complete-volumes'
    assert report['precision']['SynthSeg_actual_forward']=={'forward':'FP32'}
    if execution=='parallel':
        assert models==[]
        assert stages==['input_talairach','n4_synthseg_parallel','nu','T1_normalize','brainmask']
    else:
        assert len(models)==1
        assert stages==['input_talairach','n4','nu','T1_normalize','brainmask','SynthSeg']


def test_group_failure_blocks_dependent_normalization_and_duplicate_segmentation(tmp_path,monkeypatch):
    function,stages,models,report,subject=run_production_input_slice(tmp_path,'parallel',monkeypatch,fail_group=True)
    with pytest.raises(RuntimeError,match='own group failed'):function()
    assert stages==['input_talairach','n4_synthseg_parallel']
    assert not models
