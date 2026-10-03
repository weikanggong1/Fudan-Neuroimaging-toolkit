"""CPU provenance guard tests, not MRI benchmarks."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('official_chain', Path(__file__).with_name('official_modeling_cohort_cpu_v3.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def contract(tmp_path):
    files = {}
    for key in ('dwi','bvecs','bvals','topup_fieldcoef','topup_movpar','eddy_mask','upstream_report'):
        p = tmp_path / key; p.write_text(key); files[key] = str(p)
    p = tmp_path / 'ready.json'
    d = {'kind':'independent_official_rawprep', 'completed':True, 'subject':'CON01',
         'files':files, 'sha256':{k:m.sha(v) for k,v in files.items()}}
    p.write_text(json.dumps(d))
    return p, d


def test_pending_upstream_is_not_completed(tmp_path):
    p = tmp_path / 'pending.json'; p.write_text(json.dumps({'completed':False}))
    assert m.ready_contract(p, []) is None


def test_changed_input_digest_is_rejected(tmp_path):
    p, d = contract(tmp_path)
    Path(d['files']['dwi']).write_text('different official DWI')
    with pytest.raises(ValueError, match='readiness SHA differs'):
        m.ready_contract(p, [])


def test_symlink_to_old_fnit_namespace_is_rejected(tmp_path):
    old = tmp_path / 'old_fnit'; old.mkdir(); (old/'dwi').write_text('old')
    fresh = tmp_path / 'fresh'; fresh.mkdir(); p,d = contract(fresh)
    Path(d['files']['dwi']).unlink(); Path(d['files']['dwi']).symlink_to(old/'dwi')
    d['sha256']['dwi'] = m.sha(old/'dwi'); p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='forbidden old/FNIT'):
        m.ready_contract(p, [str(old)])


def raw_fixture(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    relative = {}
    for index in range(10):
        path = raw / f'canonical_{index}'; path.write_text(str(index))
        relative[path.name] = m.sha(path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'subject':'CON01','input_sha256':relative}]}))
    report = tmp_path / 'upstream.json'
    expected = {str(raw/name):h for name,h in relative.items()}
    report.write_text(json.dumps({'subject':'CON01','input_sha256':expected}))
    return {'files':{'upstream_report':str(report)}}, {'manifest':str(manifest),'raw_root':str(raw)}, report, expected


def test_failed_upstream_does_not_wait_forever(tmp_path):
    path = tmp_path/'failed.json'; path.write_text(json.dumps({'completed':False,'state':'failed'}))
    with pytest.raises(ValueError, match='upstream official rawprep failed'):
        m.ready_contract(path, [])


def test_exact_absolute_canonical_ten_file_binding(tmp_path):
    contract,config,report,expected = raw_fixture(tmp_path)
    result = m.validate_raw_binding(contract, config)
    assert result['canonical_raw_input_sha256'] == expected
    data = json.loads(report.read_text()); data['input_sha256'].pop(next(iter(expected))); report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='exact canonical ten-file'):
        m.validate_raw_binding(contract, config)


def test_raw_mutation_rejected_after_matching_report(tmp_path):
    contract,config,report,expected = raw_fixture(tmp_path)
    Path(next(iter(expected))).write_text('changed raw')
    with pytest.raises(ValueError, match='raw acquisition changed'):
        m.validate_raw_binding(contract, config)


def cpu_report(root):
    binary = '/public/software/apps/FSL/6.0.7.4/bin/eddy_cpu'
    paths = {'imain':root/'raw/AP.nii.gz','mask':root/'mask/nodif_brain_mask.nii.gz',
             'acqp':root/'topup/acqparams.txt','index':root/'eddy/eddy_index.txt',
             'bvecs':root/'raw/AP.bvec','bvals':root/'raw/AP.bval',
             'topup':root/'topup/fieldmap_out','out':root/'eddy/data'}
    command = [binary,*[f'--{k}={v}' for k,v in paths.items()],*m.EDDY_SCIENTIFIC_FLAGS,'--ref_scan_no=76']
    record = {'stage':'official_EDDY_CPU','returncode':0,'program_sha256':m.EDDY_BINARY_SHA['cpu'],
              'command':command,'GPU':False,'env':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'},
              'host':'CPU provenance test fixture, not benchmark','wall_seconds':1.0}
    return {'EDDY_solver':'cpu','GPU_UUID':None,'gp_seed':12345,'threads':8,
            'selection':{'ap_index':76},'commands':[{'stage':'official_topup','returncode':0},
            {'stage':'official_synthstrip_CPU','returncode':0},record]}


def test_actual_installed_cpu_binary_and_labels_accepted(tmp_path):
    report = cpu_report(tmp_path)
    actual = m.validate_eddy_solver(report,tmp_path)
    assert actual['solver']=='cpu' and actual['stage']=='official_EDDY_CPU'
    assert actual['GPU_UUID'] is None and actual['binary_sha256']==m.EDDY_BINARY_SHA['cpu']


@pytest.mark.parametrize('mutation',['uuid','missing_uuid','gpu_stage','cuda_env','binary'])
def test_cpu_cannot_be_mislabelled_as_gpu(tmp_path,mutation):
    report=cpu_report(tmp_path);stage=report['commands'][-1]
    if mutation=='uuid':report['GPU_UUID']='GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
    elif mutation=='missing_uuid':report.pop('GPU_UUID')
    elif mutation=='gpu_stage':stage['GPU']=True
    elif mutation=='cuda_env':stage['env']['CUDA_VISIBLE_DEVICES']='0'
    else:stage['command'][0]='/public/software/apps/FSL/6.0.7.4/bin/eddy_cuda10.2'
    with pytest.raises(ValueError):m.validate_eddy_solver(report,tmp_path)


def test_cpu_scientific_flags_cannot_change_or_duplicate(tmp_path):
    report=cpu_report(tmp_path);command=report['commands'][-1]['command']
    command[command.index('--niter=8')]='--niter=4'
    with pytest.raises(ValueError,match='scientific flag'):m.validate_eddy_solver(report,tmp_path)
    command[command.index('--niter=4')]='--niter=8';command.append('--initrand=12345')
    with pytest.raises(ValueError,match='duplicates'):m.validate_eddy_solver(report,tmp_path)


def test_cpu_completed_sidecar_required(tmp_path):
    with pytest.raises(ValueError,match='completed_contract missing'):
        m.validate_cpu_completed_sidecar(tmp_path/'report.json',{},tmp_path)


def test_inactive_fallback_cannot_create_new_namespace(tmp_path):
    output=tmp_path/'new_modeling'
    with pytest.raises(ValueError,match='activation has not occurred'):
        m.cpu_activation_preflight({'CPU_fallback_activation_report':str(tmp_path/'absent.json'),
                                    'output_root':str(output)})
    assert not output.exists()


def test_pending_aligned_binding_is_rejected():
    with pytest.raises(ValueError,match='pending actual aligned'):
        m.validate_case_binding({'binding_ready':False},{},{})


def aligned_binding(tmp_path):
    report=tmp_path/'report.json'
    report.write_text(json.dumps({'subject':'CON04','selection':{'ap_index':26,'pa_index':0}}))
    case={'subject':'CON04','binding_ready':True,'ready_contract':str(report),
          'upstream_runner_sha256':'producer_A','expected_selection':{'ap_index':26,'pa_index':0},
          'expected_upstream_solver':'cpu'}
    config={'manifest_sha256':'manifest','raw_root':'canonical_raw'}
    freeze={'manifest_sha256':'manifest','raw_root':'canonical_raw','source_sha256':{'runner':'producer_A'}}
    contract={'files':{'upstream_report':str(report)},'actual_EDDY':{'solver':'cpu'}}
    return case,config,freeze,contract,report


def test_explicit_case_producer_solver_and_raw_frame_binding(tmp_path):
    case,config,freeze,contract,report=aligned_binding(tmp_path)
    assert m.validate_case_binding(case,config,freeze,contract)
    report.write_text(json.dumps({'subject':'CON04','selection':{'ap_index':0,'pa_index':0}}))
    with pytest.raises(ValueError,match='raw-frame selection differs'):
        m.validate_case_binding(case,config,freeze,contract)


@pytest.mark.parametrize('mutation',['producer','solver','subject','missing_selection'])
def test_aligned_case_binding_rejects_wrong_lineage(tmp_path,mutation):
    case,config,freeze,contract,report=aligned_binding(tmp_path)
    if mutation=='producer':freeze['source_sha256']['runner']='producer_B'
    elif mutation=='solver':contract['actual_EDDY']['solver']='gpu'
    elif mutation=='subject':case['subject']='CON05'
    else:case['expected_selection']=None
    with pytest.raises(ValueError):m.validate_case_binding(case,config,freeze,contract)


def budget_stop_proof(tmp_path):
    report=tmp_path/'failed_GPU.json'
    report.write_text(json.dumps({'completed':False,'state':'failed','commands':[{
        'stage':'official_EDDY_GPU','returncode':-15,'budget_exceeded':True,
        'sampled_GPU_process_peak_bytes':43203428352}]}))
    policy='explicit_authorized_GPU_budget_stop_CPU_reference'
    marker={'activated':True,'solver':'official_CPU','runner_sha256':'producer',
            'activation_reason':policy,'verified_failed_GPU_attempts':[{'report':str(report),'sha256':m.sha(report)}]}
    config={'upstream_runner_sha256':'producer','activation_policy':policy,
            'explicit_CPU_budget_reference_authorization':True}
    return marker,config,report


def test_explicit_authorized_budget_stop_is_separate_from_oom(tmp_path):
    marker,config,report=budget_stop_proof(tmp_path)
    assert m.validate_activation_marker(marker,config)==config['activation_policy']
    config['explicit_CPU_budget_reference_authorization']=False
    with pytest.raises(ValueError,match='authorization'):m.validate_activation_marker(marker,config)


@pytest.mark.parametrize('mutation',['not_budget_stop','wrong_returncode','below_budget','wrong_reason'])
def test_budget_reference_rejects_other_activation_causes(tmp_path,mutation):
    marker,config,report=budget_stop_proof(tmp_path);data=json.loads(report.read_text())
    if mutation=='wrong_reason':marker['activation_reason']='OOM'
    elif mutation=='wrong_returncode':data['commands'][0]['returncode']=1
    elif mutation=='not_budget_stop':data['commands'][0]['budget_exceeded']=False
    else:data['commands'][0]['sampled_GPU_process_peak_bytes']=2*1024**3
    report.write_text(json.dumps(data));marker['verified_failed_GPU_attempts'][0]['sha256']=m.sha(report)
    with pytest.raises(ValueError):m.validate_activation_marker(marker,config)


def test_budget_report_waits_for_actual_verified_sidecar(tmp_path):
    report=tmp_path/'report.json'
    report.write_text(json.dumps({'completed':True,'scope':'official CPU reference: restored own successful raw TOPUP/SynthStrip lineage plus new CPU8 EDDY'}))
    assert m.ready_contract(report,[]) is None


def test_actual_budget_activation_SHA_cannot_change(tmp_path):
    activation=tmp_path/'activation.json';activation.write_text('{}')
    config={'upstream_activation_format':'official_CPU_budget_reference_v1',
            'CPU_fallback_activation_report':str(activation),'upstream_activation_sha256':'wrong'}
    with pytest.raises(ValueError,match='activation SHA changed'):m.read_bound_upstream_freeze({},config)


def test_restored_CPU_source_labels_cannot_claim_rerun(tmp_path):
    source=tmp_path/'old';source.mkdir();report=source/'report.json';report.write_text('{}')
    lineage={'source':str(source),'report':str(report),'report_sha256':m.sha(report),
             'CPU_stage_reexecuted':True,'restored_successful_own_official_TOPUP_and_SynthStrip':True}
    with pytest.raises(ValueError,match='source report/labels changed'):
        m.validate_restored_CPU_lineage({'source_CPU_stage_lineage':lineage},tmp_path)


def test_completed_budget_report_requires_exact_activation_source_SHA(tmp_path):
    case,config,freeze,contract,report=aligned_binding(tmp_path)
    config['upstream_activation_format']='official_CPU_budget_reference_v1'
    with pytest.raises(ValueError,match='producer/helper SHA differs'):
        m.validate_case_binding(case,config,freeze,contract)
    d=json.loads(report.read_text());d['source_sha256']=freeze['source_sha256'];report.write_text(json.dumps(d))
    assert m.validate_case_binding(case,config,freeze,contract)


def test_immutable_pending_alignment_sidecar_is_write_once(tmp_path):
    path=tmp_path/'bindings/sub-CON08.alignment_binding.json'
    evidence={'subject':'CON08','actual_selection':{'ap_index':52,'pa_index':0},'upstream_report_sha256':'actual_report_digest'}
    first=m.write_once_alignment_binding(path,evidence);before=path.stat().st_mtime_ns
    assert m.write_once_alignment_binding(path,evidence)['sha256']==first['sha256']
    assert path.stat().st_mtime_ns==before
    with pytest.raises(ValueError,match='Immutable pending alignment'):
        m.write_once_alignment_binding(path,dict(evidence,upstream_report_sha256='changed'))
    assert path.stat().st_mtime_ns==before


def test_pending_case_does_not_promote_prepared_or_unverified_report(tmp_path):
    producer=tmp_path/'same_producer';root=producer/'sub-CON08';root.mkdir(parents=True)
    report=root/'report.json'
    case={'subject':'CON08','binding_ready':False,'ready_contract':str(report),'upstream_runner_sha256':'pinned'}
    config={'upstream_activation_format':'official_CPU_budget_reference_v1','upstream_producer_root':str(producer),
            'upstream_runner_sha256':'pinned','forbidden_input_roots':[]}
    report.write_text(json.dumps({'completed':False,'CPU_prepared':True}))
    assert m.resolve_case_binding(case,config)==(None,None)
    report.write_text(json.dumps({'completed':True,'scope':'official CPU reference: restored own successful raw TOPUP/SynthStrip lineage'}))
    assert m.resolve_case_binding(case,config)==(None,None)
    assert 'expected_selection' not in case


@pytest.mark.parametrize('mutation',['report_path','producer_SHA'])
def test_pending_case_cannot_switch_producer_or_old_namespace(tmp_path,mutation):
    producer=tmp_path/'same_producer';case={'subject':'CON08','binding_ready':False,
        'ready_contract':str(producer/'sub-CON08/report.json'),'upstream_runner_sha256':'pinned'}
    config={'upstream_activation_format':'official_CPU_budget_reference_v1','upstream_producer_root':str(producer),'upstream_runner_sha256':'pinned'}
    if mutation=='report_path':case['ready_contract']=str(tmp_path/'old_v2/sub-CON08/report.json')
    else:case['upstream_runner_sha256']='other_producer'
    with pytest.raises(ValueError,match='same explicit CPU budget producer'):
        m.resolve_case_binding(case,config)


def test_true_verified_pending_contract_creates_exact_immutable_binding(tmp_path,monkeypatch):
    """Small provenance fixture exercises full guards; it is not an MRI benchmark."""
    import shutil
    import nibabel as nib
    import numpy as np
    raw=tmp_path/'canonical_raw';raw.mkdir();affine=np.eye(4)
    for stem,offset in [('AP',1.),('PA',3.)]:
        data=np.full((2,2,2,2),offset,dtype=np.float32);data[...,1]+=2
        nib.save(nib.Nifti1Image(data,affine),raw/f'{stem}.nii.gz')
        np.savetxt(raw/f'{stem}.bval',[[0,1000]]);np.savetxt(raw/f'{stem}.bvec',np.array([[0,1],[0,0],[0,0]]))
    for index in range(4):(raw/f'other{index}').write_text(str(index))
    raw_hashes={str(path):m.sha(path) for path in raw.iterdir()};assert len(raw_hashes)==10
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'cases':[{'subject':'CON08','input_sha256':{Path(k).name:v for k,v in raw_hashes.items()}}]}))
    source_root=tmp_path/'own_official_heldout';source=source_root/'sub-CON08';source.mkdir(parents=True)
    producer=tmp_path/'same_CPU_budget_producer';root=producer/'sub-CON08';root.mkdir(parents=True)
    for directory in (source,root):
        for name in ('raw','topup','mask','eddy'):(directory/name).mkdir()
        for stem in ('AP','PA'):
            for suffix in ('nii.gz','bval','bvec'):(directory/'raw'/f'{stem}.{suffix}').symlink_to(raw/f'{stem}.{suffix}')
        pair=np.stack([np.asarray(nib.load(raw/f'{stem}.nii.gz').dataobj)[...,0] for stem in ('AP','PA')],axis=3)
        nib.save(nib.Nifti1Image(pair,affine),directory/'topup/B0_AP_PA.nii.gz')
        nib.save(nib.Nifti1Image(np.zeros((2,2,2),dtype=np.float32),affine),directory/'topup/fieldmap_out_fieldcoef.nii.gz')
        (directory/'topup/fieldmap_out_movpar.txt').write_text('0 0 0 0 0 0\n')
        (directory/'topup/acqparams.txt').write_text('0 1 0 0.05\n0 -1 0 0.05\n')
        (directory/'eddy/eddy_index.txt').write_text('1 1\n')
        nib.save(nib.Nifti1Image(np.ones((2,2,2),dtype=np.uint8),affine),directory/'mask/nodif_brain_mask.nii.gz')
    formal=tmp_path/'actual_formal';packing=formal/'sub-CON08/connectome/preproc/topup/B0_AP_PA.nii.gz';packing.parent.mkdir(parents=True);shutil.copy2(source/'topup/B0_AP_PA.nii.gz',packing)
    shutil.copy2(raw/'AP.nii.gz',root/'eddy/data.nii.gz');shutil.copy2(raw/'AP.bvec',root/'eddy/data.eddy_rotated_bvecs')
    report=cpu_report(root);report['selection']={'ap_index':0,'pa_index':0};report['commands'][-1]['command'][-1]='--ref_scan_no=0'
    source_report=source/'CPU_prepared_report.json';source_report.write_text(json.dumps({'commands':report['commands'][:-1]}))
    source_dependencies={str(source/'raw/AP.nii.gz'):m.sha(raw/'AP.nii.gz'),str(source/'mask/nodif_brain_mask.nii.gz'):m.sha(source/'mask/nodif_brain_mask.nii.gz')}
    outputs={str(path.relative_to(root)):m.sha(path) for name in ('topup','mask','eddy') for path in (root/name).iterdir()}
    report.update(subject='CON08',completed=True,state='completed',scope='official CPU reference: restored own successful raw TOPUP/SynthStrip lineage plus new CPU8 EDDY',input_sha256=raw_hashes,output_sha256=outputs,source_sha256={'runner':'producer'},
        source_CPU_stage_lineage={'source':str(source),'report':str(source_report),'report_sha256':m.sha(source_report),'CPU_stage_reexecuted':False,'restored_successful_own_official_TOPUP_and_SynthStrip':True,'source_EDDY_inputs_sha256':source_dependencies,'actual_FNIT_packing':str(packing),'actual_FNIT_packing_sha256':m.sha(packing)},
        copied_successful_CPU_output_sha256={k:v for k,v in outputs.items() if k.startswith(('topup/','mask/'))},
        EDDY_inputs_sha256={str(root/'topup/fieldmap_out_fieldcoef.nii.gz'):outputs['topup/fieldmap_out_fieldcoef.nii.gz'],str(root/'topup/fieldmap_out_movpar.txt'):outputs['topup/fieldmap_out_movpar.txt'],str(root/'mask/nodif_brain_mask.nii.gz'):outputs['mask/nodif_brain_mask.nii.gz']})
    report_path=root/'report.json';report_path.write_text(json.dumps(report))
    sidecar={'completed':True,'subject':'CON08','EDDY_solver':'cpu','GPU_UUID':None,'GP_seed':12345,'ref_scan_no':0,'raw_sha256':raw_hashes,'output_sha256':outputs,
        'data':str(root/'eddy/data.nii.gz'),'rotated_bvecs':str(root/'eddy/data.eddy_rotated_bvecs'),'mask':str(root/'mask/nodif_brain_mask.nii.gz'),'bvals':str(root/'raw/AP.bval'),'topup_prefix':str(root/'topup/fieldmap_out')}
    (root/'completed_contract.json').write_text(json.dumps(sidecar));verified=dict(sidecar,report=str(report_path),report_sha256=m.sha(report_path));(root/'completed_contract_verified.json').write_text(json.dumps(verified))
    case={'subject':'CON08','binding_ready':False,'ready_contract':str(report_path),'upstream_runner_sha256':'producer','expected_upstream_solver':'cpu','expected_selection':None}
    config={'upstream_activation_format':'official_CPU_budget_reference_v1','upstream_producer_root':str(producer),'upstream_runner_sha256':'producer','forbidden_input_roots':[],
        'manifest':str(manifest),'manifest_sha256':m.sha(manifest),'raw_root':str(raw),'approved_restored_heldout_root':str(source_root),'formal_FNIT_packing_root':str(formal),
        'pending_alignment_binding_root':str(tmp_path/'bindings'),'CPU_fallback_activation_report':'actual_activation','upstream_activation_sha256':'actual_activation_digest'}
    freeze={'manifest_sha256':m.sha(manifest),'raw_root':str(raw),'source_sha256':{'runner':'producer'}}
    monkeypatch.setattr(m,'read_bound_upstream_freeze',lambda case,config:freeze)
    resolved,proof=m.resolve_case_binding(case,config)
    assert resolved['expected_selection']=={'ap_index':0,'pa_index':0} and resolved['binding_ready'] is True
    assert case['binding_ready'] is False and case['expected_selection'] is None
    assert proof['evidence']['upstream_report_sha256']==m.sha(report_path)
    assert proof['evidence']['verified_completed_contract_sha256']==m.sha(root/'completed_contract_verified.json')
    assert m.resolve_case_binding(case,config)[1]['sha256']==proof['sha256']
    # Actual voxel mutation is rejected even though all labels and completed flags remain.
    altered=np.asarray(nib.load(packing).dataobj).copy();altered[0,0,0,0]+=1;nib.save(nib.Nifti1Image(altered,affine),packing)
    with pytest.raises(ValueError,match='actual formal packing'):
        m.resolve_case_binding(case,config)
