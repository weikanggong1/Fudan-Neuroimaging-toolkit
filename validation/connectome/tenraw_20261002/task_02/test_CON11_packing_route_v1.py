"""Metadata/source-route guards; tiny NIfTI fixtures are not MRI benchmarks."""
import importlib.util
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('route_guard',Path(__file__).with_name('validate_CON11_packing_route_v1.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();affine=np.eye(4)
    for stem,offset in [('AP',1),('PA',7)]:
        a=np.full((2,2,2,2),offset,dtype=np.float32);a[...,1]+=2
        nib.save(nib.Nifti1Image(a,affine),raw/f'{stem}.nii.gz');np.savetxt(raw/f'{stem}.bval',[[0,0]])
    for i in range(6):(raw/f'other{i}').write_text(str(i))
    hashes={p.name:m.sha(p) for p in raw.iterdir()};manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'cases':[{'subject':'CON11','input_sha256':hashes}]}))
    old=tmp_path/'old_formal';new=tmp_path/'actual_new_formal/sub-CON11/connectome/preproc/topup';new.mkdir(parents=True)
    packing=new/'B0_AP_PA.nii.gz';pair=np.stack([np.asarray(nib.load(raw/f'{stem}.nii.gz').dataobj)[...,1] for stem in ('AP','PA')],axis=3);nib.save(nib.Nifti1Image(pair,affine),packing)
    lineage={}
    for key in ('source','configuration','driver'):
        p=tmp_path/key;p.write_text(key);lineage[key]={'path':str(p),'sha256':m.sha(p)}
    config={'raw_root':str(raw),'manifest':str(manifest),'manifest_sha256':m.sha(manifest),'formal_FNIT_packing_root':str(old)}
    route={'state':'actual_CON11_packing_verified','case_id':'sub-CON11','root_actual_origin_verified':True,'baseline_lineage':lineage,'manifest_sha256':m.sha(manifest),
        'packing':{'path':str(packing),'sha256':m.sha(packing)},'actual_raw_frame_indices':{'ap_index':1,'pa_index':1},
        'canonical_inputs':{stem:{kind:{'path':str(raw/name),'sha256':hashes[name]} for kind,name in [('dwi',f'{stem}.nii.gz'),('bvals',f'{stem}.bval')]} for stem in ('AP','PA')}}
    return route,config,lineage


def test_pending_origin_cannot_enable_modeling(tmp_path):
    assert m.validate_actual_CON11_packing_route({'state':'awaiting_actual_root_origin'},{},{}) is None
    assert list(tmp_path.iterdir())==[]


def test_new_actual_packing_route_keeps_old_namespace_untouched(tmp_path):
    route,config,lineage=fixture(tmp_path);proof=m.validate_actual_CON11_packing_route(route,config,lineage)
    assert proof['actual_raw_frame_indices']=={'ap_index':1,'pa_index':1}
    assert proof['modeling_ready'] is False
    assert not Path(config['formal_FNIT_packing_root']).exists()


@pytest.mark.parametrize('mutation',['case','lineage','packing_SHA','guessed_frame','raw_SHA','origin_unverified'])
def test_changed_or_guessed_origin_is_rejected(tmp_path,mutation):
    route,config,lineage=fixture(tmp_path)
    if mutation=='case':route['case_id']='sub-CON10'
    elif mutation=='lineage':route['baseline_lineage']=dict(lineage,driver={'path':'other','sha256':'other'})
    elif mutation=='packing_SHA':route['packing']['sha256']='wrong'
    elif mutation=='guessed_frame':route['actual_raw_frame_indices']['ap_index']=0
    elif mutation=='raw_SHA':route['canonical_inputs']['AP']['dwi']['sha256']='wrong'
    else:route['root_actual_origin_verified']=False
    with pytest.raises(ValueError):m.validate_actual_CON11_packing_route(route,config,lineage)


def test_old_path_symlink_cannot_impersonate_new_origin(tmp_path):
    route,config,lineage=fixture(tmp_path);p=Path(route['packing']['path']);alias=p.with_name('alias.nii.gz');alias.symlink_to(p)
    route['packing']['path']=str(alias)
    with pytest.raises(ValueError,match='symlink alias'):m.validate_actual_CON11_packing_route(route,config,lineage)
