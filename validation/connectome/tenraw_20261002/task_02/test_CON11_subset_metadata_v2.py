"""Provenance fixtures only; not MRI benchmarks or scientific solver tests."""
from pathlib import Path
import importlib.util,sys,json,copy
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import orchestrate_CON11_CPU_subset_v2 as m

def records(tmp_path):
 root=tmp_path/'sub-CON11';root.mkdir();data=root/'eddy/data.nii.gz';data.parent.mkdir();data.write_bytes(b'actual output fixture');rp=root/'report.json'
 r={'completed':True,'state':'completed','subject':'CON11','input_sha256':{'raw':'abc'},'output_sha256':{'eddy/data.nii.gz':m.sha(data)},'EDDY_inputs_sha256':{str(data):m.sha(data)}};rp.write_text(json.dumps(r))
 v={'completed':True,'subject':'CON11','EDDY_solver':'cpu','GPU_UUID':None,'GP_seed':12345,'ref_scan_no':0,'raw_sha256':r['input_sha256'],'output_sha256':r['output_sha256'],'report':str(rp),'report_sha256':m.sha(rp),'execution_kind':'fresh_official_CPU_rawprep','actual_selection_origin':{'receipt_SHA256':m.ORIGIN_SHA}}
 for k,p in {'data':'eddy/data.nii.gz','mask':'mask/nodif_brain_mask.nii.gz','rotated_bvecs':'eddy/data.eddy_rotated_bvecs','bvals':'raw/AP.bval','topup_prefix':'topup/fieldmap_out'}.items():v[k]=str(root/p)
 route={'case_id':'sub-CON11','subject':'CON11','report':str(rp),'report_SHA256':m.sha(rp),'output_sha256':r['output_sha256'],'execution_kind':'fresh_official_CPU_rawprep',**{k:v[k] for k in ['data','mask','rotated_bvecs','bvals','topup_prefix']}}
 return r,copy.deepcopy(v),v,rp,root,route

def test_pending_explicit_route_does_not_enable_modeling(tmp_path):
 assert m.ready(tmp_path,None,{})==(None,'waiting_explicit_route')
 d=tmp_path/'task_01';d.mkdir();(d/'explicit_actual_CPU_case_routes_v1.json').write_text(json.dumps({'cases':{'sub-CON11':{'state':'waiting_actual_verified_CPU_contract'}}}))
 assert m.ready(tmp_path,None,{})==(None,'waiting_actual_verified_CON11_CPU_contract')
 assert not (tmp_path/'task_02').exists()

def test_matching_actual_metadata_passes(tmp_path):assert m.validate_verified_records(*records(tmp_path)) is True

@pytest.mark.parametrize('kind',['subject','report_SHA','origin_SHA','restored','output_SHA','CPU_label','sidecar_field','own_path','execution_kind'])
def test_changed_actual_metadata_is_rejected(tmp_path,kind):
 r,s,v,rp,root,route=records(tmp_path)
 if kind=='subject':r['subject']='CON10'
 elif kind=='report_SHA':route['report_SHA256']='wrong'
 elif kind=='origin_SHA':v['actual_selection_origin']['receipt_SHA256']='wrong'
 elif kind=='restored':r['source_CPU_stage_lineage']={'source':'old'}
 elif kind=='output_SHA':(root/'eddy/data.nii.gz').write_bytes(b'changed')
 elif kind=='CPU_label':v['GPU_UUID']='GPU'
 elif kind=='sidecar_field':s['GP_seed']=0
 elif kind=='own_path':route['data']='outside'
 elif kind=='execution_kind':v['execution_kind']='restored'
 with pytest.raises(ValueError):m.validate_verified_records(r,s,v,rp,root,route)


def test_verified_canonical_raw_link_is_allowed_only_with_matching_SHA(tmp_path):
 r,s,v,rp,root,route=records(tmp_path);canonical=tmp_path/'canonical_AP.nii.gz';canonical.write_bytes(b'actual raw');raw=root/'raw';raw.mkdir();link=raw/'AP.nii.gz';link.symlink_to(canonical);r['input_sha256'][str(canonical)]=m.sha(canonical);s['raw_sha256']=v['raw_sha256']=r['input_sha256'];r['EDDY_inputs_sha256'][str(link)]=m.sha(link)
 assert m.validate_verified_records(r,s,v,rp,root,route) is True
 r['input_sha256'][str(canonical)]='wrong'
 with pytest.raises(ValueError):m.validate_verified_records(r,s,v,rp,root,route)

def test_field_or_mask_cannot_link_outside_own_chain(tmp_path):
 r,s,v,rp,root,route=records(tmp_path);canonical=tmp_path/'foreign';canonical.write_bytes(b'raw');link=root/'mask';link.mkdir();mask=link/'nodif_brain_mask.nii.gz';mask.symlink_to(canonical);r['input_sha256'][str(canonical)]=m.sha(canonical);s['raw_sha256']=v['raw_sha256']=r['input_sha256'];r['EDDY_inputs_sha256'][str(mask)]=m.sha(mask)
 with pytest.raises(ValueError):m.validate_verified_records(r,s,v,rp,root,route)
