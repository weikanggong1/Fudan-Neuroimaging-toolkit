import pathlib,json,numpy as np,nibabel as nib,hashlib
r=pathlib.Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001");report={'scope':'frozen_auxiliary_precision_diagnostic; cache_enabled; not_whole_speedup','production_commit':"61926c7dceae2f9097fa296e306cf1efa0a09191",'cases':{}}
for sub in ['01','02']:
 entry={'run':json.loads((r/('synth_aux_fp32_sub'+sub+'.json')).read_text()),'monitor':json.loads((r/('synth_aux_fp32_sub'+sub+'_monitor/monitor.json')).read_text()),'comparisons':{}}
 for ref in ['oldcpu','newgpu']:
  rows={}
  for name in ['entowm','mca-dura','vsinus']:
   a=r/('stage1r2_sub'+sub+'_'+ref)/'mri'/(name+'.mgz');b=r/('synth_aux_fp32_sub'+sub)/'mri'/(name+'.mgz');x,y=nib.load(a),nib.load(b);u,v=np.asarray(x.dataobj),np.asarray(y.dataobj);labs=np.union1d(u,v);rows[name]={'different_voxels':int(np.count_nonzero(u!=v)),'geometry_equal':bool(np.array_equal(x.affine,y.affine)),'dtype_equal':x.get_data_dtype()==y.get_data_dtype(),'baseline_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'label_dice':{str(int(i)):float(2*np.count_nonzero((u==i)&(v==i))/(np.count_nonzero(u==i)+np.count_nonzero(v==i))) for i in labs if i!=0}}
  entry['comparisons'][ref]=rows
 report['cases'][sub]=entry
print(json.dumps({k:{r:{n:y['different_voxels'] for n,y in z.items()} for r,z in v['comparisons'].items()} for k,v in report['cases'].items()}))
(r/'synth_aux_fp32_summary.json').write_text(json.dumps(report,indent=2)+'\n')