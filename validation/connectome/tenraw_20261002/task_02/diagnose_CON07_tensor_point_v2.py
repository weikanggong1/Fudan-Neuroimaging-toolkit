"""Trace unmodified frozen Torch tensor function at actual CON07 outlier, CPU only."""
import argparse,hashlib,importlib.util,inspect,json,sys,socket
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;out=a.output;assert not out.exists();ref=r/'task_02/official_modeling_CPU_budget_raw10_v1/sub-CON07';diag=r/'task_02/actual_CPU_budget_comparison_v3/sub-CON07/tensor_same_input';src=r/'baseline_raw_compatible_v3/src/fnit/connectome/response.py';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(src)=='258938a5f20ee7a6efc2fc1c50af5c93412030a60b78036438e8473f072e3b91';c=json.loads((ref/'consumer_contract.json').read_text());dr=json.loads((diag/'report.json').read_text());xyz=(66,42,12)
files={k:Path(c['files'][name]['path']) for k,name in [('dwi','official_corrected_dwi'),('gradient','official_gradient_mrtrix'),('mask','brain_mask')]}
for key,path in files.items():assert sha(path)==dr['input_sha256'][{'mask':'brain_mask'}.get(key,key)]
image=nib.load(files['dwi']);signal=np.asarray(image.dataobj,dtype=np.float32);mask=np.asarray(nib.load(files['mask']).dataobj)>0;assert mask[xyz];gradient=np.loadtxt(files['gradient']);assert gradient.shape==(102,4)
spec=importlib.util.spec_from_file_location('frozen_response',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);fn=m.fit_mrtrix_dhollander_tensor;torch.set_num_threads(8);source_lines,start=inspect.getsourcelines(fn);lines={start+i:line.strip() for i,line in enumerate(source_lines)}
conv=lambda x:x.detach().cpu().numpy().tolist() if isinstance(x,torch.Tensor) else x
flat_index=int(np.ravel_multi_index(xyz,mask.shape));voxels=np.flatnonzero(mask.reshape(-1));rank=int(np.flatnonzero(voxels==flat_index)[0]);block_start=rank//4096*4096;block_indices=voxels[block_start:block_start+4096];focus=rank-block_start
saved_FA=float(np.asarray(nib.load(diag/'fa_cpu.nii.gz').dataobj)[xyz]);saved_vec=np.asarray(nib.load(diag/'direction_cpu.nii.gz').dataobj)[xyz].astype(np.float64)
def trace_call(data,selected):
    capture={'iterations':[]}
    def trace(frame,event,arg):
        if frame.f_code is fn.__code__ and event=='line':
            local=frame.f_locals;line=lines.get(frame.f_lineno,'')
            if line=='if iteration < 2:':
                par=local['parameters'][selected];design=local['design'];weights=local['weights'][selected];gram=local['gram'][selected];right=local['right'][selected]
                capture['design_matrix']=conv(design);capture['clamped_measurements']=conv(local['samples'][selected]);capture['log_signal']=conv(local['log_signal'][selected]);capture['iterations'].append({'iteration':local['iteration'],'parameters_D11_D22_D33_D12_D13_D23_logS0':conv(par),'weights':conv(weights),'predicted_signal':conv((par@design.T).exp()),'gram':conv(gram),'right':conv(right),'gram_condition_number':float(torch.linalg.cond(gram)),'cholesky_info':int(local['info'][selected]),'branch':'cholesky' if int(local['info'][selected])==0 else 'lstsq_fallback'})
            if line=='principal = values.abs().argmax(dim=1)':
                capture.update(tensor=conv(local['tensor'][selected]),tensor_coefficients_after_float32_roundtrip=conv(local['d'][selected]),eigenvalues_algebraic_ascending=conv(local['values'][selected]),eigenvectors_columns=conv(local['vectors'][selected]),principal_choice='maximum absolute eigenvalue; no clipping',FA=float(local['fa'][selected]))
        return trace
    previous=sys.gettrace();sys.settrace(trace)
    try:fa,vec=fn(torch.from_numpy(np.ascontiguousarray(data)).reshape(-1,1,1,102),torch.from_numpy(gradient),torch.ones((len(data),1,1),dtype=torch.bool),batch_size=4096)
    finally:sys.settrace(previous)
    capture['output_FA']=float(fa.reshape(-1)[selected]);capture['output_direction']=conv(vec.reshape(-1,3)[selected]);values=np.array(capture['eigenvalues_algebraic_ascending']);order=np.argsort(abs(values));capture['absolute_top_eigenvalue_gap']=float(abs(values[order[-1]])-abs(values[order[-2]]));capture['relative_absolute_top_gap']=float(capture['absolute_top_eigenvalue_gap']/max(abs(values)));return capture
single=trace_call(signal[xyz][None,:],0)
# Preserve the original 4096-row batch when needed to test LAPACK/batch dependence.
block=trace_call(signal.reshape(-1,102)[block_indices],focus)
coeff=np.asarray(nib.load(ref/'tensor.nii.gz').dataobj)[xyz].astype(np.float64);d11,d22,d33,d12,d13,d23=coeff;official=np.array([[d11,d12,d13],[d12,d22,d23],[d13,d23,d33]]);ev,evec=np.linalg.eigh(official);official_FA=float(np.asarray(nib.load(ref/'fa.nii.gz').dataobj)[xyz]);official_vec=np.asarray(nib.load(ref/'direction.nii.gz').dataobj)[xyz].astype(np.float64)
angle=lambda x,y:float(np.degrees(np.arccos(np.clip(abs(np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)),0,1))))
for item in [single,block]:item['angle_vs_saved_CPU_direction_degrees']=angle(np.array(item['output_direction']),saved_vec);item['angle_vs_saved_official_direction_degrees']=angle(np.array(item['output_direction']),official_vec);item['FA_abs_difference_vs_saved_CPU']=abs(item['output_FA']-saved_FA);item['output_FA_float32_bytes_equal_saved_CPU']=np.float32(item['output_FA']).tobytes()==np.float32(saved_FA).tobytes()
formal=r/'formal_baseline_common_v3_raw_rerun_v1/baseline/sub-CON07/connectome';formal_FA=formal/'fa_dwi.nii.gz';formal_tensor_paths=list(formal.glob('*tensor*'));raw=signal[xyz].astype(np.float64)
report={'subject':'CON07','hostname':socket.gethostname(),'scope':'Actual same-input CPU outlier traced through unchanged frozen response.py; single voxel and its original4096-row batch only. Formal GPU raw-chain FA is a separate input chain. No production/math/result mutation.','source':{'path':str(src),'sha256':sha(src)},'diagnostic_source_sha256':sha(__file__),'inputs':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},'voxel_zero_based':list(xyz),'flatten_mapping':{'flat_index_C_order':flat_index,'brain_mask_rank':rank,'original_block_start_rank':block_start,'original_block_voxel_count':len(block_indices),'index_in_original_block':focus,'block_flat_indices':block_indices.tolist()},'dtype_policy':'original corrected float32; gradients float64; original function float64 IWLS and float32 tensor roundtrip; 8 CPU threads, batch4096, TF32 irrelevant on CPU','actual_measurements':raw.tolist(),'measurement_stats':{'min':float(raw.min()),'max':float(raw.max()),'positive_count':int((raw>0).sum()),'zero_count':int((raw==0).sum()),'negative_count':int((raw<0).sum()),'nonfinite_count':int((~np.isfinite(raw)).sum())},'gradient_Nx4':gradient.tolist(),'saved_CPU_FA':saved_FA,'saved_CPU_vector':saved_vec.tolist(),'single_voxel_frozen_call':single,'original_batch_frozen_call':block,'official_saved_tensor':{'path':str(ref/'tensor.nii.gz'),'sha256':sha(ref/'tensor.nii.gz'),'tensor':official.tolist(),'eigenvalues':ev.tolist(),'FA':official_FA,'vector':official_vec.tolist(),'coefficients_D11_D22_D33_D12_D13_D23':coeff.tolist()},'formal_GPU_chain_separate_FA':{'path':str(formal_FA),'sha256':sha(formal_FA),'value_at_same_grid_index':float(np.asarray(nib.load(formal_FA).dataobj)[xyz]),'tensor_candidate_paths':list(map(str,formal_tensor_paths)),'same_corrected_input':False},'limitations':['This traces original Torch calls, not native MRtrix iterative internals; official tensor differences do not identify which native fitting step diverged.','Formal baseline did not retain tensor coefficients; formal GPU FA from its different raw correction is not evidence of same-input CPU/GPU parity.','Original per-voxel and original batch are reproduced without altering thresholds, weights, signal clipping, iterations, negative eigenvalue handling or vector selection.']}
def strict_metadata(value):
    if isinstance(value,float) and not np.isfinite(value):return {'nonfinite': 'NaN' if np.isnan(value) else ('+Infinity' if value>0 else '-Infinity')}
    if isinstance(value,dict):return {key:strict_metadata(item) for key,item in value.items()}
    if isinstance(value,list):return [strict_metadata(item) for item in value]
    return value
report['metadata_nonfinite_encoding']='Actual nonfinite scalar values are explicit {nonfinite: label}; never replaced with finite values.'
report['previous_trace_attempt']={'output':str(out.with_name('CON07_tensor_point_trace_v1.json')),'state':'diagnostic_computation_completed_but_strict_JSON_serialization_failed; original empty file preserved'}
encoded=json.dumps(strict_metadata(report),indent=2,allow_nan=False)+'\n'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:f.write(encoded)
print(json.dumps({k:report[k] for k in ['measurement_stats','saved_CPU_FA']},indent=2));print(json.dumps({'single_eigenvalues':single['eigenvalues_algebraic_ascending'],'batch_eigenvalues':block['eigenvalues_algebraic_ascending'],'batch_gap':block['absolute_top_eigenvalue_gap'],'batch_saved_FA_diff':block['FA_abs_difference_vs_saved_CPU'],'batch_saved_CPU_angle':block['angle_vs_saved_CPU_direction_degrees'],'batch_branches':[x['branch'] for x in block['iterations']],'batch_gram_conditions':[x['gram_condition_number'] for x in block['iterations']]},indent=2))
