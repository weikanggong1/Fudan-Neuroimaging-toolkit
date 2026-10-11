"""真实自产距离输入：完整CPU/GPU距离、unMRI与反馈逐字段配对。"""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    for name in ['inputs','schema','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--threads',type=int,default=4);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch,numba
    from fnit.recon_all.topology_defect_signed_distance import defect_signed_distance
    from fnit.recon_all.topology_gpu_distance import gpu_defect_signed_distance
    from fnit.recon_all.topology_defect_volume_likelihood import defect_volume_likelihood
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);numba.set_num_threads(args.threads);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_device(args.device);torch.cuda.reset_peak_memory_stats(args.device)
    with np.load(args.inputs,allow_pickle=False)as arrays:
        def decode(v):
            if not isinstance(v,dict):return v
            if 'array'in v:return arrays[v['array']].copy()
            if 'scalar_dtype'in v:return np.asarray(v['value'],dtype=v['scalar_dtype'])[()]
            if 'mapping'in v:return {k:decode(x)for k,x in v['mapping'].items()}
            if 'tuple'in v:return tuple(decode(x)for x in v['tuple'])
            return [decode(x)for x in v['list']]
        prepared=decode(json.loads(args.schema.read_text()))
    rows=[];reference=None;checks=[]
    for trial in range(args.repeats):
        for backend in (['cpu','gpu']if trial%2==0 else ['gpu','cpu']):
            tick=time.perf_counter();distance=defect_signed_distance(**prepared['distance_inputs'])if backend=='cpu'else gpu_defect_signed_distance(**prepared['distance_inputs'],device=args.device)
            torch.cuda.synchronize(args.device);elapsed=time.perf_counter()-tick
            volume=defect_volume_likelihood(**prepared['volume_inputs'],signed_distance_mm=distance['distance_mm'])
            if reference is None:reference=(distance,volume)
            current={}
            for group,actual,golden in [('distance',distance,reference[0]),('volume',volume,reference[1])]:
                for key,value in golden.items():
                    if key=='scope':continue
                    a,b=np.asarray(value),np.asarray(actual[key]);same=a.shape==b.shape and a.dtype==b.dtype;finite=np.isfinite(a)&np.isfinite(b)if same else None
                    current[group+'/'+key]={'shape_dtype_equal':same,'bytes_equal':same and a.tobytes()==b.tobytes(),'different_elements':int(np.count_nonzero(a!=b))if same else None,'max_abs':float(np.max(np.abs(a[finite].astype(float)-b[finite].astype(float)),initial=0))if same else None}
            array=distance['distance_mm'];baseline=reference[0]['distance_mm'];negative_zero=np.signbit(array)&(array==0)
            rows.append({'trial':trial,'backend':backend,'complete_distance_seconds':elapsed,'distance_shape':list(array.shape),'complete_face_rows':len(distance['buffer_bounds']),'block_face_entries':distance.get('block_face_entries'),'unmri_ll':volume['unmri_ll'],'distance_unset_mismatch':int(np.count_nonzero((array==1000)!=(baseline==1000))),'negative_zero_voxels':int(negative_zero.sum()),'all_bytes_equal':all(x['bytes_equal']for x in current.values())})
            checks.append({'trial':trial,'backend':backend,'fields':current});np.save(args.output/(backend+'-'+str(trial)+'.distance.private.npy'),array,allow_pickle=False)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report={'status':'complete','inputs_sha256':digest(args.inputs),'schema_sha256':digest(args.schema),'tool_sha256':digest(__file__),'gpu_source_sha256':digest(__import__('fnit.recon_all.topology_gpu_distance',fromlist=['x']).__file__),'kernel_sha256':digest(__import__('fnit.recon_all.topology_gpu_distance_kernel',fromlist=['x']).__file__),'records':rows,'comparisons':checks,'all_outputs_byte_equal':all(x['all_bytes_equal']for x in rows),'numeric_threads':args.threads,'cpu_affinity':sorted(os.sched_getaffinity(0)),'device_uuid':str(getattr(torch.cuda.get_device_properties(args.device),'uuid','unavailable')),'forward_dtype':'float32; source RN operations; no fp_fusion, matmul or half precision','tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device),'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device),'continuous_nvml_tree_peak':'not_verified','full_GA':'not_assessed','whole_recon_all':'not_assessed','quality_gate_changed':False}
    (args.output/'report.private.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'all_outputs_byte_equal':report['all_outputs_byte_equal'],'records':rows}),flush=True)
    if not report['all_outputs_byte_equal']:raise RuntimeError('GPU distance or downstream feedback differs; retained full report')
if __name__=='__main__':main()
