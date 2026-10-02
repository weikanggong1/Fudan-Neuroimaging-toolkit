"""冻结真实输入的 GCA 评分/搜索回归；不产生完整注册声明。"""
import argparse, csv, hashlib, json, os, platform, subprocess, threading, time
from pathlib import Path
import numpy as np
import torch
from fnit.recon_all.mri_em_register import (read_gca, read_masked_input, find_stable_samples,
    find_all_samples, atlas_label_peak, estimate_image_white_matter_peak, scale_input_intensity)
from fnit.recon_all.mri_em_register_score_gpu import GCASearchScorer
from fnit.recon_all.mri_em_register_search_jit import log_sample_probability_jit
from fnit.recon_all.mri_em_register_translation_source import find_optimal_translation_source


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--commit',required=True); p.add_argument('--translation',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    snapshot=Path(__file__).resolve().parents[5]/'source_commit.txt'
    actual_commit=snapshot.read_text().strip() if snapshot.exists() else a.commit
    report={'commit':actual_commit,'dispatch_commit':a.commit,'host':platform.node(),'pid':os.getpid(),'tolerance_declared':{'score_atol':0,'translation_matrix_atol':0},
            'scope':'frozen_same_input_component_not_complete_EM_or_continuous_chain', 'torch':torch.__version__,
            'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')},
            'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'external_load':[], 'cases':[]}
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
                report['external_load'].append({'time':time.time(),'processes':processes,'gpu':gpu})
            except Exception as e: report['external_load'].append({'error':str(e),'time':time.time()})
            stop.wait(.5)
    watcher=threading.Thread(target=monitor);watcher.start()
    try:
        atlas_path=a.root/'assets/average/RB_all_2020-01-02.gca'
        from fnit.recon_all.assets import ASSET_FILES
        size,digest,_=ASSET_FILES['average/RB_all_2020-01-02.gca']
        assert atlas_path.stat().st_size==size and sha(atlas_path)==digest
        tick=time.perf_counter(); atlas=read_gca(atlas_path); stable=find_stable_samples(atlas); all_samples=find_all_samples(atlas)
        report['atlas_preparation_seconds']=time.perf_counter()-tick
        report['atlas']={'size':size,'sha256':digest,'redistributed':False}
        matrices=np.repeat(np.eye(4,dtype=np.float32)[None],256,axis=0)
        # Fixed broad candidate set exercises edges, negative rounding and nonidentity inverses.
        rng=np.random.default_rng(34)
        matrices[:,:3,:3]+=rng.uniform(-.12,.12,(256,3,3)).astype(np.float32)
        matrices[:,:3,3]=rng.uniform(-60,60,(256,3)).astype(np.float32)
        for case in ('whole_sub01_candidate_retry1','whole_sub02_candidate_retry2'):
            mri=a.root/'serial_20261001'/case/'mri';nu=mri/'nu.mgz';mask=mri/'brainmask.mgz'
            row={'case':case,'inputs':{'nu':sha(nu),'brainmask':sha(mask)},'steps':[]}
            tick=time.perf_counter(); masked=read_masked_input(nu,mask)
            peak,*_=estimate_image_white_matter_peak(atlas,masked)
            source=scale_input_intensity(masked,atlas_label_peak(atlas,2),peak)
            row['input_prepare_seconds']=time.perf_counter()-tick
            for name,samples in [('stable',stable),('all',all_samples)]:
                use=matrices if name=='stable' else matrices[:16]
                step={'sample_set':name,'sample_count':len(samples.means),'candidate_count':len(use)}
                tick=time.perf_counter(); cpu=np.array([log_sample_probability_jit(samples,source,m) for m in use],np.float32)
                step['cpu_cold_seconds']=time.perf_counter()-tick
                tick=time.perf_counter(); cpu=np.array([log_sample_probability_jit(samples,source,m) for m in use],np.float32)
                step['cpu_warm_seconds']=time.perf_counter()-tick
                tick=time.perf_counter(); scorer=GCASearchScorer(samples,source,device='cuda:0');torch.cuda.synchronize()
                step['upload_seconds']=time.perf_counter()-tick
                for mode in ('cold','warm'):
                    tick=time.perf_counter(); gpu=scorer.score_many(use);torch.cuda.synchronize()
                    step['gpu_'+mode+'_seconds']=time.perf_counter()-tick
                delta=np.abs(gpu.astype(float)-cpu.astype(float))
                step.update(different_scores=int(np.count_nonzero(cpu!=gpu)),max_error=float(delta.max()),p99_error=float(np.quantile(delta,.99)),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
                step['cpu_scores']=cpu.tolist();step['gpu_scores']=gpu.tolist()
                row['steps'].append(step)
                if step['different_scores']: row['strict_score_pass']=False
                del scorer
            if a.translation:
                tick=time.perf_counter(); before,history=find_optimal_translation_source(stable,source,np.eye(4,dtype=np.float32))
                row['cpu_translation_seconds']=time.perf_counter()-tick
                scorer=GCASearchScorer(stable,source,device='cuda:0')
                tick=time.perf_counter();after,ghistory=find_optimal_translation_source(stable,source,np.eye(4,dtype=np.float32),scorer=scorer);torch.cuda.synchronize()
                row['gpu_translation_seconds']=time.perf_counter()-tick
                row['translation']={'cpu_matrix':before.tolist(),'gpu_matrix':after.tolist(),'exact_matrix':bool(np.array_equal(before,after)),
                   'cpu_scores':[h[0] for h in history],'gpu_scores':[float(h[0]) for h in ghistory]}
                del scorer
            old_dtype=torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float64)
                scorer=GCASearchScorer(stable,source,device='cuda:0')
                default64=scorer.score_many(matrices)
            finally:
                torch.set_default_dtype(old_dtype)
            expected=np.array([log_sample_probability_jit(stable,source,m) for m in matrices],np.float32)
            row['default_float64_different_scores']=int(np.count_nonzero(default64!=expected))
            assert np.array_equal(default64,expected),'default dtype altered real-input score'
            del scorer
            report['cases'].append(row)
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        import runpy
        test_module=runpy.run_path(str(Path(__file__).resolve().parents[5]/'test_gca_score_batch.py'))
        test_module['test_gpu_coordinates_ignore_default_dtype']()
        report['boundary_default_dtype_test']='passed'
    finally:
        stop.set();watcher.join()
        source_root=Path(__file__).resolve().parents[5]/'src'
        report['source_sha256']={str(p.relative_to(source_root)):sha(p) for p in (source_root/'fnit/recon_all').glob('mri_em_register*.py')}
        report['script_sha256']=sha(__file__)
        report['overall_equivalence']='not_assessed'
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'scores.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=['case','sample_set','candidate_count','cpu_warm_seconds','gpu_warm_seconds','max_error','different_scores']);writer.writeheader()
        for row in report['cases']:
            for step in row['steps']:writer.writerow({k:(row['case'] if k=='case' else step[k]) for k in writer.fieldnames})

if __name__=='__main__': main()
