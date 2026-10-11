"""显式fresh子进程在捕获真实清理输入上回放；不改父进程缓存策略。"""
import argparse,hashlib,importlib.util,json,os,pathlib,sys,time,threading,traceback

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['checkpoint','old-output','old-report','candidate-module','output']:
  p.add_argument('--'+name,type=pathlib.Path,required=True)
 p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
 a=p.parse_args();r={'status':'running','scope':'same-input 1001-step fresh GPU replay; shared diagnostic window, not causal performance comparison','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(a.candidate_module.read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'threads':a.threads,'affinity':sorted(os.sched_getaffinity(0))}
 def save():
  t=a.output.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(a.output)
 save();entered=time.perf_counter()
 try:
  import numpy as np,torch
  from fnit.recon_all.profiling import ProcessTreeDeviceSampler,configure_cuda_allocator
  torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
  torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
  r['allocator']=configure_cuda_allocator(a.device,'enabled')
  r.update(torch_version=torch.__version__,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,autocast=torch.is_autocast_enabled('cuda'),allocator_env=os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'))
  spec=importlib.util.spec_from_file_location('fnit.recon_all.overlap_checkpoint_candidate',a.candidate_module);candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
  data=np.load(a.checkpoint);old=np.load(a.old_output);oldr=json.loads(a.old_report.read_text())
  x=torch.from_numpy(data['positions']).to(a.device);faces=torch.from_numpy(data['faces']).to(a.device)
  sampler=ProcessTreeDeviceSampler(device=a.device,parent_pid=os.getpid(),interval=.2);gpu_uuid=str(torch.cuda.get_device_properties(a.device).uuid);sampler.uuid=gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-"+gpu_uuid
  stop=threading.Event()
  def sampling():
   while not stop.wait(.2):sampler.sample_if_due(force=True)
  t=threading.Thread(target=sampling,daemon=True);t.start();report={}
  torch.cuda.synchronize(a.device);start=time.perf_counter()
  out,history=candidate.remove_overlap_sphere(x,faces,start_iteration=int(data['start_iteration']),diagnostics=report)
  torch.cuda.synchronize(a.device);r['full_cleanup_seconds_with_diagnostics']=time.perf_counter()-start
  values=out.cpu().numpy();delta=np.abs(values.astype(np.float64)-old.astype(np.float64))
  r.update(history_exact=history==oldr['old_actual_cleanup_history'],coordinate_elements_differing=int(np.count_nonzero(delta)),coordinate_max_mm=float(delta.max(initial=0)),coords_sha256=hashlib.sha256(values.tobytes()).hexdigest(),diagnostics=report)
  r['strict_gate']='failed_quality' if report['quality_status']=='failed_quality' else 'passed'
  stop.set();t.join(2);sampler.sample_if_due(force=True);r['gpu_process_sampler']=sampler.report()
  r['allocated_peak_bytes']=torch.cuda.max_memory_allocated(a.device);r['reserved_peak_bytes']=torch.cuda.max_memory_reserved(a.device)
  r['status']='complete'
 except Exception:r.update(status='failed',error=traceback.format_exc())
 r['harness_wall_seconds_including_import_load_transfer_compute_IO']=time.perf_counter()-entered;save()
 if r['status']=='failed':raise RuntimeError(r['error'])
if __name__=='__main__':main()
