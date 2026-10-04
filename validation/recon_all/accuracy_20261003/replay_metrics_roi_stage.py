"""Same-input left metrics and six ROI tables; calls frozen FNIT functions only.

Caller must hold both shared CPU/GPU locks and admit >=20GB. Each new process
runs one cold pass and one same-process warm pass; no algorithm failure retry.
"""
from __future__ import annotations
import argparse,json,os,pathlib,hashlib,shutil,sys,time,traceback
from replay_annotation_stage import declared_gpu_uuid,normalized_actual_uuid,write
MORPHS=('thickness','area','area.pial','curv','curv.pial','area.mid','volume')
INPUTS=('surf/lh.white','surf/lh.pial','label/lh.cortex.label','stats/brainvol.stats','mri/transforms/talairach.xfm')+tuple('label/lh.'+a+'.annot' for a in ('aparc','aparc.a2009s','aparc.DKTatlas','BA_exvivo','BA_exvivo.thresh'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify_inputs(manifest,checkpoint):
 entries=manifest['files']
 if set(e['relative'] for e in entries)!=set(INPUTS) or len(entries)!=len(INPUTS):raise ValueError('exact declared minimal lh input set required')
 for e in entries:
  p=checkpoint/e['relative']
  if p.stat().st_size!=e['size_bytes'] or sha(p)!=e['sha256']:raise ValueError('input changed: '+str(p))
 return entries

def main():
 started=time.perf_counter();a=argparse.ArgumentParser();a.add_argument('--checkpoint',type=pathlib.Path,required=True);a.add_argument('--input-manifest',type=pathlib.Path,required=True);a.add_argument('--assets',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--gpu-uuid',type=declared_gpu_uuid,required=True);args=a.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('sole GPU UUID environment mismatch')
 if args.output.exists() or args.output.resolve()==args.checkpoint.resolve() or args.checkpoint.resolve() in args.output.resolve().parents:raise ValueError('new independent output required')
 manifest=json.loads(args.input_manifest.read_text());entries=verify_inputs(manifest,args.checkpoint)
 args.output.mkdir(parents=True);subject=args.output/'subject';report={'status':'preparing','stage_only':True,'hemisphere':'lh','gpu_uuid_declared':args.gpu_uuid,'input_manifest_sha256':sha(args.input_manifest),'inputs':entries,'passes':[],'algorithm_retry':False,'precision_role':manifest.get('role'),'cache_scope':'process warm; fresh SurfaceStatsCache per pass, six tables reuse each cache'};report_path=args.output/'report.json'
 try:
  copytick=time.perf_counter()
  for e in entries:
   dest=subject/e['relative'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(args.checkpoint/e['relative'],dest)
   if sha(dest)!=e['sha256']:raise ValueError('copy SHA mismatch')
  for folder in ('surf','mri','label','stats','scripts'):(subject/folder).mkdir(exist_ok=True)
  report['copy_and_hash_seconds']=time.perf_counter()-copytick
  import torch,numpy as np,nibabel as nib
  import nibabel.freesurfer.io as fs
  from fnit.recon_all import native_free,profiling,surface_stats_cache
  from fnit.recon_all.thread_budget import thread_budget
  from fnit.recon_all.anatomical_stats_global import read_brain_volume_stats
  allocator=profiling.configure_cuda_allocator('cuda:0','disabled');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
  report.update(interpreter=sys.executable,torch_version=torch.__version__,nibabel_version=nib.__version__,precision={'matmul_tf32':True,'cudnn_tf32':True,'autocast':profiling.autocast_state('cuda')},allocator=allocator)
  if report['precision']['autocast']['enabled']:raise ValueError('autocast must be disabled')
  report['loaded_source_sha256']={m.__name__:{'path':m.__file__,'sha256':sha(m.__file__)} for m in [native_free,profiling,surface_stats_cache]}
  white,faces=fs.read_geometry(str(subject/'surf/lh.white'));pial,pfaces=fs.read_geometry(str(subject/'surf/lh.pial'))
  if not np.array_equal(faces,pfaces) or len(white)!=len(pial):raise ValueError('ordered white/pial faces mismatch')
  report['space']={'coordinates':'surface RAS/mm','vertices':len(white),'faces':len(faces),'ordered_faces_equal':True,'white_input_sha256':sha(subject/'surf/lh.white'),'pial_input_sha256':sha(subject/'surf/lh.pial')}
  volumes=read_brain_volume_stats(subject/'stats/brainvol.stats')
  profiler=profiling.StageProfiler(device='cuda:0',synchronize=True,allocator=allocator)
  # Same lifecycle as public four-thread wrapper. First timed stage includes context bootstrap.
  report['operation_entered']=False
  with thread_budget(threads=4) as budget:
   for passname in ('cold','warm'):
    rows=[]; values={};tick=time.perf_counter();report['operation_entered']=True;result=profiler.run('finish_metrics_lh',native_free._finish_cortical_metrics,subject,'lh',pathlib.Path('/unused_cuda_metrics_binary'),args.assets,device='cuda:0');rows.append(dict(profiler.last_row))
    props=torch.cuda.get_device_properties(0);observed=normalized_actual_uuid(str(props.uuid))
    if observed!=normalized_actual_uuid(args.gpu_uuid):raise ValueError('actual GPU UUID mismatch')
    report['gpu_uuid_actual']=observed
    for name in MORPHS:
     arr=fs.read_morph_data(str(subject/('surf/lh.'+name)))
     if arr.shape!=(len(white),) or not np.isfinite(arr).all():raise ValueError('morph numeric/order shape invalid')
     values[name]=arr
    tabledir=args.output/passname;tabledir.mkdir();np.savez_compressed(tabledir/'metrics.npz',**values)
    map_receipts={name:{'dtype':str(v.dtype),'shape':list(v.shape),'min':float(v.min()),'max':float(v.max()),'sha256':sha(subject/('surf/lh.'+name))} for name,v in values.items()}
    with surface_stats_cache.SurfaceStatsCache(device='cuda:0') as cache:
     def stage(name,fn,*pos,**kw):
      try:return profiler.run(name,fn,*pos,**kw)
      finally:rows.append(dict(profiler.last_row))
     native_free._write_hemisphere_stats(subject,'lh',volumes,cache,stage,'cuda:0');counters=dict(cache.counters)
    tables={}
    for suffix in ('aparc','aparc.a2009s','aparc.DKTatlas','aparc.pial','BA_exvivo','BA_exvivo.thresh'):
     src=subject/('stats/lh.'+suffix+'.stats');shutil.copy2(src,tabledir/src.name)
     data=[line.split() for line in src.read_text().splitlines() if line.strip() and not line.startswith('#')];names=[r[0] for r in data];numbers=np.asarray([[float(v) for v in r[1:]] for r in data],dtype=np.float64)
     if numbers.ndim!=2 or numbers.shape[1]!=9 or not np.isfinite(numbers).all():raise ValueError('ROI table invalid')
     np.savez_compressed(tabledir/(suffix+'.npz'),names=np.asarray(names),values=numbers)
     tables[suffix]={'sha256':sha(src),'rows':len(data),'numeric_dtype':str(numbers.dtype),'column_names':['NumVert','SurfArea','GrayVol','ThickAvg','ThickStd','MeanCurv','GausCurv','FoldInd','CurvInd']}
    report['passes'].append({'name':passname,'entry_seconds_including_output_serialization':time.perf_counter()-tick,'metrics_internal_seconds':result['metric_seconds'],'stages':rows,'cache_counters':counters,'maps':map_receipts,'tables':tables})
    write(report_path,report)
   report['thread_budget']=budget
  verify_inputs(manifest,args.checkpoint)
  if sha(subject/'surf/lh.white')!=report['space']['white_input_sha256'] or sha(subject/'surf/lh.pial')!=report['space']['pial_input_sha256']:raise ValueError('ordered mesh inputs changed')
  report.update(status='complete',whole_case_completed=False,entry_seconds_from_validation=time.perf_counter()-started)
 except BaseException as e:
  if 'profiler' in locals():report['failed_last_stage']=dict(profiler.last_row)
  report.update(status='failed',error=repr(e),traceback=traceback.format_exc(),entry_seconds_from_validation=time.perf_counter()-started);write(report_path,report);raise
 write(report_path,report)
if __name__=='__main__':main()
