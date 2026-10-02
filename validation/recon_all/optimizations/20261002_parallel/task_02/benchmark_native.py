"""冻结同输入：当前Conda、未改重建控制、去冗余建表候选与可选官方参考。

每个完整步骤独立复制自产前置；读取官方输出只用于结束后的隔离比较。
原MRI/表面、源码副本与日志留服务器，公开报告只含hash/统计。
"""
import argparse,hashlib,json,os,platform,shutil,time
from pathlib import Path
import nibabel as nib,nibabel.freesurfer.io as fs,numpy as np,torch,numba
from fnit.recon_all.white_preaparc_conda import run_white_preaparc
from fnit.recon_all.final_white_conda import run_final_white
from fnit.recon_all.native_free import _run_native_pial
from fnit.recon_all.assets import ASSET_FILES
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',type=Path,required=True);p.add_argument('--hemi',choices=['lh','rh'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--current',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--stages',nargs='+',choices=['white_preaparc','final_white','pial'],default=['white_preaparc','final_white','pial']);p.add_argument('--official',type=Path);p.add_argument('--official-home',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);numba.set_num_threads(4)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
hemi=a.hemi
sources=[*[f'mri/{x}.mgz' for x in ('brain.finalsurfs','wm','aseg.presurf')],*[f'surf/{hemi}.{x}' for x in ('orig','orig.premesh','white.preaparc','white')],f'surf/autodet.gw.stats.{hemi}.dat',*[f'label/{hemi}.{x}' for x in ('cortex.label','cortex+hipamyg.label','aparc.annot')]]
report=dict(commit=a.commit,host=platform.node(),subject=str(a.subject),hemi=hemi,threads=4,scope='same-input independent complete native placement steps; not continuous chain or whole case',input_sha256={x:sha(a.subject/x) for x in sources},benchmark_sha256=sha(__file__),program_python_sha256=sha(os.sys.executable),resources={},rows=[],overall_equivalence='not_assessed')
for name in ('FreeSurferColorLUT.txt','ASegStatsLUT.txt','SubCorticalMassLUT.txt','WMParcStatsLUT.txt'):
 path=a.assets/name;size,expected,_=ASSET_FILES[name];actual=sha(path);report['resources'][name]=dict(bytes=path.stat().st_size,sha256=actual,manifest_match=actual==expected and path.stat().st_size==size)
 if not report['resources'][name]['manifest_match']:raise ValueError('fixed asset mismatch: '+name)
runners=dict(white_preaparc=run_white_preaparc,final_white=run_final_white,pial=_run_native_pial)
binaries=[('current',a.current,a.assets),('control',a.control,a.assets),('candidate',a.candidate,a.assets)]
if a.official:
 if not a.official_home:raise ValueError('official-home required with official')
 binaries.append(('official',a.official,a.official_home))
def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for stage in a.stages:
 outputs={};stats={};volumes={}
 for name,binary,assets in binaries:
  total_start=time.perf_counter();subject=a.output/stage/name
  for path in sources:
   dest=subject/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.subject/path,dest)
  (subject/'scripts').mkdir();copy_seconds=time.perf_counter()-total_start
  before={str(x.relative_to(subject)):sha(x) for x in subject.rglob('*') if x.is_file()}
  started=time.perf_counter()
  if stage=='pial':result=runners[stage](binary,subject,hemi,assets,4)
  else:result=runners[stage](subject,hemi,binary,assets,threads=4)
  seconds=time.perf_counter()-started
  surface=Path(result['output']);xyz,faces=fs.read_geometry(str(surface));outputs[name]=(xyz,faces)
  after={str(x.relative_to(subject)):sha(x) for x in subject.rglob('*') if x.is_file()}
  stats[name]=after[f'surf/autodet.gw.stats.{hemi}.dat']
  if result.get('outvol'):
   image=nib.load(result['outvol']);volumes[name]=(np.asarray(image.dataobj).copy(),image.affine.copy(),str(image.get_data_dtype()))
  row=dict(stage=stage,backend=name,binary_sha256=sha(binary),surface_sha256=sha(surface),stage_wrapper_seconds=seconds,isolation_copy_seconds=copy_seconds,total_including_isolation_validation_seconds=time.perf_counter()-total_start,result=result,output_files={x:y for x,y in after.items() if before.get(x)!=y},stats_sha256=stats[name],external_loadavg=os.getloadavg())
  if name!='current':
   ref,rf=outputs['current'];same=xyz.shape==ref.shape and np.array_equal(faces,rf);delta=np.linalg.norm(xyz-ref,axis=1) if same else None
   row['comparison_to_current']=dict(ordered_correspondence=same,different_components=int(np.count_nonzero(xyz!=ref)) if same else None,max_mm=float(delta.max()) if same else None,p99_mm=float(np.quantile(delta,.99)) if same else None,mean_mm=float(delta.mean()) if same else None,stats_exact=stats[name]==stats['current'])
   if name in volumes:
    ref,ra,rd=volumes['current'];candidate,ca,cd=volumes[name];row['comparison_to_current'].update(outvol_array_exact=np.array_equal(ref,candidate),outvol_affine_exact=np.array_equal(ra,ca),outvol_dtype_exact=rd==cd)
  report['rows'].append(row);save();print(f'{stage} {name}: {seconds:.3f}s',flush=True)
  if name in ('control','candidate'):
   compare=row['comparison_to_current'];assert compare['ordered_correspondence'] and compare['different_components']==0 and compare['stats_exact']
   if name in volumes:assert compare['outvol_array_exact'] and compare['outvol_affine_exact'] and compare['outvol_dtype_exact']
