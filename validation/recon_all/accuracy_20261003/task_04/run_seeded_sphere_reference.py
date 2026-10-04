"""显式种子匹配的隔离 sphere 回放；每次取得并释放资源锁，保留重复性。"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,platform,shutil,subprocess,time
from pathlib import Path
import nibabel.freesurfer.io as fsio
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numeric(a,b):
    v,f=zip(*(fsio.read_geometry(str(x)) for x in (a,b)));aligned=v[0].shape==v[1].shape and np.array_equal(f[0],f[1]);r={'sha256':[sha(a),sha(b)],'ordered_faces_equal':bool(aligned),'vertices':[len(x) for x in v],'faces':[len(x) for x in f],'coordinates_equal':bool(aligned and np.array_equal(v[0],v[1]))}
    if aligned:
        d=np.linalg.norm(v[0]-v[1],axis=1);r['indexed_displacement_mm']={'mean':float(d.mean()),'p99':float(np.quantile(d,.99)),'max':float(d.max()),'different_vertices':int((d!=0).sum())}
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','reference-home','output-root','fnit-report','fnit-sphere'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--hemisphere',choices=('lh','rh'),required=True);p.add_argument('--seed',type=int,default=1234);p.add_argument('--threads',type=int,default=4);p.add_argument('--repetitions',type=int,default=2);p.add_argument('--lock',type=Path,default=Path('/tmp/fnit-shared-benchmark.lock'));a=p.parse_args()
    for k in ('checkpoint','reference_home','output_root','fnit_report','fnit_sphere'):setattr(a,k,getattr(a,k).resolve())
    if a.output_root.exists():raise FileExistsError('Use a new empty output directory; previous receipt is immutable')
    if a.seed!=1234:raise ValueError('Existing FNIT replay uses seed1234; other seed requires an independently bound FNIT output')
    if a.repetitions<2:raise ValueError('At least two repetitions are required to assess official repeatability')
    license_path=os.environ.get('FS_LICENSE')
    if not license_path or not Path(license_path).is_file():raise FileNotFoundError('Existing private FS_LICENSE path required')
    a.output_root.mkdir(parents=True);report_path=a.output_root/'receipt.json';fnit=json.loads(a.fnit_report.read_text());binary=a.reference_home/'bin/mris_sphere'
    r={'status':'running','pid':os.getpid(),'host':platform.node(),'script_sha256':sha(__file__),'program_sha256':sha(binary),'fnit_report_sha256':sha(a.fnit_report),'fnit_commit':fnit['args']['commit'],'fnit_seed':1234,'seed':a.seed,'threads':a.threads,'repetitions':a.repetitions,'license_exists':True,'overall_equivalence':'not_assessed','kind':'same-byte-input explicit-seed isolated sphere replay','runs':[]}
    def save():report_path.write_text(json.dumps(r,indent=2)+'\n')
    save()
    try:
        for repetition in range(a.repetitions):
            out=a.output_root/('repeat_'+str(repetition+1));surf=out/'subject/surf';label=out/'subject/label';surf.mkdir(parents=True);label.mkdir(parents=True)
            for source in (a.checkpoint/'surf').glob(a.hemisphere+'.*'):
                if source.is_file():shutil.copyfile(source,surf/source.name)
            for source in (a.checkpoint/'label').glob(a.hemisphere+'.*'):
                if source.is_file():shutil.copyfile(source,label/source.name)
            output=out/(a.hemisphere+'.sphere');inputs={k:sha(surf/(a.hemisphere+'.'+k)) for k in ('inflated','smoothwm')}
            if inputs!=fnit['input_sha256']:raise ValueError('Inflated/smoothwm SHA differs from the FNIT replay')
            cmd=[str(binary),'-threads',str(a.threads),'-seed',str(a.seed),str(surf/(a.hemisphere+'.inflated')),str(output)];item={'repetition':repetition+1,'status':'waiting_lock','command':cmd,'input_sha256':inputs,'env':{'FREESURFER_HOME':str(a.reference_home),'FREESURFER_SEED':str(a.seed),'OMP_NUM_THREADS':str(a.threads),'OPENBLAS_NUM_THREADS':str(a.threads),'MKL_NUM_THREADS':str(a.threads)},'all_private_surf_sha256':{x.name:sha(x) for x in sorted(surf.glob(a.hemisphere+'.*'))}};r['runs'].append(item);save()
            with a.lock.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);item['status']='running';tick=time.perf_counter();env=dict(os.environ,**item['env'],SUBJECTS_DIR=str(out))
                with (out/'command.log').open('w') as log:
                    child=subprocess.Popen(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT);item['pid']=child.pid;save();item['exit_code']=child.wait()
                item['command_seconds']=time.perf_counter()-tick;fcntl.flock(lock,fcntl.LOCK_UN)
            item['status']='complete' if item['exit_code']==0 and output.exists() else 'failed'
            if item['status']!='complete':save();raise RuntimeError('Official seeded sphere replay failed')
            log=(out/'command.log').read_text();item['log_sha256']=sha(out/'command.log');item['seed_warning_present']='seed not set' in log;item['seed_cli_acknowledged']=f'setting seed for random number genererator to {a.seed}' in log;item['fnit_geometry_comparison']=numeric(output,a.fnit_sphere);save()
        r['official_repeat_geometry']=numeric(a.output_root/'repeat_1'/(a.hemisphere+'.sphere'),a.output_root/'repeat_2'/(a.hemisphere+'.sphere'));r['status']='complete';save()
    except Exception as error:r.update(status='failed',error=repr(error));save();raise
if __name__=='__main__':main()
