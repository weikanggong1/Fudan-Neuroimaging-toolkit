"""仅供隔离官方同输入benchmark；不进入FNIT生产调用图。"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,shutil,subprocess,time


def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','reference-home','assets','output-root'):
        p.add_argument('--'+name,type=pathlib.Path,required=True)
    p.add_argument('--hemisphere',choices=('lh','rh'),required=True)
    p.add_argument('--stage',choices=('remesh','sphere','register'),required=True)
    a=p.parse_args()
    for name in ('checkpoint','reference_home','assets','output_root'):
        setattr(a,name,getattr(a,name).resolve())
    h=a.hemisphere;a.output_root.mkdir(parents=True,exist_ok=True)
    context=a.output_root/'subject';surf=context/'surf';surf.mkdir(parents=True,exist_ok=True)
    # 真正复制受控输入，官方程序即使写同名辅助文件也不能修改冻结来源。
    for item in (a.checkpoint/'surf').glob(h+'.*'):
        if item.is_file():shutil.copyfile(item,surf/item.name)
    label=context/'label';label.mkdir(exist_ok=True)
    for item in (a.checkpoint/'label').glob(h+'.*'):
        if item.is_file():shutil.copyfile(item,label/item.name)
    atlas=a.assets/'average'/(h+'.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif')
    name={'remesh':'mris_remesh','sphere':'mris_sphere','register':'mris_register'}[a.stage]
    binary=a.reference_home/'bin'/name;output=a.output_root/(h+'.'+a.stage)
    if a.stage=='remesh':cmd=[str(binary),'--remesh','--iters','3','--input',str(surf/(h+'.orig.premesh')),'--output',str(output)];inputs={'premesh':surf/(h+'.orig.premesh')}
    elif a.stage=='sphere':cmd=[str(binary),'-threads','4',str(surf/(h+'.inflated')),str(output)];inputs={k:surf/(h+'.'+k) for k in ('inflated','smoothwm')}
    else:cmd=[str(binary),'-threads','4',str(surf/(h+'.sphere')),str(atlas),str(output)];inputs={k:surf/(h+'.'+k) for k in ('sphere','smoothwm','sulc')};inputs['atlas']=atlas
    report={'kind':'isolated_official_same_input','stage':a.stage,'hemisphere':h,'command':cmd,'program_sha256':sha(binary),'input_sha256':{k:sha(v) for k,v in inputs.items()},'source_checkpoint_untouched':True,'load_before':os.getloadavg(),'thread_budget':4,'context_copy_boundary':'private copy preparation precedes measured official command; official wall includes complete command IO; report separate preparation'}
    env=dict(os.environ,FREESURFER_HOME=str(a.reference_home),SUBJECTS_DIR=str(context.parent),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    t0=time.perf_counter()
    with (a.output_root/'command.log').open('w') as stream:
        result=subprocess.run(cmd,cwd=a.output_root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    report.update(exit_code=result.returncode,command_wall_seconds=time.perf_counter()-t0,load_after=os.getloadavg(),status='complete' if result.returncode==0 and output.is_file() else 'failed')
    if output.is_file():report['output_sha256']=sha(output)
    (a.output_root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('stage','hemisphere','status','command_wall_seconds')},indent=2))
    if report['status']!='complete':raise SystemExit(1)

if __name__=='__main__':main()
