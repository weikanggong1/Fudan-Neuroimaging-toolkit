"""收集同输入真实阶段，先核对输入SHA与有序面，再报告坐标误差。"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import nibabel.freesurfer.io as fsio
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def geometry(a,b):
    v,f=zip(*(fsio.read_geometry(str(p)) for p in (a,b)));raw_quad=None
    if a.name.endswith('.quad'):
        from fnit.recon_all.extract_main_component_python import read_quad_core
        av,aq,_=read_quad_core(a);bv,bq,_=read_quad_core(b)
        raw_quad={'reference_quad_count':len(aq),'candidate_quad_count':len(bq),'ordered_quads_equal':bool(np.array_equal(aq,bq)),'coordinates_equal':bool(np.array_equal(av,bv)),'triangles_read_note':'nibabel returned triangles are not used as proof of FreeSurfer quad split ordering'}
    aligned=v[0].shape==v[1].shape and np.array_equal(f[0],f[1]);r={'sha256':[sha(a),sha(b)],'vertices':[len(x) for x in v],'faces':[len(x) for x in f],'ordered_faces_equal':bool(aligned),'coordinates_equal':bool(aligned and np.array_equal(v[0],v[1])),'unit':'surface RAS mm'}
    if raw_quad is not None:r['raw_quad_geometry']=raw_quad
    if aligned:
        d=np.linalg.norm(v[1]-v[0],axis=1);r['displacement_mm']={'mean':float(d.mean()),'p99':float(np.quantile(d,.99)),'max':float(d.max()),'different_vertices':int(np.count_nonzero(d))}
    else:r['indexed_displacement']='not_assessed; correspondence not proven'
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.root=a.root.resolve();sys.path.insert(0,str(a.root.parent/'baseline_runtime_816e5610/src'));status=json.loads((a.root/'queue_status.json').read_text());rows=[]
    base=a.root.parent.parent
    cps={'sub01':base/'parallel_20261002/whole_sub01_candidate_8d750e2','sub02':base/'parallel_20261002/whole_sub02_candidate_8d750e2_retry_v3'}
    for subject,cp in cps.items():
        for h in ('lh','rh'):
            tess=a.root/'frozen_v2'/subject/h/'tessellation'
            for name in ('orig.raw.quad','orig.nofix'):
                x=tess/(h+'.'+name);y=cp/'surf'/(h+'.'+name)
                if x.exists():
                    pretess=cp/'mri'/('filled-pretess'+('255' if h=='lh' else '127')+'.mgz')
                    rows.append(dict(subject=subject,hemisphere=h,stage=name,input_pretess_sha256=sha(pretess),kind='official same pretess input versus saved FNIT 8d750e2 geometry',**geometry(x,y)))
            for stage in ('remesh','sphere','register'):
                stage_root=a.root/'frozen_v2'/subject/h/stage;dirs=[stage_root/k for k in ('official','fnit')]
                if not all((x/'report.json').exists() for x in dirs):continue
                reports=[json.loads((x/'report.json').read_text()) for x in dirs]
                r={'subject':subject,'hemisphere':h,'stage':stage,'status':[x['status'] for x in reports],'same_input_sha256':reports[0].get('input_sha256')==reports[1].get('input_sha256'),'official_seconds':reports[0].get('command_wall_seconds'),'fnit_command_seconds':reports[1].get('command_wall_seconds'),'fnit_api_seconds':reports[1].get('stage',{}).get('total_seconds_including_io'),'official_program_sha256':reports[0].get('program_sha256'),'fnit_commit':reports[1].get('args',{}).get('commit')}
                if all(x['status']=='complete' for x in reports):r.update(geometry(*(x/(h+'.'+stage) for x in dirs)))
                rows.append(r)
    result={'kind':'frozen same-input checkpoint; not whole continuous validation','overall_equivalence':'not_assessed','queue_status':status['status'],'queue_pid':status['pid'],'rows':rows,'pending':[{k:v for k,v in x.items() if k in ('stage','subject','hemisphere','kind','status','pid')} for x in status['commands'] if x['status'] not in ('complete','failed')]};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
