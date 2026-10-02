"""严格/预声明算子容差分别评估实际自产受影响文件；未知类型不伪造数值比较。"""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
from fnit.recon_all.compare_subject import _numeric, _labels, _surface, _topology

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('pair_root',type=Path)
args=parser.parse_args();root=args.pair_root
run=json.loads((root/'result.json').read_text());tolerances=run['numeric_tolerances']
a,b=root/'serial/subject',root/'parallel/subject'
paths=run['results']['parallel']['published']
report={'scope':'newly produced affected files only; prefix copies excluded','run_commit':run['commit'],
        'strict_reproduction':run['strict_reproduction'],'overall_equivalence':'not_assessed','checks':{},
        'tolerance_aliases_declared_before_parallel_results':{'white.preaparc.H':'white.H','white.preaparc.K':'white.K'},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
correspondence={}
for hemi in ('lh','rh'):
    x,f=fs.read_geometry(a/f'surf/{hemi}.white');y,g=fs.read_geometry(b/f'surf/{hemi}.white')
    correspondence[hemi]=x.shape==y.shape and np.array_equal(f,g)
for relative in sorted(set(paths)):
    x,y=a/relative,b/relative
    if relative.startswith('scripts/'):continue
    if not x.exists() or not y.exists():
        report['checks'][relative]={'status':'missing'};continue
    suffix=x.name[3:] if x.name.startswith(('lh.','rh.')) else x.name
    tolerance_key={'white.preaparc.H':'white.H','white.preaparc.K':'white.K'}.get(suffix,suffix)
    tol=tolerances.get(tolerance_key,{'atol':0.,'rtol':0.})
    if x.suffix in ('.mgz','.mgh'):
        xx,yy=nib.load(x),nib.load(y);vx,vy=np.asarray(xx.dataobj),np.asarray(yy.dataobj)
        check=_numeric(vx.ravel(),vy.ravel(),{'atol':0.,'rtol':0.},50)
        check.update(dtype_equal=vx.dtype==vy.dtype,shape_equal=vx.shape==vy.shape,
                     affine_max_abs=float(np.max(np.abs(xx.affine-yy.affine))))
        if x.name=='surface.defects.mgz':check['per_label']=_labels(vx.ravel(),vy.ravel(),1.,50)
    elif x.name.endswith('.annot'):
        hemi=x.name[:2]
        if not correspondence[hemi]:check={'status':'blocked','reason':'ordered mesh correspondence absent'}
        else:
            vx,cx,nx=fs.read_annot(x,orig_ids=True);vy,cy,ny=fs.read_annot(y,orig_ids=True)
            check=_labels(vx,vy,1.,50);check['color_table_equal']=np.array_equal(cx,cy) and nx==ny
    elif x.name.endswith('.label'):
        vx,sx=fs.read_label(x,read_scalars=True);vy,sy=fs.read_label(y,read_scalars=True)
        check={'status':'passed' if np.array_equal(vx,vy) and np.array_equal(sx,sy) else 'failed',
               'ordered_vertex_ids_equal':np.array_equal(vx,vy),'scalars_equal':np.array_equal(sx,sy)}
    elif relative.startswith('surf/') and suffix in ('orig','orig.nofix','orig.premesh','orig.raw.quad','white.preaparc','smoothwm','smoothwm.nofix','inflated','inflated.nofix','sphere','sphere.reg','qsphere.nofix','topology-centered.sphere','white','pial','pial.T1'):
        xx,yy=fs.read_geometry(x),fs.read_geometry(y)
        if suffix in ('orig.nofix','orig.raw.quad','qsphere.nofix','smoothwm.nofix','inflated.nofix','topology-centered.sphere'):
            vx,fx=xx;vy,fy=yy
            comparable=vx.shape==vy.shape and np.array_equal(fx,fy)
            check=_numeric(np.zeros(len(vx)),np.linalg.norm(vy-vx,axis=1),tolerances['coordinates'],50) if comparable else {'status':'blocked','reason':'pre-fix ordered mesh differs'}
            check.update(ordered_faces_equal=comparable,reference_topology=_topology(vx,fx),candidate_topology=_topology(vy,fy),topology_scope='pre-fix defects retained; final topology gate does not apply here')
        else:check=_surface(xx,yy,tolerances['coordinates'],50)
    elif relative.startswith('surf/') and (suffix in ('thickness','area','area.pial','area.mid','volume','curv','curv.pial','sulc','avg_curv','white.preaparc.H','white.preaparc.K','inflated.H','inflated.K') or suffix.startswith('smoothwm.') and suffix.endswith('.crv')):
        if not correspondence[x.name[:2]]:check={'status':'blocked','reason':'ordered mesh correspondence absent'}
        else:check=_numeric(fs.read_morph_data(x),fs.read_morph_data(y),tol,50)
    else:
        same=x.read_bytes()==y.read_bytes()
        check={'status':'passed' if same else 'not_assessed','bytes_equal':same,
               'reason':'strict bytes only; nonstandard diagnostic text/binary not interpreted as morphometry'}
    report['checks'][relative]=check
for key in ('passed','failed','blocked','missing','not_assessed'):
    report[key+'_count']=sum(value['status']==key for value in report['checks'].values())
report['numeric_operator_regression']='passed' if all(value['status']=='passed' for value in report['checks'].values()) else 'failed_or_not_assessed'
(root/'affected_numeric_regression.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x))
print(json.dumps({key:report[key] for key in ('passed_count','failed_count','blocked_count','missing_count','not_assessed_count','numeric_operator_regression')}))
