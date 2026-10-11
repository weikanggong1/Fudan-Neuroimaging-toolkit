"""完整数字拓扑候选形状诊断：双向全顶点到三角面距离与真实T1删除区图。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.spatial import cKDTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_distances(source,target,faces,reference):
    """仅对目标有序面使用的完全相同顶点证明0；其余复用完整原距离函数。"""
    used=target[np.unique(faces)]
    upper=cKDTree(used).query(source,workers=4)[0]
    zero=upper==0
    result=np.zeros(len(source),np.float64)
    if not zero.all():
        result[~zero]=reference._point_to_mesh(source[~zero],target,faces)
    return result,{'exact_zero_from_identical_used_target_vertex':int(zero.sum()),
                   'full_triangle_search_vertices':int((~zero).sum()),
                   'total_vertices':len(source),'all_vertices_evaluated':True}


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject',type=Path,action='append',required=True)
    parser.add_argument('--topology-run',type=Path,required=True)
    parser.add_argument('--surface-distance-module',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('_topology_shape_reference',args.surface_distance_module)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report={'scope':'posthoc topology candidate shape diagnosis, not rawT1 execution',
            'distance_definition':'all source vertices to all target triangle mesh, both directions',
            'script_sha256':sha(__file__),'distance_module_sha256':sha(args.surface_distance_module),
            'overall_equivalence':'not_assessed','cases':[],'execution_status':'running'}
    save=lambda:(args.output_dir/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for case,subject in enumerate(args.subject):
        image=nib.load(str(subject/'mri/T1.mgz'));t1=np.asarray(image.dataobj)
        for hemi,value in [('lh',255),('rh',127)]:
            candidate=args.topology_run/f'case{case}_{hemi}'/f'{hemi}.orig.premesh'
            b,fb=fsio.read_geometry(str(candidate))
            comparisons=[]
            for stage in ['orig.nofix','orig.premesh']:
                original=subject/'surf'/f'{hemi}.{stage}'
                a,fa=fsio.read_geometry(str(original))
                tick=time.perf_counter()
                forward,proof_a=exact_distances(b,a,fa,reference)
                reverse,proof_b=exact_distances(a,b,fb,reference)
                comparisons.append({'reference_stage':stage,'reference_sha256':sha(original),
                    'candidate_sha256':sha(candidate),'reference_vertices':len(a),'candidate_vertices':len(b),
                    'reference_faces':len(fa),'candidate_faces':len(fb),
                    'same_index_correspondence':False,'candidate_to_reference_triangle':reference._summary(forward),
                    'reference_to_candidate_triangle':reference._summary(reverse),'forward_proof':proof_a,'reverse_proof':proof_b,
                    'diagnostic_seconds':time.perf_counter()-tick})
                np.savez(args.output_dir/f'case{case}_{hemi}_{stage}.distances.private.npz',forward=forward,reverse=reverse)
            before_image=nib.load(str(subject/'mri'/f'filled-pretess{value}.mgz'))
            after_image=nib.load(str(args.topology_run/f'case{case}_{hemi}'/'filled.corrected.mgz'))
            if t1.shape!=before_image.shape or not np.allclose(image.affine,before_image.affine,atol=1e-6,rtol=0):
                raise ValueError('T1 overlay geometry differs from candidate input grid')
            before=np.asarray(before_image.dataobj)==value;after=np.asarray(after_image.dataobj)==value
            removed=before&~after
            center=np.argwhere(before).mean(axis=0)
            slices=[int(round(center[1]+offset)) for offset in [-15,0,15]]
            fig,axes=plt.subplots(1,3,figsize=(9,3.5))
            for ax,index in zip(axes,slices):
                ax.imshow(t1[:,index,:].T,cmap='gray',origin='lower',vmin=0,vmax=160)
                ax.contour(after[:,index,:].T,levels=[.5],colors=['cyan'],linewidths=.5)
                overlay=np.zeros((*removed[:,index,:].T.shape,4));overlay[removed[:,index,:].T]=[1,.1,.05,.9]
                ax.imshow(overlay,origin='lower');ax.set_title(f'{hemi}: conformed y={index}');ax.axis('off')
            fig.suptitle(f'case {case}: repaired boundary cyan, removed voxels red')
            fig.tight_layout();figure=args.output_dir/f'case{case}_{hemi}.png';fig.savefig(figure,dpi=180);plt.close(fig)
            report['cases'].append({'case_number':case,'hemisphere':hemi,'comparisons':comparisons,
                'figure':figure.name,'figure_sha256':sha(figure),'overlay_T1_sha256':sha(subject/'mri/T1.mgz'),
                'conformed_slice_indices_y':slices,'local_anatomical_review':'figure_and_distance_values_pending_whole_downstream_metrics'})
            save()
    report['execution_status']='complete';report['diagnosis_total_seconds']=time.perf_counter()-started;save()


if __name__=='__main__':
    main()
