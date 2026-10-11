"""固定面组织候选的完整三角面距离、同索引位移和T1局部图；只读。"""
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
from diagnose_topology_volume_shapes import exact_distances
from diagnose_topology_volume_local_shape import plane_segments


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-surface',type=Path,required=True)
    parser.add_argument('--baseline-surface',type=Path,required=True)
    parser.add_argument('--GA-reference-surface',dest='GA_reference_surface',type=Path,required=True)
    parser.add_argument('--T1-volume',dest='T1_volume',type=Path,required=True)
    parser.add_argument('--surface-distance-module',type=Path,required=True)
    parser.add_argument('--baseline-distance-checkpoint',type=Path,required=True)
    parser.add_argument('--baseline-distance-report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('_topology_tissue_shape_reference',args.surface_distance_module)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    b,fb,mb=fsio.read_geometry(str(args.baseline_surface),read_metadata=True)
    c,fc,mc=fsio.read_geometry(str(args.candidate_surface),read_metadata=True)
    g,fg,mg=fsio.read_geometry(str(args.GA_reference_surface),read_metadata=True)
    if not np.array_equal(fb,fc) or b.shape!=c.shape or not all(np.array_equal(mb[k],mc[k]) for k in mb):
        raise ValueError('candidate and baseline must have identical ordered topology and volume geometry')
    old=json.loads(args.baseline_distance_report.read_text());matching=[]
    for case in old['cases']:
        matching += [item for item in case['comparisons'] if item['candidate_sha256']==sha(args.baseline_surface)
                     and item['reference_sha256']==sha(args.GA_reference_surface)]
    if len(matching)!=1 or old['distance_module_sha256']!=sha(args.surface_distance_module):
        raise ValueError('baseline distance source/target/module SHA binding missing or ambiguous')
    checkpoint=np.load(args.baseline_distance_checkpoint);bf=checkpoint['forward'];br=checkpoint['reverse']
    if bf.shape!=(len(b),) or br.shape!=(len(g),) or not np.isfinite(bf).all() or not np.isfinite(br).all():
        raise ValueError('invalid complete baseline distance checkpoint')
    for actual, expected in [(reference._summary(bf),matching[0]['candidate_to_reference_triangle']),
                             (reference._summary(br),matching[0]['reference_to_candidate_triangle'])]:
        if actual.keys()!=expected.keys() or not all(np.isclose(actual[k],expected[k],rtol=0,atol=1e-12) for k in actual):
            raise ValueError('checkpoint metrics disagree with source-bound completed baseline report')
    displacement=np.linalg.norm(c-b,axis=1);changed=np.flatnonzero(displacement)
    forward=bf.copy()
    # 固定GA目标不变时，完全相同的源坐标可以复用其完整三角面距离。
    if len(changed):forward[changed]=reference._point_to_mesh(c[changed],g,fg)
    reverse,reverse_proof=exact_distances(g,c,fc,reference)
    np.savez(args.output_dir/'complete_distances.private.npz',forward=forward,reverse=reverse,
             baseline_forward=bf,baseline_reverse=br,displacement=displacement)
    baseline_max=int(bf.argmax());candidate_max=int(forward.argmax())
    image=nib.load(str(args.T1_volume));t1=np.asarray(image.dataobj);inv=np.linalg.inv(image.header.get_vox2ras_tkr())
    to_voxel=lambda vertices:vertices@inv[:3,:3].T+inv[:3,3]
    meshes=[(to_voxel(g),fg,'orange','GA'),(to_voxel(b),fb,'cyan','digital v8'),(to_voxel(c),fc,'magenta','tissue candidate')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    figure,axes=plt.subplots(2,3,figsize=(11,7))
    local=[]
    for row,(label,vertex) in enumerate([('baseline maximum',baseline_max),('candidate maximum',candidate_max)]):
        point=to_voxel(c[vertex:vertex+1])[0];index=np.rint(point).astype(np.int64)
        if np.any(index<0) or np.any(index>=np.asarray(t1.shape)):raise ValueError('local point outside declared T1')
        for axis,ax in enumerate(axes[row]):
            others=[v for v in range(3) if v!=axis];plane=int(index[axis])
            ax.imshow(np.take(t1,plane,axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=160)
            for points,faces,color,name in meshes:
                ax.add_collection(LineCollection(plane_segments(points,faces,axis,plane),colors=color,linewidths=.8))
            ax.scatter(point[others[0]],point[others[1]],s=15,c='yellow')
            ax.set_xlim(point[others[0]]-18,point[others[0]]+18);ax.set_ylim(point[others[1]]-18,point[others[1]]+18)
            ax.set_title(f'{label}, axis {axis} voxel {plane}');ax.axis('off')
        local.append({'selection':label,'same_index_vertex':vertex,'baseline_surface_RAS_mm':b[vertex].tolist(),
                      'candidate_surface_RAS_mm':c[vertex].tolist(),'T1_voxel':point.tolist(),
                      'baseline_to_GA_triangle_mm':float(bf[vertex]),'candidate_to_GA_triangle_mm':float(forward[vertex]),
                      'coordinate_displacement_mm':float(displacement[vertex])})
    figure.suptitle('Orange: original GA; cyan: digital v8; magenta: tissue candidate')
    figure.tight_layout();figure.savefig(args.output_dir/'local_shape.png',dpi=170);plt.close(figure)
    report={'scope':'posthoc frozen local primitive shape diagnosis, not rawT1 or complete GA',
            'execution_status':'complete','script_sha256':sha(__file__),'distance_module_sha256':sha(args.surface_distance_module),
            'input_sha256':{key:sha(path) for key,path in {'candidate':args.candidate_surface,'baseline':args.baseline_surface,
                'GA_reference':args.GA_reference_surface,'T1':args.T1_volume,'baseline_distance_checkpoint':args.baseline_distance_checkpoint,
                'baseline_distance_report':args.baseline_distance_report}.items()},
            'ordered_candidate_baseline_faces_equal':True,'same_index_comparison_to_GA':False,
            'candidate_vs_baseline_same_index_displacement_mm':reference._summary(displacement),
            'baseline_to_GA_triangle_mm':reference._summary(bf),'GA_to_baseline_triangle_mm':reference._summary(br),
            'candidate_to_GA_triangle_mm':reference._summary(forward),'GA_to_candidate_triangle_mm':reference._summary(reverse),
            'candidate_forward_proof':{'all_source_vertices_evaluated':True,'unchanged_source_coordinates_with_fixed_target_reused':len(b)-len(changed),
                'changed_source_points_full_triangle_search':len(changed)},'candidate_reverse_proof':reverse_proof,
            'local_points':local,'complete_distances_npz_sha256':sha(args.output_dir/'complete_distances.private.npz'),
            'figure':'local_shape.png','figure_sha256':sha(args.output_dir/'local_shape.png'),
            'overall_metric_equivalence':'not_assessed','complete_dynamic_topology_solver':False,
            'posthoc_seconds_including_IO_and_plots':time.perf_counter()-started}
    (args.output_dir/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
