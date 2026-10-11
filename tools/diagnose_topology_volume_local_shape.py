"""只读定位数字拓扑与原GA的最大局部形状差；不修改网格或生产算法。"""
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


def closest_triangle_point(point, triangle):
    """同成熟距离公式的投影与闭合边界，返回实际最近坐标而非最近顶点。"""
    a,b,c=triangle
    ab,ac,bc=b-a,c-a,c-b
    normal=np.cross(ab,ac);normal_sq=float(normal@normal)
    projection=point-normal*float((point-a)@normal)/max(normal_sq,1e-30)
    d00,d01,d11=float(ab@ab),float(ab@ac),float(ac@ac)
    d20,d21=float((projection-a)@ab),float((projection-a)@ac)
    denominator=max(d00*d11-d01*d01,1e-30)
    u,v=(d11*d20-d01*d21)/denominator,(d00*d21-d01*d20)/denominator
    if normal_sq>1e-20 and u>=-1e-12 and v>=-1e-12 and u+v<=1+1e-12:
        return projection
    candidates=[]
    for start,vector in [(a,ab),(b,bc),(c,a-c)]:
        fraction=float((point-start)@vector)/max(float(vector@vector),1e-30)
        candidates.append(start+np.clip(fraction,0,1)*vector)
    candidates=np.asarray(candidates)
    return candidates[np.linalg.norm(candidates-point,axis=1).argmin()]


def nearest_triangle(point,vertices,faces,reference):
    triangles=vertices[faces].astype(np.float64);centers=triangles.mean(axis=1)
    radius=float(np.linalg.norm(triangles-centers[:,None],axis=2).max())
    upper=float(cKDTree(vertices[np.unique(faces)]).query(point)[0])
    ids=np.asarray(cKDTree(centers).query_ball_point(point,upper+radius+1e-7),np.int64)
    distances=reference._point_triangle_distance(point,triangles[ids])
    selected=int(ids[int(distances.argmin())]);value=float(distances.min())
    closest=closest_triangle_point(point,triangles[selected])
    if not np.isclose(np.linalg.norm(closest-point),value,rtol=1e-9,atol=1e-9):
        raise RuntimeError('closest coordinate disagrees with established full triangle distance')
    return closest,selected,value


def plane_segments(vertices,faces,axis,index):
    """实际三角面与体素截面的交线，避免最近顶点冒充表面边界。"""
    triangles=vertices[faces];delta=triangles[:,:,axis]-index
    points=np.zeros((len(faces),2,3));counts=np.zeros(len(faces),np.int32)
    for first,second in [(0,1),(1,2),(2,0)]:
        valid=((delta[:,first]<=0)&(delta[:,second]>0))|((delta[:,second]<=0)&(delta[:,first]>0))
        ids=np.flatnonzero(valid)
        fraction=-delta[ids,first]/(delta[ids,second]-delta[ids,first])
        intersection=triangles[ids,first]+fraction[:,None]*(triangles[ids,second]-triangles[ids,first])
        points[ids,counts[ids]]=intersection;counts[ids]+=1
    keep=counts==2;other=[i for i in range(3) if i!=axis]
    return points[keep][:,:,other]


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject',type=Path,action='append',required=True)
    parser.add_argument('--topology-run',type=Path,required=True)
    parser.add_argument('--shape-run',type=Path,required=True)
    parser.add_argument('--surface-distance-module',type=Path,required=True)
    parser.add_argument('--color-lut',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('_topology_local_shape_reference',args.surface_distance_module)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    names={}
    for line in args.color_lut.read_text().splitlines():
        tokens=line.split()
        if len(tokens)>1 and tokens[0].isdigit(): names[int(tokens[0])]=tokens[1]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    report={'scope':'read-only maximum local shape diagnosis, not a new production run',
            'script_sha256':sha(__file__),'surface_distance_module_sha256':sha(args.surface_distance_module),
            'color_lut_sha256':sha(args.color_lut),'whole_metric_equivalence':'not_assessed','cases':[],'execution_status':'running'}
    save=lambda:(args.output_dir/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for case,subject in enumerate(args.subject):
        image=nib.load(str(subject/'mri/T1.mgz'));t1=np.asarray(image.dataobj)
        transform=np.linalg.inv(image.header.get_vox2ras_tkr())
        label_image=nib.load(str(subject/'mri/wmparc.mgz'));labels=np.asarray(label_image.dataobj)
        if labels.shape!=t1.shape or not np.allclose(image.affine,label_image.affine,rtol=0,atol=1e-6):
            raise ValueError('label/T1 grid mismatch')
        to_voxel=lambda v:v@transform[:3,:3].T+transform[:3,3]
        for hemi,value in [('lh',255),('rh',127)]:
            candidate=args.topology_run/f'case{case}_{hemi}'/f'{hemi}.orig.premesh'
            original=subject/'surf'/f'{hemi}.orig.nofix';ga=subject/'surf'/f'{hemi}.orig.premesh'
            xyz,faces=fsio.read_geometry(str(candidate));orig,orig_faces=fsio.read_geometry(str(original));ga_xyz,ga_faces=fsio.read_geometry(str(ga))
            frozen=np.load(args.shape_run/f'case{case}_{hemi}_orig.premesh.distances.private.npz')
            forward=frozen['forward'];number=int(forward.argmax());point=xyz[number]
            ga_closest,ga_face,distance=nearest_triangle(point,ga_xyz,ga_faces,reference)
            if not np.isclose(distance,float(forward[number]),rtol=1e-12,atol=1e-12): raise RuntimeError('checkpoint distance mismatch')
            original_frozen=np.load(args.shape_run/f'case{case}_{hemi}_orig.nofix.distances.private.npz')
            original_distance=float(original_frozen['forward'][number])
            ga_to_orig_distance=float(reference._point_to_mesh(ga_closest[None],orig,orig_faces)[0])
            point_voxel=to_voxel(point);ga_voxel=to_voxel(ga_closest)
            index=np.rint(point_voxel).astype(np.int64)
            if np.any(index<0) or np.any(index>=np.asarray(t1.shape)): raise ValueError('maximum point outside T1')
            local_label=int(labels[tuple(index)])
            before=np.asarray(nib.load(str(subject/'mri'/f'filled-pretess{value}.mgz')).dataobj)==value
            after=np.asarray(nib.load(str(args.topology_run/f'case{case}_{hemi}/filled.corrected.mgz')).dataobj)==value
            deleted=before&~after
            low=np.maximum(index-3,0);high=np.minimum(index+4,t1.shape)
            local_deleted=int(deleted[tuple(slice(int(a),int(b)) for a,b in zip(low,high))].sum())
            fig,axes=plt.subplots(1,3,figsize=(11,4))
            voxel_meshes=[(to_voxel(orig),orig_faces,'white'),(to_voxel(ga_xyz),ga_faces,'orange'),(to_voxel(xyz),faces,'cyan')]
            for axis,ax in enumerate(axes):
                other=[i for i in range(3) if i!=axis];plane=int(index[axis])
                ax.imshow(np.take(t1,plane,axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=160)
                for vertices,triangles,color in voxel_meshes:
                    ax.add_collection(LineCollection(plane_segments(vertices,triangles,axis,plane),colors=color,linewidths=.7))
                red=np.zeros((*np.take(deleted,plane,axis=axis).T.shape,4));red[np.take(deleted,plane,axis=axis).T]=[1,0,0,.75];ax.imshow(red,origin='lower')
                ax.plot([point_voxel[other[0]],ga_voxel[other[0]]],[point_voxel[other[1]],ga_voxel[other[1]]],color='yellow',linewidth=1)
                ax.scatter(point_voxel[other[0]],point_voxel[other[1]],c='cyan',s=10)
                ax.set_xlim(min(point_voxel[other[0]],ga_voxel[other[0]])-20,max(point_voxel[other[0]],ga_voxel[other[0]])+20)
                ax.set_ylim(min(point_voxel[other[1]],ga_voxel[other[1]])-20,max(point_voxel[other[1]],ga_voxel[other[1]])+20)
                ax.set_title(f'voxel axis {axis}={plane}');ax.axis('off')
            fig.suptitle(f'case{case} {hemi}: max {distance:.3f}mm; cyan candidate / orange GA / white nofix; red deletion')
            fig.tight_layout();figure=args.output_dir/f'case{case}_{hemi}.maximum.png';fig.savefig(figure,dpi=180);plt.close(fig)
            report['cases'].append({'case_number':case,'hemisphere':hemi,'input_sha256':{'candidate':sha(candidate),'original_nofix':sha(original),'original_GA':sha(ga),'T1':sha(subject/'mri/T1.mgz'),'wmparc':sha(subject/'mri/wmparc.mgz'),'distance_checkpoint':sha(args.shape_run/f'case{case}_{hemi}_orig.premesh.distances.private.npz')},
                'maximum_candidate_vertex_index':number,'candidate_point_surface_RAS_mm':point.tolist(),'candidate_point_voxel_xyz':point_voxel.tolist(),
                'nearest_GA_point_surface_RAS_mm':ga_closest.tolist(),'nearest_GA_face_index':ga_face,'candidate_to_GA_triangle_distance_mm':distance,
                'same_candidate_point_to_original_nofix_triangle_mm':original_distance,'same_point_is_exact_original_nofix_vertex':bool(cKDTree(orig).query(point)[0]==0),
                'nearest_GA_point_to_original_nofix_triangle_mm':ga_to_orig_distance,'deleted_voxels_in_7_cube_at_candidate_point':local_deleted,
                'contextual_baseline_wmparc_label':local_label,'contextual_baseline_wmparc_name':names.get(local_label,'not_in_LUT'),
                'label_interpretation':'baseline79 parcellation context; not a correctness label for new topology',
                'figure':figure.name,'figure_sha256':sha(figure),'numeric_reproduction_diagnostic':'different topology, no indexed vertex equivalence'});save()
    report['execution_status']='complete';report['diagnostic_seconds_including_io_and_plotting']=time.perf_counter()-started;save()


if __name__=='__main__':
    main()
