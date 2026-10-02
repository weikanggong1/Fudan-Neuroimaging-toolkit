"""连续链产生后进行CPU质量/逐区指标诊断；不计性能，不读取官方结果。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel.freesurfer.io as fs
from fnit.recon_all.compare_subject import _topology
from fnit.recon_all.expected_outputs import paths as expected_paths
from fnit.recon_all.native_free import _validate_meshes
from fnit.recon_all.surface_roi_gpu import roi_area_thickness, roi_gray_volume
from fnit.recon_all.place_surface_collision import triangles_intersect
from scipy.spatial import cKDTree


def sphere_quality(vertices,faces):
    a,b,c=vertices[faces[:,0]],vertices[faces[:,1]],vertices[faces[:,2]]
    sign=np.einsum('ij,ij->i',np.cross(b-a,c-a),a+b+c)
    return {'negative_faces':int(np.count_nonzero(sign<0)),
            'zero_orientation_faces':int(np.count_nonzero(sign==0)),
            'radius_min_mm':float(np.linalg.norm(vertices,axis=1).min()),
            'radius_max_mm':float(np.linalg.norm(vertices,axis=1).max())}


def cross_surface_faces(white,pial,faces,cortex):
    """全量cortex面双网格查询；原生tol的接触也计数，无抽样/近邻数量截断。"""
    mask=np.zeros(len(white),bool);mask[cortex]=True
    ids=np.flatnonzero(mask[faces].all(axis=1))
    w,p=white[faces[ids]],pial[faces[ids]]
    wc,pc=w.mean(axis=1),p.mean(axis=1)
    wr=np.linalg.norm(w-wc[:,None],axis=2).max(axis=1)
    pr=np.linalg.norm(p-pc[:,None],axis=2).max(axis=1)
    pl,ph=p.min(axis=1),p.max(axis=1)
    tree=cKDTree(pc)
    pairs=[];coincident=0;tested=0
    for start in range(0,len(w),4096):
        lists=tree.query_ball_point(wc[start:start+4096],wr[start:start+4096]+pr.max()+1e-5)
        for offset,near in enumerate(lists):
            i=start+offset
            candidates=np.asarray(near,dtype=np.int64)
            if not len(candidates):continue
            near=np.sum((pc[candidates]-wc[i])**2,axis=1)<=(pr[candidates]+wr[i]+1e-5)**2
            candidates=candidates[near]
            low,high=w[i].min(axis=0),w[i].max(axis=0)
            candidates=candidates[np.all(pl[candidates]<=high+1e-5,axis=1)&np.all(low<=ph[candidates]+1e-5,axis=1)]
            for j in candidates:
                if i==j and np.array_equal(w[i],p[j]):coincident+=1;continue
                tested+=1
                if triangles_intersect(w[i].astype(np.float32),p[j].astype(np.float32)):
                    pairs.append((int(ids[i]),int(ids[j])))
    return {'cortex_face_count':len(ids),'candidate_pairs_tested':tested,
            'intersecting_pairs_count':len(pairs),'intersecting_face_pairs':pairs,
            'coincident_corresponding_faces':coincident,
            'scope':'all faces with all three vertices in cortex; native 1e-6 plane tolerance, touch-inclusive; no pair sampling'}


def white_pial_quality(white,pial,faces):
    # 对应面法向符号仅是局部逆向诊断，不等于完整两网格交叉检测。
    tri=white[faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    displacement=(pial-white)[faces].mean(axis=1)
    signed=np.einsum('ij,ij->i',normal,displacement)
    return {'reversed_prism_faces':int(np.count_nonzero(signed<0)),
            'same_vertex_distance_max_mm':float(np.linalg.norm(pial-white,axis=1).max()),
            'white_pial_crossings':'see full cortex triangle-pair check',
            'diagnostic_scope':'oriented corresponding-face displacement, not full crossings'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair_root',type=Path)
    args=parser.parse_args();root=args.pair_root
    report={'quality':{},'regions':{},'strict138':{},'overall_equivalence':'not_assessed'}
    rows=[]
    for mode in ('serial','parallel'):
        subject=root/mode/'subject'
        report['quality'][mode]={'existing_pipeline_mesh_validation':_validate_meshes(subject),'hemispheres':{}}
        report['strict138'][mode]={'expected':len(expected_paths()),'paths':{
            relative:{'present':(subject/relative).is_file(),
                      'source':'recomputed affected chain' if relative.startswith(('surf/','label/'))
                                or relative in ('mri/surface.defects.mgz','mri/mrisps.wpa.mgz','mri/mrisps.white.mgz')
                                else 'frozen MRI prefix or absent downstream whole-run output'}
            for relative in expected_paths()},'scope':'stage-only completeness diagnostic; copied prefix is not whole-run evidence'}
        report['regions'][mode]={}
        for hemi in ('lh','rh'):
            white,faces=fs.read_geometry(subject/f'surf/{hemi}.white')
            pial,pfaces=fs.read_geometry(subject/f'surf/{hemi}.pial')
            quality={'white':_topology(white,faces),'pial':_topology(pial,pfaces)}
            if np.array_equal(faces,pfaces):
                quality['white_pial']=white_pial_quality(white,pial,faces)
                quality['cortex_white_pial_crossings']=cross_surface_faces(white,pial,faces,fs.read_label(subject/f'label/{hemi}.cortex.label'))
            for sphere in ('sphere','sphere.reg'):
                xyz,sfaces=fs.read_geometry(subject/f'surf/{hemi}.{sphere}')
                quality[sphere]=sphere_quality(xyz,sfaces)
            report['quality'][mode]['hemispheres'][hemi]=quality
            for atlas in ('aparc','aparc.a2009s','aparc.DKTatlas'):
                annotation=subject/f'label/{hemi}.{atlas}.annot'
                thickness=subject/f'surf/{hemi}.thickness'
                base=roi_area_thickness(subject/f'surf/{hemi}.white',annotation,thickness,device='cpu')
                volumes=roi_gray_volume(subject/f'surf/{hemi}.white',subject/f'surf/{hemi}.pial',thickness,annotation,device='cpu')
                key=f'{hemi}.{atlas}'
                report['regions'][mode][key]={name:{'vertices':count,'area_mm2':area,
                    'thickness_mean_mm':mean,'thickness_std_mm':std,'gray_volume_no_th3_mm3':volumes.get(name)}
                    for name,(count,area,mean,std) in base.items()}
    for atlas,regions in report['regions']['serial'].items():
        for name,a in regions.items():
            b=report['regions']['parallel'][atlas].get(name,{})
            row={'atlas':atlas,'region':name}
            for metric,value in a.items():
                row['serial_'+metric]=value;row['parallel_'+metric]=b.get(metric)
                if value is not None and b.get(metric) is not None:row['difference_'+metric]=b[metric]-value
            rows.append(row)
    report['regional_differences']=rows
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root/'quality_and_regions.json').write_text(json.dumps(report,indent=2))
    with (root/'regional_differences.csv').open('w') as stream:
        fields=sorted(set().union(*(row.keys() for row in rows)))
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
