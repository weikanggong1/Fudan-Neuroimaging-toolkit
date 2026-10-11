#!/usr/bin/env python3
"""只读定位首patch的三维相交，区分保留外部面和新面，不移动坐标。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.spatial import cKDTree

from diagnose_recon_topology_retessellation import reconstruct_candidate
from fnit.recon_all.mris_remove_intersection_python import _triangles_intersect
from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intersection_pairs(*, vertices, faces):
    """复用成熟triangle谓词；完整保守球包围/box候选，不漏new/outside。"""
    triangles = np.asarray(vertices,np.float64)[faces]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles-centers[:,None],axis=2).max(axis=1)
    pairs = cKDTree(centers).query_pairs(2*float(radii.max())+1e-5,output_type='ndarray')
    a,b = pairs.T
    pairs = pairs[np.sum((centers[a]-centers[b])**2,axis=1)<=(radii[a]+radii[b]+1e-5)**2]
    a,b = pairs.T;low,high = triangles.min(axis=1),triangles.max(axis=1)
    pairs=pairs[np.all(low[a]<=high[b]+1e-5,axis=1)&np.all(low[b]<=high[a]+1e-5,axis=1)]
    a,b=pairs.T
    pairs=pairs[np.all(faces[a,:,None]!=faces[b,None,:],axis=(1,2))]
    return np.asarray([(int(a),int(b)) for a,b in pairs if _triangles_intersect(triangles[a],triangles[b])],np.int32).reshape(-1,2)


def main():
    start=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-surface',type=Path,required=True)
    parser.add_argument('--intensity-volume',type=Path,required=True)
    parser.add_argument('--stage-report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError('new output directory required')
    args.output_dir.mkdir(parents=True)
    report=json.loads(args.stage_report.read_text())
    if sha(args.input_surface)!=report['input_sha256']['orig_nofix'] or sha(args.intensity_volume)!=report['input_sha256']['MRI']:
        raise ValueError('same source surface and MRI hashes required')
    xyz,old_faces,metadata=fsio.read_geometry(str(args.input_surface),read_metadata=True)
    faces,_,counts=reconstruct_candidate(faces=old_faces,nvertices=len(xyz),report=report)
    tick=time.perf_counter();old_pairs=intersection_pairs(vertices=xyz,faces=old_faces);old_seconds=time.perf_counter()-tick
    tick=time.perf_counter();pairs=intersection_pairs(vertices=xyz,faces=faces);new_seconds=time.perf_counter()-tick
    if len(np.unique(pairs))!=report['geometric_intersecting_faces']:
        raise ValueError('complete geometry replay differs from preserved producer')
    outside=report['outside_face_source_indices'];split=len(outside)
    retained=set(report['retained_inside_source_indices'])
    rows=[]
    for a,b in pairs:
        rows.append({'candidate_face_indices':[int(a),int(b)],
                     'face_origin':['preserved_outside' if f<split else 'new_patch' for f in (a,b)],
                     'original_face_indices':[outside[f] if f<split else None for f in (a,b)],
                     'triangle_source_vertex_indices':faces[[a,b]].tolist(),
                     'retained_inside_vertices':sorted(set(map(int,faces[[a,b]].ravel()))&retained),
                     'shares_source_vertex':bool(np.any(faces[a,:,None]==faces[b,None,:]))})
    result={'scope':'read-only preserved failed producer geometry; no repair, no surface written',
            'input_sha256':report['input_sha256'],'producer_source_sha256':report['source_sha256'],
            'stage_report_sha256':sha(args.stage_report),'diagnostic_source_sha256':sha(__file__),
            'reconstruction_source_sha256':sha(Path(__file__).with_name('diagnose_recon_topology_retessellation.py')),
            'triangle_predicate_source_sha256':sha(Path(__import__('fnit.recon_all.mris_remove_intersection_python',fromlist=['x']).__file__)),
            'reconstructed_counts':counts.__dict__,'original_intersection_pairs':old_pairs.tolist(),
            'original_intersecting_faces':len(np.unique(old_pairs)),
            'candidate_intersecting_faces':len(np.unique(pairs)),
            'candidate_intersection_pairs':rows,
            'preserved_only_pairs':sum(a<split and b<split for a,b in pairs),
            'new_only_pairs':sum(a>=split and b>=split for a,b in pairs),
            'new_outside_pairs':sum((a>=split)!=(b>=split) for a,b in pairs),
            'original_geometry_seconds':old_seconds,'candidate_geometry_seconds':new_seconds,
            'total_seconds_including_io':time.perf_counter()-start,'overall_metric_equivalence':'not_assessed'}
    # NumPy整数计数转换为普通JSON整数，保持实际数值。
    for key in ('preserved_only_pairs','new_only_pairs','new_outside_pairs'):result[key]=int(result[key])
    (args.output_dir/'intersection_diagnosis.private.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not len(pairs):return
    image=nib.load(str(args.intensity_volume));matrix=surface_ras_to_voxel(image.header,metadata)
    ids=np.unique(pairs);tri_voxel=np.c_[xyz[faces[ids].ravel()],np.ones(3*len(ids))]@matrix.T
    tri_voxel=tri_voxel[:,:3].reshape(-1,3,3)
    center=np.rint(tri_voxel.mean(axis=(0,1))).astype(int)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(1,3,figsize=(12,4));volume=np.asarray(image.dataobj)
    lines=np.concatenate((tri_voxel[:,[0,1]],tri_voxel[:,[1,2]],tri_voxel[:,[2,0]]))
    for ax,cut,dims in zip(axes,range(3),((1,2),(0,2),(0,1))):
        ax.imshow(np.take(volume,int(center[cut]),axis=cut).T,origin='lower',cmap='gray',vmin=0,vmax=150)
        ax.add_collection(LineCollection(lines[:,:,list(dims)],colors='red',linewidths=.7))
        ax.set_xlim(center[dims[0]]-20,center[dims[0]]+20);ax.set_ylim(center[dims[1]]-20,center[dims[1]]+20)
        ax.set_title(f'MRI axis {cut}={center[cut]}');ax.set_xlabel('voxel');ax.set_ylabel('voxel')
    fig.suptitle('Public ds000114 first complete patch: rejected 3D intersecting triangles (projection)')
    fig.tight_layout();fig.savefig(args.output_dir/'intersecting_patch_T1.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
