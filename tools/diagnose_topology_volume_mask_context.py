"""最大拓扑形状差的自产影像上下文；只读八邻接体素，不能作为参考修补。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from collections import deque
from scipy import ndimage
import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject',type=Path,action='append',required=True)
    parser.add_argument('--local-shape-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--core-depth-mm',type=float,default=2.0)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.core_depth_mm<=0 or not np.isfinite(args.core_depth_mm): raise ValueError('positive finite core depth required')
    local=json.loads(args.local_shape_report.read_text())
    if local['execution_status']!='complete': raise ValueError('complete point checkpoint required')
    result={'scope':'posthoc baseline-producer image context; no output repair and no new production run',
            'script_sha256':sha(__file__),'local_shape_report_sha256':sha(args.local_shape_report),
            'context_interpretation':'existing FNIT baseline masks and labels, not official truth or whole equivalence',
            'cases':[],'execution_status':'running'}
    for row in local['cases']:
        subject=args.subject[row['case_number']]
        low=np.floor(np.asarray(row['candidate_point_voxel_xyz'])).astype(np.int64)
        indices=np.asarray([low+[x,y,z] for x in range(2) for y in range(2) for z in range(2)])
        volumes={};reference=nib.load(str(subject/'mri/T1.mgz'))
        for name in ['T1','brainmask','aseg.presurf','brain','brain.finalsurfs','wm','filled',('filled-pretess255' if row['hemisphere']=='lh' else 'filled-pretess127')]:
            path=subject/'mri'/f'{name}.mgz';image=nib.load(str(path));data=np.asarray(image.dataobj)
            if data.shape!=reference.shape or not np.allclose(image.affine,reference.affine,rtol=0,atol=1e-6):
                raise ValueError('context volume geometry mismatch: '+name)
            values=data[tuple(indices.T)].astype(float)
            volumes[name]={'input_sha256':sha(path),'dtype':str(data.dtype),'corner_adjacent_values':values.tolist(),
                           'nonzero_adjacent_voxels':int(np.count_nonzero(values)),
                           'minimum':float(values.min()),'maximum':float(values.max())}
        value=255 if row['hemisphere']=='lh' else 127
        pretess_path=subject/'mri'/f'filled-pretess{value}.mgz'
        target=np.asarray(nib.load(str(pretess_path)).dataobj)==value
        depth=ndimage.distance_transform_edt(target,sampling=reference.header.get_zooms()[:3])
        starts=[tuple(map(int,index)) for index in indices if target[tuple(index)]]
        if not starts: raise ValueError('maximum candidate point has no original foreground corner; this diagnostic requires preserved target branch')
        seed=min(starts,key=lambda x:(-depth[x],x));parents={seed:None};queue=deque([seed]);finish=None
        while queue:
            index=queue.popleft()
            if depth[index]>=args.core_depth_mm: finish=index;break
            for axis in range(3):
                for offset in [-1,1]:
                    other=list(index);other[axis]+=offset;other=tuple(other)
                    if all(0<=other[k]<target.shape[k] for k in range(3)) and target[other] and other not in parents:
                        parents[other]=index;queue.append(other)
        path=[]
        if finish is not None:
            index=finish
            while index is not None: path.append(index);index=parents[index]
            path.reverse()
        path_array=np.asarray(path,np.int64)
        result['cases'].append({'case_number':row['case_number'],'hemisphere':row['hemisphere'],
            'original_branch_connectivity':{'foreground_value':value,'pretess_sha256':sha(pretess_path),'seed_voxel':list(seed),
                'core_depth_mm':args.core_depth_mm,'core_depth_role':'diagnostic geometric depth, not equivalence or output filtering threshold',
                'reached_core':finish is not None,'six_connected_BFS_visited_voxels':len(parents),
                'shortest_six_connected_steps_to_core':len(path)-1 if path else None,
                'path_voxel_xyz':path_array.tolist(),'path_foreground_depth_mm':[float(depth[x]) for x in path]},
            'point_voxel_xyz':row['candidate_point_voxel_xyz'],'eight_voxel_indices':indices.tolist(),
            'candidate_to_GA_triangle_distance_mm':row['candidate_to_GA_triangle_distance_mm'],
            'same_point_is_exact_original_nofix_vertex':row['same_point_is_exact_original_nofix_vertex'],
            'volumes':volumes})
    result['execution_status']='complete';result['diagnostic_seconds_including_io']=time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
