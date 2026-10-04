"""不同拓扑的精确点到三角面距离及双向面积加权注释Dice，隔离诊断。"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, platform, time
from pathlib import Path
import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit
from scipy.spatial import cKDTree


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@njit(cache=True,fastmath=False)
def triangle_projection(point,a,b,c):
    # 先计算投影；落在面内时返回投影，否则比较三条闭合边。
    ab=b-a;ac=c-a;ap=point-a
    d00=np.dot(ab,ab);d01=np.dot(ab,ac);d11=np.dot(ac,ac)
    denominator=d00*d11-d01*d01
    if denominator>1e-24:
        d20=np.dot(ap,ab);d21=np.dot(ap,ac)
        u=(d11*d20-d01*d21)/denominator;v=(d00*d21-d01*d20)/denominator
        if u>=0 and v>=0 and u+v<=1:
            closest=a+u*ab+v*ac
            return np.dot(point-closest,point-closest),np.array([1-u-v,u,v])
    starts=(a,b,c);ends=(b,c,a);best=np.inf;weights=np.zeros(3)
    for edge in range(3):
        vector=ends[edge]-starts[edge];length=np.dot(vector,vector)
        t=min(1.,max(0.,np.dot(point-starts[edge],vector)/length)) if length>0 else 0.
        closest=starts[edge]+t*vector;distance=np.dot(point-closest,point-closest)
        if distance<best:
            best=distance;weights[:]=0.;weights[edge]=1-t;weights[(edge+1)%3]=t
    return best,weights


@njit(cache=True,fastmath=False)
def find_closest(points,triangles,offsets,ids):
    distances=np.empty(len(points));faces=np.empty(len(points),np.int64);weights=np.empty((len(points),3))
    for i in range(len(points)):
        best=np.inf;best_id=-1
        for j in range(offsets[i],offsets[i+1]):
            face=ids[j];distance,w=triangle_projection(points[i],triangles[face,0],triangles[face,1],triangles[face,2])
            # KD候选按面编号排序，等距稳定采用首面。
            if distance<best:best=distance;best_id=face;weights[i]=w
        if best_id<0:raise ValueError('No candidate triangle')
        distances[i]=np.sqrt(best);faces[i]=best_id
    return distances,faces,weights


def point_mesh_projection(source,target,faces):
    triangles=target[faces].astype(np.float64);centers=triangles.mean(1)
    radius=float(np.linalg.norm(triangles-centers[:,None],axis=2).max())
    vertex_tree=cKDTree(target[np.unique(faces)]);face_tree=cKDTree(centers)
    distances=[];face_ids=[];barycentric=[]
    for start in range(0,len(source),1024):
        points=source[start:start+1024].astype(np.float64);upper=vertex_tree.query(points,workers=4)[0]
        candidates=face_tree.query_ball_point(points,upper+radius+1e-7,workers=4,return_sorted=True)
        offsets=np.r_[0,np.cumsum([len(x) for x in candidates])].astype(np.int64)
        ids=np.array([i for item in candidates for i in item],np.int64)
        d,f,w=find_closest(points,triangles,offsets,ids);distances.append(d);face_ids.append(f);barycentric.append(w)
    return np.concatenate(distances),np.concatenate(face_ids),np.concatenate(barycentric)


def summary(d):return {'mean_mm':float(d.mean()),'p99_mm':float(np.quantile(d,.99)),'max_mm':float(d.max()),'max_source_vertex':int(np.argmax(d)),'over_0_1_mm':int((d>.1).sum())}


def areas(vertices,faces):
    tri=vertices[faces];third=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/6
    weights=np.zeros(len(vertices));np.add.at(weights,faces.ravel(),np.repeat(third,3));return weights


def region_dice(original,projected,weights):
    rows=[]
    for code in np.union1d(original,projected):
        a=original==code;b=projected==code;na=int(a.sum());nb=int(b.sum());intersection=int((a&b).sum());aa=float(weights[a].sum());ab=float(weights[b].sum());ai=float(weights[a&b].sum())
        rows.append({'annotation_id':int(code),'source_vertices':na,'projected_vertices':nb,'vertex_dice':2*intersection/(na+nb),'source_area_mm2':aa,'projected_area_mm2':ab,'area_weighted_dice':2*ai/(aa+ab) if aa+ab else None})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--stages',nargs='+',default=['orig.nofix','orig.premesh','orig','sphere','sphere.reg']);p.add_argument('--lock',type=Path,default=Path('/tmp/fnit-shared-benchmark.lock'));p.add_argument('--code-commit',required=True);p.add_argument('--reference-distance-helper',type=Path)
    a=p.parse_args();helper=None
    if a.reference_distance_helper:
        import importlib.util
        spec=importlib.util.spec_from_file_location('reference_distance',a.reference_distance_helper);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    a.output_root.mkdir(parents=True,exist_ok=True);r={'status':'running','kind':'different-topology spatial diagnostic','host':platform.node(),'pid':os.getpid(),'script_sha256':sha(__file__),'code_commit':a.code_commit,'reference':str(a.reference),'candidate':str(a.candidate),'overall_equivalence':'not_assessed','rows':[],'annotation_metric':'Each direction separately: closest closed triangle; vertex label at maximum closest-point barycentric weight, tie first triangle corner; area weights use source orig triangles. This is resampled Dice, not same-index Dice, and does not prove native vertex correspondence.'}
    def save():(a.output_root/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    save()
    for hemi in ('lh','rh'):
        for stage in a.stages:
            files=[root/'surf'/(hemi+'.'+stage) for root in (a.reference,a.candidate)]
            if not all(p.exists() for p in files):r['rows'].append({'hemisphere':hemi,'stage':stage,'status':'missing','exists':[x.exists() for x in files]});save();continue
            with a.lock.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);tick=time.perf_counter();v,f=zip(*(fsio.read_geometry(str(x)) for x in files));row={'hemisphere':hemi,'stage':stage,'status':'running','sha256':[sha(x) for x in files],'ordered_faces_equal':bool(v[0].shape==v[1].shape and np.array_equal(f[0],f[1]))};r['rows'].append(row);save()
                maps=[]
                for source,target in ((0,1),(1,0)):
                    d,ids,bary=point_mesh_projection(v[source],v[target],f[target])
                    if helper:
                        expected=helper._point_to_mesh(v[source][:64],v[target],f[target]);error=float(np.max(np.abs(expected-d[:64])));row['distance_validation_'+str(source)]={'actual_input_vertices':64,'max_difference_mm':error,'declared_atol_mm':1e-9,'reference_helper_sha256':sha(a.reference_distance_helper)}
                        if error>1e-9:raise RuntimeError('Independent exact triangle distance check failed')
                    label=('reference_to_candidate' if source==0 else 'candidate_to_reference');row[label+'_triangle_distance']=summary(d);cache=a.output_root/(hemi+'.'+stage+'.'+label+'.npz');np.savez_compressed(cache,distance_mm=d,triangle_id=ids,barycentric=bary);row[label+'_projection_sha256']=sha(cache);maps.append((ids,bary))
                if stage in ('orig','sphere.reg'):
                    row['annotations']=[]
                    orig=[fsio.read_geometry(str(root/'surf'/(hemi+'.orig'))) for root in (a.reference,a.candidate)]
                    if not all(len(x[0])==len(v[i]) and np.array_equal(x[1],f[i]) for i,x in enumerate(orig)):raise ValueError('Orig surface ordering does not match the projection surface')
                    for atlas in ('aparc','aparc.a2009s','aparc.DKTatlas'):
                        afiles=[root/'label'/(hemi+'.'+atlas+'.annot') for root in (a.reference,a.candidate)]
                        item={'atlas':atlas,'exists':[x.exists() for x in afiles]};row['annotations'].append(item)
                        if not all(item['exists']):continue
                        annotations=[fsio.read_annot(str(x),orig_ids=True) for x in afiles];ann=[x[0] for x in annotations];item['label_names']={str(int(code)):name.decode('utf-8') for code,name in zip(annotations[0][1][:,-1],annotations[0][2])};item['sha256']=[sha(x) for x in afiles]
                        for source,target in ((0,1),(1,0)):
                            ids,bary=maps[source];chosen=f[target][ids,np.argmax(bary,axis=1)];projected=ann[target][chosen];weights=areas(*orig[source]);direction='reference_domain' if source==0 else 'candidate_domain';item[direction]=region_dice(ann[source],projected,weights)
                row.update(status='complete',diagnostic_wall_seconds=time.perf_counter()-tick);save();fcntl.flock(lock,fcntl.LOCK_UN)
    r['status']='complete';save()
if __name__=='__main__':main()
