"""同一顶点批量KD查询与原逐面路径完全相同，保留相交与共享点规则。"""
import numpy as np
from scipy.spatial import cKDTree
from fnit.recon_all.place_surface_collision import _moved_face_geometry, _candidate_collision, _vertex_collision_batch

def test_batch_keeps_exact_ordered_face_candidates_and_collision_result():
    xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[0,0,.5],[2,0,.5],[0,2,.5],[0,0,2]],np.float32)
    faces=np.array([[0,1,2],[3,4,5],[0,2,6]],np.int32)
    points=xyz[faces];centers=points.mean(axis=1,dtype=np.float64)
    maximum=float(np.linalg.norm(points.astype(np.float64)-centers[:,None],axis=2).max())
    tree=cKDTree(centers)
    outcomes=[]
    for vertex in range(len(xyz)):
        incident=np.flatnonzero(np.any(faces==vertex,axis=1))
        for shift in ([-.3,0,0],[0,0,.3],[0,0,1],[0,0,0]):
            endpoint=np.float32(xyz[vertex]+np.array(shift,np.float32))
            baseline=False
            batch_centers=[];batch_radii=[];single_lists=[]
            for face_id in incident:
                moved,center,radius,low,high=_moved_face_geometry(xyz,faces,face_id,vertex,endpoint)
                near=np.asarray(tree.query_ball_point(center,radius+maximum+1.),np.int32)
                baseline=baseline or bool(_candidate_collision(xyz,faces,moved,faces[face_id],low,high,near))
                batch_centers.append(center);batch_radii.append(radius);single_lists.append(near)
            if len(incident):
                lists=tree.query_ball_point(np.asarray(batch_centers),np.asarray(batch_radii)+maximum+1.,return_sorted=False)
                for old,new in zip(single_lists,lists):np.testing.assert_array_equal(old,new)
            result=_vertex_collision_batch(xyz,faces,incident,vertex,endpoint,tree,maximum)
            assert result==baseline
            outcomes.append(result)
    assert any(outcomes) and not all(outcomes)
