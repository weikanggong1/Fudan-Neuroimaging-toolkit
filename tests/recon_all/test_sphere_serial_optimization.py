"""静态拓扑缓存及有序SSE行并行必须保持精度与缓存失效行为。"""
import numpy as np
import pytest
from numba import njit
from fnit.recon_all.place_surface_normals import (
    CoordinateNormalCache, FaceNormalTopology, initial_vertex_normals,
)
from fnit.recon_all.sphere_standard_line_search import _distance_sse
from fnit.recon_all.sphere_standard_unfold import _sphere_radius_units, _spherical_distance
def test_normal_cache_recomputes_coordinates_and_rejects_changed_faces():
    vertices=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]],np.float32)
    faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]],np.int32)
    context=FaceNormalTopology(faces,4)
    for xyz in (vertices,vertices*np.array([1.,2.,3.],np.float32)):
        np.testing.assert_array_equal(initial_vertex_normals(xyz,faces,topology=context),
                                      initial_vertex_normals(xyz,faces))
    with pytest.raises(ValueError):
        initial_vertex_normals(vertices,faces[::-1],topology=context)
    faces[0,0]=3
    with pytest.raises(ValueError):
        initial_vertex_normals(vertices,faces,topology=context)
    assert context.faces[0,0]==0


def test_coordinate_normal_cache_reuses_identity_without_changing_float32_result():
    vertices=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]],np.float32)
    faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]],np.int32)
    context=FaceNormalTopology(faces,4)
    cache=CoordinateNormalCache(context)
    first=cache.evaluate(vertices)
    assert cache.evaluate(vertices) is first
    np.testing.assert_array_equal(first, initial_vertex_normals(vertices,faces,topology=context))

    changed=vertices.copy()
    changed[0,0]=2
    second=cache.evaluate(changed)
    assert second is not first
    np.testing.assert_array_equal(second, initial_vertex_normals(changed,faces,topology=context))

    # Explicit invalidation is required if a caller mutates an array in place.
    vertices[0,0]=2
    cache.clear()
    third=cache.evaluate(vertices)
    np.testing.assert_array_equal(third, initial_vertex_normals(vertices,faces,topology=context))

@njit
def reference(xyz,offsets,neighbors,distances,scale):
    radius,unit=_sphere_radius_units(xyz)
    total=0.
    for vertex in range(len(xyz)):
        row=0.
        for index in range(offsets[vertex],offsets[vertex+1]):
            current=_spherical_distance(xyz,radius,unit,vertex,neighbors[index])
            delta=scale*np.float64(current)-np.float64(distances[index])
            row+=delta*delta
        total+=row
    return total
def test_sse_parallel_rows_preserve_serial_total():
    rng=np.random.default_rng(17)
    xyz=rng.normal(size=(128,3)).astype(np.float32)
    xyz*=100/np.linalg.norm(xyz,axis=1)[:,None]
    neighbors=rng.integers(0,128,size=128*12,dtype=np.int32)
    distances=rng.uniform(0,15,size=len(neighbors)).astype(np.float32)
    offsets=np.arange(129,dtype=np.int64)*12
    assert _distance_sse(xyz,offsets,neighbors,distances,.71)==reference(xyz,offsets,neighbors,distances,.71)
