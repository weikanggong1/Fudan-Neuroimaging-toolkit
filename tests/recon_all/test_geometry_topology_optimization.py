"""动态remesh拓扑顺序、重复面项和零接受压缩语义回归。"""
import numpy as np
from fnit.recon_all.mris_remesh_python import Mesh, initial_topology


def reference(faces):
    index, vertices, adjacent, face_edges = {}, [], [], []
    for ti, (a,b,c) in enumerate(faces):
        row=[]
        for a,b in ((a,b),(b,c),(c,a)):
            key=(a,b) if a<b else (b,a)
            ei=index.get(key)
            if ei is None:
                ei=len(vertices); index[key]=ei; vertices.append(list(key)); adjacent.append([ti])
            else:
                adjacent[ei].append(ti)
            row.append(ei)
        face_edges.append(row)
    return index,vertices,adjacent,face_edges


def test_topology_preserves_first_occurrence_and_duplicate_incidence():
    faces=[[2,0,1],[3,2,1],[0,0,2],[2,0,1]]
    assert initial_topology(faces)==reference(faces)
    assert initial_topology([])==reference([])


def test_zero_acceptance_still_removes_orphan_vertices():
    points=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(4.,4.,4.)]
    mesh=Mesh(points,[[0,1,2]])
    assert mesh.collapse_pass(0.)==0
    assert mesh.points==points[:3]
    assert mesh.faces==[[0,1,2]]


def test_zero_acceptance_keeps_existing_order():
    points=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)]
    mesh=Mesh(points,[[0,1,2]])
    before=reference(mesh.faces)
    assert mesh.collapse_pass(0.)==0
    assert (mesh.edge_index,mesh.edge_vertices,mesh.edge_faces,mesh.face_edges)==before
    np.testing.assert_array_equal(mesh.face_normals(),[[0.,0.,1.]])


def test_rebuild_preserves_first_corner_and_vertex_neighbor_insertion_order():
    faces=[[2,0,1],[3,2,1],[0,0,2],[2,0,1]]
    mesh=Mesh([(float(i),0.,0.) for i in range(5)],faces)
    expected_faces=[[] for _ in mesh.points];expected_locals=[[] for _ in mesh.points]
    for ti,face in enumerate(faces):
        for local,vertex in enumerate(face):
            if ti not in expected_faces[vertex]:
                expected_faces[vertex].append(ti);expected_locals[vertex].append(local)
    expected_neighbors=[{} for _ in mesh.points]
    for ei,(a,b) in enumerate(mesh.edge_vertices):
        expected_neighbors[a][b]=ei;expected_neighbors[b][a]=ei
    assert mesh.vertex_faces==expected_faces
    assert mesh.vertex_local==expected_locals
    assert [list(row.items()) for row in mesh.vertex_edges]==[list(row.items()) for row in expected_neighbors]
    assert mesh.vertex_faces[4]==[] and mesh.vertex_edges[4]=={}
