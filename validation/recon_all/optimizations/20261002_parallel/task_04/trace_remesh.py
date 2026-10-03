"""非计时真实 remesh 轨迹核对：记录每个拆/缩边pass及每轮平滑后的完整数组哈希。"""
import argparse, hashlib, json, os, pathlib, sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=pathlib.Path, required=True)
    parser.add_argument('--input', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()/'src'))
    import nibabel.freesurfer.io as fsio
    import numpy as np
    from fnit.recon_all import mris_remesh_python as remesh
    rows = []
    load_before = os.getloadavg()
    def snapshot(kind, points, faces, accepted=None):
        vertices = np.asarray(points, dtype='<f8').reshape(-1,3)
        triangles = np.asarray(faces, dtype='<i4').reshape(-1,3)
        rows.append({'kind':kind, 'accepted':accepted, 'vertices':len(vertices), 'faces':len(triangles), 'coordinates_float64_sha256':hashlib.sha256(vertices.tobytes()).hexdigest(), 'ordered_faces_sha256':hashlib.sha256(triangles.tobytes()).hexdigest()})
    original_split, original_collapse, original_smooth = remesh.split_pass, remesh.Mesh.collapse_pass, remesh.smooth
    def split(*items, **kwargs):
        accepted = original_split(*items, **kwargs)
        snapshot('split_pass', items[0], items[1], accepted)
        return accepted
    def collapse(mesh, *items, **kwargs):
        accepted = original_collapse(mesh, *items, **kwargs)
        snapshot('collapse_pass', mesh.points, mesh.faces, accepted)
        return accepted
    def smooth(mesh, *items, **kwargs):
        result = original_smooth(mesh, *items, **kwargs)
        snapshot('smooth', mesh.points, mesh.faces)
        return result
    remesh.split_pass, remesh.Mesh.collapse_pass, remesh.smooth = split, collapse, smooth
    vertices, faces = fsio.read_geometry(str(args.input))
    output_vertices, output_faces = remesh.remesh_geometry(vertices, faces, iterations=3)
    report = {'kind':'real_data_correctness_trace_not_timing', 'source_module_sha256':hashlib.sha256(pathlib.Path(remesh.__file__).read_bytes()).hexdigest(), 'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), 'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(), 'rows':rows, 'output_float32_sha256':hashlib.sha256(output_vertices.astype('<f4').tobytes()).hexdigest(), 'output_ordered_faces_sha256':hashlib.sha256(output_faces.astype('<i4').tobytes()).hexdigest(), 'load_before':load_before, 'load_after':os.getloadavg(), 'scope':'hash snapshots after every complete pass; per-edge transient states not recorded; no elapsed performance claim'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
