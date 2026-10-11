"""读取候选25次平滑/40次MRI拟合的实际输入和三阶段输出，仅诊断。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_match_capture(*, capture_path):
    """FNITMAT1→独立数组；保留源vtotal、入射面顺序、首轮实际法向。

    capture_path为完整源观察器文件；返回header(int32/10：N,F,邻接条目,
    入射条目,selectedV/F,ninside,phase,smoothType,match开关)、parameters
    (float32/4：white/gray/k1/k2 mean)、MRI-only ras_to_voxel(float64/4x4)、
    vertex_fields(float32/Nx27：orig/current/fx/normals/e1/e2/k1/k2/area/
    val/val2/val2bak/curvbak/undefval/old_undefval)，surface RAS/mm；
    flags(int32/Nx2，marked/rip)、topology(int32/Nx3，vnum/vtotal/nsizeCur)、
    faces(int32/Fx3)/face_rip，以及有序neighbors、incident faces/corners CSR、
    selectedV/F和三phase编号0before/1smooth/2after。没有默认值或写入；
    格式/截断/索引/CSR/尾部错误抛ValueError。参考后态仅事后回归比较，
    不能供生产算法补状态；MRI体素另由同brain.mgz的SHA绑定。
    """
    raw = Path(capture_path).read_bytes()
    if raw[:8] != b'FNITMAT1':
        raise ValueError('complete FNITMAT1 required')
    position = 8
    def read(dtype, count, shape=None):
        nonlocal position
        dt = np.dtype(dtype)
        if count < 0 or position+dt.itemsize*count > len(raw):
            raise ValueError('truncated candidate match capture')
        data = np.frombuffer(raw, dt, count=count, offset=position).copy()
        position += dt.itemsize*count
        return data.reshape(shape) if shape is not None else data
    header = read('<i4', 10)
    n, f, ne, ni, sv, sf, inside, phase, smooth, match = map(int, header)
    if (n < 1 or min(f, ne, ni, sv, sf, inside) < 0 or max(header) > 100000000
            or phase not in (0, 1, 2) or smooth not in (0, 1, 2, 3, 4)
            or match not in (0, 1) or inside > sv):
        raise ValueError('invalid match capture dimensions or source parameters')
    fields = dict(header=header, parameters=read('<f4', 4),
                  ras_to_voxel=read('<f8', 16, (4, 4)),
                  vertex_fields=read('<f4', n*27, (n, 27)),
                  flags=read('<i4', n*2, (n, 2)),
                  topology=read('<i4', n*3, (n, 3)))
    face_state = read('<i4', f*4, (f, 4))
    fields['faces'] = face_state[:, :3].copy()
    fields['face_rip'] = face_state[:, 3].copy()
    fields['neighbor_offsets'] = read('<i4', n+1)
    fields['neighbors'] = read('<i4', ne)
    fields['incident_offsets'] = read('<i4', n+1)
    off = fields['incident_offsets']
    if off[0] != 0 or off[-1] != ni or np.any(np.diff(off) < 0):
        raise ValueError('invalid ordered incident CSR')
    ids, corners = np.empty(ni, np.int32), np.empty(ni, np.int32)
    for v in range(n):
        count = int(off[v+1]-off[v])
        ids[off[v]:off[v+1]] = read('<i4', count)
        corners[off[v]:off[v+1]] = read('<i4', count)
    fields['incident_faces'], fields['incident_corners'] = ids, corners
    fields['selected_vertices'] = read('<i4', sv)
    fields['selected_faces'] = read('<i4', sf)
    if position != len(raw):
        raise ValueError('unexpected candidate match trailing data')
    offsets = fields['neighbor_offsets']
    topology = fields['topology']
    if (offsets[0] != 0 or offsets[-1] != ne or np.any(np.diff(offsets) < 0)
            or np.any(topology < 0) or np.any(topology[:, 0] > topology[:, 1])
            or not np.array_equal(np.diff(offsets), topology[:, 1])):
        raise ValueError('invalid source ordered neighbor CSR')
    for name in ['faces', 'neighbors', 'selected_vertices']:
        a = fields[name]
        if a.size and (a.min() < 0 or a.max() >= n):
            raise ValueError(name+' index out of range')
    for name in ['incident_faces', 'selected_faces']:
        a = fields[name]
        if a.size and (a.min() < 0 or a.max() >= f):
            raise ValueError(name+' index out of range')
    if corners.size and (corners.min() < 0 or corners.max() >= 3):
        raise ValueError('incident corner out of range')
    for v in range(n):
        a, b = off[v:v+2]
        if not np.all(fields['faces'][ids[a:b], corners[a:b]] == v):
            raise ValueError('source incident corner does not identify vertex')
    if not np.isfinite(fields['vertex_fields'][:, :12]).all() or not np.isfinite(fields['ras_to_voxel']).all():
        raise ValueError('nonfinite source match coordinate/normal input')
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError('new candidate match diagnostic checkpoint required')
    fields = load_match_capture(capture_path=a.capture)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **fields)
    print(json.dumps(dict(scope='candidate smooth/MRI-match reference, not production input',
        capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),
        dimensions=fields['header'].tolist())))


if __name__ == '__main__':
    main()
