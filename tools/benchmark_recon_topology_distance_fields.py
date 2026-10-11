"""解析真实首次缺陷有向距离的有序输入/输出；不供生产读取参考结果。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_distance_capture(*, capture_path):
    """FNITDST1/2 → 独立数组，保留原始候选、拓扑、线程和 IEEE 字段。

    capture_path 是独立固定源码 benchmark 观察器写出的完整文件。
    返回 dict：orig/current XYZ 是 tkregister surface RAS/mm，法向和
    angle 加权伪法向无量纲；incident_offsets/faces/corners 是原 vt.f/n
    顺序，不按面编号重建；buffer 面及其 bbox 为原 OMP 追加顺序。
    局部 volume shape 为 XYZ；volume_geometry 仅为距离存储头，不能推断
    算法实际采样变换。FNITDST2 的 algorithm_volume_geometry 是真正
    mri_defect 的 start 和坐标到索引乘数，不是影像 affine 的 spacing。
    老格式没有此字段；不能将其存储头默认替代算法输入。before/after
    距离为 float32/mm，未处理
    值为源 NPY；本检查点不能作为生产输入或完整 GA 通过证据。
    错格式、截断/尾部、非法 shape/索引/有序 corner 映射抛 ValueError。
    """
    raw = Path(capture_path).read_bytes()
    magic = raw[:8]
    if magic not in (b'FNITDST1', b'FNITDST2'):
        raise ValueError('complete FNITDST1/2 benchmark capture required')
    position = 8

    def read(dtype, count, shape=None):
        nonlocal position
        dtype = np.dtype(dtype)
        if count < 0 or position + dtype.itemsize * count > len(raw):
            raise ValueError('truncated distance capture')
        data = np.frombuffer(raw, dtype, count=count, offset=position).copy()
        position += count * dtype.itemsize
        return data.reshape(shape) if shape is not None else data

    header = read('<i4', 10)
    if np.any(header < 0) or np.any(header > 100000000):
        raise ValueError('invalid distance capture dimensions')
    n, f, entries, buffered, queries, width, height, depth, kind, threads = map(int, header)
    if min(n, f, width, height, depth, threads) < 1:
        raise ValueError('positive mesh/volume/thread dimensions required')
    fields = dict(header=header, volume_geometry=read('<f4', 9))
    if magic == b'FNITDST2':
        fields['algorithm_volume_dimensions'] = read('<i4', 4)
        fields['algorithm_volume_geometry'] = read('<f4', 9)
        if np.any(fields['algorithm_volume_dimensions'][:3] < 1):
            raise ValueError('positive algorithm volume dimensions required')
    fields.update(settings=read('<i4', 5), vertex_fields=read('<f4', n * 6, (n, 6)), vertex_rip=read('<i4', n),
        faces=read('<i4', f * 3, (f, 3)), face_normals=read('<f4', f * 3, (f, 3)),
        face_rip=read('<i4', f), incident_offsets=read('<i4', n + 1),
        incident_faces=read('<i4', entries), incident_corners=read('<i4', entries),
        buffer_bounds=read('<i4', buffered * 7, (buffered, 7)),
        buffer_coordinates=read('<f4', buffered * 9, (buffered, 9)),
        buffer_other_faces=read('<i4', buffered * 3, (buffered, 3)),
        query_vertices=read('<i4', queries), query_pseudonormals=read('<f4', queries * 3, (queries, 3)))
    shape = (width, height, depth)
    fields['distance_before'] = read('<f4', width * height * depth, shape)
    fields['distance_after'] = read('<f4', width * height * depth, shape)
    if position != len(raw):
        raise ValueError('unexpected distance capture trailing data')
    offsets = fields['incident_offsets']
    if offsets[0] != 0 or offsets[-1] != entries or np.any(np.diff(offsets) < 0):
        raise ValueError('invalid source incident-face CSR')
    for array, limit in ((fields['faces'], n), (fields['incident_faces'], f),
            (fields['incident_corners'], 3), (fields['buffer_bounds'][:, 0], f),
            (fields['buffer_other_faces'], f), (fields['query_vertices'], n)):
        if array.size and (array.min() < 0 or array.max() >= limit):
            raise ValueError('distance capture index out of range')
    owner = np.repeat(np.arange(n), np.diff(offsets))
    if np.any(fields['faces'][fields['incident_faces'], fields['incident_corners']] != owner):
        raise ValueError('source incident face/corner ownership mismatch')
    if np.unique(fields['query_vertices']).size != queries:
        raise ValueError('duplicate source query vertex IDs')
    requested = np.unique(fields['faces'][fields['buffer_bounds'][:, 0]])
    if not np.array_equal(np.sort(fields['query_vertices']), requested):
        raise ValueError('pseudo-normal queries do not cover buffered faces')
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('new independent distance diagnostic checkpoint required')
    fields = load_distance_capture(capture_path=args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    print(json.dumps(dict(scope='same-input source distance checkpoint; no own distance/GA claim',
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        fields_checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        vertices=int(fields['header'][0]), faces=int(fields['header'][1]),
        buffered_faces=int(fields['header'][3]), pseudo_queries=int(fields['header'][4]),
        distance_shape=list(fields['distance_after'].shape), threads=int(fields['header'][9]),
        dp_fitness_size=int(fields['settings'][4])), indent=2))


if __name__ == '__main__':
    main()
