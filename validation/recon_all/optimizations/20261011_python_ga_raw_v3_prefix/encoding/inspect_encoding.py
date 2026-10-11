"""只读定位两次 FNIT 自产 MGZ 的字节差异；不输出路径或标签内容。"""
from pathlib import Path
import gzip
import hashlib
import json
import math
import os
import platform
import resource
import struct
import time
import zlib

import nibabel as nib

VOLUMES = ('rawavg.mgz', 'orig.mgz', 'tmp/nu0.mgz', 'nu.mgz', 'T1.mgz',
           'brainmask.mgz', 'synthseg.rca.mgz', 'norm.mgz', 'brain.mgz',
           'aseg.presurf.mgz', 'wm.seg.mgz', 'wm.asegedit.mgz', 'wm.mgz', 'filled.mgz')


def inspect(*, subject_directory, filename):
    """MGZ→固定五区段 SHA/typed tag；只读，不解码数组或调用阶段程序。"""
    path = subject_directory / 'mri' / filename
    encoded = path.read_bytes()
    raw = gzip.decompress(encoded)
    image = nib.load(path)
    if not isinstance(image, nib.MGHImage) or len(image.shape) != 3:
        raise ValueError('3D MGH input required')
    offset = int(image.header.get_data_offset())
    end = offset + math.prod(image.shape) * image.get_data_dtype().itemsize
    if len(raw) < end + 20:
        raise ValueError('truncated data or pulse footer')
    regions = {'header': raw[:offset], 'voxels': raw[offset:end],
               'pulse_footer': raw[end:end+20], 'tags': raw[end+20:]}
    tags, position = [], end + 20
    colortable_name = None
    while position < len(raw):
        if len(raw)-position < 4:
            raise ValueError('incomplete typed tag')
        number = struct.unpack_from('>i', raw, position)[0]
        position += 4
        if number == 1:
            # 旧 CTAB 不带 int64 长度。按 FNIT 已有 writer 的 version2
            # 格式走完全部条目，不能将其头两项误当普通标签长度。
            begin = position
            def integer():
                nonlocal position
                if position+4 > len(raw):
                    raise ValueError('truncated color table')
                v = struct.unpack_from('>i', raw, position)[0]
                position += 4
                return v
            def string():
                nonlocal position
                size = integer()
                if size < 0 or size > len(raw)-position:
                    raise ValueError('invalid color table string length')
                v = raw[position:position+size]
                position += size
                return v
            version, maximum = integer(), integer()
            if version != -2 or maximum < 0:
                raise ValueError('unsupported old color table version')
            colortable_name = string()
            count = integer()
            if count < 0 or count > (len(raw)-position)//24:
                raise ValueError('invalid color table count')
            for _ in range(count):
                integer(); string()
                for _ in range(4):integer()
            payload = raw[begin:position]
        else:
            if len(raw)-position < 8:
                raise ValueError('incomplete typed tag length')
            size = struct.unpack_from('>q', raw, position)[0]
            position += 8
            if size < 0 or size > len(raw)-position:
                raise ValueError('invalid tag length')
            payload = raw[position:position+size]
            position += size
        tags.append((number, payload))
    expected = str(subject_directory / 'mri/transforms/talairach.xfm').encode() + b'\0'
    provenance = None
    if colortable_name is not None:
        paths = (subject_directory/'mri/wm.mgz', subject_directory/'mri/aseg.presurf.mgz',
                 subject_directory/'mri/transforms/talairach.lta')
        provenance = (b'fnit.recon_all.fill_cutting_plane_python '+
                      b' '.join(str(p).encode() for p in paths)+b' '+
                      colortable_name.rstrip(b'\0')+b' '+
                      str(subject_directory/'mri/filled.mgz').encode()+b'\0')
    sha = lambda b: hashlib.sha256(b).hexdigest()
    result = {'file_sha256': sha(encoded), 'decoded_sha256': sha(raw),
              'regions': {k: {'sha256': sha(v), 'bytes': len(v)} for k, v in regions.items()},
              'tags': [{'id': n, 'bytes': len(v), 'sha256': sha(v),
                        'matches_own_transform_path': v == expected if n == 31 else None,
                        'matches_own_fill_provenance': v == provenance if n == 3 else None}
                       for n, v in tags]}
    if hashlib.sha256(path.read_bytes()).hexdigest() != result['file_sha256']:
        raise RuntimeError('input changed during read')
    return result


def compare(*, baseline_subject, candidate_subject):
    """返回十四份编码诊断；差异归因只在完整区段和TAG验证成立时记录。"""
    started = time.perf_counter()
    rows = {}
    for name in VOLUMES:
        left = inspect(subject_directory=baseline_subject, filename=name)
        right = inspect(subject_directory=candidate_subject, filename=name)
        equal = {k: left['regions'][k] == right['regions'][k] for k in left['regions']}
        only_own_path = (all(equal[k] for k in ('header','voxels','pulse_footer'))
                        and not equal['tags']
                        and len(left['tags']) == len(right['tags']) == 1
                        and all(t['id'] == 31 and t['matches_own_transform_path'] is True
                                for t in left['tags'] + right['tags']))
        changed_tags = [(a,b) for a,b in zip(left['tags'],right['tags'])
                        if a['sha256'] != b['sha256']]
        only_provenance = (all(equal[k] for k in ('header','voxels','pulse_footer'))
                           and bool(changed_tags)
                           and len(left['tags']) == len(right['tags'])
                           and [a['id'] for a in left['tags']] == [a['id'] for a in right['tags']]
                           and all(a['id'] in (3,31) and
                                   ((a['matches_own_fill_provenance'] is True and b['matches_own_fill_provenance'] is True)
                                    if a['id']==3 else
                                    (a['matches_own_transform_path'] is True and b['matches_own_transform_path'] is True))
                                   for a,b in changed_tags))
        rows[name] = {'baseline': left, 'candidate': right, 'regions_equal': equal,
                      'only_own_transform_path_tag_differs': only_own_path,
                      'only_verified_own_path_provenance_differs': only_provenance}
    return {'schema_version': 1, 'scope': 'read-only actual FNIT prefix encoding; not scientific equivalence',
            'volumes': rows, 'seconds': time.perf_counter()-started,
            'all_inputs_unchanged_during_read': True,
            'runtime': {'python': platform.python_version(), 'nibabel': nib.__version__,
                        'zlib': zlib.ZLIB_RUNTIME_VERSION, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                        'rss_high_water_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        'CUDA_VISIBLE_DEVICES_empty': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                        'GPU_computation': False}}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-subject', type=Path, required=True)
    parser.add_argument('--candidate-subject', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    report = compare(baseline_subject=arguments.baseline_subject,
                     candidate_subject=arguments.candidate_subject)
    report['diagnostic_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'complete', 'file_sha256': hashlib.sha256(arguments.output.read_bytes()).hexdigest(),
                      'bytes': arguments.output.stat().st_size, 'seconds': report['seconds']}))
