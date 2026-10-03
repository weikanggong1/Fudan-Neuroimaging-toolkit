#!/usr/bin/env python3
"""只读检查配对原生 id24 体素、统计表和既有 FNIT 标签来源；不运行 MRI。"""
import argparse, hashlib, json, time
from pathlib import Path
import nibabel as nib
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case-id', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite frozen diagnostic')
    start = time.perf_counter()
    own = sha(Path(__file__))
    paths, before, chains = {}, {}, {}
    names = ['synthseg.rca.mgz', 'aseg.auto_noCCseg.mgz', 'aseg.auto.mgz',
             'aseg.presurf.mgz', 'aseg.presurf.hypos.mgz', 'aseg.mgz']
    for chain, subject in [('reference', args.reference), ('candidate', args.candidate)]:
        volumes = []
        for name in names:
            path = subject / 'mri' / name
            if not path.is_file():
                continue
            key = chain + '/mri/' + name
            paths[key] = path
            before[key] = sha(path)
            image = nib.load(str(path))
            labels = np.asarray(image.dataobj)
            count = int(np.count_nonzero(labels == 24))
            voxel_volume = float(np.prod(image.header.get_zooms()[:3]))
            volumes.append(dict(file=name, sha256=before[key], shape=[int(x) for x in image.shape],
                                voxel_volume_mm3=voxel_volume, id24_count=count,
                                id24_binary_volume_mm3=count * voxel_volume))
        path = subject / 'stats/aseg.stats'
        key = chain + '/stats/aseg.stats'
        paths[key] = path
        before[key] = sha(path)
        rows = [line.split() for line in path.read_text().splitlines()
                if not line.startswith('#') and len(line.split()) >= 5 and line.split()[1] == '24']
        if len(rows) != 1:
            raise ValueError('Need exactly one original id24 stats row')
        row = rows[0]
        csf = dict(label_id=int(row[1]), voxel_count=int(row[2]),
                   partial_volume_mm3=float(row[3]), structure_name=row[4])
        final = [v for v in volumes if v['file'] == 'aseg.mgz']
        if len(final) != 1 or csf['voxel_count'] != final[0]['id24_count']:
            raise ValueError('Original stats NVoxels disagrees with its saved segmentation')
        commands = []
        log = subject / 'scripts/recon-all.log'
        if log.is_file():
            paths[chain + '/scripts/recon-all.log'] = log
            before[chain + '/scripts/recon-all.log'] = sha(log)
            commands = list(dict.fromkeys(line.strip() for line in log.read_text(errors='replace').splitlines()
                                          if 'mri_segstats' in line and 'aseg.stats' in line))
        chains[chain] = dict(volumes=volumes, stats_sha256=before[key], id24_stats=csf,
                             voxel_count_matches_stats=True, saved_reference_commands=commands)
    source_files = ['recon_all/native_free.py', 'recon_all/segstats_aseg_python.py',
                    'recon_all/segstats_wmparc_python.py', 'recon_all/surf2volseg_fix_python.py']
    for name in source_files:
        key = 'fnit_source/' + name
        path = args.source / 'src/fnit' / name
        paths[key], before[key] = path, sha(path)
    after = {key: sha(path) for key, path in paths.items()}
    if before != after or sha(Path(__file__)) != own:
        raise ValueError('Read-only diagnostic input or source changed')
    report = dict(case_id=args.case_id, status='complete', label_id=24, helper_sha256=own,
                  input_sha256=before, input_after_sha256=after, inputs_unchanged=True,
                  chains=chains, own_source_unchanged=True,
                  interpretation='Original counts differ before stats. Native FNIT copies SynthSeg segmentation into aseg.auto_noCCseg; the mature stats function retains label IDs and applies partial-volume weights using norm. A common CSF anatomical label domain is not established, and no label was normalized or removed.',
                  timing_scope='Independent CPU saved-file diagnostic; no MRI rerun; excluded from production whole wall',
                  diagnostic_wall_seconds=time.perf_counter() - start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], sha256=sha(args.output), seconds=report['diagnostic_wall_seconds'])))


if __name__ == '__main__':
    main()
