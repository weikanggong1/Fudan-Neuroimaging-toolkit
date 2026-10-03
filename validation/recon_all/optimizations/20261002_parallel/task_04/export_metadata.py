"""仅打包已完成验证的 JSON/CSV；不递归打包私有影像、表面或缓存。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    task = args.task_root.resolve()
    completion = json.loads((task/'completed_summary/completion.json').read_text())
    if (completion['paired_count'], completion['chain_count'], completion['official_count']) != (12, 4, 12):
        raise RuntimeError('Required real executions are incomplete')
    if not all(completion[k] for k in ('all_paired_exact', 'all_chains_propagated', 'all_official_same_input')):
        raise RuntimeError('Required real comparisons failed')
    files = list((task/'completed_summary').glob('*.json')) + list((task/'completed_summary').glob('*.csv'))
    for subject in ('sub01', 'sub02'):
        for hemi in ('lh', 'rh'):
            for stage in ('remesh', 'sphere', 'register'):
                for variant in ('baseline', 'candidate'):
                    stage_dir = task/'cold_pairs'/subject/hemi/stage/variant
                    files.extend(stage_dir/name for name in ('report.json', 'monitor/monitor.json', 'monitor/gpu_samples.csv'))
                files.append(task/'reference'/subject/hemi/stage/'report.json')
            chain = task/'chains'/subject/hemi
            files.extend(chain/name for name in ('provenance.json', 'output/report.json', 'output/monitor/monitor.json', 'output/monitor/gpu_samples.csv'))
    files = sorted(set(files))
    manifest = {}
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in files:
            data = path.read_bytes()
            relative = path.relative_to(task).as_posix()
            manifest[relative] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
        data = (json.dumps({'files': manifest, 'scope': 'numerical metadata only; private surfaces, images, licenses and caches excluded'}, indent=2)+'\n').encode()
        info = tarfile.TarInfo('metadata_manifest.json')
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    print(json.dumps({'archive_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'file_count': len(files)}))


if __name__ == '__main__':
    main()
