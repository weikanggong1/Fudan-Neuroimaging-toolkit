"""导出本轮 WM 已完成收据及公开 T1 衍生脑图；不复制 MRI、许可或参考程序。"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def numeric(value):
    if isinstance(value, dict):
        return {key: numeric(item) for key, item in value.items() if not isinstance(item, str)}
    if isinstance(value, list):
        return [numeric(item) for item in value]
    return (type(value).__name__, value) if value is None or isinstance(value, (int, float, bool)) else 'string'


def redact(value, root):
    if isinstance(value, dict):
        return {key: '<private-host>' if key == 'hostname_private' else redact(item, root)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), '<FNIT_ROOT>')
    return value


def collect(args):
    spec = importlib.util.spec_from_file_location('wm_regression_compare', args.source / 'tools/benchmark_recon_wm_remove_paths.py')
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    args.output.mkdir(parents=True, exist_ok=False)
    exports = []

    def export(src, relative):
        value = json.loads(src.read_text()); safe = redact(value, args.fnit_root)
        if numeric(value) != numeric(safe): raise ValueError('numeric type/value changed during export')
        dest = args.output / relative; write(dest, safe)
        exports.append({'relative_path': str(relative), 'original_sha256': sha(src),
                        'published_sha256': sha(dest), 'bytes': dest.stat().st_size,
                        'numeric_type_value_preserved': True})
        return safe

    for version, src in [('v1', args.v1_run), ('v2', args.v2_run)]:
        for path in sorted(src.rglob('*.json')):
            export(path, Path('reports') / version / path.relative_to(src))
    export(args.source / 'SOURCE_MANIFEST.json', Path('source_manifest_v2.json'))
    v1_source = args.fnit_root / 'workspaces' / args.v1_run.name / 'SOURCE_MANIFEST.json'
    export(v1_source, Path('source_manifest_v1.json'))
    result = {'scope': 'frozen same-input complete WM file stage; no raw-T1 or whole evaluation',
              'overall_metric_equivalence': 'not_assessed', 'source_manifest_v2_sha256': sha(args.source / 'SOURCE_MANIFEST.json'),
              'v1_exit_code': int((args.v1_run / 'EXIT_CODE').read_text()),
              'v2_exit_code': int((args.v2_run / 'EXIT_CODE').read_text()), 'cases': {}}
    import nibabel as nib
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for case, cpu_root in [('sub07', args.v1_run / 'sub07-cpu'), ('sub06', args.v2_run / 'sub06-cpu')]:
        complete = json.loads((cpu_root / 'COMPLETE.json').read_text())
        if complete['status'] != 'complete': raise ValueError('not complete')
        native = cpu_root / '2-native/output.mgz'; cpu = cpu_root / '1-cpu/output.mgz'
        hybrid_root = args.v1_run / (case + '-hybrid'); hybrid = hybrid_root / '0-hybrid/output.mgz'
        comparisons = {'cpu_vs_native': tool.compare(cpu,native),
            'hybrid_vs_native': tool.compare(hybrid,native),
            'hybrid_vs_cpu': tool.compare(hybrid,cpu),
            'native_repeat': tool.compare(cpu_root / '3-native/output.mgz',native),
            'cpu_repeat': tool.compare(cpu_root / '4-cpu/output.mgz',cpu)}
        rows = []
        for i, rec in enumerate(complete['records']):
            api = json.loads((cpu_root/f'{i}-{rec["backend"]}'/'report.json').read_text())
            rows.append({'backend': rec['backend'], 'order': i, 'API_seconds': api['file_api_wall_seconds'],
                         'fresh_CLI_seconds': rec['fresh_CLI_wall_seconds'],
                         'import_seconds': api['import_seconds'], 'output_sha256': api['output_sha256'],
                         'original_storage_dtype': api['original_storage_dtype'],
                         'input_sha256': api['input_sha256'],
                         'input_unchanged': api['input_sha256'] == api['input_sha256_after']})
        hybrid_complete = json.loads((hybrid_root/'COMPLETE.json').read_text())
        hybrid_api = json.loads((hybrid_root/'0-hybrid/report.json').read_text())
        rows.append({'backend':'hybrid','order': 'separate_gpu_window',
                     'API_seconds': hybrid_api['file_api_wall_seconds'],
                     'fresh_CLI_seconds': hybrid_complete['records'][0]['fresh_CLI_wall_seconds'],
                     'import_seconds': hybrid_api['import_seconds'], 'output_sha256': hybrid_api['output_sha256'],
                     'original_storage_dtype': hybrid_api['original_storage_dtype'],
                     'input_sha256': hybrid_api['input_sha256'],
                     'input_unchanged': hybrid_api['input_sha256'] == hybrid_api['input_sha256_after']})
        result['cases'][case] = {'comparisons': comparisons,'stage_times': rows,
            'active_path_real_benchmark': complete['active_path_real_benchmark'],
            'path_fill_count_comparison': complete['path_fill_count_comparison'],
            'hybrid_cuda': hybrid_api['cuda'], 'hybrid_memory': hybrid_complete['records'][0]['memory_samples'],
            'complete_report_original_sha256': sha(cpu_root/'COMPLETE.json'),
            'hybrid_report_original_sha256': sha(hybrid_root/'COMPLETE.json'),
            'strict_decoded_stage_passed': all(x['different_voxels']==0 and x['affine_exact'] and x['dtype_equal'] and x['vox2ras_tkr_exact'] for x in comparisons.values()),
            'strict_file_bytes_passed': all(x['raw_bytes_equal'] for x in comparisons.values())}
        wm = np.asarray(nib.load(str(cpu)).dataobj); ref = np.asarray(nib.load(str(native)).dataobj)
        mri = args.fnit_root/'runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1'/f'{case}-candidate/subject/mri'
        brain = np.asarray(nib.load(str(mri/'brain.mgz')).dataobj)
        foreground=np.argwhere(wm>=5); centre=np.rint(foreground.mean(axis=0)).astype(int)
        figure,axes=plt.subplots(2,3,figsize=(11,7))
        for axis in range(3):
            bg=np.take(brain,centre[axis],axis=axis).T; mask=np.take(wm>=5,centre[axis],axis=axis).T
            diff=np.take(np.abs(wm.astype(np.int16)-ref.astype(np.int16)),centre[axis],axis=axis).T
            axes[0,axis].imshow(bg,cmap='gray',origin='lower',vmin=0,vmax=140)
            axes[0,axis].contour(mask,levels=[.5],colors=['#35c5fb'],linewidths=.45)
            axes[0,axis].set_title(f'{case}, axis {axis}, voxel {centre[axis]}')
            axes[1,axis].imshow(diff,origin='lower',cmap='magma',vmin=0,vmax=1)
            axes[1,axis].set_title(f'CPU - fixed source; max={diff.max()}')
            for row in range(2): axes[row,axis].axis('off')
        figure.suptitle('Public ds000114 derived T1 / WM overlay; same-input stage diagnostic')
        figure.tight_layout(); path=args.output/'plots'/f'{case}_wm_paths.png';path.parent.mkdir(exist_ok=True)
        figure.savefig(path,dpi=130);plt.close(figure)
        exports.append({'relative_path':str(path.relative_to(args.output)),'published_sha256':sha(path),
                        'bytes':path.stat().st_size,'source':'authorized public ds000114-derived brain/WM from frozen79a41cdd'})
    write(args.output/'summary.json',result)
    import csv
    with (args.output/'stage_times.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['case','backend','order','API_seconds','fresh_CLI_seconds','import_seconds'])
        w.writeheader()
        for case, value in result['cases'].items():
            for row in value['stage_times']:w.writerow(dict(case=case,**{k:row[k] for k in w.fieldnames[1:]}))
    exports.extend({'relative_path':str(p.relative_to(args.output)), 'published_sha256':sha(p),'bytes':p.stat().st_size}
                   for p in [args.output/'summary.json',args.output/'stage_times.csv'])
    failure={'status':'failed_benchmark_driver','v1_exit_code':1,'algorithm_failure':False,
        'location':'sub06-cpu baseline cached Numba environment load',
        'error':'ModuleNotFoundError: No module named <dynamic>',
        'fix':'register explicit importlib module name in sys.modules before exec; independent v2 cache',
        'v1_successful_stage_outputs_preserved':True,'v2_source_algorithm_modules_identical_to_v1':True,
        'failed_log_original_sha256':sha(args.v1_run/'sub06-cpu/0-baseline.log')}
    write(args.output/'v1_failure.json',failure)
    exports.append({'relative_path':'v1_failure.json','published_sha256':sha(args.output/'v1_failure.json'),
                    'bytes':(args.output/'v1_failure.json').stat().st_size})
    write(args.output/'PUBLIC_EXPORT_MANIFEST.json',{'source_root':'private FNIT task run',
        'allowed_payload':'machine JSON/CSV and two public-derived PNG; no MRI/license/weights/programs/source CPP',
        'files':exports})
    print(json.dumps({c: {'pass':v['strict_decoded_stage_passed'],'active':v['active_path_real_benchmark'],
        'times':[(r['backend'],r['API_seconds']) for r in v['stage_times']]} for c,v in result['cases'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fnit-root','source','v1-run','v2-run','output']:p.add_argument('--'+name,type=Path,required=True)
    collect(p.parse_args())
