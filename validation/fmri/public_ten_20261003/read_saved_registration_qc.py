"""只读摘录正式整链保存的 MSM solver/output QC；不重建临时基线或运行 MRI。"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    case = args.case_root.resolve(strict=True)
    output = args.output_root.resolve()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError('new output root already exists')
    source = args.source.resolve(strict=True)
    protected = (case, source, Path(__file__).resolve().parent)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('output overlaps an original case or reader source')
    paths = {'formal_report': case / 'report/report.public.json',
             'files_binding': case / 'report/files.private.json',
             'formal_source_manifest': case / 'report/source.private.json',
             'native_qc_module': source / 'src/fnit/msm/msmsulc.py',
             'reader': Path(__file__).resolve()}
    initial = {key: sha256(path) for key, path in paths.items()}
    formal = json.loads(paths['formal_report'].read_text())
    files = json.loads(paths['files_binding'].read_text())
    manifest = json.loads(paths['formal_source_manifest'].read_text())
    if (initial['formal_source_manifest'] != formal['source_sha256']
            or initial['native_qc_module'] != manifest['src/fnit/msm/msmsulc.py']):
        raise ValueError('actual native QC source differs from its original source manifest')
    if (formal.get('status') != 'complete' or formal.get('subject') != case.name
            or formal.get('source_revision') != '1128bc52c7a0233266e5b8a8d7dc0b382994e676'
            or formal.get('frames') != 180 or formal.get('backend') != 'fnit'
            or any(formal.get(key) is not True for key in (
                'raw_inputs_unchanged', 'configuration_unchanged',
                'source_unchanged_during_run', 'driver_unchanged', 'provenance_guards_passed'))):
        raise ValueError('original whole API completion and provenance guards are required')
    metadata = Path(files['metadata']).resolve(strict=True)
    if not metadata.is_relative_to(case / 'derivatives') or not metadata.is_file():
        raise ValueError('saved metadata escapes the original derivatives')
    paths['metadata'] = metadata
    before = {key: sha256(path) for key, path in paths.items()}
    if any(before[key] != value for key, value in initial.items()):
        raise ValueError('original report or reader changed during input selection')
    started = time.perf_counter()
    saved = json.loads(metadata.read_text())['FNIT']['RegistrationDetails']['Hemispheres']
    selected = {}
    for hemisphere in ('L', 'R'):
        values = {}
        for key in ('folded_solver_faces', 'folded_output_faces',
                    'minimum_solver_orientation_ratio', 'minimum_output_orientation_ratio',
                    'degenerate_input_faces'):
            value = saved[hemisphere][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('saved QC scalar is not finite numeric data')
            if key.endswith('_faces') and (not isinstance(value, int) or value < 0):
                raise ValueError('saved face count is not a nonnegative integer')
            values[key] = value
        selected[hemisphere] = values
    after = {key: sha256(path) for key, path in paths.items()}
    if before != after:
        raise ValueError('original saved input or reader changed')
    result = {'status': 'measured_saved_production_qc', 'case_id': case.name,
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_revision': formal['source_revision'], 'input_sha256_before': before,
              'input_sha256_after': after, 'input_guards_equal': True,
              'raw_input_sha256': formal['input_sha256'],
              'original_source_manifest_sha256': formal['source_sha256'],
              'original_whole_source_guard_passed': formal['source_unchanged_during_run'],
              'reader_sha256': before['reader'], 'hemispheres': selected,
              'native_qc_module_sha256': before['native_qc_module'],
              'source_function': 'fnit.msm.msmsulc._native_output_qc',
              'folded_solver_faces_phase': 'Native output after final _sphere_warp interpolation, before writing float32; not the optimizer DATA/control sphere final state.',
              'folded_output_faces_phase': 'Same native output in its saved float32 precision.',
              'baseline': 'Original production normalized rotated sphere; differs from posthoc own-native sphere.',
              'scope': 'Exact numeric fields already saved by the original whole API. No geometry change, MRI/MSM rerun, temporary baseline reconstruction or recovery of unavailable solver coordinates.',
              'cuda_initialized': False, 'diagnostic_wall_seconds': time.perf_counter() - started}
    args.output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_root / 'report.public.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'report_sha256': sha256(args.output_root / 'report.public.json')}))


if __name__ == '__main__':
    main()
