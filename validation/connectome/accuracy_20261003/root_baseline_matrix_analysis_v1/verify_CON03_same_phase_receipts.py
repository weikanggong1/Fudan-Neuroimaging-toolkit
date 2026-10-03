"""Read-only CPU identity receipt for completed actual B/C CON03; no metric recomputation."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1')
CONFIG_SHA = 'f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c'
SCIENCE = {'baseline': '328c398496c5b90f459381ca462ba6597cbbe5f2e9700f0b1a273f1408962bac',
           'candidate': 'a27fe1ad0aca34c23b62017dc0bacb6b7a4c44d3423bf855a840509ffc1b6e82'}
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'formal_frozen_v1/tools_source'))
from tools import analyze_connectome_accuracy_cohort as analysis
import nibabel as nib
import numpy as np

driver = analysis.driver
configuration_path = ROOT / 'formal_frozen_v1/accuracy_configuration.json'
assert driver.cohort.sha256(configuration_path) == CONFIG_SHA
config = json.loads(configuration_path.read_bytes())
driver.verify_tools(config)
manifest, bindings = driver.bound(config['raw_manifest']), driver.bound(config['input_bindings'])
case = driver.validate_plan(config, manifest, bindings)['sub-CON03']
anatomy = bindings['cases']['sub-CON03']['anatomy']['directory']
reports = {'baseline': ROOT / 'root_baseline_matrix_analysis_v1/sub-CON03/case_report.json',
           'candidate': ROOT / 'root_matrix_analysis_v3/sub-CON03/report.json'}
receipt = {'scope': 'actual completed formal B/C CON03; frozen completed-run/export gates; existing metric reports',
           'utc': driver.cohort.utc(), 'command': [sys.executable, *sys.argv],
           'configuration_sha256_before': CONFIG_SHA,
           'verification_script_sha256': driver.cohort.sha256(Path(__file__).resolve()),
           'frozen_analysis_sha256': driver.cohort.sha256(Path(analysis.__file__).resolve()), 'cases': {}}
for version in ('baseline', 'candidate'):
    gpu, wall, gpu_path = analysis.completed_run(config, case, version, anatomy)
    assert gpu['source_before']['source_fingerprint'] == SCIENCE[version]
    assert gpu['source_after']['source_fingerprint'] == SCIENCE[version]
    exports = {name: analysis.verified_export(wall, name) for name in
               ('tracks.tck', 'wm_fod_normalized.nii.gz', 'track_metrics.npz')}
    job = Path(config['run_root']) / version / 'sub-CON03'
    report = json.loads(reports[version].read_bytes())
    result = report if version == 'baseline' else report['cases']['sub-CON03']
    metric_root = (ROOT / 'root_baseline_matrix_analysis_v1/sub-CON03' if version == 'baseline' else
                   ROOT / 'root_matrix_analysis_v3/sub-CON03/sub-CON03')
    paths = [reports[version], gpu_path, job / 'raw_bids_wall.json', *exports.values(),
             *(metric_root / name for name in ('matrix_envelope.json', 'population_envelope.json', 'population.png'))]
    before = {str(path): {'sha256': driver.cohort.sha256(path), 'size_bytes': path.stat().st_size} for path in paths}
    for name in ('matrix_envelope.json', 'population_envelope.json', 'population.png'):
        assert before[str(metric_root / name)] == result['outputs'][name]
    population = json.loads((metric_root / 'population_envelope.json').read_bytes())
    assert population['input_paths']['fnit_0'] == str(exports['tracks.tck'])
    assert population['input_sha256']['fnit_0'] == before[str(exports['tracks.tck'])]['sha256']
    assert population['grid_path'] == str(exports['wm_fod_normalized.nii.gz'])
    assert population['grid_sha256'] == before[str(exports['wm_fod_normalized.nii.gz'])]['sha256']
    official = driver.bound(bindings['cases']['sub-CON03']['official_reference_manifest'])
    official_tck = {}
    for seed in official['seeds']:
        path = Path(population['input_paths'][f'official_{seed}'])
        digest = driver.cohort.sha256(path)
        assert digest == population['input_sha256'][f'official_{seed}'] == official['outputs'][str(seed)]['tracks_sha256']
        official_tck[str(path)] = digest
    # Read actual serialized endpoints and saved returned-result scalars; no new tracking or scalar calculation.
    tracks = list(nib.streamlines.load(exports['tracks.tck'], lazy_load=False).streamlines)
    with np.load(exports['track_metrics.npz']) as scalars:
        endpoints = np.stack([np.stack((track[0], track[-1])) for track in tracks]) if tracks else np.empty((0, 2, 3), np.float32)
        assert endpoints.dtype == scalars['endpoints'].dtype and np.array_equal(endpoints, scalars['endpoints'])
        assert all(len(scalars[name]) == len(tracks) for name in ('weights', 'lengths', 'mean_fa', 'endpoints'))
    assert len(tracks) == result['track_count'] == population['track_counts']['fnit_0']
    analysis.completed_run(config, case, version, anatomy)
    for name in exports:
        assert analysis.verified_export(wall, name) == exports[name]
    after = {str(path): {'sha256': driver.cohort.sha256(path), 'size_bytes': path.stat().st_size} for path in paths}
    assert before == after
    assert {str(path): driver.cohort.sha256(Path(path)) for path in official_tck} == official_tck
    receipt['cases'][version] = {'source_before': SCIENCE[version], 'source_after': SCIENCE[version],
                               'anatomy_directory': anatomy, 'track_count': len(tracks),
                               'endpoint_bits_equal': True, 'scalar_counts_equal': True,
                               'actual_files_before': before, 'actual_files_after': after,
                               'official_TCK_sha256_before_after': official_tck,
                               'gpu_memory': gpu['gpu_memory'], 'memory_budget': gpu['memory_budget'],
                               'raw_dwi_cli_seconds': gpu['raw_dwi_cli_total_runtime_seconds'],
                               'same_run_export_seconds': wall['post_timing_result_export']['seconds']}
assert driver.cohort.sha256(configuration_path) == CONFIG_SHA
receipt['configuration_sha256_after'] = CONFIG_SHA
receipt['status'] = 'strict_bindings_passed'
output = ROOT / 'root_baseline_matrix_analysis_v1_controller/CON03_same_phase_binding_receipt.json'
assert not output.exists()
driver.cohort.atomic_json(output, receipt)
print(json.dumps({'receipt_path': str(output), 'sha256': driver.cohort.sha256(output),
                  'tracks': {v: r['track_count'] for v, r in receipt['cases'].items()},
                  'status': receipt['status']}))
