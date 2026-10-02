"""独立原始病例对照工具的 CPU 合同回归；不作为科学 benchmark。"""
import importlib.util
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from tools import benchmark_connectome_raw_cohort_envelope as whole
from tools.connectome_repeat_common import NAMES, check_metadata, sha256


def raw_tool():
    path = Path(__file__).parents[2] / 'tools/reference/benchmark_connectome_raw_official.py'
    spec = importlib.util.spec_from_file_location('official_raw_contract_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_record(path):
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha256(path)}


def test_original_official_raw_case_binding_requires_actual_input_sha_and_success(tmp_path):
    tool = raw_tool()
    files = []
    for name, kind in [('T1.nii.gz', 'raw_t1w'), ('AP.nii.gz', 'raw_dwi'),
                       ('PA.nii.gz', 'reverse_pe'), ('AP.bvec', 'bvec'), ('AP.json', 'dwi_json'),
                       ('dataset_description.json', 'dataset_description')]:
        path = tmp_path / name
        path.write_bytes(('contract fixture:' + name).encode())
        files.append({**file_record(path), 'kind': kind})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'dataset': 'fixture_only', 'snapshot': 'fixture', 'license': 'fixture',
        'cases': [{'case_id': 'sub-fixture', 'subject': 'fixture', 'session': 'preop', 'input_files': files}]}))
    producer = tmp_path / 'rawprep.json'
    report = {'completed': True, 'subject': 'fixture', 'session': 'preop',
              'commands': [{'returncode': 0, 'command': ['/fixture/eddy_cpu'],
                            'program_sha256': 'e' * 64, 'host': 'fixture', 'wall_seconds': 1.}],
              'input_sha256': {item['path']: item['sha256'] for item in files if item['kind'] != 'dataset_description'}}
    producer.write_text(json.dumps(report))
    anatomy = {'raw_t1w': file_record(tmp_path / 'T1.nii.gz')}
    dwi = {'upstream_official_rawprep_report': str(producer),
           'upstream_official_rawprep_report_sha256': sha256(producer)}
    result = tool.raw_binding(manifest, sha256(manifest), 'sub-fixture', anatomy, dwi)
    assert result['rawprep_canonical_dwi_coverage_verified'] is True
    assert result['official_eddy_solver']['mode'] == 'cpu'
    assert result['official_eddy_solver']['gpu_uuid'] is None
    report['input_sha256'].pop(str(tmp_path / 'PA.nii.gz'))
    producer.write_text(json.dumps(report)); dwi['upstream_official_rawprep_report_sha256'] = sha256(producer)
    with pytest.raises(ValueError, match='provenance differs'):
        tool.raw_binding(manifest, sha256(manifest), 'sub-fixture', anatomy, dwi)
    report['completed'] = False; report['state'] = 'failed'; report['error'] = {'type': 'fixture_failure'}
    producer.write_text(json.dumps(report)); dwi['upstream_official_rawprep_report_sha256'] = sha256(producer)
    with pytest.raises(ValueError, match='incomplete/failed'):
        tool.raw_binding(manifest, sha256(manifest), 'sub-fixture', anatomy, dwi)


def test_official_native_fa_record_preserves_nan_and_header_dtype(tmp_path):
    tool = raw_tool()
    path = tmp_path / 'fa.nii.gz'
    data = np.zeros((4, 4, 4), dtype=np.float32); data[1, 2, 3] = np.nan
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    _, image, actual = tool.image_record(file_record(path), 'originalFA')
    assert actual['nonfinite_count'] == 1
    assert np.isnan(np.asarray(image.dataobj)[1, 2, 3])
    reader = tmp_path / 'reader.json'
    record = {'json': str(reader)}
    info = {'name': str(path), 'datatype': 'Float32LE', 'strides': [1, 2, 3],
            'size': [4, 4, 4], 'spacing': [1., 1., 1.], 'transform': np.eye(4).tolist(),
            'intensity_offset': 0., 'intensity_scale': 1., 'format': 'NIfTI-1'}
    reader.write_text(json.dumps(info))
    assert tool.native_readback(record, actual)['status'].startswith('native_source')
    info['datatype'] = 'Float64LE'; reader.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='dtype'):
        tool.native_readback(record, actual)


def item(directory, digest, rows=None):
    arrays = {name: np.eye(2) for name in NAMES}
    nodes = rows or [['1', '1001', 'lh', 'regionA'], ['2', '2001', 'rh', 'regionB']]
    return arrays, {'directory': str(directory), 'nodes': 2, 'atlas_sha256': digest, 'node_rows': nodes}


def test_independent_wholechain_semantics_do_not_relax_fixed_input_atlas_contract(tmp_path):
    official = [item(tmp_path / f'o{i}', 'a' * 64) for i in range(3)]
    fnit = [item(tmp_path / 'f0', 'b' * 64)]
    result = whole.semantic_identity(official, fnit, 'fixture')
    assert result['node_rows']['status'] == 'verified_equal_semantics'
    assert result['cross_arm_atlas_sha256']['official'] != result['cross_arm_atlas_sha256']['fnit']
    with pytest.raises(ValueError, match='atlas_sha256 differs'):
        check_metadata(official + fnit, 'original_fixed_input_gate')
    changed = [item(tmp_path / 'f1', 'b' * 64, [['1', '1002', 'lh', 'different'], ['2', '2001', 'rh', 'regionB']])]
    with pytest.raises(ValueError, match='semantic mismatch'):
        whole.semantic_identity(official, changed, 'fixture')
    official[1][1]['atlas_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='atlas_sha256 differs'):
        whole.semantic_identity(official, fnit, 'fixture')


def test_fnit_raw_producer_actual_sha_and_duplicate_records_are_not_accepted():
    rows = [{'path': '/tmp/raw_contract_fixture', 'sha256': 'a' * 64, 'actual_sha256': 'b' * 64}]
    with pytest.raises(ValueError, match='not verified unchanged'):
        whole.file_map(rows, verified=True)
    with pytest.raises(ValueError, match='duplicate'):
        whole.file_map(rows * 2)


def test_readonly_program_loader_failure_is_recorded_not_ready(tmp_path):
    # Protocol fixture only: preserve an actual failure return code; not a scientific tolerance.
    path = Path(__file__).parents[2] / 'tools/reference/preflight_connectome_raw_reference.py'
    spec = importlib.util.spec_from_file_location('official_raw_readonly_fixture', path)
    preflight = importlib.util.module_from_spec(spec); spec.loader.exec_module(preflight)
    executable = tmp_path / 'version_fixture'
    executable.write_text('#!/bin/sh\necho fixture-loader-unavailable >&2\nexit 9\n')
    executable.chmod(0o700)
    result = preflight.inspect_program(raw_tool(), executable,
        {'path': str(executable), 'sha256': sha256(executable)}, {'CUDA_VISIBLE_DEVICES': ''})
    assert result['returncode'] == 9
    assert result['actual_version_output'] == 'fixture-loader-unavailable'
    assert result['readonly_invocation_ready'] is False


def test_actual_five_tissue_undefined_channel_spacing_is_metadata_only(tmp_path):
    tool = raw_tool()
    path = tmp_path / 'actual_five_tissue_protocol_fixture.nii.gz'
    values = np.zeros((4, 4, 4, 5), dtype=np.float32)
    values[1, 2, 3, 0] = np.nan
    image = nib.Nifti1Image(values, np.eye(4))
    image.header['pixdim'][4] = np.nan
    nib.save(image, path)
    original_sha = sha256(path)
    _, read_image, source = tool.image_record(file_record(path), '5TT')
    assert source['spacing'] == [1., 1., 1., None]
    assert source['undefined_nonspatial_spacing'][0]['axis'] == 3
    assert source['nonfinite_count'] == 1
    assert np.isnan(np.asarray(read_image.dataobj)[1, 2, 3, 0])
    json.dumps(source, allow_nan=False)
    reader = tmp_path / 'mrinfo.json'
    actual = {'name': str(path), 'datatype': 'Float32LE', 'strides': [1, 2, 3, 4],
              'size': [4, 4, 4, 5], 'spacing': [1., 1., 1., float('nan')],
              'transform': np.eye(4).tolist(), 'intensity_offset': 0., 'intensity_scale': 1.}
    reader.write_text(json.dumps(actual))
    reader_bytes = reader.read_bytes()
    readback = tool.native_readback({'json': str(reader)}, source)
    json.dumps(readback, allow_nan=False)
    assert readback['mrinfo_json']['spacing'][3] is None
    assert readback['undefined_nonspatial_spacing'][0]['source_value'] == 'NaN'
    assert reader.read_bytes() == reader_bytes and sha256(path) == original_sha


@pytest.mark.parametrize('spacing', [[float('nan'), 1., 1., 1.], [1., 0., 1.],
                                    [1., 1., 1., float('inf')], [1., 1., 1., -1.]])
def test_nonspatial_metadata_fix_does_not_relax_spatial_geometry_or_invalid_values(spacing):
    with pytest.raises(ValueError, match='spacing'):
        raw_tool().spacing_metadata(spacing)
