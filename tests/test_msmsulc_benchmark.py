"""Metrics and privacy regressions; generated data here are test fixtures only."""
import importlib.util
from pathlib import Path

import nibabel as nib
from nibabel.cifti2.cifti2_axes import BrainModelAxis, SeriesAxis
import numpy as np
import pytest


def tool():
    path = Path(__file__).parents[1]/'tools/benchmark_msmsulc.py'
    spec = importlib.util.spec_from_file_location('msmsulc_benchmark_test', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def write_sphere(path, vertices, faces):
    nib.save(nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(np.asarray(vertices, np.float32), intent=1008),
        nib.gifti.GiftiDataArray(np.asarray(faces, np.int32), intent=1009),
    ]), path)


def test_sphere_metrics_require_same_native_topology(tmp_path):
    a, b = tmp_path/'a.surf.gii', tmp_path/'b.surf.gii'
    vertices = [[100, 0, 0], [0, 100, 0], [0, 0, 100]]
    write_sphere(a, vertices, [[0, 1, 2]])
    write_sphere(b, vertices, [[1, 0, 2]])
    with pytest.raises(ValueError, match='native vertex order'):
        tool().sphere_metrics(a, b, a)


def test_sphere_metrics_use_stable_zero_angle_and_count_folds(tmp_path):
    a, b = tmp_path/'a.surf.gii', tmp_path/'b.surf.gii'
    vertices = [[100, 0, 0], [0, 100, 0], [0, 0, 100]]
    write_sphere(a, vertices, [[0, 1, 2]])
    identical = tool().sphere_metrics(a, a, a)
    assert identical['sphere_angle_deg']['maximum'] == 0
    assert identical['fnit_folded_faces'] == 0
    write_sphere(b, [vertices[1], vertices[0], vertices[2]], [[0, 1, 2]])
    assert tool().sphere_metrics(b, a, a)['fnit_folded_faces'] == 1


def test_cifti_corr_is_across_time_then_average(tmp_path):
    axis = BrainModelAxis.from_surface([0, 1, 2], 3, name='CORTEX_LEFT')
    time = SeriesAxis(0, .7, 4)
    x = np.array([[0, 10, 2], [1, 20, 2], [2, 30, 2], [3, 40, 2]], np.float32)
    y = np.array([[0, 40, 2], [1, 30, 2], [2, 20, 2], [3, 10, 2]], np.float32)
    paths = [tmp_path/'a.dtseries.nii', tmp_path/'b.dtseries.nii']
    for path, values in zip(paths, [x, y]):
        nib.save(nib.Cifti2Image(values, header=nib.Cifti2Header.from_axes((time, axis))), path)
    report = tool().cifti_metrics(*paths, chunk_size=1)
    cortex = report['structures']['CIFTI_STRUCTURE_CORTEX_LEFT']
    assert cortex['valid_nonconstant_grayordinates'] == 2
    assert cortex['mean_temporal_r'] == pytest.approx(0, abs=1e-14)
    assert cortex['mean_absolute_difference'] == pytest.approx(np.abs(x-y).mean())


def test_source_report_export_drops_paths_and_input_hashes():
    private = {'seconds': 2, 'source_path': '/private/sub-a', 'input_sha256': 'abcdef',
               'folded_output_faces': 1, 'folded_solver_faces': 1,
               'minimum_output_orientation_ratio': -1.5,
               'minimum_solver_orientation_ratio': -1.6, 'degenerate_input_faces': 0,
               'stages': [{'control_points': 162, 'iterations': [{'changed': 3, 'seconds': 1}],
                           'private_path': '/private/other'}]}
    safe = tool().safe_registration_report({'L': private, 'R': private})
    assert safe['L']['stages'][0]['iterations'][0] == {'changed': 3, 'seconds': 1}
    assert safe['L']['folded_output_faces'] == safe['L']['folded_solver_faces'] == 1
    assert safe['L']['minimum_output_orientation_ratio'] == -1.5
    assert safe['L']['minimum_solver_orientation_ratio'] == -1.6
    assert safe['L']['degenerate_input_faces'] == 0
    assert 'private' not in str(safe) and 'abcdef' not in str(safe)


def test_legacy_source_cannot_claim_requested_config_applied():
    def old_function(inputs, output_dir, *, device='cpu'):
        pass
    options, report = tool().registration_options({'config_file': '/private/config'}, old_function)
    assert options == {}
    assert report['requested'] and not report['applied']
    assert '/private' not in str(report)


def test_configuration_role_rejects_conflicting_options():
    def function(inputs, output_dir, *, device='cpu', config=None):
        pass
    with pytest.raises(ValueError, match='only one'):
        tool().registration_options({'config_file': '/private/a', 'config_options': {}}, function)
    with pytest.raises(ValueError, match='device CLI'):
        tool().registration_options({'run_options': {'device': 'cuda:0'}}, function)


def test_peak_measurement_retains_values_before_source_resets(tmp_path):
    from types import SimpleNamespace
    state = {'allocated': 7, 'reserved': 12}
    def reset(device=None):
        state.update(allocated=0, reserved=0)
    cuda = SimpleNamespace(is_available=lambda: True,
                           reset_peak_memory_stats=reset,
                           max_memory_allocated=lambda device=None: state['allocated'],
                           max_memory_reserved=lambda device=None: state['reserved'])
    fake_torch = SimpleNamespace(cuda=cuda)
    native = SimpleNamespace(optimize=lambda *args: b'')
    with tool().Measure(fake_torch, SimpleNamespace(), native, tmp_path) as measure:
        fake_torch.cuda.reset_peak_memory_stats()
        state.update(allocated=4, reserved=15)
        fake_torch.cuda.reset_peak_memory_stats()
    assert measure.peak_allocated_before_source_reset == 7
    assert measure.peak_reserved_before_source_reset == 15
    assert state == {'allocated': 0, 'reserved': 0}


@pytest.mark.parametrize('position_api', [False, True])
def test_measure_finds_affine_cost_module_after_wrapping_initialization(tmp_path, monkeypatch, position_api):
    from types import ModuleType, SimpleNamespace
    import sys
    benchmark = tool()
    affine_module = ModuleType('fnit_test_affine_measurement')
    class RigidCost:
        def __call__(self, value):
            return value + 1
    if position_api:
        class RigidCost(RigidCost):
            def evaluate_positions(self, value):
                return super().__call__(value)
            def __call__(self, value):
                return self.evaluate_positions(value)
    def initialize():
        cost = affine_module._RigidCost()
        first = cost.evaluate_positions(1) if position_api else cost(1)
        return first + cost(2)
    initialize.__module__ = affine_module.__name__
    affine_module._RigidCost = RigidCost
    monkeypatch.setitem(sys.modules, affine_module.__name__, affine_module)
    msm = SimpleNamespace(_affine_initialization=initialize)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    native = SimpleNamespace(optimize=lambda *args: b'')
    with benchmark.Measure(fake_torch, msm, native, tmp_path) as measure:
        assert msm._affine_initialization() == 5
        assert msm._affine_initialization.__module__ == affine_module.__name__
    assert measure.counts['affine_initialization'] == 1
    assert measure.counts['rigid_cost_evaluation'] == 2
    assert msm._affine_initialization is initialize
    assert RigidCost()(1) == 2


def test_projection_recomputes_area_surfaces_from_each_registered_sphere(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import fnit.fmri.surface_fmriprep as projection
    benchmark = tool()
    geometries = {}
    for name in ['left', 'right']:
        geometries[name] = {field: '/private/'+name+'/'+field for field in (
            'white', 'pial', 'midthickness', 'registered_sphere', 'native_roi',
            'atlas_sphere', 'atlas_midthickness', 'atlas_roi')}
    case = {'projection': {**geometries, 'clean_t1w': '/private/t1w',
                           'clean_mni': '/private/mni', 'left_label': '/private/l',
                           'right_label': '/private/r', 'hcp_dseg': '/private/dseg',
                           'official_cifti': '/private/old.dtseries.nii'}}
    commands = []; supplied = {}
    monkeypatch.setattr(benchmark.shutil, 'which', lambda command: '/bin/wb_command')
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda arguments, **kwargs: commands.append(arguments))
    def run(**kwargs):
        supplied.update(kwargs)
        return SimpleNamespace(dtseries=tmp_path/'result.dtseries.nii')
    monkeypatch.setattr(projection, 'run_fmriprep_surface_projection', run)
    benchmark.project(case, {'L': '/private/fnit.L.sphere', 'R': '/private/fnit.R.sphere'}, tmp_path)
    assert len(commands) == 2
    for command, name, hemi in zip(commands, ['left', 'right'], ['L', 'R']):
        assert command[1] == '-surface-resample'
        assert command[2] == geometries[name]['midthickness']
        assert command[3] == '/private/fnit.'+hemi+'.sphere'
        assert command[4] == geometries[name]['atlas_sphere']
        assert command[5] == 'BARYCENTRIC'
        assert str(supplied[name].atlas_midthickness) == command[6]
        assert supplied[name].registered_sphere == command[3]
    assert 'official_cifti' not in supplied


def test_measure_records_native_wls_payload_once_without_extra_gpu_transfer(tmp_path):
    from types import SimpleNamespace
    benchmark = tool()
    payload = np.zeros((2, 3, 3), np.float64)
    seen = []
    def native_cost(buffer, rows, width, sigma):
        seen.append(buffer)
        return 4.0
    native = SimpleNamespace(source_wls_cost=native_cost)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with benchmark.Measure(fake_torch, SimpleNamespace(), native, tmp_path) as measure:
        assert native.source_wls_cost(payload, 2, 3, 1.) == 4.
    assert seen == [payload]
    assert native.source_wls_cost is native_cost
    assert measure.counts['affine_source_wls'] == 1
    assert measure.counts['affine_wls_host_payload_bytes'] == payload.nbytes
    assert measure.counts['affine_wls_query_slots'] == 6
    assert measure.times['affine_source_wls']['calls'] == 1


def test_measure_separates_native_rotation_cache_from_label_application(tmp_path):
    from types import SimpleNamespace
    benchmark = tool()
    prior = np.zeros((5, 3), np.float64)
    centre = np.ones(3, np.float64)
    seen = []
    def build(buffer, raw_centre, count):
        seen.append((buffer, raw_centre, count))
        return bytes(count*9*8)
    native = SimpleNamespace(source_rotation_matrices=build)
    def prepare(points, origin):
        return native.source_rotation_matrices(points, origin, len(points))
    msm = SimpleNamespace(_rotation_matrices=prepare)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with benchmark.Measure(fake_torch, msm, native, tmp_path) as measure:
        result = msm._rotation_matrices(prior, centre)
    assert len(seen) == 1 and seen[0][0] is prior and seen[0][1] is centre
    assert native.source_rotation_matrices is build
    assert measure.counts['source_rotation_matrices'] == 1
    assert measure.counts['control_rotation_points'] == 5
    assert measure.counts['control_rotation_host_input_bytes'] == prior.nbytes+centre.nbytes
    assert measure.counts['control_rotation_host_result_bytes'] == len(result)
    assert measure.times['source_rotation_matrices']['calls'] == 1
    assert measure.counts['label_rotation_preparation'] == 1
    assert measure.times['label_rotation_preparation']['calls'] == 1
    assert measure.counts['label_rotation'] == 0
