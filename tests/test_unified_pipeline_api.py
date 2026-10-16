"""统一公开入口保留实现对象、算法选择及参数/结果契约。"""

import importlib
import inspect

import numpy as np
import pytest


@pytest.mark.parametrize(('module_name', 'canonical', 'original'), [
    ('topup', 'FNIT_topup', 'TorchTOPUP'),
    ('eddy', 'FNIT_eddy', 'TorchEDDY'),
    ('dtifit', 'FNIT_dtifit', 'TorchDTIFIT'),
    ('bedpostx', 'FNIT_bedpostx', 'TorchBEDPOSTX'),
    ('probtrackx', 'FNIT_probtrackx2', 'TorchProbtrackX'),
    ('dmri_pipeline', 'FNIT_dMRIPipeline', 'DMRIPipeline'),
    ('connectome', 'FNIT_UKBconnectomePipeline', 'UKBConnectome_pipeline'),
    ('fmri', 'FNIT_fMRIVolume_pipeline', 'fMRIVolume_pipeline'),
    ('fmri', 'FNIT_fMRISurface_pipeline', 'fMRISurface_pipeline'),
    ('melodic', 'FNIT_melodic', 'decompose_spatial_ica'),
    ('melodic', 'FNIT_melodic_bids', 'run_melodic_bids'),
])
def test_canonical_alias_preserves_implementation_and_signature(module_name, canonical, original):
    module = importlib.import_module(f'fnit.{module_name}')
    actual = getattr(module, canonical)
    expected = getattr(module, original)
    assert actual is expected
    assert inspect.signature(actual) == inspect.signature(expected)
    assert canonical in module.__all__


@pytest.mark.parametrize('method', ['msmsulc', 'msmall'])
def test_newmsm_forwards_the_same_inputs_configuration_and_execution(tmp_path, monkeypatch, method):
    import fnit.msm as msm
    seen = {}
    inputs = {'L': object(), 'R': object()}
    config = object()
    expected = {'L': tmp_path / 'L.surf.gii', 'R': tmp_path / 'R.surf.gii'}

    def register(actual_inputs, actual_output, **options):
        assert actual_inputs is inputs
        assert actual_output is tmp_path
        seen.update(options)
        return expected

    monkeypatch.setattr(msm, 'run_' + method, register)
    actual = msm.FNIT_newmsm(inputs, tmp_path, method=method, config=config,
                            device='cpu', execution='reference', parallel=False,
                            cpu_threads=3)
    assert actual is expected
    assert seen == dict(config=config, device='cpu', execution='reference',
                        parallel=False, cpu_threads=3,
                        **({'qc_policy': 'report'} if method == 'msmsulc' else {}))


def test_newmsm_qc_policy_is_forwarded_only_to_sulc(tmp_path, monkeypatch):
    import fnit.msm as msm
    seen = []
    monkeypatch.setattr(msm, 'run_msmsulc', lambda *args, **options: seen.append(options))
    msm.FNIT_newmsm({}, tmp_path, qc_policy='error')
    assert seen[0]['qc_policy'] == 'error'
    with pytest.raises(ValueError, match='only'):
        msm.FNIT_newmsm({}, tmp_path, method='msmall', qc_policy='repair')


def test_newmsm_rejects_invalid_method_before_reading_inputs(tmp_path):
    from fnit.msm import FNIT_newmsm
    with pytest.raises(ValueError, match='method'):
        FNIT_newmsm(None, tmp_path, method='unknown')


@pytest.mark.parametrize(('options', 'selected'), [
    ({}, 'amico'), ({'method': 'amico'}, 'amico'),
    ({'method': 'classic'}, 'classic'), ({'fit_method': 'classic'}, 'classic'),
    ({'method': 'classic', 'fit_method': 'classic'}, 'classic'),
])
def test_noddi_reuses_the_model_configuration_and_all_inherited_methods(options, selected):
    from fnit.amico_noddi import AMICONODDIConfig, FNIT_noddi, TorchAMICONODDI
    config = AMICONODDIConfig(lut_batch_size=17)
    model = FNIT_noddi('cpu', config=config, **options)
    assert model.config is config
    assert model.fit_method == model.method == selected
    assert model.device.type == 'cpu'
    assert FNIT_noddi.__call__ is TorchAMICONODDI.__call__
    assert FNIT_noddi.run is TorchAMICONODDI.run


@pytest.mark.parametrize('options', [
    {'method': 'invalid'}, {'fit_method': 'invalid'},
    {'method': 'amico', 'fit_method': 'classic'},
])
def test_noddi_rejects_invalid_or_conflicting_selection_before_device_initialization(options, monkeypatch):
    from fnit.amico_noddi import FNIT_noddi
    import fnit.amico_noddi.core as core
    monkeypatch.setattr(core, 'configure_device', lambda *_: pytest.fail('device initialization should not run'))
    with pytest.raises(ValueError):
        FNIT_noddi(**options)


def test_mshbm_profile_controls_and_input_objects_are_preserved(monkeypatch):
    import fnit.mshbm as mshbm
    assets = object()
    profiles = [object(), object()]
    expected = (np.array([1, 2], dtype=np.uint8), [{'outer': 2}])
    seen = {}

    def parcellate(actual_profiles, actual_assets, **options):
        assert actual_profiles is profiles
        assert actual_assets is assets
        seen.update(options)
        return expected

    monkeypatch.setattr(mshbm, 'parcellate', parcellate)
    assert mshbm.FNIT_mshbm(profiles, assets, w=111, c=22, max_outer=3,
                           max_em=4, max_m=5, max_lambda=6) is expected
    assert seen == dict(w=111, c=22, max_outer=3, max_em=4, max_m=5, max_lambda=6)


def test_mshbm_timeseries_preparation_keeps_censor_and_shared_assets(monkeypatch):
    import fnit.mshbm as mshbm
    assets, series, censor, profiles = object(), object(), object(), object()
    seen = {}
    monkeypatch.setattr(mshbm, 'load_assets', lambda: assets)

    def prepare(actual_series, actual_assets, *, censor):
        assert actual_series is series
        assert actual_assets is assets
        seen['censor'] = censor
        return profiles

    def parcellate(actual_profiles, actual_assets, **options):
        assert actual_profiles is profiles
        assert actual_assets is assets
        seen.update(options)
        return 'result'

    monkeypatch.setattr(mshbm, 'profiles_from_timeseries', prepare)
    monkeypatch.setattr(mshbm, 'parcellate', parcellate)
    assert mshbm.FNIT_mshbm(series, input_kind='timeseries', censor=censor) == 'result'
    assert seen['censor'] is censor
    assert seen['max_outer'] == 50


def test_mshbm_volume_reuses_the_existing_orchestration_and_all_options(monkeypatch):
    import fnit.mshbm as mshbm
    data, assets, censor = object(), object(), object()
    expected, seen = object(), {}

    def volume(*args, **options):
        seen.update(args=args, **options)
        return expected

    monkeypatch.setattr(mshbm, 'parcellate_volume', volume)
    actual = mshbm.FNIT_mshbm(data, assets, input_kind='volume', censor=censor,
                             left_surface='L.surf.gii', right_surface='R.surf.gii',
                             cortical_mask='cortex.nii.gz', output_dir='outputs',
                             w=123, c=45, device='cpu', frame_chunk=5,
                             max_distance_mm=2.5)
    assert actual is expected
    assert seen == dict(args=(data, 'L.surf.gii', 'R.surf.gii', 'cortex.nii.gz', 'outputs'),
                        assets=assets, censor=censor, w=123, c=45, device='cpu',
                        frame_chunk=5, max_distance_mm=2.5)


@pytest.mark.parametrize('options', [
    {'input_kind': 'invalid'}, {'censor': [1]},
    {'left_surface': 'L.surf.gii'}, {'output_dir': 'outputs'},
    {'device': 'cpu'}, {'frame_chunk': 4},
    {'input_kind': 'volume'},
    dict(input_kind='volume', left_surface='L', right_surface='R',
         cortical_mask='mask', output_dir='out', max_m=1),
])
def test_mshbm_rejects_inapplicable_options_before_loading_assets(options, monkeypatch):
    import fnit.mshbm as mshbm
    monkeypatch.setattr(mshbm, 'load_assets', lambda: pytest.fail('asset loading should not run'))
    with pytest.raises(ValueError):
        mshbm.FNIT_mshbm(None, **options)
