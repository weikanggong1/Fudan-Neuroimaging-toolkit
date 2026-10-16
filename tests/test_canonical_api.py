"""Canonical imports retain lazy loading and the mature numerical entry points."""
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from fnit._public_api import PRIMARY_APIS, PUBLIC_APIS


@pytest.mark.parametrize('name', PRIMARY_APIS)
def test_canonical_root_and_feature_are_the_same_object(name):
    import fnit
    module, attribute = PUBLIC_APIS[name]
    expected = getattr(importlib.import_module('fnit.' + module), attribute)
    assert getattr(fnit, name) is expected
    assert callable(expected)
    assert inspect.signature(getattr(fnit, name)) == inspect.signature(expected)


def test_canonical_registry_is_lightweight_and_discoverable():
    source = Path(__file__).resolve().parents[1] / 'src'
    code = "import sys; import fnit; assert all(n.startswith('FNIT_') for n in fnit.__all__); assert 'FNIT_flirt' in dir(fnit); assert not {'torch','nibabel','h5py','SimpleITK'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(source)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_all_requested_primary_names_have_installed_console_entries():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root/'pyproject.toml').read_text())
    scripts = project['project']['scripts']
    for name in PRIMARY_APIS:
        module, function = scripts[name].split(':')
        assert callable(getattr(importlib.import_module(module), function))


@pytest.mark.parametrize('name', PRIMARY_APIS)
def test_every_canonical_cli_help_is_available(name, capsys):
    from fnit.api_cli import run_cli
    try:
        status = run_cli(name, ['--help'])
    except SystemExit as error:
        assert error.code == 0
    else:
        assert status in (None, 0)
    assert 'usage:' in capsys.readouterr().out.lower()


@pytest.mark.parametrize('name,args,expected', [
    ('FNIT_newmsm',['--method','msmall','--inputs-json','inputs.json'], ['msmall','--inputs-json','inputs.json']),
    ('FNIT_newmsm',['msmsulc','--inputs-json','inputs.json'], ['msmsulc','--inputs-json','inputs.json']),
    ('FNIT_newmsm',['--inputs-json','inputs.json'], ['msmsulc','--inputs-json','inputs.json']),
    ('FNIT_noddi',['--method','classic','--data','data.nii.gz'], ['--fit-method','classic','--data','data.nii.gz']),
    ('FNIT_noddi',['--method=classic'], ['--fit-method=classic']),
])
def test_method_arguments_reach_the_existing_parser(name,args,expected,monkeypatch):
    from fnit import api_cli
    calls = []
    monkeypatch.setattr(api_cli,'import_module',lambda name: SimpleNamespace(main=lambda argv: calls.append(argv)))
    api_cli.run_cli(name,args)
    assert calls == [expected]


@pytest.mark.parametrize('name,args',[
    ('FNIT_newmsm',['--method','invalid']),
    ('FNIT_newmsm',['msmall','--method','msmsulc']),
    ('FNIT_newmsm',['--method']),
    ('FNIT_noddi',['--method','classic','--fit-method','amico']),
])
def test_invalid_or_conflicting_methods_fail_before_parser_or_data(name,args,monkeypatch):
    from fnit import api_cli
    monkeypatch.setattr(api_cli,'import_module',lambda name: pytest.fail('must validate before imports'))
    with pytest.raises(ValueError): api_cli.run_cli(name,args)


def test_canonical_command_through_main_dispatch(monkeypatch):
    from fnit import cli,api_cli
    calls=[]
    monkeypatch.setattr(api_cli,'run_cli',lambda name,args:calls.append((name,args)))
    cli.main(['FNIT_flirt','--help'])
    assert calls == [('FNIT_flirt',['--help'])]


def test_removed_preparation_helpers_cannot_be_imported():
    import fnit.robust_register as module
    assert not hasattr(module,'prepare_subregion_alignment_target')
    assert not hasattr(module,'reflect_atlas_header')
    with pytest.raises(ModuleNotFoundError):importlib.import_module('fnit.robust_register.preparation')


def test_console_does_not_treat_result_object_as_error(monkeypatch):
    from fnit import api_cli
    monkeypatch.setattr(api_cli,'import_module',lambda name:SimpleNamespace(main=lambda args:{'scientific_result':True}))
    assert api_cli.run_cli('FNIT_topup',[]) == 0


@pytest.mark.parametrize('arguments', [
    ['--device', 'cpu', 'msmall', '--inputs-json', 'inputs.json', '--output-dir', 'outputs'],
    ['--config', 'schedule.cnf', '--no-parallel', 'msmsulc', '--cpu-threads', '2'],
    ['--device=cpu', '--config=schedule.cnf', 'msmall'],
    ['--dev', 'cpu', 'msmsulc', '--out', 'outputs'],
    ['--', 'msmall'],
])
def test_msm_legacy_position_is_preserved_after_options(arguments, monkeypatch):
    from fnit import api_cli
    calls = []
    monkeypatch.setattr(api_cli, 'import_module', lambda name:
                        SimpleNamespace(main=lambda argv: calls.append(argv)))
    api_cli.run_cli('FNIT_newmsm', arguments)
    assert calls == [arguments]


@pytest.mark.parametrize('option', [
    '--inputs-json', '--output-dir', '--device', '--config',
    '--execution', '--qc-policy', '--cpu-threads',
])
@pytest.mark.parametrize('value', ['msmsulc', 'msmall'])
def test_msm_option_values_do_not_count_as_a_method(option, value, monkeypatch):
    from fnit import api_cli
    calls = []
    monkeypatch.setattr(api_cli, 'import_module', lambda name:
                        SimpleNamespace(main=lambda argv: calls.append(argv)))
    arguments = [option, value]
    api_cli.run_cli('FNIT_newmsm', arguments)
    assert calls == [['msmsulc', *arguments]]
    calls.clear()
    api_cli.run_cli('FNIT_newmsm', ['--method', 'msmall', *arguments])
    assert calls == [['msmall', *arguments]]


def test_msm_position_after_options_conflicts_with_explicit_method(monkeypatch):
    from fnit import api_cli
    monkeypatch.setattr(api_cli, 'import_module', lambda name: pytest.fail('no execution'))
    with pytest.raises(ValueError, match='exactly once'):
        api_cli.run_cli('FNIT_newmsm', ['--device', 'cpu', 'msmall', '--method', 'msmsulc'])


def test_msm_position_after_options_reaches_the_real_parser(tmp_path, monkeypatch):
    import fnit.msm as msm
    import fnit.msm.cli as msm_cli
    from fnit.api_cli import run_cli
    inputs = {'L': object(), 'R': object()}
    observed = []
    monkeypatch.setattr(msm_cli, 'load_inputs', lambda *_: inputs)

    def register(actual_inputs, output_dir, **options):
        assert actual_inputs is inputs
        observed.append((output_dir, options))
        return {'L': tmp_path / 'L.surf.gii', 'R': tmp_path / 'R.surf.gii'}

    monkeypatch.setattr(msm, 'run_msmall', register)
    arguments = ['--device', 'cpu', 'msmall', '--inputs-json', 'inputs.json',
                 '--output-dir', str(tmp_path), '--config', 'msmsulc',
                 '--cpu-threads', '3', '--no-parallel']
    assert run_cli('FNIT_newmsm', arguments) == 0
    assert observed == [(str(tmp_path), dict(device='cpu', config='msmsulc',
                                           execution='optimized', parallel=False,
                                           cpu_threads=3))]
