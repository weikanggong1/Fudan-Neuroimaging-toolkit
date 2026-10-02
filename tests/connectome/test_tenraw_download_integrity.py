"""Test source binding and corrupted resume detection, not MRI benchmarking."""
import hashlib
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'validation/connectome/tenraw_20261002/task_01/download_new_raw.py'
spec = importlib.util.spec_from_file_location('tenraw_download', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_git_blob_uses_literal_nul():
    assert module.git_blob_sha(b'') == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'


def test_same_size_corrupted_resume_is_rejected(tmp_path):
    path = tmp_path/'input.nii.gz'
    original = b'original compressed source'
    path.write_bytes(original)
    expected = {'algorithm':'md5','bytes':len(original),'hash':hashlib.md5(original).hexdigest()}
    sha = hashlib.sha256(original).hexdigest()
    assert module.verify_file(path, expected, sha)['sha256'] == sha
    path.write_bytes(b'X'+original[1:])
    with pytest.raises(ValueError, match='annex content'):
        module.verify_file(path, expected, sha)


def test_metadata_is_bound_to_git_blob_as_well_as_sha256(tmp_path):
    path=tmp_path/'AP.json';data=b'{"PhaseEncodingDirection":"j-"}\n';path.write_bytes(data)
    expected={'algorithm':'sha256','bytes':len(data),'hash':hashlib.sha256(data).hexdigest(),
              'git_blob_sha':module.git_blob_sha(data)}
    module.verify_file(path,expected)
    expected['git_blob_sha']='0'*40
    with pytest.raises(ValueError,match='Git text blob'):
        module.verify_file(path,expected)


def test_resume_rejects_unbound_old_cache_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['download_new_raw.py','--output-root',str(tmp_path),'--resume'])
    def network_forbidden(*args, **kwargs):
        raise AssertionError('resume provenance must be checked before network')
    monkeypatch.setattr(module,'urlopen',network_forbidden)
    with pytest.raises(ValueError,match='original download_manifest'):
        module.main()


def description_fixture():
    directory = SCRIPT.parent
    frozen = (directory / 'dataset_description.snapshot.json').read_bytes()
    import json
    provenance = json.loads((directory / 'dataset_description_provenance.json').read_text())
    return frozen, provenance


def test_existing_mutable_description_is_bound_to_its_actual_source(tmp_path):
    frozen, provenance = description_fixture()
    path = tmp_path / 'dataset_description.json'
    path.write_bytes((SCRIPT.parent / 'dataset_description.actual_s3.json').read_bytes())
    result = module.verify_top_level_description(path, frozen, provenance['snapshot_git']['git_blob_sha'], provenance)
    assert result['source_kind'] == 'mutable_s3_download'
    assert result['DatasetDOI'].endswith('v5.0.1')
    assert result['matches_frozen_snapshot_bytes'] is False


def test_existing_frozen_description_is_verified_as_git(tmp_path):
    frozen, provenance = description_fixture()
    path = tmp_path / 'dataset_description.json'; path.write_bytes(frozen)
    result = module.verify_top_level_description(path, frozen, provenance['snapshot_git']['git_blob_sha'], provenance)
    assert result['source_kind'] == 'immutable_git_blob'
    assert result['DatasetDOI'].endswith('v5.0.0')
    assert result['matches_frozen_snapshot_bytes'] is True


def test_same_size_mutated_root_description_is_rejected(tmp_path):
    frozen, provenance = description_fixture()
    data = (SCRIPT.parent / 'dataset_description.actual_s3.json').read_bytes()
    changed = data.replace(b'v5.0.1', b'v5.0.2'); assert len(changed) == len(data)
    path = tmp_path / 'dataset_description.json'; path.write_bytes(changed)
    with pytest.raises(ValueError, match='size/SHA/source'):
        module.verify_top_level_description(path, frozen, provenance['snapshot_git']['git_blob_sha'], provenance)


def test_mutable_description_without_snapshot_bound_provenance_is_rejected(tmp_path):
    frozen, provenance = description_fixture()
    path = tmp_path / 'dataset_description.json'
    path.write_bytes((SCRIPT.parent / 'dataset_description.actual_s3.json').read_bytes())
    with pytest.raises(ValueError, match='not bound to frozen'):
        module.verify_top_level_description(path, frozen, provenance['snapshot_git']['git_blob_sha'], {})


def test_verify_existing_cli_rejects_changed_root_before_acquisition_checks(tmp_path, monkeypatch):
    import base64
    import io
    import json
    frozen, provenance = description_fixture()
    root = tmp_path / 'raw'; root.mkdir()
    actual = (SCRIPT.parent / 'dataset_description.actual_s3.json').read_bytes()
    (root / 'dataset_description.json').write_bytes(actual.replace(b'v5.0.1', b'v5.0.2'))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'snapshot': module.SNAPSHOT, 'license': 'CC0', 'cases': []}))
    entry = {'path': 'dataset_description.json', 'sha': provenance['snapshot_git']['git_blob_sha'], 'size': len(frozen)}
    def frozen_api_only(request, **options):
        url = request.full_url
        if '/git/trees/' in url:
            payload = {'sha': 'tree', 'tree': [entry]}
        elif '/git/blobs/' in url:
            payload = {'content': base64.b64encode(frozen).decode()}
        else:
            raise AssertionError('invalid top-level metadata must fail before acquisition or README fetching')
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(module, 'urlopen', frozen_api_only)
    monkeypatch.setattr('sys.argv', ['download_new_raw.py', '--output-root', str(root), '--verify-existing',
                                  '--expected-manifest', str(manifest)])
    with pytest.raises(ValueError, match='size/SHA/source'):
        module.main()
