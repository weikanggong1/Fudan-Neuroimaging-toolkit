"""Publication audit checks use generic synthetic text, not MRI benchmarks."""
from importlib.util import module_from_spec, spec_from_file_location
import json
import gzip
import io
import marshal
from pathlib import Path
import struct
import tarfile
import subprocess
import zlib


def module():
    path=Path(__file__).parents[2]/'tools'/'audit_public_privacy.py'
    spec=spec_from_file_location('privacy_publication',path)
    result=module_from_spec(spec);spec.loader.exec_module(result)
    return result


def repository(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    return tmp_path


def stage(root):subprocess.run(['git','-C',str(root),'add','.'],check=True)


def chunk(kind,body):
    return struct.pack('>I',len(body))+kind+body+struct.pack('>I',zlib.crc32(kind+body)&0xffffffff)


def test_report_omits_matches_and_private_policy(tmp_path):
    root=repository(tmp_path)
    (root/'report.txt').write_text('sample-host appears twice: sample-host\n')
    stage(root)
    result=module().audit(root,[r'sample-host'])
    assert result['matching_files']==1
    assert result['records'][0]['matches']=={'pattern_0':2}
    assert 'sample-host' not in json.dumps(result)
    assert 'report.txt' not in json.dumps(result)


def test_png_audits_text_metadata_and_ignores_pixel_bytes(tmp_path):
    root=repository(tmp_path)
    signature=b'\x89PNG\r\n\x1a\n'
    (root/'clean.png').write_bytes(signature+chunk(b'IDAT',b'sample-host')+chunk(b'IEND',b''))
    (root/'metadata.png').write_bytes(signature+chunk(b'tEXt',b'Comment\0sample-host')+chunk(b'IEND',b''))
    stage(root)
    result=module().audit(root,[r'sample-host'])
    assert result['files_scanned']==2
    assert result['matching_files']==1


def test_include_untracked_inspects_new_publications(tmp_path):
    root=repository(tmp_path)
    (root/'staged.txt').write_text('public')
    stage(root)
    (root/'new.txt').write_text('sample-host')
    assert module().audit(root,[r'sample-host'])['matching_files']==0
    assert module().audit(root,[r'sample-host'],include_untracked=True)['matching_files']==1


def test_compressed_log_and_header_are_both_audited(tmp_path):
    root=repository(tmp_path)
    stream=io.BytesIO()
    with gzip.GzipFile(filename='sample-host',fileobj=stream,mode='wb',mtime=0) as handle:
        handle.write(b'sample-host in a log')
    (root/'record.log.gz').write_bytes(stream.getvalue())
    stage(root)
    result=module().audit(root,[r'sample-host'])
    assert result['records'][0]['matches']=={'pattern_0':2}
    assert 'sample-host' not in json.dumps(result)


def test_profile_paths_are_audited_as_strings(tmp_path):
    root=repository(tmp_path)
    (root/'run.pstats').write_bytes(marshal.dumps({('sample-host/file.py',1,'compute'):(1,1,.1,.1,{})}))
    stage(root)
    result=module().audit(root,[r'sample-host'])
    assert result['matching_files']==1
    assert result['records'][0]['matches']=={'pattern_0':1}


def test_tar_members_and_text_are_audited(tmp_path):
    root=repository(tmp_path)
    archive=root/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as handle:
        payload=b'sample-host in archived JSON'
        item=tarfile.TarInfo('sample-host/report.txt');item.size=len(payload)
        handle.addfile(item,io.BytesIO(payload))
    stage(root)
    result=module().audit(root,[r'sample-host'])
    assert result['matching_files']==1
    assert result['records'][0]['matches']=={'pattern_0':2}
    assert 'sample-host' not in json.dumps(result)
