"""Capability environment and stale output must not masquerade as native execution."""
import hashlib,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
from fnit.recon_all.mri_em_register_cached_conda import run_cached_em_register


def run_case(write_output):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);mri=root/'mri';mri.mkdir();(mri/'transforms').mkdir()
        for path in (mri/'nu.mgz',mri/'brainmask.mgz',root/'atlas.gca'):path.write_text('input existence test')
        output=mri/'transforms/talairach.lta';output.write_text('old LTA')
        binary=root/'fake_native'
        text='#!'+sys.executable+'''\nimport json,os,sys
from pathlib import Path
if os.environ.get('FNIT_GCA_QUERY_CAPABILITIES')=='1':
 print(json.dumps({'fnit_gca_cached_search':True,'full_native_em':True,'version':2,'reduction':'upstream_ROMP_partials'}));sys.exit(0)
Path('environment_seen.json').write_text(json.dumps({'query':os.environ.get('FNIT_GCA_QUERY_CAPABILITIES'),'scorer':os.environ.get('FNIT_GCA_SCORER'),'omp':os.environ.get('OMP_NUM_THREADS')}))
'''
        if write_output:text+="Path(sys.argv[-1]).write_text('1 4 4\\n1 0 0 0\\n0 1 0 0\\n0 0 1 0\\n0 0 0 1\\n')\n"
        binary.write_text(text);binary.chmod(0o755)
        digest=hashlib.sha256(binary.read_bytes()).hexdigest()
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'8','FNIT_GCA_QUERY_CAPABILITIES':'1'}):
            if write_output:
                assert run_cached_em_register(binary,mri,root/'atlas.gca',root,binary_sha256=digest)==output
                assert output.read_text().startswith('1 4 4')
            else:
                try:run_cached_em_register(binary,mri,root/'atlas.gca',root,binary_sha256=digest)
                except FileNotFoundError:pass
                else:raise AssertionError('zero exit with no new output was accepted')
                assert output.read_text()=='old LTA'
            assert os.environ['OMP_NUM_THREADS']=='8'
        seen=json.loads((mri/'environment_seen.json').read_text())
        assert seen=={'query':None,'scorer':'cpu_cached','omp':'4'}
        assert not list((mri/'transforms').glob('.fnit-cached-em-*'))

def test_polluted_environment_generates_fresh_output():
    run_case(True)

def test_zero_exit_without_new_output_preserves_old_lta():
    run_case(False)

if __name__=='__main__':
    run_case(True);run_case(False)
    print('2 polluted-environment/stale-LTA launcher tests passed')
