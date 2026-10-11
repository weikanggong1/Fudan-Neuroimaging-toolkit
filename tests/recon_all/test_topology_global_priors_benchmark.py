"""Prepared-only producer CLI boundary: separate after state, full output gate."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest


def tool():
    path=Path(__file__).parents[2]/'tools/benchmark_recon_topology_global_sff.py'
    spec=importlib.util.spec_from_file_location('global_priors_benchmark',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def inputs():
    return dict(smoothed_original_vertices=np.array([[0,0,0],[3,4,5],[-1,0,2],[-2,1,4]],np.float32),
        metric_vertex_normals=np.tile([1,2,3],(4,1)).astype(np.float32),
        neighbor_offsets=np.array([0,3,3,3,3],np.int32),neighbors=np.array([1,2,3],np.int32),
        vertex_rip=np.zeros(4,np.int32))


def test_dot_cli_separate_after_reference_and_source_binding(tmp_path,monkeypatch):
    module=tool();before=tmp_path/'before.npz';reference=tmp_path/'after.npz';manifest=tmp_path/'source.json'
    np.savez(before,**inputs());manifest.write_text('{}')
    expected=module.global_normal_dot_prior(**inputs())
    np.savez(reference,bins=expected['bins'],counts=expected['counts'])
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['benchmark','--input',str(before),'--reference',str(reference),
        '--source-manifest',str(manifest),'--output',str(output),'--mode','dot','--threads','2'])
    assert module.main()==0
    report=json.loads((output/'report.private.json').read_text())
    assert report['mode']=='dot' and report['strict_output_reproduction']
    assert len(report['executed_module_sha256'])==64
    assert report['whole_ga'] is False and report['equivalence']=='not_assessed'


@pytest.mark.parametrize('bad_before,bad_reference',[(True,False),(False,True)])
def test_after_fields_not_accepted_as_before_and_empty_after_not_pass(tmp_path,monkeypatch,bad_before,bad_reference):
    module=tool();values=inputs()
    if bad_before:values['reference_counts']=np.zeros(100,np.float32)
    before=tmp_path/'before.npz';reference=tmp_path/'after.npz';manifest=tmp_path/'source.json'
    np.savez(before,**values);np.savez(reference) if bad_reference else np.savez(reference,bins=np.zeros(100,np.float32),counts=np.zeros(100,np.float32))
    manifest.write_text('{}')
    monkeypatch.setattr(sys,'argv',['benchmark','--input',str(before),'--reference',str(reference),
        '--source-manifest',str(manifest),'--output',str(tmp_path/'out'),'--mode','dot','--threads','2'])
    with pytest.raises(ValueError):module.main()


def test_value_equal_signed_zero_is_still_bit_diagnostic():
    result=tool()._compare({'field':np.array([0.],np.float32)},
                          {'field':np.array([-0.],np.float32)})['field']
    assert result['differing_values']==0 and result['differing_bytes']==1
    assert result['byte_equal'] is False
