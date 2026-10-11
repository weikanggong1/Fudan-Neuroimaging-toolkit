"""typed导出合同fixture；元数据为schema测试，不是MRI benchmark。"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fnit.recon_all import place_surface_residual_joint as joint
from tools.audit_white_pial_crossing_exact import exact_clip_relation

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'validation/recon_all/optimizations/20261011_pial_joint_projection'
SHARED=ROOT/'validation/recon_all/optimizations/20261011_white_pial_crossing/export_qc.py'
spec=importlib.util.spec_from_file_location('joint_export_contract',DIRECTORY/'export_joint.py')
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)


def fixture():
    """小面对真实resolver诊断+仅schema测试的闭合拓扑元数据。"""
    xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[3,-1,-1],[3,-1,1],[.5,.5,0]],np.float32)
    faces=np.array([[0,1,2],[3,4,5]],np.int32)
    fixed=np.array([True,True,True,True,True,False])
    result,trial,diagnosis=joint.resolve_joint_residual_intersections(vertices=xyz,faces=faces,fixed_vertices=fixed,white_vertices=xyz,exact_relation=exact_clip_relation)
    geometry={}
    for name,value in (('result',result),('trial',trial)):
        movement=np.linalg.norm(value.astype(float)-xyz,axis=1);changed=movement[movement!=0]
        old=joint._face_normals(xyz,faces);new=joint._face_normals(value,faces)
        geometry[name]={'finite':True,'ordered_faces_equal':True,'fixed_different_elements':0,'different_elements':int(np.count_nonzero(value!=xyz)),
            'changed_vertices':len(changed),'maximum_displacement_mm':float(movement.max()),'p99_displacement_mm':float(np.quantile(movement,.99)),
            'changed_vertex_p99_mm':float(np.quantile(changed,.99)) if len(changed) else 0.,'minimum_area_ratio':float((np.linalg.norm(new,axis=1)/np.linalg.norm(old,axis=1)).min()),
            'zero_area_faces':0,'nonpositive_initial_normal_dot_faces':0}
    digest='a'*64
    report={'schema_version':1,'execution_status':'complete_independent_joint_diagnosis','scope':'frozen own seven-input residual stage; not raw T1 or original program comparison',
        'threads':4,'cpu_affinity':[112,113,114,115],'gpu_computation':False,'gpu_memory_bytes':None,'coordinate_space':'surface_RAS','coordinate_unit':'mm',
        'inputs_unchanged':True,'original_stage_program_called':False,'strict_reproduction':'independent_algorithm_not_source_reproduction','overall_metric_equivalence':'not_assessed',
        'production_default_change':False,'nonmanifold_vertex_links':0,'edge_orientation_conflicts':0,'peak_process_cpu_rss_bytes':1,
        'wall_seconds_before_report_write':1.,'trial_retained':True,'input_sha256':{k:digest for k in exporter.INPUTS},
        'source_version_bindings':{k:digest for k in ('fnit.recon_all.place_surface_residual_joint','fnit.recon_all.place_surface_residual_cleanup','fnit.recon_all.place_surface_residual_projection','fnit.recon_all.white_pial_crossing_python')},
        'mesh':{'vertices':6,'faces':8,'fixed_vertices':5,'outside_cortex_elements_different_from_white_at_entry':0},
        'topology_shared_by_entry_trial_and_result':{'vertices':6,'faces':8,'edges':12,'components':1,'boundary_edges':0,'nonmanifold_edges':0,'degenerate_index_faces':0,'duplicate_faces':0,'euler':2,'valid':True},
        'diagnosis':diagnosis,'geometry':geometry,
        'bidirectional_vertex_to_triangle':{name:{'xyz_to_entry':{'mean_mm':1.,'p99_mm':1.,'max_mm':1.},'entry_to_xyz':{'mean_mm':1.,'p99_mm':1.,'max_mm':1.},'calculation':'existing_bidirectional_vertex_to_triangle','seconds':1.} for name in ('result','trial')},
        'output_sha256':{name:digest for name in ('result.private.npy','result.pial','trial.private.npy','trial.pial')},
        'versions':{'python':'3.12.1','numpy':'2.3.1','numba':'0.61.0','scipy':'1.15.1','nibabel':'5.3.1'}}
    for name in ('protocol_sha256','script_sha256','terminal_sha256','exact_report_sha256','preflight_source_sha256','exact_audit_sha256','distance_source_sha256'):report[name]=digest
    timer={'schema_version':1,'scope':'single frozen Python benchmark cold CLI; not end-to-end recon-all','command_sha256':digest,'timer_script_sha256':digest,
        'gpu_memory_bytes':None,'simultaneous_parent_child_peak_cpu_memory_bytes':None,'execution_status':'complete','child_exit_code':0,'cold_cli_wall_seconds':2.,'max_child_cpu_rss_bytes':1}
    return report,timer


class JointExportContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base_report,cls.base_timer=fixture()

    def export_fixture(self,report=None,timer=None,complete_change=None):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);report=copy.deepcopy(self.base_report if report is None else report);timer=copy.deepcopy(self.base_timer if timer is None else timer)
            raw=json.dumps(report,sort_keys=True,allow_nan=True).encode();(base/'report.json').write_bytes(raw)
            (base/'timer.json').write_text(json.dumps(timer,allow_nan=True))
            complete={'execution_status':report['execution_status'],'stage_quality':report['diagnosis']['status'],'report_sha256':hashlib.sha256(raw).hexdigest(),
                'output_sha256':report['output_sha256'],'wall_seconds_after_report_write':1.1}
            if complete_change:complete.update(complete_change)
            (base/'COMPLETE.json').write_text(json.dumps(complete));output=base/'safe.json'
            args=dict(report_path=base/'report.json',timer_path=base/'timer.json',complete_path=base/'COMPLETE.json',schema_source=SHARED,output=output)
            try:
                result=exporter.export(**args)
            except (ValueError,KeyError,TypeError) as error:
                self.assertFalse(output.exists());raise error
            self.assertTrue(output.exists());return result

    def test_declared_fixture_preserves_every_producer_value(self):
        result=self.export_fixture()
        for key,value in self.base_report.items():self.assertEqual(result[key],value)
        self.assertEqual(result['cold_cli'],self.base_timer)

    def test_rejected_trial_retained_including_same_index_cross_surface_pair(self):
        report=copy.deepcopy(self.base_report);r=report['diagnosis']
        r['decision']='rejected_by_complete_quality_gates';r['status']='failed_residual'
        r['source_directed_pairs_after']=r['source_directed_pairs_before'];r['float64_pairs_after']=r['float64_pairs_before']
        r['self_qc_final']=copy.deepcopy(r['self_qc_before']);r['final_coordinates_equal_entry']=True
        r['trial_checks']['no_new_white_proper_pairs']=False
        g=r['white_pial_guard'];g['new_proper_pairs']=[[0,0]];g['trial']['proper_crossing_pairs']+=1
        g['trial_zero_proper']=False;g['final_zero_proper']=g['initial']['proper_crossing_pairs']==0
        g['final_zero_nonmedial_contact']=g['initial_nonmedial_exact_contacts']==0
        e=r['exact_affected_self_guard'];e['final_proper_pairs']=e['initial_proper_pairs'];e['final_coplanar_area_pairs']=e['initial_coplanar_area_pairs']
        report['geometry']['result']={'finite':True,'ordered_faces_equal':True,'fixed_different_elements':0,'different_elements':0,'changed_vertices':0,
            'maximum_displacement_mm':0.,'p99_displacement_mm':0.,'changed_vertex_p99_mm':0.,'minimum_area_ratio':1.,'zero_area_faces':0,'nonpositive_initial_normal_dot_faces':0}
        report['bidirectional_vertex_to_triangle']['result']={'xyz_to_entry':{'mean_mm':0.,'p99_mm':0.,'max_mm':0.},'entry_to_xyz':{'mean_mm':0.,'p99_mm':0.,'max_mm':0.},'calculation':'bitwise_equal_geometry_identity'}
        result=self.export_fixture(report=report)
        self.assertTrue(result['trial_retained']);self.assertEqual(result['diagnosis']['white_pial_guard']['new_proper_pairs'],[[0,0]])
        self.assertEqual(result['geometry']['result']['different_elements'],0)

    def test_no_trial_empty_family_keeps_unmeasured_guards(self):
        report=copy.deepcopy(self.base_report)
        xyz=np.array([[0,0,0],[2,0,0],[0,2,0],[3,-1,-1],[3,-1,1],[.5,.5,0]],np.float32)
        faces=np.array([[0,1,2],[3,4,5]],np.int32)
        _,trial,diagnosis=joint.resolve_joint_residual_intersections(vertices=xyz,faces=faces,fixed_vertices=np.ones(6,bool),white_vertices=xyz,exact_relation=exact_clip_relation)
        self.assertIsNone(trial);report['diagnosis']=diagnosis;report['trial_retained']=False;report['mesh']['fixed_vertices']=6
        report['geometry']={'result':{'finite':True,'ordered_faces_equal':True,'fixed_different_elements':0,'different_elements':0,'changed_vertices':0,
            'maximum_displacement_mm':0.,'p99_displacement_mm':0.,'changed_vertex_p99_mm':0.,'minimum_area_ratio':1.,'zero_area_faces':0,'nonpositive_initial_normal_dot_faces':0}}
        report['bidirectional_vertex_to_triangle']={'result':{'xyz_to_entry':{'mean_mm':0.,'p99_mm':0.,'max_mm':0.},'entry_to_xyz':{'mean_mm':0.,'p99_mm':0.,'max_mm':0.},'calculation':'bitwise_equal_geometry_identity'}}
        report['output_sha256']={k:v for k,v in report['output_sha256'].items() if k.startswith('result')}
        result=self.export_fixture(report=report)
        self.assertIsNone(result['diagnosis']['white_pial_guard']);self.assertIsNone(result['diagnosis']['exact_affected_self_guard'])
        self.assertEqual(result['diagnosis']['optimizer']['lp_calls'],0)

    def test_unknown_private_type_domain_route_and_count_fail_without_output(self):
        mutations=[
            lambda d:d.update(private_path='/private/test'),
            lambda d:d['source_version_bindings'].update({'private/source.py':'a'*64}),
            lambda d:d['source_version_bindings'].pop('fnit.recon_all.white_pial_crossing_python'),
            lambda d:d.update(threads=True),
            lambda d:d.update(gpu_memory_bytes=0),
            lambda d:d['versions'].update(python='private/path'),
            lambda d:d['input_sha256'].pop(next(iter(exporter.INPUTS))),
            lambda d:d['topology_shared_by_entry_trial_and_result'].update(edges=11),
            lambda d:d['diagnosis']['parameters'].update(clearance_mm=.001),
            lambda d:d['diagnosis']['self_qc_final'].update(source_directed_pairs=4),
            lambda d:d['diagnosis']['exact_before'][0].update(pair=[1,0]),
            lambda d:d['diagnosis']['exact_before'].append(copy.deepcopy(d['diagnosis']['exact_before'][0])),
            lambda d:d['diagnosis']['optimizer'].update(lp_calls=2),
            lambda d:d['diagnosis']['optimizer'].update(decision='selected_family_lp_failed'),
            lambda d:d['diagnosis']['family']['fixed_face_selections'][0].update(selected_plane='private/path'),
            lambda d:d['diagnosis']['family'].update(existing_fixed_contact_margin_changed=True),
            lambda d:d['diagnosis']['exact_affected_self_guard']['initial'].update(exact_tested_pairs=0),
            lambda d:d['diagnosis']['exact_affected_self_guard'].update(final_proper_pairs=9),
            lambda d:d['diagnosis']['white_pial_guard'].update(final_zero_proper=True),
            lambda d:d['diagnosis']['trial_checks'].update(no_new_exact_affected_proper_pairs=False),
            lambda d:d['diagnosis'].update(decision='rejected_by_complete_quality_gates'),
            lambda d:d['geometry']['result'].update(fixed_different_elements=1),
            lambda d:d['geometry']['trial'].update(maximum_displacement_mm=float('inf')),
            lambda d:d['output_sha256'].update({'private/path':'a'*64}),
        ]
        for mutate in mutations:
            report=copy.deepcopy(self.base_report);mutate(report)
            with self.subTest(index=mutations.index(mutate)):
                with self.assertRaises(ValueError):self.export_fixture(report=report)

    def test_cold_cli_false_exit_unknown_scope_and_complete_bindings_reject(self):
        for change in ({'child_exit_code':False},{'child_exit_code':1},{'execution_status':'running'},{'gpu_memory_bytes':0},{'private':1},{'cold_cli_wall_seconds':True}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):self.export_fixture(timer=self.base_timer|change)
        for change in ({'report_sha256':'b'*64},{'stage_quality':'passed'},{'output_sha256':{}},{'private':1}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):self.export_fixture(complete_change=change)


if __name__=='__main__':unittest.main()
