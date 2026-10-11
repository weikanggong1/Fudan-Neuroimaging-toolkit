"""联合候选预定协议的输入合同，不生成或代替MRI结果。"""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import hashlib
from types import SimpleNamespace
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('joint_runner',ROOT/'tools/benchmark_recon_pial_joint_projection.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class JointProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.actual=json.loads((ROOT/'validation/recon_all/optimizations/20261011_pial_joint_projection/protocol.json').read_text())

    def test_valid_predeclared_protocol(self):
        self.assertIs(runner.validate_protocol(self.actual),self.actual)
        self.assertEqual(self.actual['expected_group_movable_vertices']*3,45)
        self.assertEqual(self.actual['terminal_sha256'],'e6440bc08203f01c3fdbb9845a8f5b86b28402c4913edcb45f2b3bc2cae0906f')

    def test_changed_scope_threshold_types_and_source_maps_rejected(self):
        def field(name,value):return lambda d:d.__setitem__(name,value)
        cases=[('extra key',field('unexpected','private path')),('missing key',lambda d:d.pop('threads')),
            ('bool schema',field('schema_version',True)),('bool budget',field('threads',True)),('wrong budget',field('threads',8)),
            ('bool iterations',lambda d:d['parameters'].__setitem__('optimizer_iterations',True)),
            ('bool displacement',lambda d:d['parameters'].__setitem__('max_displacement_mm',True)),
            ('loosened area',lambda d:d['parameters'].__setitem__('minimum_area_ratio',.1)),
            ('more attempts',lambda d:d['parameters'].__setitem__('optimizer_iterations',200)),
            ('unknown params',lambda d:d['parameters'].__setitem__('max_sweeps',8)),
            ('partial input',lambda d:d['input_sha256'].pop('mri/wm.mgz')),
            ('partial source',lambda d:d['source_sha256'].pop('fnit.recon_all.place_surface_residual_joint')),
            ('path source',lambda d:d['source_sha256'].__setitem__('fnit.recon_all.place_surface_residual_joint','/private/module.py')),
            ('bad digest',field('terminal_sha256','not a sha')),('wrong group',field('expected_group_fixed_vertices',19)),
            ('bool expected pairs',field('expected_float64_pairs',False)),('cuda enabled',field('cuda_visible_devices','0')),
            ('wrong affinity',field('cpu_affinity',[0,1,2,3])),('production default',field('production_default_change',True)),
            ('equivalence',field('overall_metric_equivalence','passed')),('scope',field('scope','raw T1 whole complete')),
            ('equivalence thresholds',field('candidate_constraints_are_not_overall_equivalence_thresholds',False)),
            ('partial exact guard',field('exact_self_guard','only_source_positive_rows'))]
        cases += [('missing secondary closure',lambda d:d['declared_source_files'].pop('fnit.recon_all.topology_volume_python')),
            ('path traversal',lambda d:d['declared_source_files']['fnit.recon_all.topology_volume_python'].__setitem__('relative_path','../outside.py')),
            ('unknown closure',lambda d:d['declared_source_files'].__setitem__('fnit.unknown',{})),
            ('unknown HiGHS threads',field('highs_thread_configuration','exactly 4')),
            ('wrong BLAS env',lambda d:d['thread_environment'].__setitem__('OPENBLAS_NUM_THREADS','8')),
            ('old diagnostic contract',field('diagnostic_contract_revision',1))]
        for name,change in cases:
            with self.subTest(name=name):
                settings=copy.deepcopy(self.actual);change(settings)
                with self.assertRaises(ValueError):runner.validate_protocol(settings)

    def test_actual_source_prefix_hash_and_complete_closure_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);installed=base/'installed';overlay=base/'overlay';declared=copy.deepcopy(self.actual['declared_source_files']);loaded={}
            for name,item in declared.items():
                p=(overlay if item['location']=='overlay' else installed)/item['relative_path'];p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('# synthetic source binding contract\n');item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();loaded[name]=SimpleNamespace(__file__=p)
            args=dict(declared=declared,loaded=loaded,installed_directory=installed,overlay_directory=overlay)
            sha,origins=runner.checked_source_bindings(**args)
            self.assertEqual(set(sha),runner.STAGE_MODULES);self.assertEqual(set(origins),runner.STAGE_MODULES)
            for change in ('missing module','unexpected module','wrong secondary hash','outside same bytes','symlink escape'):
                altered=copy.deepcopy(declared);modules=loaded.copy();key='fnit.recon_all.compare_subject'
                if change=='missing module':modules.pop(key)
                if change=='unexpected module':modules['fnit.unknown']=modules[key]
                if change=='wrong secondary hash':altered[key]['sha256']='b'*64
                if change in ('outside same bytes','symlink escape'):
                    outside=base/'outside.py';outside.write_bytes(Path(modules[key].__file__).read_bytes())
                    if change=='outside same bytes':modules[key]=SimpleNamespace(__file__=outside)
                    else:
                        original=Path(modules[key].__file__);original.unlink();original.symlink_to(outside)
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):runner.checked_source_bindings(**(args|{'declared':altered,'loaded':modules}))

    def test_finite_degenerate_geometry_can_be_written_and_marked_unassessed(self):
        from fnit.recon_all.mris_remove_intersection_python import _write_preserving_surface
        from nibabel import freesurfer
        entry=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32);trial=entry.copy();trial[2]=trial[1]
        faces=np.array([[0,1,2]],np.int32);fixed=np.array([True,True,False])
        geometry=runner.diagnostic_geometry(vertices=trial,entry=entry,faces=faces,fixed_vertices=fixed)
        self.assertEqual(geometry['zero_area_faces'],1);self.assertEqual(geometry['minimum_area_ratio'],0.)
        distance=SimpleNamespace(_point_to_mesh=mock.Mock(side_effect=AssertionError('unmeasured rejected geometry')))
        report=runner.candidate_distances(vertices=trial,entry=entry,faces=faces,distance_module=distance,basic_rejected=True)
        self.assertEqual(report,{'xyz_to_entry':None,'entry_to_xyz':None,'calculation':'not_assessed_basic_geometry_rejected'})
        with tempfile.TemporaryDirectory() as directory:
            white=Path(directory)/'entry.white';pial=Path(directory)/'trial.pial';freesurfer.write_geometry(str(white),entry,faces)
            _write_preserving_surface(white,pial,trial);xyz,ordered=freesurfer.read_geometry(str(pial))
            np.testing.assert_array_equal(xyz.astype(np.float32).view(np.uint32),trial.view(np.uint32));np.testing.assert_array_equal(ordered,faces)


if __name__=='__main__':unittest.main()
