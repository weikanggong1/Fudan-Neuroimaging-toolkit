"""联合候选预定协议的输入合同，不生成或代替MRI结果。"""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

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
        for name,change in cases:
            with self.subTest(name=name):
                settings=copy.deepcopy(self.actual);change(settings)
                with self.assertRaises(ValueError):runner.validate_protocol(settings)


if __name__=='__main__':unittest.main()
