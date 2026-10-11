"""独立边裁剪覆盖精确接触和短交线，不使用原 interval 决策。"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from fnit.recon_all.white_pial_crossing_python import _exact_relation, _rational_triangle, _exact_coplanar_kind

spec = importlib.util.spec_from_file_location('white_pial_exact_audit', Path(__file__).resolve().parents[2]/'tools/audit_white_pial_crossing_exact.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class ExactClipAuditTests(unittest.TestCase):
    def test_relations_and_reversed_winding(self):
        flat = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        fixtures = [
            (np.array([[.5,.5,-1.],[.5,.5,1.],[1.,.5,0.]]), 1),
            (flat.copy(), 3),
            (flat+np.array([0.,0.,1e-15]), 0),
            (np.array([[0.,0.,0.],[0.,-1.,1.],[0.,-1.,-1.]]), 2),
            (flat+np.array([3.,0.,0.]), 0),
        ]
        for second, expected in fixtures:
            for a, b in ((flat,second),(second,flat),(flat[::-1],second),(flat,second[::-1])):
                with self.subTest(expected=expected): self.assertEqual(audit.exact_clip_relation(a,b), expected)

    def test_tiny_positive_is_not_contact(self):
        first=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        second=np.array([[1.,0.,-1e-15],[1.,0.,1e-15],[1.,1e-15,0.]])
        self.assertEqual(audit.exact_clip_relation(first,second),1)

    def test_translated_rotated_shared_edge(self):
        first=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        second=np.array([[0.,0.,0.],[2.,0.,0.],[1.,0.,1.]])
        rotation=np.array([[.8,0.,.6],[0.,1.,0.],[-.6,0.,.8]])
        offset=np.array([73.331,12.717,-61.723])
        a=first@rotation+offset;b=second@rotation+offset
        self.assertEqual(audit.exact_clip_relation(a,b),2)

    def test_independent_interval_against_exact_kernel(self):
        rng=np.random.default_rng(71)
        for slot in range(100):
            a=rng.normal(size=(3,3)).astype(np.float32).astype(np.float64)
            b=rng.normal(size=(3,3)).astype(np.float32).astype(np.float64)
            ra,rb=_rational_triangle(a),_rational_triangle(b)
            expected=_exact_relation(ra,rb)[0]
            if expected==3: expected=3 if _exact_coplanar_kind(ra,rb) else 0
            with self.subTest(slot=slot): self.assertEqual(audit.exact_clip_relation(a,b),expected)

    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError): audit.exact_clip_relation(np.zeros((3,3)),np.eye(3))


if __name__=='__main__': unittest.main()
