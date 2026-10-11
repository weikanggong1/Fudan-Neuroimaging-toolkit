"""自相交精确名单审查：固定/共有顶点不豁免真实内部交叉。"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[2]/"tools"
def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS/(name+".py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

audit = load("audit_pial_self_pairs_exact")
exact = load("audit_white_pial_crossing_exact")


class SelfPairExactAuditTests(unittest.TestCase):
    def run_pair(self, first, second, *, shared=(), fixed=False):
        vertices = list(first); ids = []
        for i, corner in enumerate(second):
            if i in dict(shared): ids.append(dict(shared)[i])
            else: ids.append(len(vertices)); vertices.append(corner)
        xyz = np.asarray(vertices, np.float64); faces = np.array([[0, 1, 2], ids], np.int32)
        before = xyz.copy(); original_faces = faces.copy()
        result = audit.audit_pairs(vertices=xyz, faces=faces, fixed_vertices=np.full(len(xyz), fixed),
                                   source_pairs={(0, 1), (1, 0)}, float64_pairs={(0, 1)},
                                   exact_relation=exact.exact_clip_relation)
        np.testing.assert_array_equal(xyz, before); np.testing.assert_array_equal(faces, original_faces)
        return result["rows"][0], result

    def test_classifies_all_positive_lists_without_assuming_proper(self):
        flat = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        fixtures = [
            (np.array([[.5, .5, -1.], [.5, .5, 1.], [1., .5, 0.]]), "proper"),
            (flat+np.array([0., 0., 1e-15]), "none"),
            (np.array([[0., 0., 0.], [0., -1., 1.], [0., -1., -1.]]), "edge_point_contact"),
            (flat.copy(), "coplanar_area_overlap"),
            (np.array([[0., 0., 0.], [2., 0., 0.], [1., -1., 0.]]), "coplanar_edge_point_contact"),
        ]
        for second, kind in fixtures:
            with self.subTest(kind=kind):
                row, result = self.run_pair(flat, second)
                self.assertEqual(row["exact_kind"], kind)
                self.assertEqual(result["float64_nonproper_pairs"], int(kind != "proper"))
                self.assertEqual(row["shared_vertex_ids"], 0)

    def test_shared_vertex_can_have_true_internal_crossing(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        second = np.array([[0., 0., 0.], [1., 1., -1.], [1., 1., 1.]])
        row, _ = self.run_pair(first, second, shared=((0, 0),), fixed=True)
        self.assertEqual(row["shared_vertex_ids"], 1)
        self.assertEqual(row["exact_kind"], "proper")
        self.assertTrue(row["all_corners_fixed"])

    def test_shared_edge_and_reversed_coplanar_winding(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        second = np.array([[0., 0., 0.], [2., 0., 0.], [1., 0., 1.]])
        row, _ = self.run_pair(first, second, shared=((0, 0), (1, 1)))
        self.assertEqual(row["exact_kind"], "edge_point_contact")
        self.assertEqual(row["shared_vertex_ids"], 2)
        for a, b in ((first, first[::-1]), (first[::-1], first)):
            with self.subTest(winding=True):
                row, _ = self.run_pair(a, b)
                self.assertEqual(row["exact_kind"], "coplanar_area_overlap")

    def test_tiny_positive_preserved(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        second = np.array([[1., 0., -1e-15], [1., 0., 1e-15], [1., 1e-15, 0.]])
        row, _ = self.run_pair(first, second)
        self.assertEqual(row["exact_kind"], "proper")

    def test_separate_components_and_empty_list(self):
        first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        second = np.array([[.5, .5, -1.], [.5, .5, 1.], [1., .5, 0.]])
        xyz = np.concatenate([first, second, first+10., second+10.]); faces = np.arange(12).reshape(4, 3)
        result = audit.audit_pairs(vertices=xyz, faces=faces, fixed_vertices=np.zeros(12, bool),
                                   source_pairs=set(), float64_pairs={(0, 1), (2, 3)}, exact_relation=exact.exact_clip_relation)
        self.assertEqual(len(result["components"]), 2)
        self.assertEqual(result["exact_classes"]["proper"], 2)
        empty = audit.audit_pairs(vertices=xyz, faces=faces, fixed_vertices=np.zeros(12, bool),
                                  source_pairs=set(), float64_pairs=set(), exact_relation=exact.exact_clip_relation)
        self.assertEqual(empty["union_pairs"], 0); self.assertEqual(empty["components"], [])

    def test_invalid_pair_and_masks_rejected(self):
        xyz = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],[.5,.5,-1.],[.5,.5,1.],[1.,.5,0.]])
        faces = np.arange(6).reshape(2,3)
        for kwargs in ({"float64_pairs": {(1, 0)}}, {"source_pairs": {(0, 0)}}, {"fixed_vertices": np.zeros(6, np.int32)}):
            args = dict(vertices=xyz, faces=faces, fixed_vertices=np.zeros(6, bool), source_pairs=set(), float64_pairs={(0,1)}, exact_relation=exact.exact_clip_relation)
            args.update(kwargs)
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError): audit.audit_pairs(**args)


if __name__ == "__main__": unittest.main()
