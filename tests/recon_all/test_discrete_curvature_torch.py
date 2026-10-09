"""离散曲率的拓扑、空间和符号契约；真实精度另由冻结网格回归验证。"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from fnit.recon_all.discrete_curvature_torch import DiscreteCurvatureTopology


class DiscreteCurvatureTest(unittest.TestCase):
    def setUp(self):
        self.vertices = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                                      [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32)
        self.faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                               [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], dtype=np.int64)

    def test_closed_ordered_ring_covers_each_incident_face_once(self):
        context = DiscreteCurvatureTopology(faces=self.faces, nvertices=6, device="cpu")
        for vertex in range(6):
            actual = context.face_ids[vertex, :context.degree[vertex]].tolist()
            expected = np.flatnonzero((self.faces == vertex).any(axis=1)).tolist()
            self.assertEqual(sorted(actual), expected)
            self.assertEqual(actual[0], expected[0])

    def test_gaussian_curvature_and_orientation(self):
        context = DiscreteCurvatureTopology(faces=self.faces, nvertices=6, device="cpu")
        result = context.evaluate(vertices=self.vertices)
        # 正八面体每点四张等边面；离散K的积分为4*pi，查表角带已声明近似。
        vertex_area = torch.full((6,), 4 * np.sqrt(3) / 6, dtype=torch.float32)
        self.assertAlmostEqual(float((result["K"] * vertex_area).sum()), 4 * np.pi, places=3)
        reversed_context = DiscreteCurvatureTopology(faces=self.faces[:, ::-1].copy(), nvertices=6, device="cpu")
        reversed_result = reversed_context.evaluate(vertices=self.vertices)
        torch.testing.assert_close(result["K"], reversed_result["K"], rtol=0, atol=2e-6)
        torch.testing.assert_close(result["H"], -reversed_result["H"], rtol=0, atol=2e-6)

    def test_closed_mesh_with_different_vertex_degrees(self):
        vertices = torch.cat((self.vertices, self.vertices[self.faces[0]].mean(0, keepdim=True)))
        split = np.array([[0, 2, 6], [2, 4, 6], [4, 0, 6]], dtype=np.int64)
        faces = np.concatenate((split, self.faces[1:]))
        context = DiscreteCurvatureTopology(faces=faces, nvertices=7, device="cpu")
        self.assertGreater(int(context.degree.max()), int(context.degree.min()))
        result = context.evaluate(vertices=vertices)
        self.assertTrue(bool(torch.isfinite(result["K1"]).all()))

    def test_scale_uses_mm_inverse_and_mm_inverse_squared(self):
        context = DiscreteCurvatureTopology(faces=self.faces, nvertices=6, device="cpu")
        first = context.evaluate(vertices=self.vertices)
        # 符号判断沿1mm单位法向偏移；对该凸闭合网格不会跨越符号边界。
        second = context.evaluate(vertices=self.vertices * 2)
        torch.testing.assert_close(first["K"], second["K"] * 4, rtol=0, atol=1e-6)
        torch.testing.assert_close(first["H"], second["H"] * 2, rtol=0, atol=1e-6)

    def test_invalid_topology_and_changed_coordinate_contract(self):
        with self.assertRaises(ValueError):
            DiscreteCurvatureTopology(faces=self.faces[:1], nvertices=6, device="cpu")
        context = DiscreteCurvatureTopology(faces=self.faces, nvertices=6, device="cpu")
        with self.assertRaises(ValueError):
            context.evaluate(vertices=self.vertices.double())
        with self.assertRaises(ValueError):
            context.evaluate(vertices=self.vertices[:5])
        invalid = self.vertices.clone()
        invalid[0, 0] = torch.nan
        with self.assertRaises(ValueError):
            context.evaluate(vertices=invalid)


if __name__ == "__main__":
    unittest.main()
