"""距离诊断输入完整性契约；模拟数据只检验解析，不代替真实 benchmark。"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.benchmark_recon_topology_distance_fields import load_distance_capture


class DistanceCaptureContracts(unittest.TestCase):
    def make(self, *, corrupt_corner=False, missing_query=False, algorithm_geometry=False):
        faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], '<i4')
        order = [1, 3, 0, 2]  # 原 incident 顺序有意不按 face ID 排序。
        entries = [(f, int(np.flatnonzero(faces[f] == v)[0])) for v in range(4)
            for f in order if v in faces[f]]
        corner = np.asarray([c for _, c in entries], '<i4')
        if corrupt_corner:
            corner[0] = (corner[0] + 1) % 3
        query = np.arange(3 if missing_query else 4, dtype='<i4')
        bounds = np.zeros((4, 7), '<i4'); bounds[:, 0] = [3, 1, 0, 2]
        parts = [b'FNITDST1', np.asarray([4, 4, 12, 4, len(query), 2, 2, 2, 3, 4], '<i4'),
            np.asarray([0, 0, 0, 2, 2, 2, 1, 1, 1], '<f4'), np.asarray([1, 1, 1, 4, 8], '<i4'),
            np.arange(24, dtype='<f4').reshape(4, 6), np.zeros(4, '<i4'), faces,
            np.ones((4, 3), '<f4'), np.zeros(4, '<i4'), np.arange(0, 13, 3, dtype='<i4'),
            np.asarray([f for f, _ in entries], '<i4'), corner, bounds,
            np.zeros((4, 9), '<f4'), np.ones((4, 3), '<i4'), query,
            np.ones((len(query), 3), '<f4'), np.full((2, 2, 2), 1000, '<f4'),
            np.arange(8, dtype='<f4').reshape(2, 2, 2)]
        if algorithm_geometry:
            parts[0] = b'FNITDST2'
            parts[3:3] = [np.asarray([2, 2, 2, 0], '<i4'),
                np.asarray([-25, -84, 5, 2, 2, 2, 1, 1, 1], '<f4')]
        return b''.join(p if isinstance(p, bytes) else p.tobytes() for p in parts)

    def load(self, raw):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.bin'; path.write_bytes(raw)
            return load_distance_capture(capture_path=path)

    def test_original_incident_and_buffer_order_preserved(self):
        f = self.load(self.make())
        np.testing.assert_array_equal(f['incident_faces'][:3], [1, 3, 0])
        np.testing.assert_array_equal(f['buffer_bounds'][:, 0], [3, 1, 0, 2])
        self.assertEqual(f['distance_after'][1, 1, 1], 7)
        self.assertEqual(f['settings'][4], 8)

    def test_truncated_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            self.load(self.make()[:-4])

    def test_trailing_data_rejected(self):
        with self.assertRaisesRegex(ValueError, 'trailing'):
            self.load(self.make() + b'padding')

    def test_wrong_face_corner_owner_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.load(self.make(corrupt_corner=True))

    def test_missing_query_vertex_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cover'):
            self.load(self.make(missing_query=True))

    def test_algorithm_grid_is_separate_from_storage_header(self):
        f = self.load(self.make(algorithm_geometry=True))
        np.testing.assert_array_equal(f['volume_geometry'][:6], [0, 0, 0, 2, 2, 2])
        np.testing.assert_array_equal(f['algorithm_volume_geometry'][:6], [-25, -84, 5, 2, 2, 2])
        self.assertNotIn('algorithm_volume_geometry', self.load(self.make()))


if __name__ == '__main__':
    unittest.main()
