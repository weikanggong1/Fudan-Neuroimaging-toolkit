"""实际 before/after 捕获格式的混合字段解析与完整性合同；非 benchmark。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('base_capture',ROOT/'tools/benchmark_recon_topology_genetic_base.py')
capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)


def fixture():
    # One triangle, explicitly interleaved face int fields and float area.
    h=np.array([3,1,6,3,0,3,1,13,2,1,1,0,0,3,0,0,1,1],'<i4')
    fields=np.arange(126,dtype='<f4').reshape(3,42)
    parts=[b'FNITBAS1',h.tobytes(),np.array([1,256,256,256,1],'<i4').tobytes(),
           np.arange(18,dtype='<f8').tobytes(),np.array([3],'<i4').tobytes(),b'abc',
           np.array([1,1,0],'<i4').tobytes(),np.arange(16,dtype='<f4').tobytes(),
           np.array([4,4,0],'<i4').tobytes(),np.eye(4,dtype='<f8').tobytes(),
           fields.tobytes(),np.zeros((3,3),'<i4').tobytes(),
           np.array([[1,1,2,2,0,0,1]]*3,'<i4').tobytes(),
           np.array([0,1,2,0],'<i4').tobytes(),np.array([0.75],'<f4').tobytes(),
           np.array([0,2,4,6],'<i4').tobytes(),np.array([1,2,0,2,0,1],'<i4').tobytes(),
           np.array([0,1,2,3],'<i4').tobytes()]
    for i in range(3):parts += [np.array([0],'<i4').tobytes(),np.array([i],'<i4').tobytes()]
    parts += [np.array([0,1,0],'<i4').tobytes(),np.array([0,1,0],'<i4').tobytes(),
              np.array([1],'<i4').tobytes(),np.array([1],'<i4').tobytes(),
              np.full(3,-1,'<i4').tobytes(),np.full(1,-1,'<i4').tobytes()]
    return b''.join(parts),fields


class CaptureContracts(unittest.TestCase):
    def test_complete_mixed_state_and_ordered_arrays(self):
        data,fields=fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'before.bin';p.write_bytes(data);r=capture.load_base_capture(capture_path=p)
        s=r['surface_state']
        self.assertEqual(s['current_vertices'].tobytes(),fields[:,:3].tobytes())
        self.assertEqual(s['k2'].tobytes(),fields[:,29].tobytes())
        self.assertEqual(s['old_undefval'].tobytes(),fields[:,41].tobytes())
        self.assertEqual(float(s['face_area'][0]),0.75)
        np.testing.assert_array_equal(s['neighbor_storage'][1],[0,2])
        np.testing.assert_array_equal(s['incident_corners'][2],[2])
        np.testing.assert_array_equal(r['source_defect_order'],[1])
        self.assertEqual(r['source_metadata']['volume_geometry']['filename_bytes'].tobytes(),b'abc')
        self.assertIsNone(r['source_metadata']['volume_geometry']['register_mat'])

    def test_truncated_trailing_and_inconsistent_dimensions_rejected(self):
        data,_=fixture()
        bad=bytearray(data);bad[8:12]=np.array([0],'<i4').tobytes()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'before.bin'
            for version in (data[:-1],data+b'x',bytes(bad),b'WRONG000'+data[8:]):
                p.write_bytes(version)
                with self.assertRaises(ValueError):capture.load_base_capture(capture_path=p)


if __name__=='__main__':unittest.main()
