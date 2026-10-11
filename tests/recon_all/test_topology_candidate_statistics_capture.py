"""只检验诊断二进制格式；模拟缓冲不作为真实精度 benchmark。"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from benchmark_recon_topology_candidate_statistics import load_statistics_capture


class StatisticsCapture(unittest.TestCase):
    def buffer(self):
        parts=[b'FNITRST1',struct.pack('<4If',3,2,1,1,-97.75)]
        for dtype,values in [('<i4',[1,2,3]),('<i4',[0,2,3]),('<f4',[1,2,3]),
                ('<i4',[0,0,-1,1]),('<i4',[1]),('<i4',[0,1,3]),
                ('<i4',[0,5]),('<i4',[1,5]),('<f4',[0,4]),('<f4',[1,4])]:
            parts.append(np.array(values,dtype=dtype).tobytes())
        return b''.join(parts)

    def read(self,raw):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'statistics.bin';path.write_bytes(raw)
            return load_statistics_capture(capture_path=path)

    def test_complete_typed_fields_and_copy(self):
        a=self.read(self.buffer())
        self.assertEqual(a['candidate_fitness'],np.float32(-97.75))
        np.testing.assert_array_equal(a['mapped_status'],[[0,0],[-1,1]])
        self.assertEqual(a['counts_after'].dtype,np.int32)
        self.assertEqual(a['fitness_before'].dtype,np.float32)
        self.assertEqual(a['edges'].shape,(1,3))
        a['marks_before'][0]=99;self.assertEqual(a['marks_after'][0],0)

    def test_bad_magic_truncated_extra_and_overlarge_count(self):
        raw=self.buffer()
        for broken in [b'WRONGMAG'+raw[8:],raw[:27],raw[:-1],raw+b'\0',
                raw[:8]+struct.pack('<4If',2**32-1,2,1,1,-97.75)+raw[28:]]:
            with self.subTest(size=len(broken)),self.assertRaises(ValueError):self.read(broken)


if __name__=='__main__':unittest.main()
