"""二进制字段和RNG请求契约，不作为真实影像 benchmark。"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from benchmark_recon_topology_ga_ordering import load_ordering_capture,evaluate_ordering


class OrderingCapture(unittest.TestCase):
    def mutation_buffer(self):
        parts=[b'FNITGAO1',struct.pack('<6Id',0,2,0,2,0,0,0.)]
        for values in [[0,1],[0,1],[0,1],[0,0,0],[],[0,0]]:
            parts.append(np.asarray(values,'<i4').tobytes())
        parts.append(np.asarray([[0,1,.1],[0,1,.9]],'<f8').tobytes())
        return b''.join(parts)

    def read(self,raw):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'capture.bin';p.write_bytes(raw)
            return load_ordering_capture(capture_path=p)

    def test_complete_mutation_fields_and_recorded_calls(self):
        fields=self.read(self.mutation_buffer());out=evaluate_ordering(fields=fields)
        np.testing.assert_array_equal(out['ordering'],fields['ordering_after'])
        np.testing.assert_array_equal(out['position_map'],fields['position_map_after'])
        np.testing.assert_array_equal(out['random_draws'],fields['random_draws'])
        self.assertEqual(fields['ordering_before'].dtype,np.int32)
        fields['ordering_before'][0]=99;self.assertEqual(fields['ordering_after'][0],0)

    def test_complete_crossover_and_parameter_match(self):
        raw=b'FNITGAO1'+struct.pack('<6Id',1,2,0,1,0,0,.2)
        raw+=np.asarray([[0,1],[1,0],[0,1]],'<i4').tobytes()
        raw+=np.asarray([[0,1,.2]],'<f8').tobytes()
        fields=self.read(raw);out=evaluate_ordering(fields=fields)
        np.testing.assert_array_equal(out['ordering'],[0,1])
        with self.assertRaises(ValueError):self.read(raw[:32]+struct.pack('<d',.3)+raw[40:])

    def test_bad_magic_alias_truncation_tail_and_wrong_rng_interval(self):
        raw=self.mutation_buffer()
        for broken in [b'BADMAGIC'+raw[8:],raw[:39],raw[:-1],raw+b'\0',
                raw[:8]+struct.pack('<6Id',0,2,0,2,0,1,0.)+raw[40:],
                raw[:8]+struct.pack('<6Id',0,2**32-1,0,2,0,0,0.)+raw[40:]]:
            with self.subTest(size=len(broken)),self.assertRaises(ValueError):self.read(broken)
        fields=self.read(raw);fields['random_draws'][0,1]=2.
        with self.assertRaises(ValueError):evaluate_ordering(fields=fields)
        fields=self.read(raw);fields['random_draws']=np.concatenate([fields['random_draws'],[[0,1,.5]]])
        with self.assertRaises(ValueError):evaluate_ordering(fields=fields)


if __name__=='__main__':unittest.main()
