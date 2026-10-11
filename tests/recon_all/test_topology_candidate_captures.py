"""候选检查点格式契约；边界fixture不代替真实MRI benchmark。"""
import tempfile,unittest
from pathlib import Path
import numpy as np
from tools.benchmark_recon_topology_neighbor_fields import load_neighbor_capture
from tools.benchmark_recon_topology_volume_prepare_fields import load_volume_preparation_capture
from tools.benchmark_recon_topology_retess_fields import load_retessellation_capture


def pack(value,dtype):return np.asarray(value,dtype=dtype).tobytes()


class CandidateCaptureContracts(unittest.TestCase):
    def load(self,raw,loader):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'capture.bin';p.write_bytes(raw);return loader(capture_path=p)
    def neighborhood(self,offsets=None):
        return b'FNITNBR1'+pack([2,2,1],'<i4')+pack([0,1,2] if offsets is None else offsets,'<i4')+pack([[1,1,1],[1,1,1]],'<i4')+pack([1,0],'<i4')+pack(np.zeros((2,15)),'<f4')+pack([0],'<i4')
    def prepare(self,phase,edge=1):
        data=b'FNITPDF1'+pack([2,1,1,1,1,2,2,phase],'<i4')+pack([0,0,0,2,2,2,.5,.5,.5],'<f4')+pack([2,0,0,85,62,-3,-4],'<f4')+pack(np.eye(4),'<f8')+pack(np.zeros((2,3)),'<f4')+pack([[0,edge]],'<i4')+pack([0,1,.4,.6,0,1,.2,.8],'<f4')+pack([255],'u1')
        if phase:data+=pack([-3,-4],'<f4')
        return data
    def retess(self,offsets=None,order=None):
        data=b'FNITRTS1'+pack([3,1,6,3,1,2,2,1,0,0,0],'<i4')+pack(np.ones((3,9)),'<f4')+pack(np.zeros((3,3)),'<i4')+pack([[0,1,2]],'<i4')+pack([0,2,4,6] if offsets is None else offsets,'<i4')+pack([1,2,0,2,0,1],'<i4')+pack([[0,1,0,0],[1,2,0,0],[2,0,0,0]],'<i4')+pack([0,1,2] if order is None else order,'<i4')+pack([0,0,0,0],'<i4')+pack([[0,0]],'<i4')+pack([1,2],'<i4')+pack([1,2],'<i4')+pack([0,0,0,0],'<i4')
        return data
    def test_original_vtotal_csr_and_independent_arrays(self):
        a=self.load(self.neighborhood(),load_neighbor_capture);self.assertEqual(a['neighbors'].tolist(),[1,0]);a['neighbors'][0]=0
        b=self.load(self.neighborhood(),load_neighbor_capture);self.assertEqual(b['neighbors'][0],1)
    def test_vtotal_csr_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'vnum|vtotal'):
            self.load(self.neighborhood([0,0,2]),load_neighbor_capture)
    def test_prepare_phases_preserve_zero_hist_bin_size(self):
        a=self.load(self.prepare(0),load_volume_preparation_capture);b=self.load(self.prepare(1),load_volume_preparation_capture)
        self.assertNotIn('white_log_pdf',a);self.assertEqual(a['parameters'][1],0);self.assertEqual(b['white_log_pdf'].shape,(1,1,1));self.assertEqual(b['white_log_pdf'].item(),-3)
    def test_prepare_endpoint_range_rejected(self):
        with self.assertRaisesRegex(ValueError,'endpoint'):
            self.load(self.prepare(0,2),load_volume_preparation_capture)
    def test_prepare_missing_phase_fields_rejected(self):
        with self.assertRaisesRegex(ValueError,'truncated'):
            self.load(self.prepare(1)[:-4],load_volume_preparation_capture)
    def test_complete_retess_before_preserves_base_order(self):
        a=self.load(self.retess(),load_retessellation_capture);self.assertEqual(a['faces'].tolist(),[[0,1,2]]);self.assertEqual(a['selected_vertices'].size,0)
    def test_retess_neighbor_csr_rejected(self):
        with self.assertRaisesRegex(ValueError,'CSR'):
            self.load(self.retess([0,2,1,6]),load_retessellation_capture)
    def test_retess_order_must_cover_full_edge_domain(self):
        with self.assertRaisesRegex(ValueError,'permutation'):
            self.load(self.retess(order=[0,0,2]),load_retessellation_capture)
    def test_extra_trailing_fields_rejected(self):
        for raw,loader in [(self.prepare(0),load_volume_preparation_capture),(self.retess(),load_retessellation_capture),(self.neighborhood(),load_neighbor_capture)]:
            with self.subTest(loader=loader.__name__),self.assertRaisesRegex(ValueError,'trailing'):
                self.load(raw+b'\0',loader)


if __name__=='__main__':unittest.main()
