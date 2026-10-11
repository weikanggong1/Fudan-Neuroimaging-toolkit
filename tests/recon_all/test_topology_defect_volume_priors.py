"""局部 MRI 准备的截断、未量化概率查表及原边采样合同。"""
import unittest
import numpy as np
from fnit.recon_all.topology_defect_volume_priors import defect_volume_priors, _statistics


class VolumePriorContracts(unittest.TestCase):
    def arguments(self):
        x, y, z = np.indices((20, 20, 20))
        white = np.full(256, 1e-9, np.float32)
        gray = white.copy()
        white[100], gray[50] = 1, 1
        white[13] = np.float32(1e-4)
        return dict(original_vertices=np.array([[5, 5, 5], [8, 5, 5], [5, 8, 5]], np.float32),
            candidate_edges=np.array([[0, 1], [1, 2]], np.int32),
            volume=(x+y+z).astype(np.uint8), ras_to_voxel=np.eye(4, dtype=np.float32),
            white_histogram_bins=np.arange(256, dtype=np.float32)+np.float32(.5),
            white_histogram_counts=white, white_histogram_bin_size=1,
            gray_histogram_bins=np.arange(256, dtype=np.float32)+np.float32(.5),
            gray_histogram_counts=gray, gray_histogram_bin_size=1)

    def test_half_bin_statistics_use_peak_threshold_and_ordered_mean(self):
        args=self.arguments()
        values=_statistics(args['white_histogram_bins'],args['white_histogram_counts'],np.float32(1))
        self.assertEqual(tuple(float(x) for x in values),(100.,0.,0.))

    def test_zero_histogram_bin_size_preserves_source_metadata(self):
        args=self.arguments()
        args['white_histogram_bins']=np.arange(256,dtype=np.float32)
        args['gray_histogram_bins']=np.arange(256,dtype=np.float32)
        args['white_histogram_bin_size']=args['gray_histogram_bin_size']=0
        result=defect_volume_priors(**args)
        self.assertEqual((result['white_mean'],result['gray_mean']),(100.,50.))

    def test_probability_uses_float_sample_before_uint8_truncation(self):
        args=self.arguments()
        result=defect_volume_priors(**args)
        self.assertEqual(result['volume_shape'],(10,10,4))
        np.testing.assert_array_equal(result['volume_origin_mm'],[4,4,4])
        self.assertEqual(result['defect_intensity_volume'][1,0,0],12)
        # Surface (4.5,4,4) samples 12.5, source nint=13 for logP.
        self.assertAlmostEqual(float(result['white_log_probability'][1,0,0]),float(np.log(args['white_histogram_counts'][13])),places=6)
        self.assertEqual(result['gray_mean'],50.)
        self.assertEqual(result['gray_log_probability'][1,0,0],0.)

    def test_source_vertical_edge_keeps_last_point_without_new_rule(self):
        args=self.arguments()
        args['candidate_edges']=np.array([[0,2]],np.int32)
        result=defect_volume_priors(**args)
        self.assertEqual(result['initial_edge_mask'].sum(),1)
        self.assertEqual(result['initial_edge_mask'][2,8,2],1)

    def test_zero_length_edge_and_input_immutability(self):
        args=self.arguments()
        args['candidate_edges']=np.array([[0,0]],np.int32)
        before={k:v.copy() for k,v in args.items() if isinstance(v,np.ndarray)}
        result=defect_volume_priors(**args)
        self.assertEqual(result['initial_edge_mask'].sum(),1)
        self.assertEqual(result['defect_intensity_volume'].dtype,np.uint8)
        for name,array in before.items():np.testing.assert_array_equal(array,args[name])

    def test_resolution_defaults_to_two_and_accepts_explicit_positive_scale(self):
        args=self.arguments()
        a=defect_volume_priors(**args)
        b=defect_volume_priors(**args,volume_resolution=2)
        self.assertEqual(a['defect_intensity_volume'].tobytes(),b['defect_intensity_volume'].tobytes())
        self.assertEqual(defect_volume_priors(**args,volume_resolution=1)['volume_shape'],(5,5,2))

    def test_invalid_inputs_fail_before_sampling_without_reference_fallback(self):
        for name,value in [('candidate_edges',np.array([],np.int32).reshape(0,2)),
                ('volume',np.zeros((2,2,2),np.float32)),
                ('ras_to_voxel',np.full((4,4),np.nan,np.float32)),
                ('white_histogram_counts',np.zeros(256,np.float32)),
                ('white_histogram_bin_size',-1),
                ('volume_resolution',0)]:
            args=self.arguments();args[name]=value
            with self.assertRaises(ValueError):defect_volume_priors(**args)


if __name__=='__main__':unittest.main()
