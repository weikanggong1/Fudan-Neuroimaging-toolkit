"""FP32 coordinate boundary/nint contract; real T1 regression is separate."""
import numpy as np
from fnit.recon_all.ca_normalize_python import prior_to_source_coordinates

def test_translation_not_scaled_and_half_integers():
 prior=np.array([[0,0,0],[1,1,1]],np.int32)
 matrix=np.eye(4,dtype=np.float32);matrix[:3,3]=[.5,-.5,.5]
 # Inverse coordinates [-.5,+.5,-.5] and [1.5,2.5,1.5]: ties away from zero.
 expected=np.array([[-1,1,-1],[2,3,2]],np.int32)
 np.testing.assert_array_equal(prior_to_source_coordinates(prior,matrix,2),expected)

def test_fp32_coordinate_is_promoted_before_half_addition():
 matrix=np.eye(4,dtype=np.float32)
 matrix[0,3]=-np.nextafter(np.float32(.5),np.float32(0))
 result=prior_to_source_coordinates(np.zeros((1,3),np.int32),matrix,2)
 assert result[0,0]==0 # FP32 +.5 would incorrectly produce exactly 1.

def test_float_boundary_keeps_original_atlas_coordinate():
 prior=np.array([[1,1,1]],np.int32)
 matrix=np.eye(4,dtype=np.float32);matrix[0,3]=-253
 # Exactly 255 is in volume, even though it is the last voxel.
 np.testing.assert_array_equal(prior_to_source_coordinates(prior,matrix,2,source_shape=(256,256,256)),[[255,2,2]])
 matrix[0,3]=-np.float32(253.1)
 # 255.1 rounds to 255 but is out of the float domain: retain initial [2,2,2].
 np.testing.assert_array_equal(prior_to_source_coordinates(prior,matrix,2,source_shape=(256,256,256)),[[2,2,2]])
 matrix[0,3]=np.float32(2.1)
 np.testing.assert_array_equal(prior_to_source_coordinates(prior,matrix,2,source_shape=(256,256,256)),[[2,2,2]])

if __name__=='__main__':
 test_translation_not_scaled_and_half_integers()
 test_fp32_coordinate_is_promoted_before_half_addition()
 test_float_boundary_keeps_original_atlas_coordinate()
 print('3 native coordinate contract tests passed')
