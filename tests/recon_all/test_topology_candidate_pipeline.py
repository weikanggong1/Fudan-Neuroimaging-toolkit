"""候选状态事务和源incidence的契约；不作为真实MRIbenchmark。"""
import unittest
from unittest.mock import patch
import numpy as np
from fnit.recon_all.topology_candidate_geometry import candidate_incidents
from fnit.recon_all.topology_candidate_pipeline import candidate_face_check, evaluate_candidate


class CandidatePipelineContracts(unittest.TestCase):
    def topology(self):
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
        neighbors=np.array([1,2,3,0,2,3,0,1,3,0,1,2],np.int32)
        return dict(faces=faces,neighbor_offsets=np.arange(0,13,3,dtype=np.int32),ordered_neighbors=neighbors,
            **candidate_incidents(faces=faces,vertex_count=4))

    def test_closed_incidence_passes_and_unbacked_neighbor_fails(self):
        args=self.topology();args.pop('incident_corners')
        result=candidate_face_check(**args,patch_vertices=np.array([0,2],np.int32))
        self.assertEqual(result['source_result'],1)
        # The real migration risk was an edge with no face support. Preserve
        # it in the CSR and explicitly reject, rather than reconstructing it away.
        args['ordered_neighbors']=np.array([1,2,3,4,0,2,3,0,1,3,0,1,2,0],np.int32)
        args['neighbor_offsets']=np.array([0,4,7,10,13,14],np.int32)
        args['incident_offsets']=np.r_[args['incident_offsets'],args['incident_offsets'][-1]].astype(np.int32)
        result=candidate_face_check(**args,patch_vertices=np.array([0],np.int32))
        self.assertEqual(result['source_result'],-1)
        np.testing.assert_array_equal(result['first_invalid'],[0,4,0])

    def test_boundary_edge_counts_one_and_missing_face_row_is_not_hidden(self):
        faces=np.array([[0,1,2]],np.int32)
        args=dict(faces=faces,neighbor_offsets=np.array([0,2,4,6],np.int32),ordered_neighbors=np.array([1,2,0,2,0,1],np.int32),
            incident_offsets=np.array([0,1,2,3],np.int32),incident_faces=np.array([0,0,0],np.int32),patch_vertices=np.array([0],np.int32))
        result=candidate_face_check(**args)
        self.assertEqual(result['source_result'],-1)
        np.testing.assert_array_equal(result['first_invalid'],[0,1,1])
        with self.assertRaises(ValueError):candidate_face_check(**dict(args,neighbor_offsets=np.array([0,2,4,7])))
        with self.assertRaises(ValueError):candidate_face_check(**dict(args,patch_vertices=np.array([.9])))

    def test_same_state_keeps_discarded_fx_external_fields_and_does_not_alias(self):
        topology=self.topology();xyz=np.array([[3,0,0],[0,3,0],[0,0,3],[-1,-1,-1]],np.float32)
        initial=dict(original_vertices=xyz,vertex_normals=np.full((4,3),.25,np.float32),
            tangent_e1=np.ones((4,3),np.float32),tangent_e2=np.full((4,3),2,np.float32),
            flattened_vertices=np.full((4,3),7,np.float32),k1=np.arange(4,dtype=np.float32),k2=np.arange(4,dtype=np.float32)+5,
            vertex_area=np.zeros(4,np.float32),curvbak=np.array([10,20,30,40],np.float32),faces=topology['faces'],face_area=np.zeros(4,np.float32))
        changed=xyz.copy();changed[0,0]+=1
        geo=dict(topology,original_vertices=changed,selected_vertices=np.array([0],np.int32),selected_faces=np.array([0],np.int32),
            selected_vertex_normals=np.array([[0,0,1]],np.float32),face_normals=np.ones((4,3),np.float32))
        retess=dict(base_faces=topology['faces'],inside_vertices=np.array([0,1],np.int32),inside_discarded=np.array([False,True]))
        priors=dict(white_mean=100.,gray_mean=60.,volume_shape=(7,7,7),volume_origin_mm=np.zeros(3,np.float32),
            volume_index_scale_per_mm=np.ones(3,np.float32),defect_intensity_volume=np.ones((7,7,7),np.uint8),
            white_log_probability=np.zeros((7,7,7),np.float32),gray_log_probability=np.zeros((7,7,7),np.float32),
            initial_white_mean_log_probability=-1.,initial_gray_mean_log_probability=-2.)
        with patch('fnit.recon_all.topology_candidate_pipeline.candidate_geometry',return_value=geo):
            result=evaluate_candidate(surface_state=initial,retessellation_inputs=retess,volume=np.ones((8,8,8),np.uint8),
                ras_to_voxel=np.eye(4),local_priors=priors,gray_white_pdf=np.ones((256,256,1),np.float32),
                k1_mean=-.1,k2_mean=-.2,normal_dot_histogram={},k1_histogram={},k2_histogram={},
                joint_curvature_pdf=np.ones((1,1,1),np.float32),joint_origin_mm_inv=np.zeros(2),joint_spacing_mm_inv=np.ones(2),
                mri_weight=0,volume_unmri_weight=0,normal_dot_weight=0,quadratic_curvature_weight=0)
        np.testing.assert_array_equal(result['flattened_vertices'][0],xyz[0])
        np.testing.assert_array_equal(result['flattened_vertices'][1:],np.full((3,3),7,np.float32))
        np.testing.assert_array_equal(result['vertex_curvbak'],[1,20,30,40])
        np.testing.assert_array_equal(result['sff']['k1'],initial['k1'])
        np.testing.assert_array_equal(result['sff']['tangent_e1'],initial['tangent_e1'])
        np.testing.assert_array_equal(initial['curvbak'],[10,20,30,40])
        self.assertFalse(np.shares_memory(initial['vertex_normals'],result['vertex_normals']))
        self.assertEqual(result['fitness']['executed_components'],[])
        self.assertEqual(result['fitness']['fitness'],0.)


if __name__=='__main__':unittest.main()
