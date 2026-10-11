"""真实nibabel格式往返与失败回执合同；不作为真实T1 benchmark。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

import benchmark_recon_topology_genetic_repair as tool


class PremeshFileContracts(unittest.TestCase):
    def fixture(self,root):
        xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int32)
        geometry=dict(head=np.array([2,0,20]),valid='1',filename='own.mgz',
            volume=np.array([2,2,2]),voxelsize=np.ones(3),xras=np.array([-1,0,0]),
            yras=np.array([0,0,-1]),zras=np.array([0,1,0]),cras=np.zeros(3))
        for name in ('own.nofix','own.qsphere'):
            fsio.write_geometry(str(root/name),xyz,faces,volume_info=geometry)
        nib.save(nib.MGHImage(np.ones((2,2,2),np.uint8),np.eye(4)),root/'brain.mgz')
        kwargs=dict(input_surface=root/'own.nofix',input_qsphere=root/'own.qsphere',
            brain=root/'brain.mgz',hemisphere='lh',output=root/'result',threads=1,seed=1234,
            population_size=10,max_unchanged=10,niters=-1,initial_selection=True,
            vertex_eliminate=True,use_overlap=False,normal_status='surface',soap_iterations=5,
            checkpoint_each_defect=False)
        return xyz,faces,geometry,kwargs

    def test_writes_self_input_geometry_footer_without_internal_register_mat(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);xyz,faces,geometry,kwargs=self.fixture(root)
            result=dict(surface_state={'current_vertices':xyz,'faces':faces,'nfaces':4},
                metadata={'volume_geometry':dict(geometry,register_mat=None)},
                raw_to_base_vertex_translation=np.arange(4,dtype=np.int32),
                base_to_compact_vertex_translation=np.arange(4,dtype=np.int32),
                original_domain_defect_labels=np.zeros(4,np.int32),current_cleanup_vertices=xyz,
                execution_status='complete',complete_ga=True,native_program_called=False,
                quality={'status':'passed'},per_defect=[],random_draws=0,current_cleanup={},
                standard_policy=True,final_mesh_quality='not_assessed',
                required_next_steps=('standard_remesh','original_intersection_cleanup','final_mesh_quality'),seconds={})
            with patch.object(tool,'repair_genetic_topology',return_value=result),patch.object(tool,'dump_state_checkpoint')as dump:
                dump.side_effect=lambda path,state:path.write_bytes(b'control-only')
                report=tool.run(**kwargs)
            saved,saved_faces,saved_geometry=fsio.read_geometry(str(root/'result/lh.orig.premesh'),read_metadata=True)
            np.testing.assert_array_equal(saved,xyz);np.testing.assert_array_equal(saved_faces,faces)
            for key,value in geometry.items():np.testing.assert_array_equal(saved_geometry[key],value)
            self.assertEqual(report['final_mesh_quality'],'not_assessed')
            self.assertEqual(report['whole_recon_all'],'not_assessed')
            self.assertIsNone(report['gpu_memory_bytes'])
            self.assertTrue((root/'result/COMPLETE.json').is_file())

    def test_stage_failure_has_no_complete_or_surface_output(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);_,_,_,kwargs=self.fixture(root)
            with patch.object(tool,'repair_genetic_topology',side_effect=RuntimeError('own stage failed')):
                with self.assertRaisesRegex(RuntimeError,'own stage failed'):tool.run(**kwargs)
            self.assertTrue((root/'result/failure.json').is_file())
            self.assertFalse((root/'result/COMPLETE.json').exists())
            self.assertFalse((root/'result/lh.orig.premesh').exists())


if __name__=='__main__':unittest.main()
