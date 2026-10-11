"""数字拓扑候选的局部判据、完整生长与标签空间契约；不是真实benchmark。"""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.freesurfer.io as fsio
from scipy import ndimage

from fnit.recon_all.topology_volume_python import correct_binary_topology, _components, write_volume_topology_defects


class DigitalTopologyContracts(unittest.TestCase):
    def test_simple_face_attachment_and_disconnected_foreground(self):
        local=np.zeros(27,bool)
        local[12]=True
        self.assertEqual(_components(local,True),1)
        self.assertEqual(_components(local,False),1)
        local[14]=True
        self.assertEqual(_components(local,True),2)

    def test_solid_cube_preserves_all_foreground_and_repeats_order(self):
        target=np.zeros((13,13,13),bool)
        target[3:10,3:10,3:10]=True
        first,a=correct_binary_topology(target=target)
        second,b=correct_binary_topology(target=target)
        np.testing.assert_array_equal(first,target)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(a['accepted_order_sha256'],b['accepted_order_sha256'])
        self.assertEqual(json.loads(json.dumps(a))['seed_voxel_index_xyz'],a['seed_voxel_index_xyz'])

    def test_handle_and_cavity_use_full_subset_and_one_foreground_component(self):
        axes=np.indices((19,19,19)).astype(float)-9
        radial=np.sqrt(axes[0]**2+axes[1]**2)
        target=(radial-5)**2+axes[2]**2 <= 2.5**2
        result,report=correct_binary_topology(target=target)
        self.assertTrue(np.all(~result|target))
        self.assertGreater(report['removed_foreground_voxels'],0)
        self.assertEqual(ndimage.label(result,ndimage.generate_binary_structure(3,1))[1],1)
        self.assertEqual(ndimage.label(~result,ndimage.generate_binary_structure(3,3))[1],1)

    def test_invalid_target_fails_before_growth(self):
        for target,spacing in ((np.zeros((9,9,9),bool),(1,1,1)),
                               (np.ones((9,9,9),bool),(1,1,1)),
                               (np.zeros((9,9),bool),(1,1,1))):
            with self.assertRaises(ValueError):
                correct_binary_topology(target=target,spacing_mm=spacing)

    def test_deleted_regions_map_to_original_corner_vertices(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            data=np.zeros((9,9,9),np.uint8);data[3:7,3:7,3:7]=255
            original=nib.MGHImage(data,np.eye(4))
            corrected=data.copy();corrected[4,4,4]=0
            nib.save(original,str(root/'before.mgz'));nib.save(nib.MGHImage(corrected,np.eye(4)),str(root/'after.mgz'))
            corners=np.asarray([[3.5,3.5,3.5],[3.5,3.5,4.5],[3.5,4.5,3.5],[1.5,1.5,1.5]])
            matrix=original.header.get_vox2ras_tkr()
            xyz=corners@matrix[:3,:3].T+matrix[:3,3]
            info={'head':np.asarray([2,0,20]),'valid':'1','filename':'unit-only',
                  'volume':np.asarray(data.shape),'voxelsize':original.header['delta'],
                  'xras':original.header['Mdc'][:,0],'yras':original.header['Mdc'][:,1],
                  'zras':original.header['Mdc'][:,2],'cras':original.header['Pxyz_c']}
            fsio.write_geometry(str(root/'orig.nofix'),xyz,np.asarray([[0,1,2],[0,2,3]]),volume_info=info)
            result=write_volume_topology_defects(input_volume=root/'before.mgz',corrected_volume=root/'after.mgz',
                source_surface=root/'orig.nofix',value=255,output_defect_labels=root/'defect_labels')
            np.testing.assert_array_equal(fsio.read_morph_data(str(root/'defect_labels')),[1,1,1,0])
            self.assertEqual(result['deleted_voxels'],1)
            self.assertEqual(result['deleted_regions_6'],1)
            self.assertEqual(result['nonzero_label_vertices'],3)


if __name__=='__main__':
    unittest.main()
