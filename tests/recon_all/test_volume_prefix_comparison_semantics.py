"""WM contracts verify the existing >=5 definition; these fixtures are not MRI benchmarks."""
from pathlib import Path
import runpy
import numpy as np


def metrics():
 return runpy.run_path(str(Path(__file__).resolve().parents[2]/'tools/compare_recon_volume_prefix_python.py'))['wm_metrics']


def test_wm_intensity_change_keeps_main_mask():
 row=metrics()(np.array([0,73,110],np.uint8),np.array([0,90,129],np.uint8))
 assert row['main_wm_mask']['dice']==1 and row['main_wm_mask']['different_membership_voxels']==0
 assert row['stored_value_diagnostic']['values']['73']['dice']==0
 assert 'labels' not in row


def test_wm_values_one_to_four_do_not_enter_main_mask():
 row=metrics()(np.array([0,1,2,3,4,5],np.uint8),np.array([4,0,2,1,5,4],np.uint8))
 assert row['main_wm_mask']['reference_count']==1 and row['main_wm_mask']['candidate_count']==1
 assert row['main_wm_mask']['different_membership_voxels']==2 and row['main_wm_mask']['dice']==0
 assert row['foreground']['reference_count']==5


def test_wm_edit_markers_follow_existing_main_mask():
 row=metrics()(np.array([0,1,255,110],np.uint8),np.array([1,0,110,255],np.uint8))
 assert row['main_wm_mask']['dice']==1 and row['main_wm_mask']['different_membership_voxels']==0
 assert row['stored_value_diagnostic']['values']['255']['dice']==0
 assert row['main_wm_mask']['threshold']==5
