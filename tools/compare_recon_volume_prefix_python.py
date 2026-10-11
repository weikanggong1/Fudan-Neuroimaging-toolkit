"""两条独立自产prefix的逐体素/几何/LTA和标签Dice；不写入生产输出。"""
import argparse,hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np
from fnit.recon_all.ca_normalize_python import read_voxel_lta
def mask_metrics(left,right):
 total=int(left.sum()+right.sum())
 return {'dice':2*int(np.count_nonzero(left&right))/total if total else 1.0,
         'reference_count':int(left.sum()),'candidate_count':int(right.sum()),
         'different_membership_voxels':int(np.count_nonzero(left!=right))}


def stored_value_dice(first,second):
 result={}
 for value in np.union1d(first,second):
  if value==0:continue
  left=first==value;right=second==value
  result[str(int(value))]=mask_metrics(left,right)
 return result


def wm_metrics(first,second):
 """WM contains intensities and edit markers; its existing main mask is >=5."""
 return {'semantics':'WM intensity/edit-marker volume; stored integers are not anatomical partitions',
         'foreground':dict(mask_metrics(first!=0,second!=0),definition='nonzero diagnostic only'),
         'main_wm_mask':dict(mask_metrics(first>=5,second>=5),definition='stored value >=5; existing WM_MIN_VAL=5',threshold=5),
         'stored_value_diagnostic':{'definition':'spatial overlap at each stored integer; intensity/edit-marker diagnostic, not anatomical label Dice',
                                    'values':stored_value_dice(first,second)}}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--producer-source-commit',required=True,help='既有生产输出的冻结源码提交')
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 report={'scope':'same_raw_T1_complete_prefix_comparison','strict_tolerance_declared':0,
         'overall_equivalence':'not_assessed','degradation':'not_assessed','volumes':{},
         'producer_source_commit':a.producer_source_commit,
         'comparison_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'LTA_reader_source_sha256':hashlib.sha256(Path(read_voxel_lta.__code__.co_filename).read_bytes()).hexdigest()}
 receipts=[root.parent/'metrics.public.json' for root in (a.reference,a.candidate)]
 if all(path.exists() for path in receipts):
  producer_receipts=[json.loads(path.read_text()) for path in receipts]
  assert all(row.get('execution_complete') and row.get('exit_code')==0 for row in producer_receipts),'successful producers required'
  report['producer_receipt_sha256']=[hashlib.sha256(path.read_bytes()).hexdigest() for path in receipts]
  report['producer_source_sha256']=[row['source_sha256'] for row in producer_receipts]
  raw_hashes=[row.get('input_sha256') for row in producer_receipts]
  report['raw_input_sha256']=raw_hashes
  report['same_raw_input_verified']=bool(raw_hashes[0] and raw_hashes[0]==raw_hashes[1])
 else:raise FileNotFoundError('both successful producer receipts required')
 label_names={'synthseg.rca','aseg.presurf','entowm','filled','filled.auto'}
 wm_names={'wm.seg','wm.asegedit','wm'}
 for name in ('orig','synthstrip','tmp/nu0','nu','T1','brainmask','synthseg.rca','norm','ctrl_pts','aseg.presurf','brain','entowm','antsdn.brain','wm.seg','wm.asegedit','wm','filled','filled.auto'):
  paths=[root/'mri'/f'{name}.mgz' for root in (a.reference,a.candidate)]
  images=[nib.load(str(path)) for path in paths];first,second=[np.asarray(image.dataobj) for image in images]
  shape_equal=first.shape==second.shape
  row={'reference_shape':list(first.shape),'candidate_shape':list(second.shape),
       'reference_dtype':str(images[0].get_data_dtype()),'candidate_dtype':str(images[1].get_data_dtype()),'shape_equal':shape_equal,'affine_equal':bool(np.array_equal(images[0].affine,images[1].affine)),
       'dtype_equal':images[0].get_data_dtype()==images[1].get_data_dtype(),
       'sha256':[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]}
  if shape_equal:
   delta=np.abs(second.astype(np.float64)-first.astype(np.float64))
   row.update(different_voxels=int(np.count_nonzero(second!=first)),voxel_count=first.size,
              max_error=float(delta.max()),p99_error=float(np.quantile(delta,.99)),
              signed_mean_difference=float(np.mean(second.astype(np.float64)-first.astype(np.float64))),
              finite=bool(np.isfinite(first).all() and np.isfinite(second).all()))
   signed=second.astype(np.float64)-first.astype(np.float64)
   support=(first!=0)|(second!=0)
   changed=first!=second
   row['mean_absolute_difference']=float(delta.mean())
   row['nonzero_support']={'elements':int(support.sum()),'signed_mean_difference':float(signed[support].mean()) if support.any() else 0.0,'mean_absolute_difference':float(delta[support].mean()) if support.any() else 0.0,'p99_error':float(np.quantile(delta[support],.99)) if support.any() else 0.0}
   row['changed_elements']={'elements':int(changed.sum()),'positive':int(np.count_nonzero(signed>0)),'negative':int(np.count_nonzero(signed<0)),'signed_mean_difference':float(signed[changed].mean()) if changed.any() else 0.0,'mean_absolute_difference':float(delta[changed].mean()) if changed.any() else 0.0,'p99_error':float(np.quantile(delta[changed],.99)) if changed.any() else 0.0}
   if name in label_names:
    row['foreground']=mask_metrics(first!=0,second!=0)
    row['labels']=stored_value_dice(first,second)
   elif name in wm_names:
    row.update(wm_metrics(first,second))
  report['volumes'][name]=row
 matrices=[read_voxel_lta(root/'mri/transforms/talairach.lta') for root in (a.reference,a.candidate)]
 report['LTA']={'reference':matrices[0].tolist(),'candidate':matrices[1].tolist(),
                'different_elements':int(np.count_nonzero(matrices[0]!=matrices[1])),
                'max_error':float(np.abs(matrices[0]-matrices[1]).max())}
 initial_matrices=[read_voxel_lta(root/'mri/transforms/talairach.xfm.lta') for root in (a.reference,a.candidate)]
 report['initial_Talairach_LTA']={'different_elements':int(np.count_nonzero(initial_matrices[0]!=initial_matrices[1])),'max_error':float(np.abs(initial_matrices[0]-initial_matrices[1]).max())}
 report['same_upstream_before_N4_verified']=report['same_raw_input_verified'] and report['initial_Talairach_LTA']['different_elements']==0 and all(report['volumes'][name]['different_voxels']==0 and report['volumes'][name]['shape_equal'] and report['volumes'][name]['affine_equal'] and report['volumes'][name]['dtype_equal'] for name in ('orig','synthstrip'))
 report['strict_decoded_prefix_passed']=report['LTA']['different_elements']==0 and all(row.get('different_voxels')==0 and row['shape_equal'] and row['affine_equal'] and row['dtype_equal'] for row in report['volumes'].values())
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
