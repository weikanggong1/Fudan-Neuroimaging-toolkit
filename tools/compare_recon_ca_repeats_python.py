"""核对相同CA输入的两次Python/原版独立运行，报告decoded重复性。"""
import argparse,hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for name in ('first','second','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
receipts=[json.loads((root/'metrics.public.json').read_text()) for root in (a.first,a.second)]
if not all(r['execution_complete'] for r in receipts):raise ValueError('both CA executions must be complete')
if receipts[0]['input_sha256']!=receipts[1]['input_sha256']:raise ValueError('repeat inputs differ')
if receipts[0]['reference_binary_sha256']!=receipts[1]['reference_binary_sha256']:raise ValueError('reference program differs')
if receipts[0]['source_sha256']!=receipts[1]['source_sha256']:raise ValueError('candidate source differs')
report={'scope':'two_completed_same_input_CA_repeats','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':receipts[0]['input_sha256'],'reference_binary_sha256':receipts[0]['reference_binary_sha256'],'strict_tolerance':0,'overall_equivalence':'not_assessed','volumes':{}}
for mode in ('python','native-reference'):
 for name in ('norm','ctrl_pts'):
  paths=[root/mode/f'{name}.mgz' for root in (a.first,a.second)]
  images=[nib.load(str(path)) for path in paths];arrays=[np.asarray(image.dataobj) for image in images]
  same_shape=arrays[0].shape==arrays[1].shape
  row={'shape_equal':same_shape,'affine_equal':bool(np.array_equal(images[0].affine,images[1].affine)),'dtype_equal':images[0].get_data_dtype()==images[1].get_data_dtype(),'output_sha256':[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]}
  if same_shape:
   different=int(np.count_nonzero(arrays[0]!=arrays[1]));row.update(different_elements=different,max_error=float(np.abs(arrays[1].astype(np.float64)-arrays[0].astype(np.float64)).max()) if different else 0.0,finite=bool(np.isfinite(arrays[0]).all() and np.isfinite(arrays[1]).all()))
  report['volumes'][f'{mode}/{name}']=row
report['strict_repeated_decoded_passed']=all(r.get('different_elements')==0 and r['shape_equal'] and r['affine_equal'] and r['dtype_equal'] and r['finite'] for r in report['volumes'].values())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
