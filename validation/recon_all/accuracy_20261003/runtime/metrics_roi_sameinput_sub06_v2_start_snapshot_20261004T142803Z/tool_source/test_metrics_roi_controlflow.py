"""Frozen-env CPU control-flow check, hidden CUDA; all numeric operators mocked.

Real function signatures, thread_budget, StageProfiler and six-table dispatcher
are used. This is interface verification, never a scientific benchmark.
"""
import argparse,contextlib,hashlib,json,os,pathlib,sys,tempfile,types
from unittest import mock

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preflight requires CUDA hidden before torch import')
 import torch,numpy as np,nibabel.freesurfer.io as fs
 import replay_metrics_roi_stage as child
 from fnit.recon_all import native_free,profiling,surface_area_gpu,surface_roi_gpu,anatomical_stats_file
 if torch.cuda.is_initialized():raise ValueError('CPU preflight CUDA already initialized')
 gpu='GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba';calls=[]
 with tempfile.TemporaryDirectory() as temporary:
  root=pathlib.Path(temporary);checkpoint=root/'checkpoint'
  for name in child.INPUTS:
   p=checkpoint/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU interface test '+name)
  for name in ['white','pial']:fs.write_geometry(str(checkpoint/('surf/lh.'+name)),np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32),np.array([[0,1,2]],np.int32))
  volume_names=['BrainSegVol','BrainSegVolNotVent','SupraTentorialVol','SupraTentorialVolNotVent','SubCortGrayVol','lhCortexVol','rhCortexVol','CortexVol','TotalGrayVol','lhCerebralWhiteMatterVol','rhCerebralWhiteMatterVol','CerebralWhiteMatterVol','MaskVol','SupraTentorialVolNotVentVox','BrainSegVolNotVentSurf','VentricleChoroidVol']
  (checkpoint/'stats/brainvol.stats').write_text(''.join('# Measure x, '+name+', x, 1.0, mm^3\n' for name in volume_names))
  entries=[{'relative':name,'size_bytes':(checkpoint/name).stat().st_size,'sha256':child.sha(checkpoint/name)} for name in child.INPUTS];manifest=root/'inputs.json';manifest.write_text(json.dumps({'files':entries}));out=root/'output'
  def metrics(binary,subject,hemi,assets,*,device):
   calls.append('metrics')
   for name in ['thickness','area','area.pial','curv','curv.pial']:fs.write_morph_data(str(subject/('surf/lh.'+name)),np.ones(3,np.float32))
   return {name:0.0 for name in ['thickness','area','area.pial','curv','curv.pial']}
  def mid(area_white,area_pial,output,*,device='cuda:0'):
   calls.append('mid_area');fs.write_morph_data(str(output),np.ones(3,np.float32))
  def volume(white,pial,cortex,output,*,device='cuda:0'):
   calls.append('volume');fs.write_morph_data(str(output),np.ones(3,np.float32))
  def stats(subject,hemi,atlas,surface,brainvol_stats,output,*,device='cuda:0',cache=None):
   calls.append('stats:'+atlas+':'+surface);output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text('# CPU MOCK\nregion 3 1 1 1 1 1 1 1 1\n');return output
  with contextlib.ExitStack() as stack:
   stack.enter_context(mock.patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden in CPU interface check')))
   stack.enter_context(mock.patch.object(torch.cuda,'get_device_properties',return_value=types.SimpleNamespace(uuid=gpu[4:])))
   stack.enter_context(mock.patch.object(native_free,'_run_surface_metrics',mock.create_autospec(native_free._run_surface_metrics,side_effect=metrics)))
   stack.enter_context(mock.patch.object(surface_area_gpu,'mid_area_map',mock.create_autospec(surface_area_gpu.mid_area_map,side_effect=mid)))
   stack.enter_context(mock.patch.object(surface_roi_gpu,'vertex_volume_map',mock.create_autospec(surface_roi_gpu.vertex_volume_map,side_effect=volume)))
   stack.enter_context(mock.patch.object(anatomical_stats_file,'write_anatomical_stats',mock.create_autospec(anatomical_stats_file.write_anatomical_stats,side_effect=stats)))
   # Keep real global CUDA_VISIBLE_DEVICES='', only child's declaration lookup is mocked.
   stack.enter_context(mock.patch.object(child,'os',types.SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES':gpu})))
   stack.enter_context(mock.patch.object(sys,'argv',['replay_metrics_roi_stage.py','--checkpoint',str(checkpoint),'--input-manifest',str(manifest),'--assets',str(root),'--output',str(out),'--gpu-uuid',gpu]))
   child.main()
  report=json.loads((out/'report.json').read_text());assert report['status']=='complete' and [p['name'] for p in report['passes']]==['cold','warm'];assert calls.count('metrics')==2 and sum(c.startswith('stats:') for c in calls)==12
  assert not torch.cuda.is_initialized();assert report['thread_budget']['restoration_complete'];assert len(list(out.glob('*/metrics.npz')))==2;assert len(list(out.glob('*/*.stats')))==12
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({'status':'passed','scope':'CPU interface/control-flow only; operators mocked; not benchmark','cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'cuda_initialized':torch.cuda.is_initialized(),'calls':calls,'real_threads':report['thread_budget'],'loaded_source_sha256':report['loaded_source_sha256'],'runner_sha256':child.sha(pathlib.Path(child.__file__))},indent=2)+'\n')
if __name__=='__main__':main()
