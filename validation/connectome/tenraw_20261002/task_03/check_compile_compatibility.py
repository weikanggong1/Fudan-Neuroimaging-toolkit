"""Optional compile_arc API/Dynamo smoke; never default lossless benchmark."""
import argparse
import importlib
import hashlib
import json
from pathlib import Path
import torch
from run_tracking_checkpoint import load_tracking

parser = argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--device',default='cuda')
parser.add_argument('--backend',default='inductor')
args = parser.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = True
module = load_tracking(args.source)
fod = torch.zeros((7,7,7,45),device=args.device)
fod[...,0] = 4
five = torch.zeros((7,7,7,5),device=args.device)
five[...,2] = 1
inverse = torch.eye(4,dtype=torch.float64,device=args.device)
compiled = torch.compile(module._ifod2_arc_probability, fullgraph=True, dynamic=True, backend=args.backend)
records = []
for batch,count in [(2,3),(3,4)]:
    position = torch.full((batch,3),3.,device=args.device)
    prior = torch.zeros((batch,3),device=args.device); prior[:,0] = 1
    angle = torch.linspace(.01,.3,count,device=args.device)
    directions = torch.stack((angle.cos(),angle.sin(),torch.zeros_like(angle)),-1)[None].expand(batch,-1,-1).contiguous()
    half_log_start = torch.ones(batch,device=args.device)
    inputs = (position,prior,directions,half_log_start,fod,five,inverse,inverse)
    kwargs = dict(lmax=8,step_mm=.5,cutoff=.1,power=.5)
    # Production initial-direction evaluation warms SH before compiled growth.
    eager = module._ifod2_arc_probability(*inputs,**kwargs)
    actual = compiled(*inputs,**kwargs)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.synchronize()
    assert len(actual) == len(eager)
    differences = []
    for a,b in zip(eager,actual):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(b).all()
        differences.append({'neq':int((a!=b).sum()),'max':float((a-b).abs().max())})
    records.append({'batch':batch,'count':count,'output_differences_diagnostic_only':differences})
public_execution = None
public_count = None
if torch.device(args.device).type == 'cuda':
    anatomy = importlib.import_module('tracking_ab.anatomy')
    public_five = torch.zeros((9,9,9,5),device=args.device)
    public_five[1,:,:,0] = 1
    public_five[2:7,:,:,2] = 1
    public_five[7,:,:,0] = 1
    public_fod = torch.zeros((9,9,9,45),device=args.device); public_fod[...,0] = 4
    public_tracks = module.probabilistic_tractography(
        public_fod,inverse,public_five,inverse,anatomy.gmwmi_from_five_tissue(public_five),
        n_seeds=4,seed=11,batch_size=4,step_mm=.5,max_length_mm=8.,compile_arc=True)
    torch.cuda.synchronize()
    assert isinstance(public_tracks.paths,tuple) and public_tracks.seeds_attempted == 4
    public_execution = True
    public_count = len(public_tracks.paths)
report = {'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':args.device,'backend':args.backend,'public_compile_arc_execution_pass':public_execution,'public_accepted_tracks_diagnostic_only':public_count,'scope':'generated optional compile API/Dynamo compatibility; not real benchmark or lossless assertion',
          'execution_pass':True,'torch':torch.__version__,'results':records,
          'source_sha256':{name:hashlib.sha256((args.source/(name+'.py')).read_bytes()).hexdigest() for name in ['fod','tracking']}}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
