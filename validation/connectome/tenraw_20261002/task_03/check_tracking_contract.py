"""Generated anatomy tests full RNG/path contract; never a real-data benchmark."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from run_tracking_checkpoint import load_tracking

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
a = load_tracking(args.baseline)
b = load_tracking(args.candidate)
shape = (22,11,11)
five = torch.zeros((*shape,5), device='cuda')
five[1,1:10,1:10,0] = 1
five[2:20,1:10,1:10,2] = 1
five[20,1:10,1:10,0] = 1
five = five.repeat_interleave(2,0).repeat_interleave(2,1).repeat_interleave(2,2)
anatomy_affine = torch.diag(torch.tensor([.5,.5,.5,1.],device='cuda'))
anatomy_affine[:3,3] = -.25
gm, wm = five[...,0]+five[...,1], five[...,2]
gradient_squared = torch.zeros_like(gm)
for axis in range(3):
    negative_gm, positive_gm = gm.roll(1,axis), gm.roll(-1,axis)
    negative_wm, positive_wm = wm.roll(1,axis), wm.roll(-1,axis)
    negative_gm.select(axis,0).copy_(gm.select(axis,0))
    positive_gm.select(axis,-1).copy_(gm.select(axis,-1))
    negative_wm.select(axis,0).copy_(wm.select(axis,0))
    positive_wm.select(axis,-1).copy_(wm.select(axis,-1))
    difference = torch.minimum((positive_gm-negative_gm).abs(), (positive_wm-negative_wm).abs())*.5
    difference.select(axis,0).mul_(2)
    difference.select(axis,-1).mul_(2)
    gradient_squared += difference.square()
gmwmi = gradient_squared.sqrt()
index = torch.arange(256,device='cuda',dtype=torch.float32)
z = 1-2*(index+.5)/256
angle = index*(math.pi*(3-math.sqrt(5)))
directions = torch.stack(((1-z.square()).sqrt()*angle.cos(), (1-z.square()).sqrt()*angle.sin(),z),-1)
coefficient = torch.linalg.lstsq(a.real_sh(directions,4), (5/(4*math.pi))*directions[:,0].pow(4)).solution
fod = torch.zeros((*shape,15),device='cuda')
fod[2:20,1:10,1:10] = coefficient
fa = torch.full(shape,.6,device='cuda')
kwargs = dict(n_seeds=100,lmax=4,fa=fa,seed=11,cutoff=.01,power=2.,max_angle_degrees=20)
left = a.probabilistic_tractography(fod,torch.eye(4,device='cuda'),five,anatomy_affine,gmwmi,**kwargs)
right = b.probabilistic_tractography(fod,torch.eye(4,device='cuda'),five,anatomy_affine,gmwmi,**kwargs)
assert len(left.paths) == len(right.paths) and len(left.paths)>1
assert all(torch.equal(x,y) for x,y in zip(left.paths,right.paths))
for field in ['endpoints','lengths_mm','mean_fa','accepted_seeds']:
    assert torch.equal(getattr(left,field),getattr(right,field)),field
assert left.seeds_attempted == right.seeds_attempted
report = {'scope':'generated contract regression, not benchmark','accepted':len(left.paths),
          'points':sum(len(path) for path in left.paths),'neq':0,'seeds_attempted':100,'seed':11,
          'source_sha256': {variant: {name: hashlib.sha256((directory / (name+'.py')).read_bytes()).hexdigest() for name in ['fod','tracking']} for variant,directory in [('baseline',args.baseline),('candidate',args.candidate)]}}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
