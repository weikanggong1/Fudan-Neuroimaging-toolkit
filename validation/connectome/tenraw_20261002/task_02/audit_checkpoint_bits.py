"""CPU byte-level comparison, including signed zeros and NaN payloads."""
import argparse
import json
from pathlib import Path
import torch
from benchmark_modeling import sha
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(8);left=torch.load(a.reference,map_location='cpu',weights_only=True);right=torch.load(a.candidate,map_location='cpu',weights_only=True)
if left.keys()!=right.keys():raise ValueError('checkpoint keys differ')
results={}
for key,l in left.items():
    r=right[key]
    if l.dtype!=r.dtype or l.shape!=r.shape:raise ValueError(f'{key} shape/dtype differs')
    lbytes=l.reshape(-1).contiguous().view(torch.uint8).reshape(l.numel(),l.element_size());rbytes=r.reshape(-1).contiguous().view(torch.uint8).reshape(r.numel(),r.element_size())
    results[key]={'dtype':str(l.dtype),'shape':list(l.shape),'bitwise_neq_values':int((lbytes!=rbytes).any(dim=1).sum())}
report={'scope':'full saved CPU checkpoint byte comparison; signed zero and NaN payload included','harness_sha256':sha(__file__),'reference_sha256':sha(a.reference),'candidate_sha256':sha(a.candidate),'results':results,'passed':all(v['bitwise_neq_values']==0 for v in results.values())}
Path(a.output).write_text(json.dumps(report,indent=2));print(a.output)
if not report['passed']:raise RuntimeError('checkpoint byte comparison failed')
