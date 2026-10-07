"""Compare retained whole-graph candidate raw tensors with isolated TF controls."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sr_first_layer import metrics,digest

p=argparse.ArgumentParser(description=__doc__)
for name in ('isolated-second-dir','candidate-trace-dir','reference-trace-dir','report'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
if a.report.exists():p.error('preserve earlier evidence')
actual=a.candidate_trace_dir/'0'
ref=a.isolated_second_dir
result={'raw':metrics(np.load(ref/'raw.npy',mmap_mode='r'),np.load(actual/'first_raw.npy',mmap_mode='r')),
        'elu':metrics(np.load(ref/'elu.npy',mmap_mode='r'),np.load(actual/'first_elu.npy',mmap_mode='r')),
        'driver_sha256':digest(__file__),'scope':'whole-graph candidate vs isolated raw+ELU reference, real branch0'}
reference=json.loads((a.reference_trace_dir/'report.public.json').read_text())
candidate=json.loads((a.candidate_trace_dir/'report.public.json').read_text())
result['branches']={i:{'inputs_equal':r['input_array_sha256']==candidate['branches'][i]['input_array_sha256'],
    'first_conv_elu_equal':r['stages']['unet_conv_downarm_0_0']['array_sha256']==candidate['branches'][i]['stages']['unet_conv_downarm_0_0']['array_sha256'],
    'second_conv_elu_equal':r['stages']['unet_conv_downarm_0_1']['array_sha256']==candidate['branches'][i]['stages']['unet_conv_downarm_0_1']['array_sha256']} for i,r in reference['branches'].items()}
a.report.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
