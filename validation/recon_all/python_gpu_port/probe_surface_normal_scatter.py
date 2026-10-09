"""捕获成熟_normals实际面对贡献，固定贡献复测原CUDA index_add_；非性能实现。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import nibabel.freesurfer.io as fs
import numpy as np
import torch
from probe_surface_metric_repeatability import array_report,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-surface',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    device=torch.device(args.device)
    if device.type!='cuda' or device.index is None:raise ValueError('indexed CUDA required')
    sys.path.insert(0,str(args.source_root/'src'))
    from fnit.recon_all import surface_thickness_gpu as module
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    vertices,faces=fs.read_geometry(str(args.input_surface))
    xyz=torch.as_tensor(np.float32(vertices),device=device);triangles=torch.as_tensor(np.int64(faces),device=device)
    report={'scope':'same_real_mesh_capture_actual_normal_inputs_and_fixed_contribution_CUDA_scatter_replay',
        'hostname':platform.node(),'torch':torch.__version__,'device':str(device),
        'script_sha256':sha(__file__),'input_surface_sha256':sha(args.input_surface),
        'normal_source_sha256':sha(module.__file__),'TF32_matmul':True,'TF32_cudnn':True,
        'half_precision':False,'FNIT_module_source_changed':False,
        'instrumentation':'process-local Tensor.index_add_ observer calls original operator unchanged and restored in finally',
        'status':'started','observed_normal_runs':[],'fixed_contribution_runs':[]}
    original=torch.Tensor.index_add_;captures=[];current_capture=[];normals=[]

    def observe(self,dim,index,source,**kwargs):
        if dim!=0:raise ValueError('unexpected normal scatter dimension')
        current_capture.append((index.detach().clone(),source.detach().clone()))
        return original(self,dim,index,source,**kwargs)

    try:
        torch.Tensor.index_add_=observe
        with torch.inference_mode():
            for iteration in range(3):
                current_capture=[];normal=module._normals(xyz,triangles)
                torch.cuda.synchronize(device);captures.append(current_capture);normals.append(normal.cpu().numpy())
                if len(current_capture)!=3:raise ValueError('exactly three source corner contributions required')
                report['observed_normal_runs'].append({'iteration':iteration,
                    **array_report(normals[-1],normals[0]),
                    'corner_contributions':[{'corner':corner,
                        'indices':array_report(pair[0].cpu().numpy(),captures[0][corner][0].cpu().numpy()),
                        'face_normals':array_report(pair[1].cpu().numpy(),captures[0][corner][1].cpu().numpy())}
                        for corner,pair in enumerate(current_capture)]})
    finally:
        torch.Tensor.index_add_=original
    sums=[];normalized=[]
    with torch.inference_mode():
        for iteration in range(3):
            total=torch.zeros_like(xyz)
            for indices,contribution in captures[0]:original(total,0,indices,contribution)
            normal=torch.nn.functional.normalize(total,dim=1)
            torch.cuda.synchronize(device);sums.append(total.cpu().numpy());normalized.append(normal.cpu().numpy())
            report['fixed_contribution_runs'].append({'iteration':iteration,
                'sum':array_report(sums[-1],sums[0]),'normal':array_report(normalized[-1],normalized[0])})
    report['all_actual_source_contributions_exact']=all(
        corner[name]['different_elements']==0 for row in report['observed_normal_runs']
        for corner in row['corner_contributions'] for name in ('indices','face_normals'))
    report['fixed_input_scatter_differences_detected']=any(
        row['sum']['different_elements']>0 for row in report['fixed_contribution_runs'])
    report['original_Tensor_method_restored']=torch.Tensor.index_add_ is original
    report['status']='complete';args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
