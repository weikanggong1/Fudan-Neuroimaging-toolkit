"""Exact CPU checkpoint comparison; full-array P99 has no Torch 2^24 limit."""
import numpy as np
import torch

def compare_tensor_dicts(reference,candidate):
    if reference.keys()!=candidate.keys():raise ValueError('checkpoint keys differ')
    errors={}
    for key,left in reference.items():
        right=candidate[key]
        if left.device.type!='cpu' or right.device.type!='cpu':raise ValueError('comparison must run on CPU')
        if left.shape!=right.shape or left.dtype!=right.dtype:raise ValueError(f'{key} shape/dtype differs')
        if left.dtype==torch.bool:
            errors[key]={'neq':int((left!=right).sum())};continue
        finite=torch.isfinite(left)&torch.isfinite(right)
        neq=int((left[finite]!=right[finite]).sum())
        nonfinite=int(((torch.isfinite(left)!=torch.isfinite(right))|(torch.isnan(left)!=torch.isnan(right))|(torch.isposinf(left)!=torch.isposinf(right))|(torch.isneginf(left)!=torch.isneginf(right))).sum())
        if neq:
            delta=(left[finite].double()-right[finite].double()).abs()
            metrics={'max':float(delta.max()),'p99':float(np.quantile(delta.numpy(),.99,method='linear')),'rmse':float(delta.square().mean().sqrt())}
        else:metrics={'max':0.0,'p99':0.0,'rmse':0.0}
        errors[key]={'neq':neq,'nonfinite_mismatch':nonfinite,**metrics}
    return errors
