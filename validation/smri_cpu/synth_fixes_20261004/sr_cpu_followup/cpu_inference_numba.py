"""CPU-only diagnostic forward; mature production/GPU forward stays unchanged."""
import torch
from torch.nn import functional as F
from elu_numba import elu_torch
from bn_numba import batch_norm_torch


def cpu_inference(model, image, *, elu_function=elu_torch):
    skips=[]
    value=image.to(memory_format=torch.channels_last_3d)
    for level,(convs,norm) in enumerate(zip(model.down,model.down_bn)):
        value=elu_function(convs[0](value))
        value=elu_function(convs[1](value))
        skips.append(value)
        value=batch_norm_torch(value,norm)
        if level<len(model.down)-1:
            value=F.max_pool3d(value,2)
    for convs,norm,skip in zip(model.up,model.up_bn,reversed(skips[:-1])):
        value=F.interpolate(value,scale_factor=2,mode='nearest')
        value=torch.cat((skip,value),dim=1)
        value=elu_function(convs[0](value))
        value=elu_function(convs[1](value))
        value=batch_norm_torch(value,norm)
    return model.likelihood(value)
