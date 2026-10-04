"""CUDA启动微型诊断的无影像目标；只在独立benchmark调用。"""

def noop(**kwargs):
    """返回初始化后设备信息；不读取影像/参考，不承担recon-all计算。"""
    import torch
    import os
    disabled='PYTORCH_NO_CUDA_MEMORY_CACHING' in os.environ
    return {'device': str(torch.cuda.current_device()),
            'initialized': torch.cuda.is_initialized(),
            'torch_stats_status': 'unavailable_allocator_disabled' if disabled else 'available',
            'allocated_bytes': None if disabled else torch.cuda.memory_allocated(),
            'free_total_bytes': list(torch.cuda.mem_get_info())}
