"""完整 GA 显式距离后端/设备验证；无全局回调与隐式回退。"""
from functools import partial

DISTANCE_BACKENDS = ('serial', 'parallel', 'gpu_defect_signed_distance')


def validate_distance_policy(*, distance_backend, distance_device=None):
    """返回规范设备字符串；CPU 不接设备，GPU 必须显式可用 cuda:N。"""
    if not isinstance(distance_backend, str) or distance_backend not in DISTANCE_BACKENDS:
        raise ValueError('distance_backend must be serial, parallel or gpu_defect_signed_distance')
    if distance_backend != 'gpu_defect_signed_distance':
        if distance_device is not None:
            raise ValueError('distance_device is only valid for GPU distance_backend')
        return None
    import torch
    try:
        target = torch.device(distance_device) if distance_device is not None else None
    except (ValueError, TypeError, RuntimeError) as error:
        raise ValueError('distance_device requires explicit cuda:N') from error
    if target is None or target.type != 'cuda' or target.index is None:
        raise ValueError('distance_device requires explicit cuda:N')
    if not torch.cuda.is_available() or target.index >= torch.cuda.device_count():
        raise RuntimeError('requested CUDA distance device is unavailable; no CPU fallback')
    return str(target)


def resolve_distance_evaluator(*, distance_backend, distance_device=None, distance_evaluator=None):
    """返回成熟核兼容 backend 和显式 callable；既有 CPU callable 保持兼容。"""
    device = validate_distance_policy(distance_backend=distance_backend, distance_device=distance_device)
    if distance_backend != 'gpu_defect_signed_distance':
        return distance_backend, distance_evaluator
    if distance_evaluator is not None:
        raise ValueError('GPU backend owns its evaluator; do not supply distance_evaluator')
    from .topology_gpu_distance import gpu_defect_signed_distance
    return 'serial', partial(gpu_defect_signed_distance, device=device)
