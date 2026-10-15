"""GA 状态后端显式接合；旧独立基线保留 legacy，packed 缺实现明确失败。"""
import inspect


def validate_state_backend(state_backend):
    if not isinstance(state_backend,str)or state_backend not in ('legacy','packed'):
        raise ValueError('state_backend must be legacy or packed')
    return state_backend


def state_backend_arguments(*,state_backend,target):
    """只给兼容显式入口传参；packed 不允许退回旧函数。"""
    validate_state_backend(state_backend)
    if 'state_backend' in inspect.signature(target).parameters:
        return {'state_backend':state_backend}
    if state_backend=='packed':
        raise RuntimeError('packed state_backend requires the explicit task5 state API; no legacy fallback')
    return {}
