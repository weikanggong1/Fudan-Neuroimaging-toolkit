"""Extract mature FNIT weight and normal callback definitions for source audits.

This utility executes no tensor arithmetic while extracting definitions.
The isolated X0 experiment is described in the adjacent README.
"""
import ast
import copy
import hashlib
from pathlib import Path


def _sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def nodes(registration_path, spline_path):
    reg = ast.parse(Path(registration_path).read_text())
    sp = ast.parse(Path(spline_path).read_text())
    system = next(n for n in reg.body if isinstance(n, ast.ClassDef) and n.name == '_LevelSystem')
    linearize = next(n for n in system.body if isinstance(n, ast.FunctionDef) and n.name == 'linearize')
    callbacks = {n.name: n for n in linearize.body if isinstance(n, ast.FunctionDef) and n.name in ('data_normal', 'bend_normal')}
    start = next(i for i, n in enumerate(linearize.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'spatial_weights' for t in n.targets))
    finish = next(i for i, n in enumerate(linearize.body[start:], start) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cpu_normal' for t in n.targets))
    weights = linearize.body[start:finish]
    funcs = {n.name: n for n in reg.body if isinstance(n, ast.FunctionDef) and n.name in ('_pack', '_unpack')}
    funcs.update({n.name: n for n in sp.body if isinstance(n, ast.FunctionDef) and n.name in ('expand_coefficients', 'adjoint_field')})
    bending = next(n for n in sp.body if isinstance(n, ast.ClassDef) and n.name == 'BendingOperator')
    funcs['bending_normal'] = next(n for n in bending.body if isinstance(n, ast.FunctionDef) and n.name == 'normal')
    return callbacks, weights, funcs


def bindings(registration_path, spline_path):
    callbacks, weights, funcs = nodes(registration_path, spline_path)
    return {**{k: _sha(v) for k, v in callbacks.items()}, 'weight_setup': _sha(ast.Module(body=weights, type_ignores=[])), **{k: _sha(v) for k, v in funcs.items()}}


def compiled(registration_path, spline_path, environment):
    callbacks, weights, funcs = nodes(registration_path, spline_path)
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in ('self', 'coefficients', 'gradient_fsl', 'mask', 'count')], kwonlyargs=[], kw_defaults=[], defaults=[])
    setup = ast.FunctionDef(name='weight_setup', args=args, body=copy.deepcopy(weights) + [ast.Return(value=ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in ('spatial_weights', 'cross_weights', 'scale_weight')], ctx=ast.Load()))], decorator_list=[])
    scope = dict(environment)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[setup], type_ignores=[])), '<mature-weight-setup>', 'exec'), scope)
    weight_setup = scope['weight_setup']
    result = {}
    for name, node in funcs.items():
        local = dict(environment)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])), '<mature-'+name+'>', 'exec'), local)
        result[name] = local[node.name]
    factory = ast.FunctionDef(name='callbacks', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[copy.deepcopy(callbacks[n]) for n in ('data_normal', 'bend_normal')] + [ast.Return(value=ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in ('data_normal', 'bend_normal')], ctx=ast.Load()))], decorator_list=[])
    def callback_factory(callback_environment):
        local = dict(callback_environment)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(factory)], type_ignores=[])), '<mature-normal-callbacks>', 'exec'), local)
        return local['callbacks']()
    return weight_setup, result, callback_factory
