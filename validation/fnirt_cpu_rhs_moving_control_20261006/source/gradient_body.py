"""Compile bound FNIT gradient statements; never create a normal callback.

This module uses only the standard library. Compilation creates function
definitions; scientific calls require the separately approved worker.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path


def identity(nodes):
    tree = ast.Module(body=nodes, type_ignores=[])
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def assigned_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def method(source, name):
    tree = ast.parse(Path(source).read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_LevelSystem")
    return next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)


def sections(registration_source, assembly_source):
    evaluation = method(registration_source, "evaluate")
    start = next(i for i, node in enumerate(evaluation.body) if assigned_name(node) == "count")
    end = next(i for i, node in enumerate(evaluation.body) if assigned_name(node) == "ssd")
    scalar_body = evaluation.body[start:end + 1]
    if [assigned_name(node) for node in scalar_body] != ["count", None, "scaled_fixed", "residual", "weight", "ssd"]:
        raise RuntimeError("mature residual/SSD statement sequence changed")
    projection = next(node for node in evaluation.body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Name) and node.test.id == "derivatives")
    if len(projection.body) != 1:
        raise RuntimeError("mature derivative projection changed")

    linear = method(registration_source, "linearize")
    if assigned_name(linear.body[0]) != "state":
        raise RuntimeError("mature linearize entry changed")
    stop = next(i for i, node in enumerate(linear.body) if assigned_name(node) == "gradient")
    lm_body = linear.body[1:stop + 1]
    if any(isinstance(node, ast.FunctionDef) and node.name not in ("adjoint", "bend_normal") for node in lm_body):
        raise RuntimeError("unexpected gradient-only closure")
    if any(assigned_name(node) in ("spatial_weights", "cross_weights", "cpu_normal") for node in lm_body):
        raise RuntimeError("normal-system statement entered gradient prefix")

    tree = ast.parse(Path(assembly_source).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    first = next(i for i, node in enumerate(main.body) if assigned_name(node) == "count")
    last = next(i for i, node in enumerate(main.body) if assigned_name(node) == "fsl_order_gradient")
    fsl_body = main.body[first:last + 1]
    if [assigned_name(node) for node in fsl_body] != ["count", "mask", "residual", "jte_product", "coefficient_gradient", None, "scale_gradient", "fsl_order_gradient"]:
        raise RuntimeError("saved FSL-order control sequence changed")
    adapter = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "fixed_evaluate")
    cost = next(node for node in adapter.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].slice, ast.Constant) and node.targets[0].slice.value == "cost")
    return {"scalar_body": scalar_body, "projection_body": projection.body,
            "lm_gradient_body": lm_body, "fsl_order_body": fsl_body, "fixed_lambda_cost_body": [cost]}


def bindings(registration_source, assembly_source):
    bodies = sections(registration_source, assembly_source)
    tree = ast.parse(Path(registration_source).read_text())
    coordinates = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "_fsl_displacement_coordinates")
    return {name: identity(body) for name, body in bodies.items()} | {
        "coordinate_function": identity([coordinates]),
        "lm_closure_order": [node.name for node in bodies["lm_gradient_body"] if isinstance(node, ast.FunctionDef)],
        "excluded": ["evaluate entry", "spatial_weights", "cross_weights", "cpu_normal", "data_normal", "matvec", "diagonal"]}


def factory(name, arguments, body, returned):
    node = ast.FunctionDef(name=name,
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=arg) for arg in arguments],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=copy.deepcopy(body) + [ast.Return(value=ast.parse(returned, mode="eval").body)],
        decorator_list=[])
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def compiled(registration_source, assembly_source, namespace):
    bodies = sections(registration_source, assembly_source)
    specifications = {
        "state_scalars": (("self", "scale", "warped", "mask"), "scalar_body", "(count, scaled_fixed, residual, ssd)"),
        "project_derivatives": (("state", "self", "gradient_voxels"), "projection_body", "state['gradient_fsl']"),
        "lm_gradient": (("self", "coefficients", "scale", "state"), "lm_gradient_body", "gradient"),
        "fsl_order_gradient": (("system", "coefficients", "state", "target_lambda"), "fsl_order_body", "(fsl_order_gradient, jte_product, scale_gradient)"),
        "fixed_lambda_cost": (("result", "target_lambda"), "fixed_lambda_cost_body", "result['cost']"),
    }
    output = {}
    for name, (arguments, key, returned) in specifications.items():
        tree = factory(name, arguments, bodies[key], returned)
        local = dict(namespace)
        exec(compile(tree, str(registration_source) + ":gradient-only/" + name, "exec"), local)
        output[name] = local[name]
    return output
