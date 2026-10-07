"""Compile only source-bound PCG prefixes; never a loop, callback or solve."""
import ast
import __future__
import copy
import hashlib
from pathlib import Path


SLOTS = {
    "current_torch": ("inverse_diagonal", "residual", "preconditioned", "direction", "rz", "rhs_norm", "floor"),
    "owned_cpu": ("right", "weights", "residual", "z", "direction", "rho", "right_norm"),
}


def source_prefix(path, kind):
    name = "preconditioned_conjugate_gradient" if kind == "current_torch" else "preconditioned_conjugate_gradient_cpu"
    tree = ast.parse(Path(path).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    stops = [index for index, node in enumerate(function.body) if isinstance(node, ast.For)]
    if len(stops) != 1:
        raise RuntimeError("bound PCG body no longer has one expected outer loop")
    body = function.body[:stops[0]]
    if any(isinstance(node, (ast.For, ast.While, ast.AsyncFor)) for statement in body for node in ast.walk(statement)):
        raise RuntimeError("loop in supposed finite initialization prefix")
    return function, body


def ast_bindings(optimizer_path, candidate_path):
    paths = {"current_torch": optimizer_path, "owned_cpu": candidate_path}
    return {kind: hashlib.sha256(ast.dump(ast.Module(body=source_prefix(path, kind)[1], type_ignores=[]),
                                         include_attributes=False).encode()).hexdigest()
            for kind, path in paths.items()}


def compile_prefix(path, kind, namespace):
    original, body = source_prefix(path, kind)
    function = copy.deepcopy(original)
    function.name = "_bound_initial_" + kind
    function.decorator_list = []
    function.body = copy.deepcopy(body) + [ast.Return(value=ast.Dict(
        keys=[ast.Constant(value=name) for name in SLOTS[kind]],
        values=[ast.Name(id=name, ctx=ast.Load()) for name in SLOTS[kind]]))]
    functions = []
    if kind == "owned_cpu":
        tree = ast.parse(Path(path).read_text())
        functions.append(copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cpu_vector")))
    module = ast.fix_missing_locations(ast.Module(body=functions + [function], type_ignores=[]))
    space = dict(namespace)
    # Preserve optimizer.py's postponed annotations and make the stdlib-only
    # preparation check independent of the helper interpreter's version.
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), space)
    return space[function.name]
