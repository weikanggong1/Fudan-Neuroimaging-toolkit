"""Restore immutable arrays and compile only the bound mature callback bodies.

No FNIRT evaluation, linearization, geometry calculation or solver is called.
Scientific dependencies are supplied by the authorized worker after binding.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace


CALLBACK_ORDER = ("bend_normal", "data_normal", "matvec")
CONTEXT_NAMES = ("self", "coefficients", "coefficient_shape", "bend_factor",
                 "spatial_weights", "cross_weights", "scale_weight", "count", "cpu_normal")
FREE_VARIABLES = {
    "bend_normal": ("bend_factor", "coefficient_shape", "coefficients", "self"),
    "data_normal": ("coefficient_shape", "count", "cpu_normal", "cross_weights", "scale_weight", "self", "spatial_weights"),
    "matvec": ("bend_normal", "data_normal"),
}


def ast_identity(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def callback_nodes(source_path):
    module = ast.parse(Path(source_path).read_text())
    klass = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "_LevelSystem")
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "linearize")
    functions = [node for node in method.body if isinstance(node, ast.FunctionDef) and node.name in CALLBACK_ORDER]
    if tuple(node.name for node in functions) != CALLBACK_ORDER:
        raise RuntimeError("mature callback definition order changed")
    return functions


def csc_node(source_path):
    tree = ast.parse(Path(source_path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "column_matvec")
    # The body is reused exactly; the probe compiles without writing a cache.
    node.decorator_list = []
    return node


def source_ast_binding(registration_source, csc_source):
    return {"callback_order": list(CALLBACK_ORDER),
            "callbacks": {node.name: ast_identity(node) for node in callback_nodes(registration_source)},
            "strict_csc_body_without_decorator": ast_identity(csc_node(csc_source)),
            "free_variables": {name: list(values) for name, values in FREE_VARIABLES.items()}}


def compile_callbacks(source_path, context, registration):
    assignments = [ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                              value=ast.Subscript(value=ast.Name(id="context", ctx=ast.Load()),
                                                  slice=ast.Constant(value=name), ctx=ast.Load()))
                   for name in CONTEXT_NAMES]
    factory = ast.FunctionDef(
        name="_restored_callback_factory",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="context")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=assignments + callback_nodes(source_path) +
             [ast.Return(value=ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Load()) for name in CALLBACK_ORDER], ctx=ast.Load()))],
        decorator_list=[],
    )
    tree = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    namespace = dict(registration.__dict__)
    exec(compile(tree, str(source_path), "exec"), namespace)
    functions = namespace[factory.name](context)
    for name, function in zip(CALLBACK_ORDER, functions):
        if function.__code__.co_freevars != FREE_VARIABLES[name]:
            raise RuntimeError("restored closure free-variable order changed: " + name)
    return functions


def compile_csc(source_path, np, njit):
    tree = ast.fix_missing_locations(ast.Module(body=[csc_node(source_path)], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return njit(cache=False, fastmath=False)(namespace["column_matvec"])


def tensor_value_hash(tensor, np):
    array = tensor.detach().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def restore_tensors(checkpoint_path, metadata, np, torch):
    tensors, layouts = {}, {}
    with np.load(checkpoint_path, allow_pickle=False) as packed:
        if set(packed.files) != set(metadata):
            raise RuntimeError("checkpoint schema mismatch")
        for name in metadata:
            value, record = packed[name], metadata[name]
            if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
                raise RuntimeError("checkpoint shape/dtype mismatch: " + name)
            if not np.isfinite(value).all():
                raise RuntimeError("checkpoint nonfinite: " + name)
            itemsize = value.dtype.itemsize
            if any(byte_stride < 0 or byte_stride % itemsize for byte_stride in record["strides"]):
                raise RuntimeError("checkpoint stride cannot be restored: " + name)
            strides = tuple(byte_stride // itemsize for byte_stride in record["strides"])
            source = torch.from_numpy(value.copy(order="K"))
            restored = torch.empty_strided(tuple(value.shape), strides, dtype=source.dtype, device="cpu")
            restored.copy_(source)
            restored_bytes = list(stride * itemsize for stride in restored.stride())
            expected_hash = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            if restored_bytes != record["strides"] or tensor_value_hash(restored, np) != expected_hash:
                raise RuntimeError("checkpoint logical values/layout failed to restore: " + name)
            if restored.requires_grad or restored.device.type != "cpu":
                raise RuntimeError("checkpoint device/grad policy: " + name)
            tensors[name] = restored
            layouts[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                             "npz_loaded_byte_strides": list(value.strides),
                             "recorded_byte_strides": record["strides"],
                             "restored_byte_strides": restored_bytes,
                             "logical_value_sha256": expected_hash,
                             "restored_values_bitexact": True}
    # Every saved array is copied into separate storage. The original alias
    # graph is not claimed to survive NPZ; callbacks only read these operands.
    pointers = [value.untyped_storage().data_ptr() for value in tensors.values() if value.numel()]
    if len(set(pointers)) != len(pointers):
        raise RuntimeError("checkpoint arrays unexpectedly share mutable storage")
    return tensors, layouts


def restored_arm(kind, tensors, stage1, registration):
    if kind not in ("optimized", "reference"):
        raise ValueError("unknown controlled callback arm")
    from fnit.fnirt.spline import BendingOperator
    geometry = stage1["geometry"]
    checkpoint = stage1["checkpoint"]
    if checkpoint["normal_cache"] != {"packed_layout": None, "scratch": None, "layout_copy_bytes": 0}:
        raise RuntimeError("stage1 unexpectedly performed a packed callback")
    bending = BendingOperator.__new__(BendingOperator)
    bending.execution = "optimized"
    bending.shape = tuple(geometry["level_shape"])
    bending.knot_spacing = tuple(geometry["knot_spacing"])
    bending.voxel_sizes = tuple(geometry["level_voxel_sizes"])
    bending.control_shape = tuple(geometry["coefficient_shape"])
    multipliers = checkpoint["bending_multipliers"]
    bending.operators = tuple((tuple(tensors[f"bending_basis_{term}_{axis}"] for axis in range(3)), multiplier)
                              for term, multiplier in enumerate(multipliers))
    bending._normal_grams = tuple((tuple(tensors[f"bending_gram_{term}_{axis}"] for axis in range(3)), multiplier)
                                  for term, multiplier in enumerate(multipliers))
    bending._diagonal = tensors["bending_diagonal"]
    state = checkpoint["state_scalars"]
    coefficients = tensors["coefficients"]
    if coefficients.dtype != registration.torch.float64 or tuple(coefficients.shape[1:]) != bending.control_shape:
        raise RuntimeError("bound coefficients are not the declared F64 control grid")
    self = SimpleNamespace(bases=tuple(tensors[f"basis_{axis}"] for axis in range(3)),
                           bending=bending, fixed=tensors["fixed"], estimate_scale=True)
    weights = tuple(tuple(tensors[f"spatial_weight_{row}_{column}"] for column in range(3)) for row in range(3))
    cross = tuple(tensors[f"cross_weight_{axis}"] for axis in range(3))
    cpu_normal = None
    if kind == "optimized":
        from fnit.fnirt._normal_cpu import SpatialNormalCPU
        cpu_normal = SpatialNormalCPU(weights, cross, state["count"])
        if cpu_normal.scratch is not None or cpu_normal._layout is not None:
            raise RuntimeError("new CPU operator unexpectedly has a packed cache")
    context = {"self": self, "coefficients": coefficients, "coefficient_shape": bending.control_shape,
               "bend_factor": state["effective_lambda"] / state["count"],
               "spatial_weights": weights, "cross_weights": cross,
               "scale_weight": tensors["scale_weight"], "count": state["count"], "cpu_normal": cpu_normal}
    functions = compile_callbacks(registration.__file__, context, registration)
    return functions[-1], cpu_normal, bending
