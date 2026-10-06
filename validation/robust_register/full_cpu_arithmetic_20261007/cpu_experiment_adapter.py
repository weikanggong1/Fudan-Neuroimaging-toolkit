"""独立 CPU robust-register 实验入口；复用已验收的保序计算候选。

先调用 load_cpu_candidate()，随后复用返回的同一实例处理多个被试。每个实例
拥有独立的实验包命名空间及进程内 JIT 缓存；实例中的调用按顺序执行。用
with 或 close() 释放该实例的实验模块；关闭后不得再次调用。此文件
不提供 GPU 接口，不修改 FNIT 默认模块、GEMS、线程数或 TF32 全局设置。

完整精度记录仍限于 README 中的一例 rigid6→affine12 真实输入；不能由此
推断未知形状、13 参数、秩亏、不同强度分布或 GPU 的等价性。
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from threading import RLock
from uuid import uuid4


_BASELINE = "validation/robust_register/rigid_affine_20261006/candidate_source/robust_register"
_INVERSE = "validation/robust_register/inverse_real_20261006/candidate_overlay"
_HELPERS = "validation/robust_register/same_ab_irls_cpu_20261007/source"
_CENTROID = "validation/robust_register/centroid_serial_cpu_probe_20261006/centroid_serial.py"
_SOLVER_SHA256 = "2b312f6bc32852c24df7e5323a89a5cc9400a9be6f15090e36370bc0f197b933"


def _load_module(name, filename, *, search_paths=None):
    specification = importlib.util.spec_from_file_location(
        name, filename, submodule_search_locations=search_paths)
    if specification is None or specification.loader is None:
        raise ImportError("cannot load the isolated CPU experiment module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _make_cpu_solver(solver, helpers):
    """改动三个 CPU 分派位置；移除分派后须还原完整原函数 AST。"""
    path = Path(solver.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != _SOLVER_SHA256:
        raise ValueError("the original experimental solver version differs")
    selected = []
    for name in ("weighted_qr", "robust_regression"):
        original = inspect.unwrap(getattr(solver, name))
        if Path(inspect.getsourcefile(original)).resolve() != path:
            raise ValueError("the original solver function provider differs")
        tree = ast.parse(inspect.getsource(original), filename=str(path))
        if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
                or tree.body[0].name != name or tree.body[0].decorator_list):
            raise ValueError("expected one undecorated original solver function")
        if any(isinstance(node, ast.Name) and node.id.startswith("_fnit_cpu_")
               for node in ast.walk(tree.body[0])):
            raise ValueError("CPU dispatch namespace collides with original source")
        selected.append(tree.body[0])
    original_dump = [ast.dump(function, include_attributes=False) for function in selected]
    counts = {"qr": 0, "residual": 0, "error": 0}

    class CPUDispatch(ast.NodeTransformer):
        def __init__(self, function_name):
            self.function_name = function_name

        def visit_Assign(self, node):
            if (self.function_name == "weighted_qr"
                    and ast.dump(node.targets[0], include_attributes=False)
                    == ast.dump(ast.parse("q, r = None").body[0].targets[0],
                                include_attributes=False)):
                if ast.unparse(node.value) != "torch.linalg.qr(matrix, mode='reduced')":
                    raise ValueError("the original reduced QR site differs")
                branch = ast.parse("""if design.device.type == 'cpu':
    _fnit_cpu_value = _fnit_cpu_qr(matrix, rhs)
    if _fnit_cpu_value is not None:
        solution = _fnit_cpu_value
        if not bool(torch.isfinite(solution).all()):
            raise ValueError('weighted QR returned a nonfinite step')
        return solution
""").body[0]
                branch._fnit_cpu_overlay = "qr"
                counts["qr"] += 1
                return [ast.copy_location(branch, node), node]
            if (self.function_name != "robust_regression" or len(node.targets) != 1
                    or not isinstance(node.targets[0], ast.Name)):
                return node
            target = node.targets[0].id
            if (target == "current_residual" and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Sub)):
                if ast.unparse(node.value) != "residual - design @ parameters":
                    raise ValueError("the original residual site differs")
                kind, call = "residual", "_fnit_cpu_residual(design, residual, parameters)"
            elif target == "error":
                kind, call = "error", "_fnit_cpu_error(current_residual, weights)"
            else:
                return node
            branch = ast.parse(
                "if design.device.type == 'cpu':\n    _fnit_cpu_value = " + call
                + "\n    if _fnit_cpu_value is None:\n        pass\n    else:\n        "
                + target + " = _fnit_cpu_value\nelse:\n    pass").body[0]
            branch.body[1].body = [copy.deepcopy(node)]
            branch.orelse = [copy.deepcopy(node)]
            branch._fnit_cpu_overlay = kind
            branch._fnit_cpu_original_assignment = copy.deepcopy(node)
            counts[kind] += 1
            return ast.copy_location(branch, node)

    for function in selected:
        CPUDispatch(function.name).visit(function)
    if counts != {"qr": 1, "residual": 1, "error": 1}:
        raise ValueError("CPU dispatch must change exactly three original sites")

    class StripCPUDispatch(ast.NodeTransformer):
        def visit_If(self, node):
            kind = getattr(node, "_fnit_cpu_overlay", None)
            if kind == "qr":
                return None
            if kind in ("residual", "error"):
                original = node._fnit_cpu_original_assignment
                expected = ast.dump(original, include_attributes=False)
                if (ast.dump(node.body[1].body[0], include_attributes=False) != expected
                        or ast.dump(node.orelse[0], include_attributes=False) != expected):
                    raise ValueError("CPU fallback or original GPU expression differs")
                return copy.deepcopy(original)
            return self.generic_visit(node)

    restored = [StripCPUDispatch().visit(copy.deepcopy(function)) for function in selected]
    if [ast.dump(function, include_attributes=False) for function in restored] != original_dump:
        raise ValueError("the CPU adapter changed unrelated original mathematics")
    namespace = dict(solver.__dict__)
    namespace.update(_fnit_cpu_qr=helpers["qr"], _fnit_cpu_residual=helpers["residual"],
                     _fnit_cpu_error=helpers["error"])
    tree = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    candidate, original = namespace["robust_regression"], solver.robust_regression

    def cpu_regression(design, residual, *, saturation=50.):
        # The original callable stays independent of all new CPU controls.
        if design.device.type != "cpu" or residual.device.type != "cpu":
            return original(design, residual, saturation=saturation)
        if saturation != 50. or design.shape[1] not in (6, 12):
            return original(design, residual, saturation=saturation)
        result = candidate(design, residual, saturation=saturation)
        result.report["baseline_solver_metadata"] = result.report["solver"]
        result.report["solver"] = "CPU_Float_LINPACK_candidate_with_original_guard_fallback"
        result.report["CPU_arithmetic_candidate"] = {
            "configured": True, "individual_helper_use_traced": False,
            "unsupported_None_uses_original_expression": True,
            "rank_deficient_helper_ValueError_preserved": True,
        }
        return result

    return cpu_regression


class CPURegistrationCandidate:
    """同实例复用 CPU 配准函数和 JIT 缓存；每个实例的调用串行执行。"""

    def __init__(self, repository_root):
        self.repository_root = Path(repository_root).resolve()
        self._namespace = "fnit._robust_cpu_arithmetic_experiment_" + uuid4().hex
        self._lock = RLock()
        self._closed = False
        self._centroid_module = None
        import fnit
        expected_fnit = self.repository_root / "src/fnit/__init__.py"
        if Path(fnit.__file__).resolve() != expected_fnit.resolve():
            raise ImportError("import fnit from this checkout before loading the CPU candidate")
        baseline = self.repository_root / _BASELINE
        overlay = self.repository_root / _INVERSE
        try:
            self._package = _load_module(
                self._namespace, baseline / "__init__.py",
                search_paths=[str(overlay), str(baseline)])
            self._registration = __import__(self._namespace + ".registration", fromlist=["registration"])
            solver = __import__(self._namespace + "._solver", fromlist=["_solver"])
            helpers = {}
            for key, filename, callable_name in (
                ("qr", "cpu_linpack_qr_candidate.py", "try_cpu_float_linpack"),
                ("residual", "cpu_ordered_residual_candidate.py", "try_cpu_ordered_residual"),
                ("error", "cpu_weighted_error_candidate.py", "try_serial_weighted_error"),
            ):
                module = _load_module(self._namespace + "._CPU_" + key,
                                     self.repository_root / _HELPERS / filename)
                helpers[key] = getattr(module, callable_name)
            self._original_regression = self._registration.robust_regression
            self._registration.robust_regression = _make_cpu_solver(solver, helpers)
            self._original_centroid = self._registration._centroid
            self._registration._centroid = self._cpu_centroid
        except BaseException:
            for name in tuple(sys.modules):
                if name == self._namespace or name.startswith(self._namespace + "."):
                    sys.modules.pop(name, None)
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("the CPU experiment instance has been closed")

    def close(self):
        """释放本实例的命名空间和 helper/JIT 引用，不改其他实例或默认模块。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._registration._centroid = self._original_centroid
            self._registration.robust_regression = self._original_regression
            for name in tuple(sys.modules):
                if name == self._namespace or name.startswith(self._namespace + "."):
                    sys.modules.pop(name, None)
            self._centroid_module = None
            self._package = None
            self._registration = None
            self._original_centroid = None
            self._original_regression = None

    def __enter__(self):
        with self._lock:
            self._ensure_open()
            return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        return False

    def _cpu_centroid(self, values):
        torch, np = self._registration.torch, self._registration.np
        original = self._original_centroid
        if isinstance(values, torch.Tensor) and values.device.type != "cpu":
            return original(values)
        if (type(values) is not torch.Tensor or values.dtype != torch.float32
                or values.ndim != 3 or values.layout != torch.strided
                or values.is_nested or values.requires_grad
                or values.is_neg() or values.is_conj() or sys.byteorder != "little"
                or torch._C._functorch.is_functorch_wrapped_tensor(values)
                or torch.autograd.forward_ad.unpack_dual(values).tangent is not None):
            return original(values)
        if not bool(torch.isfinite(values).all()):
            return original(values)
        mass = values.double()
        total = mass.sum()
        if float(total) <= 0:
            return original(values)
        packed = np.array(values.detach().numpy().ravel(order="F"),
                          dtype=np.float32, copy=True)
        packed.setflags(write=False)
        if self._centroid_module is None:
            self._centroid_module = _load_module(
                self._namespace + "._CPU_centroid", self.repository_root / _CENTROID)
        result = self._centroid_module.centroid_serial(
            packed, *map(int, values.shape), np.float64(0.0))
        return np.asarray(result, dtype=np.float64)

    def _cpu_device(self, device):
        selected = self._registration.torch.device(device)
        if selected.type != "cpu":
            raise ValueError("this independent experiment accepts only device='cpu'")
        return selected

    def cpu_robust_register(self, source, target, *, device="cpu", **parameters):
        """单阶段；其他参数与原实验 robust_register 完全相同。"""
        with self._lock:
            self._ensure_open()
            selected = self._cpu_device(device)
            result = self._registration.robust_register(
                source, target, device=selected, **parameters)
            result.report["CPU_experimental_adapter"] = {
                "default_backend_replaced": False, "GPU_interface": False,
                "JIT_modules_reused_within_this_instance": True,
                "input_specific_native_equivalence": "not_assessed_by_this_call",
            }
            return result

    def cpu_robust_rigid_affine(self, source, target, *, stage_directory,
                               device="cpu", **parameters):
        """Rigid→MGH保存重载→affine；stage_directory 必须不存在。"""
        with self._lock:
            self._ensure_open()
            selected = self._cpu_device(device)
            result = self._registration.robust_rigid_affine(
                source, target, stage_directory=stage_directory,
                device=selected, **parameters)
            for arm in ("rigid", "affine"):
                result[arm].report["CPU_experimental_adapter"] = {
                    "default_backend_replaced": False, "GPU_interface": False,
                    "JIT_modules_reused_within_this_instance": True,
                    "input_specific_native_equivalence": "not_assessed_by_this_call",
                }
            return result


def load_cpu_candidate(repository_root=None):
    """加载一次，返回可重复调用的 CPURegistrationCandidate。

    repository_root 是包含 src/fnit 与 validation 的 FNIT checkout 根目录。
    从本文件正式发布位置导入时可省略；从外部路径加载草稿必须显式传入。
    应先用该 checkout 的 editable 安装或 PYTHONPATH 导入 fnit。每实例各自
    缓存三个 helper 模块及质心 JIT；不在每个被试调用中重新创建实例。
    推荐 with load_cpu_candidate(...) as candidate，退出时只关闭该实例。
    """
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[3]
    return CPURegistrationCandidate(root)
