"""CPU 验证边界：保存首个 A/b，随后停止原配准求解。

这是验证候选；使用指定源码身份和用户回调，不改变原数学或 GPU 接口。
"""
import ast
import hashlib
import inspect
from pathlib import Path


class FirstAbCaptured(Exception):
    """Expected private boundary after durable real state is saved by the caller."""


def make_first_ab_cpu_tap(registration_module, *, source_binding, save_state):
    """Return a candidate-only callable; GPU dispatch is deliberately unavailable.

    source_binding supplies exact path, resolved_path, bytes and sha256 of the
    frozen original FNIT registration module. save_state receives live original
    locals once; its finite caller owns Float32 raw writes/provider checkpoint.
    The caller must verify/retain the untouched original module and CPU helper.
    """
    path = Path(registration_module.__file__)
    raw = path.read_bytes()
    actual = dict(path=str(path), resolved_path=str(path.resolve()), bytes=len(raw),
                  sha256=hashlib.sha256(raw).hexdigest())
    if actual != {k: source_binding[k] for k in actual}:
        raise ValueError("different frozen original registration provider")
    original = inspect.unwrap(registration_module.robust_register)
    if Path(inspect.getsourcefile(original)).resolve() != path.resolve():
        raise ValueError("different original function provider")
    source = inspect.getsource(original)
    tree = ast.parse(source, filename=str(path))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "robust_register":
        raise ValueError("unexpected original callable AST")
    function = functions[0]
    # Keep the original inference context when exec creates the tapped callable.
    expected = ast.parse("@torch.inference_mode()\ndef robust_register(): pass\n").body[0].decorator_list
    if [ast.dump(n, include_attributes=False) for n in function.decorator_list] != [
            ast.dump(n, include_attributes=False) for n in expected]:
        raise ValueError("different original torch.inference_mode decorator")
    if any(isinstance(n, ast.Name) and n.id.startswith("_fnit_ab_") for n in ast.walk(function)):
        raise ValueError("boundary names collide with original namespace")
    inserted = []

    class AddBoundary(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    and [n.id if isinstance(n, ast.Name) else None for n in node.targets[0].elts] == ["a", "b", "rows"]
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "analytic_system"):
                inserted.append(node.lineno)
                extra = ast.parse("_fnit_ab_save_state(locals())\nraise _fnit_ab_stop()\n").body
                return [node] + [ast.copy_location(n, node) for n in extra]
            return node

    AddBoundary().visit(function)
    if len(inserted) != 1:
        raise ValueError("original first analytic_system boundary is not unique")
    function.name = "_fnit_ab_original_prefix"
    ast.fix_missing_locations(tree)
    namespace = dict(registration_module.__dict__)
    namespace.update(_fnit_ab_save_state=save_state, _fnit_ab_stop=FirstAbCaptured)
    exec(compile(tree, str(path), "exec"), namespace)
    prefix = namespace[function.name]
    used = False

    def once(source_image, target_image, **parameters):
        nonlocal used
        if used:
            raise RuntimeError("one candidate prefix only; no automatic replay")
        if parameters.get("device") != "cpu" or parameters.get("mode") != "rigid":
            raise ValueError("this private tap only binds the reviewed CPU rigid case")
        used = True
        try:
            prefix(source_image, target_image, **parameters)
        except FirstAbCaptured:
            return dict(boundary="after_first_analytic_system_before_regression",
                        original_math_body_unchanged=True, analytic_system_calls=1,
                        registration_QR_calls=0, optimizer_updates=0, GPU_calls=0)
        raise RuntimeError("first analytic_system boundary was not captured")

    return once
