"""CPU IRLS boundary observer validated on one saved real analytic A/b system.

This is a validation callable, not a production CLI or default solver. It wraps
one exact source provider, retains original arithmetic AST, and records original
median/sum/QR/IRLS results through the caller's save_event callback.
"""
import ast
import hashlib
import inspect
from pathlib import Path


def make_solver_tap(solver_module, *, source_binding, save_event):
    path = Path(solver_module.__file__)
    raw = path.read_bytes()
    actual = dict(path=str(path), resolved_path=str(path.resolve()), bytes=len(raw),
                  sha256=hashlib.sha256(raw).hexdigest())
    if actual != {k: source_binding[k] for k in actual}:
        raise ValueError("different frozen original solver provider")
    selected = []
    for name in ("weighted_qr", "robust_regression"):
        original = inspect.unwrap(getattr(solver_module, name))
        if Path(inspect.getsourcefile(original)).resolve() != path.resolve():
            raise ValueError("different original solver function provider")
        tree = ast.parse(inspect.getsource(original), filename=str(path))
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("expected one original function")
        function = tree.body[0]
        if function.name != name or function.decorator_list:
            raise ValueError("different original standalone solver AST")
        if any(isinstance(n, ast.Name) and n.id.startswith("_fnit_solver_") for n in ast.walk(function)):
            raise ValueError("observer namespace collides with original math")
        selected.append(function)
    original_dump = [ast.dump(f, include_attributes=False) for f in selected]
    insertion_counts = {}
    targets = {"weighted_qr": {"matrix", "rhs", "solution"},
               "robust_regression": {"center", "sigma", "normalized", "weights", "parameters", "current_residual", "w2", "error"}}

    class InsertEvents(ast.NodeTransformer):
        def __init__(self, function_name):
            self.function_name = function_name
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (self.function_name == "robust_regression" and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "sum"):
                label = "sw" if len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == "w2" else "swr"
                insertion_counts[label] = insertion_counts.get(label, 0) + 1
                return ast.copy_location(ast.Call(func=ast.Name(id="_fnit_solver_pass_sum", ctx=ast.Load()),
                    args=[ast.Constant(value=label), node], keywords=[]), node)
            return node
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            names = [x.id for x in node.targets if isinstance(x, ast.Name)]
            if len(names) == 1 and names[0] in targets[self.function_name]:
                label = self.function_name + ":" + names[0]
                insertion_counts[label] = insertion_counts.get(label, 0) + 1
                # Skip initial clone before the IRLS loop, not a numerical event.
                if label == "robust_regression:current_residual" and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "clone":
                    return node
                event = ast.parse("_fnit_solver_save(" + repr(label) + ", locals())").body[0]
                return [node, ast.copy_location(event, node)]
            return node
        def visit_For(self, node):
            node = self.generic_visit(node)
            if self.function_name == "robust_regression" and isinstance(node.target, ast.Name) and node.target.id == "iteration":
                event = ast.parse("_fnit_solver_save('iteration_begin', locals())").body[0]
                node.body.insert(0, ast.copy_location(event, node))
            return node
        def visit_Return(self, node):
            if self.function_name == "robust_regression":
                event = ast.parse("_fnit_solver_save('selected_final_state', locals())").body[0]
                return [ast.copy_location(event, node), node]
            return node

    for function in selected:
        InsertEvents(function.name).visit(function)
    expected = {"weighted_qr:matrix": 1, "weighted_qr:rhs": 1, "weighted_qr:solution": 1,
                "robust_regression:center": 1, "robust_regression:sigma": 1,
                "robust_regression:normalized": 1, "robust_regression:weights": 2,
                "robust_regression:parameters": 1, "robust_regression:current_residual": 2,
                "robust_regression:w2": 1, "robust_regression:error": 1, "sw": 1, "swr": 1}
    if insertion_counts != expected:
        raise ValueError("different original solver boundaries")

    class StripEvents(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "_fnit_solver_pass_sum":
                return self.visit(node.args[1])
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_fnit_solver_save":
                return None
            return node

    import copy
    restored = [StripEvents().visit(copy.deepcopy(f)) for f in selected]
    if [ast.dump(f, include_attributes=False) for f in restored] != original_dump:
        raise ValueError("observer changed an original mathematical AST")
    namespace = dict(solver_module.__dict__)
    namespace["_fnit_solver_save"] = save_event
    def pass_sum(label, result):
        save_event(label, {"result": result})  # actual original sum, evaluated once
        return result
    namespace["_fnit_solver_pass_sum"] = pass_sum
    original_median = solver_module.median_even
    def observed_median(values):
        result = original_median(values)  # original callable each time; no extra median
        save_event("median_return", {"result": result})
        return result
    namespace["median_even"] = observed_median
    tree = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    observed = namespace["robust_regression"]
    used = False
    def once(design, residual, *, saturation=50.):
        nonlocal used
        if used:
            raise RuntimeError("one real standalone CPU solve only")
        if str(design.device) != "cpu" or str(residual.device) != "cpu" or saturation != 50.:
            raise ValueError("only the frozen CPU sat50 profile")
        used = True
        return observed(design, residual, saturation=saturation)
    return once
