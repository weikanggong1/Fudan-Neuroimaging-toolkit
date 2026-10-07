"""Read the mature diagonal suffix and add identity observation taps.

This diagnostic helper does not evaluate or sample a registration state.
Its caller supplies verified CPU operands and the mature source environment.
"""
import ast
import copy
import hashlib


def _dump(value):
    return ast.dump(value, include_attributes=False)


def _source_parts(registration_path, spline_path):
    reg = ast.parse(open(registration_path).read())
    spline = ast.parse(open(spline_path).read())
    level = next(n for n in reg.body if isinstance(n, ast.ClassDef) and n.name == '_LevelSystem')
    linear = next(n for n in level.body if isinstance(n, ast.FunctionDef) and n.name == 'linearize')
    start = next(i for i, n in enumerate(linear.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'diagonal_parts' for t in n.targets))
    tail = copy.deepcopy(linear.body[start:-1])
    if not (isinstance(linear.body[-1], ast.Return) and isinstance(tail[-1], ast.Assign)
            and isinstance(tail[-1].targets[0], ast.Name) and tail[-1].targets[0].id == 'diagonal'):
        raise ValueError('mature diagonal suffix changed')
    design = next(n for n in spline.body if isinstance(n, ast.FunctionDef) and n.name == 'design_diagonal')
    pack = next(n for n in reg.body if isinstance(n, ast.FunctionDef) and n.name == '_pack')
    return tail, copy.deepcopy(design), pack


def bindings(registration_path, spline_path):
    tail, design, pack = _source_parts(registration_path, spline_path)
    return {'diagonal_suffix': hashlib.sha256(_dump(ast.Module(body=tail, type_ignores=[])).encode()).hexdigest(),
            'design_diagonal': hashlib.sha256(_dump(design).encode()).hexdigest(),
            'pack': hashlib.sha256(_dump(pack).encode()).hexdigest(),
            'normal_evaluate_gradient_bodies_excluded': True}


class _Taps(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'torch' and node.func.attr == 'stack'
                and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == 'diagonal_parts'):
            return ast.Call(func=ast.Name(id='capture', ctx=ast.Load()),
                            args=[ast.Constant('candidate_data_half_xyz'), node], keywords=[])
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name) and node.left.id == 'bend_factor'
                and isinstance(node.right, ast.Subscript) and isinstance(node.right.value, ast.Call)
                and isinstance(node.right.value.func, ast.Attribute) and node.right.value.func.attr == 'diagonal'):
            return ast.Call(func=ast.Name(id='capture', ctx=ast.Load()),
                            args=[ast.Constant('candidate_regularizer_half_shared_xyz'), node], keywords=[])
        return node


def compiled(registration_path, spline_path, environment):
    tail, design, _ = _source_parts(registration_path, spline_path)
    design_module = ast.fix_missing_locations(ast.Module(body=[design], type_ignores=[]))
    design_env = dict(environment); exec(compile(design_module, '<bound mature design_diagonal>', 'exec'), design_env)
    function = ast.parse('def mature_diagonal_only(self, coefficients, gradient_fsl, mask, count, bend_factor):\n    pass\n').body[0]
    function.body = [_Taps().visit(n) for n in tail]
    function.body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id='diagonal_scale', ctx=ast.Load()),
                                                         ast.Name(id='diagonal', ctx=ast.Load())], ctx=ast.Load())))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    diagonal_env = dict(environment); exec(compile(module, '<bound mature diagonal suffix>', 'exec'), diagonal_env)
    return design_env['design_diagonal'], diagonal_env['mature_diagonal_only']
