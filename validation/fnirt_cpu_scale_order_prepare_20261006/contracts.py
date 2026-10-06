"""Standard-library immutable bindings for one scalar-only control."""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct


def bound(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'bytes': int(path.stat().st_size), 'sha256': digest.hexdigest()}


def write_json(path, value):
    def builtin(item):
        if type(item) in (str, int, float, bool, type(None)):
            return
        if type(item) is list:
            for child in item:
                builtin(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                builtin(child)
            return
        raise TypeError('JSON values must be built-in scalar/list/dict types')
    builtin(value)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_paths(root, expected):
    result = {}
    for name, record in expected['files'].items():
        relative = Path(record['relative_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'invalid input path')
        result[name] = Path(root) / relative
    return result


def check_bindings(root, expected):
    values, failures = {}, []
    for section in ('production_source', 'source_extra'):
        for relative, record in expected[section].items():
            key = section + '/' + relative
            values[key] = bound(Path(root) / 'repo' / relative)
            if values[key] != record:
                failures.append(key)
    for name, path in file_paths(root, expected).items():
        key = 'input/' + name
        values[key] = bound(path)
        if values[key] != {part: expected['files'][name][part] for part in ('bytes', 'sha256')}:
            failures.append(key)
    return values, failures


def check_freeze(workspace):
    workspace = Path(workspace)
    freeze = json.loads((workspace / 'freeze.public.json').read_text())
    values = {name: bound(workspace / name) for name in freeze['files']}
    return values, [name for name in values if values[name] != freeze['files'][name]]


def git_head(repo):
    git = Path(repo) / '.git'
    if git.is_file():
        git = (Path(repo) / git.read_text().strip().split(':', 1)[1].strip()).resolve()
    common = git if not (git / 'commondir').is_file() else (git / (git / 'commondir').read_text().strip()).resolve()
    value = (git / 'HEAD').read_text().strip()
    if value.startswith('ref: '):
        ref = value[5:]
        if (common / ref).is_file():
            value = (common / ref).read_text().strip()
        else:
            rows = [line.split()[0] for line in (common / 'packed-refs').read_text().splitlines()
                    if line and not line.startswith(('#', '^')) and line.split()[1] == ref]
            require(len(rows) == 1, 'unresolved canonical reference')
            value = rows[0]
    require(len(value) == 40 and all(v in '0123456789abcdef' for v in value), 'invalid HEAD')
    return value


def source_baselines(root, namespace):
    registration = Path(root) / 'repo/src/fnit/fnirt/registration.py'
    assembly = Path(root) / 'repo/validation/fnirt_shared_followup_20261006/source/assembly_shared_v2.py'
    tree = ast.parse(registration.read_text())
    level = next(v for v in tree.body if isinstance(v, ast.ClassDef) and v.name == '_LevelSystem')
    evaluate = next(v for v in level.body if isinstance(v, ast.FunctionDef) and v.name == 'evaluate')
    ssd = next(v for v in evaluate.body if isinstance(v, ast.Assign)
               and len(v.targets) == 1 and isinstance(v.targets[0], ast.Name) and v.targets[0].id == 'ssd')
    main = next(v for v in ast.parse(assembly.read_text()).body if isinstance(v, ast.FunctionDef) and v.name == 'main')
    scale = next(v for v in main.body if isinstance(v, ast.Assign)
                 and len(v.targets) == 1 and isinstance(v.targets[0], ast.Name) and v.targets[0].id == 'scale_gradient')
    require(isinstance(ssd.value, ast.BinOp) and isinstance(ssd.value.left, ast.Call), 'SSD source expression schema')
    require(isinstance(scale.value, ast.BinOp) and isinstance(scale.value.left, ast.UnaryOp)
            and isinstance(scale.value.left.operand, ast.Call), 'scale source expression schema')
    SSD_product = ssd.value.left.func.value
    scale_product = scale.value.left.operand.func.value
    identities = {name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                  for name, node in [('baseline_ssd', ssd), ('baseline_scale_half', scale),
                                     ('SSD_FP32_product', SSD_product), ('scale_FP32_product', scale_product)]}
    output = {}
    # Product expressions are extracted unchanged and computed once. Only the
    # repeated expression in each baseline reduction is replaced by this exact
    # shared operand; no product, dtype, normalization or reduction is changed.
    for name, original, returned in [('baseline_ssd', ssd, 'ssd'),
                                      ('baseline_scale_half', scale, 'scale_gradient')]:
        node = copy.deepcopy(original)
        call = node.value.left if name == 'baseline_ssd' else node.value.left.operand
        call.func.value = ast.Name(id='product', ctx=ast.Load())
        function = ast.FunctionDef(name=name,
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=v) for v in ['product', 'count']], kwonlyargs=[],
                               kw_defaults=[], defaults=[]),
            body=[node, ast.Return(value=ast.Name(id=returned, ctx=ast.Load()))], decorator_list=[])
        code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        local = dict(namespace)
        exec(compile(code, str(registration) + ':scalar-only/' + name, 'exec'), local)
        output[name] = local[name]
    function = ast.FunctionDef(name='shared_FP32_products',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=v) for v in ['system', 'residual', 'mask', 'weight']],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Return(value=ast.Tuple(elts=[copy.deepcopy(scale_product), copy.deepcopy(SSD_product)], ctx=ast.Load()))],
        decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    local = dict(namespace)
    exec(compile(code, str(registration) + ':scalar-only/products', 'exec'), local)
    output['shared_FP32_products'] = local['shared_FP32_products']
    return output, identities


def fixed_header(path, expected):
    with gzip.open(path, 'rb') as stream:
        header = stream.read(348)
    require(len(header) == 348 and struct.unpack_from('<i', header, 0)[0] == 348, 'reference header schema')
    record = {'header348_bytes': len(header), 'header348_sha256': hashlib.sha256(header).hexdigest(),
              'shape': [int(v) for v in struct.unpack_from('<8h', header, 40)[1:4]],
              'qform_code': int(struct.unpack_from('<h', header, 252)[0]),
              'sform_code': int(struct.unpack_from('<h', header, 254)[0]),
              'srow_x': [float(v) for v in struct.unpack_from('<4f', header, 280)],
              'srow_y': [float(v) for v in struct.unpack_from('<4f', header, 296)],
              'srow_z': [float(v) for v in struct.unpack_from('<4f', header, 312)]}
    for key in record:
        require(record[key] == expected[key], 'reference header changed: ' + key)
    require(record['srow_x'][:3] == [-8.0, 0.0, 0.0]
            and record['srow_y'][:3] == [0.0, 8.0, 0.0]
            and record['srow_z'][:3] == [0.0, 0.0, 8.0], 'forward radiological X scan not established')
    return record
