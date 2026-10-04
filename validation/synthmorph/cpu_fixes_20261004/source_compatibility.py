"""Audit unchanged GPU formulas against the frozen published source."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


def dump(value):
    return ast.dump(value, include_attributes=False)


def functions(tree, prefix=''):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result[prefix + node.name] = node
        elif isinstance(node, ast.ClassDef):
            result.update(functions(node, prefix + node.name + '.'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    roots = [Path(args.baseline), Path(args.candidate)]
    rows = {}
    parsed = {}
    for relative in ('src/fnit/synthmorph/models.py', 'src/fnit/synthmorph/pipeline.py',
                     'src/fnit/synthmorph/spatial.py', 'src/fnit/_world_resampling.py',
                     'src/fnit/_transforms.py', 'src/fnit/_nib.py'):
        contents = [(root / relative).read_bytes() for root in roots]
        rows[relative] = {'sha256': [hashlib.sha256(value).hexdigest() for value in contents],
                          'same_file': contents[0] == contents[1]}
        if relative.endswith(('models.py', 'pipeline.py')):
            pair = [functions(ast.parse(value)) for value in contents]
            exceptions = ({'AffineNetwork.forward'} if relative.endswith('models.py') else
                          {'_tensor', '_tensor_frames', 'SynthMorph.__call__'})
            unchanged = {name: dump(node) == dump(pair[1][name])
                         for name, node in pair[0].items() if name not in exceptions}
            rows[relative]['unchanged_functions'] = unchanged
            parsed[Path(relative).name] = pair
    old, new = parsed['pipeline.py']
    def header_branch(function):
        return next(node for node in ast.walk(function)
                    if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                    and node.test.id == 'header_only')
    previous_cuda_sampler = header_branch(old['SynthMorph.__call__']).orelse
    current_cpu_branch = header_branch(new['SynthMorph.__call__']).orelse[0]
    gpu_sampler_equal = [dump(node) for node in previous_cuda_sampler] == [
        dump(node) for node in current_cpu_branch.orelse]
    center_choice = next(node for node in ast.walk(parsed['models.py'][1]['AffineNetwork.forward'])
                         if isinstance(node, ast.IfExp))
    decode_choice = next(node for node in ast.walk(new['_image_data'])
                         if isinstance(node, ast.IfExp))
    normalized_forward = copy.deepcopy(parsed['models.py'][1]['AffineNetwork.forward'])
    normalized_forward.body = [node for node in normalized_forward.body
                               if not (isinstance(node, ast.Assign)
                                       and any(isinstance(target, ast.Name) and target.id == 'center_function'
                                               for target in node.targets))]
    for node in ast.walk(normalized_forward):
        if isinstance(node, ast.Name) and node.id == 'center_function':
            node.id = 'barycenter'
    guards = {
        'unchanged_cuda_final_sampler': gpu_sampler_equal,
        'cpu_final_sampler_guard': ast.unparse(current_cpu_branch.test) == "self.device.type == 'cpu'",
        'joint_reduction_guard': ast.unparse(center_choice.test) == "mid_space and moving.device.type == 'cpu'",
        'cuda_reduction_original_function': ast.unparse(center_choice.orelse) == 'barycenter',
        'remaining_affine_forward_formulas_equal': dump(normalized_forward) == dump(parsed['models.py'][0]['AffineNetwork.forward']),
        'rigid_inverse_cpu_guard': any(isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.device.type == 'cpu' and self.model == 'rigid'"
            for node in ast.walk(new['SynthMorph.__call__'])),
        'cpu_decode_guard': ast.unparse(decode_choice.test) == "torch.device(device).type == 'cpu'",
        'cuda_decode_original_proxy': ast.unparse(decode_choice.orelse) == 'image.dataobj',
    }
    all_passed = all(guards.values()) and all(
        all(row['unchanged_functions'].values()) if 'unchanged_functions' in row else row['same_file']
        for row in rows.values())
    report = {'scope': 'AST audit of CPU-only changes; all other network, CUDA sampler and shared-world source unchanged',
              'files': rows, 'guards': guards, 'all_gates_passed': all_passed,
              'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not all_passed:
        raise RuntimeError('source compatibility audit failed')


if __name__ == '__main__':
    main()
