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
            exceptions = ({'AffineNetwork.forward', 'FeatureDetector.forward'} if relative.endswith('models.py') else
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
    input_choice = next(node for node in new['SynthMorph.__call__'].body
                        if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "self.device.type == 'cpu' and self.model == 'joint'")
    gpu_sampler_equal = [dump(node) for node in previous_cuda_sampler] == [
        dump(node) for node in current_cpu_branch.orelse]
    center_choice = next(node for node in ast.walk(parsed['models.py'][1]['AffineNetwork.forward'])
                         if isinstance(node, ast.IfExp))
    choices = {node.targets[0].id: node.value
               for node in parsed['models.py'][1]['AffineNetwork.forward'].body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and isinstance(node.value, ast.IfExp)}
    decode_choice = next(node for node in ast.walk(new['_image_data'])
                         if isinstance(node, ast.IfExp))
    normalized_forward = copy.deepcopy(parsed['models.py'][1]['AffineNetwork.forward'])
    normalized_forward.body = [node for node in normalized_forward.body
                               if not (isinstance(node, ast.Assign)
                                       and any(isinstance(target, ast.Name) and target.id in ('center_function', 'fit_function', 'inverse_function', 'sqrt_function', 'cpu_inference')
                                               for target in node.targets))]
    feature_choice = next(node for node in normalized_forward.body
                          if isinstance(node, ast.If) and ast.unparse(node.test) == 'cpu_inference')
    position = normalized_forward.body.index(feature_choice)
    normalized_forward.body[position:position + 1] = feature_choice.orelse
    weight_choice = next(node for node in normalized_forward.body
                         if isinstance(node, ast.If)
                         and ast.unparse(node.test) == 'cpu_inference')
    position = normalized_forward.body.index(weight_choice)
    normalized_forward.body[position:position + 1] = weight_choice.orelse
    center_composition_choice = next(node for node in normalized_forward.body
                                    if isinstance(node, ast.If)
                                    and ast.unparse(node.test) == 'cpu_inference')
    position = normalized_forward.body.index(center_composition_choice)
    normalized_forward.body[position:position + 1] = center_composition_choice.orelse
    class EstablishedFunctions(ast.NodeTransformer):
        def visit_Name(self, node):
            replacement = {'center_function':'barycenter', 'fit_function':'fit_affine',
                           'inverse_function':'torch.linalg.inv', 'sqrt_function':'matrix_sqrt'}.get(node.id)
            return ast.parse(replacement, mode='eval').body if replacement else node
    normalized_forward = EstablishedFunctions().visit(normalized_forward)
    detector_forward = copy.deepcopy(parsed['models.py'][1]['FeatureDetector.forward'])
    detector_choice = detector_forward.body.pop(0)
    detector_forward.args.kwonlyargs = []
    detector_forward.args.kw_defaults = []
    guards = {
        'unchanged_cuda_final_sampler': gpu_sampler_equal,
        'cpu_final_sampler_guard': ast.unparse(current_cpu_branch.test) == "self.device.type == 'cpu'",
        'joint_CPU_input_sampler_guard': ast.unparse(input_choice.test) == "self.device.type == 'cpu' and self.model == 'joint'",
        'cuda_and_other_modes_input_sampler_original': len(input_choice.orelse) == 1 and ast.unparse(input_choice.orelse[0]) == 'input_sampler = transform',
        'joint_CPU_inference_features_guard': ast.unparse(feature_choice.test) == 'cpu_inference',
        'cuda_training_and_linear_features_original': len(feature_choice.orelse) == 1 and ast.unparse(feature_choice.orelse[0]) == 'feat1, feat2 = (self.detector(moving), self.detector(fixed))',
        'detector_CPU_inference_guard': ast.unparse(detector_choice.test) == "cpu_joint_inference and x.device.type == 'cpu' and (x.dtype == torch.float32) and (not torch.is_grad_enabled()) and torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled",
        'detector_leaf_policy_guard': isinstance(detector_choice.body[-1], ast.If) and ast.unparse(detector_choice.body[-1].test) == 'supported_inference(self, x)',
        'detector_remaining_body_original': dump(detector_forward) == dump(parsed['models.py'][0]['FeatureDetector.forward']),
        'joint_reduction_guard': ast.unparse(center_choice.test) == 'cpu_inference',
        'cuda_reduction_original_function': ast.unparse(center_choice.orelse) == "_cpu_joint_barycenter_legacy if mid_space and moving.device.type == 'cpu' else barycenter",
        'cpu_confidence_guard': ast.unparse(weight_choice.test) == 'cpu_inference',
        'cpu_center_composition_guard': ast.unparse(center_composition_choice.test) == 'cpu_inference',
        'cpu_fit_inverse_and_sqrt_guard': all(ast.unparse(choices[name].test) == 'cpu_inference'
                                        for name in ('fit_function','inverse_function','sqrt_function')),
        'cuda_fit_and_inverse_original_functions': all(ast.unparse(choices[name].orelse) == original
            for name,original in [('fit_function','fit_affine'),('inverse_function','torch.linalg.inv'),('sqrt_function','matrix_sqrt')]),
        'remaining_affine_forward_formulas_equal': dump(normalized_forward) == dump(parsed['models.py'][0]['AffineNetwork.forward']),
        'rigid_inverse_cpu_guard': any(isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.device.type == 'cpu' and self.model == 'rigid'"
            for node in ast.walk(new['SynthMorph.__call__'])),
        'cpu_decode_guard': ast.unparse(decode_choice.test) == "torch.device(device).type == 'cpu'",
        'cuda_decode_original_proxy': ast.unparse(decode_choice.orelse) == 'image.dataobj',
    }
    policy = parsed['models.py'][1]['_cpu_joint_inference_enabled']
    guards['CPU_policy_rejects_cuda_training_grad_and_non_F32'] = ast.unparse(policy.body[0].test) == "not mid_space or module.training or moving.device.type != 'cpu' or (fixed.device.type != 'cpu') or (moving.dtype != torch.float32) or (fixed.dtype != torch.float32) or torch.is_grad_enabled()"
    guards['CPU_policy_uses_detector_policy_for_both_inputs'] = ast.unparse(policy.body[-1].value) == 'supported_inference(module.detector, moving) and supported_inference(module.detector, fixed)'
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
