"""Measure the complete native objective/gradient effect of native-smoothed alphas."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('shared', 'native-alpha', 'mesh', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    sys.path.insert(0, '/public/software/apps/Freesurfer/8.2.0-1/python/lib/python3.8/site-packages/samseg/gems')
    import gemsbindings as gems
    data = np.load(args.shared)
    reference = np.asfortranarray(data['reference'].astype(np.float64))
    positions = np.asfortranarray(data['vertices'].astype(np.float64))
    native_alpha = np.load(args.native_alpha)
    if not np.array_equal(native_alpha, native_alpha.astype(np.float32).astype(np.float64)):
        raise RuntimeError('native getter alpha values are not exactly FP32 representable')
    collection = gems.KvlMeshCollection()
    collection.read(str(args.mesh))
    collection.set_positions(reference, [positions])
    collection.k = float(data['stiffness'])
    mesh = collection.get_mesh(0)
    image = gems.KvlImage(np.asfortranarray(data['image']))
    transform = np.eye(4)
    transform[:3, :3] = data['boundary_transform'].astype(np.float32).astype(np.float64)
    calculator = gems.KvlCostAndGradientCalculator(
        typeName='AtlasMeshToIntensityImage', images=[image], boundaryCondition='Sliding',
        transform=gems.KvlTransform(np.asfortranarray(transform)),
        means=np.asfortranarray(data['means']), variances=np.asfortranarray(data['variances']),
        mixtureWeights=np.ones(len(data['means']), dtype=np.float32),
        numberOfGaussiansPerClass=np.ones(len(data['means']), dtype=np.int32))
    results = {}
    gradients = {}
    for name, alphas in [('fnit_alphas', data['alphas']), ('native_alphas', native_alpha.astype(np.float32))]:
        mesh.alphas = np.asfortranarray(alphas)
        assert np.array_equal(mesh.alphas, alphas)
        cost, gradient = calculator.evaluate_mesh_position(mesh)
        results[name] = float(cost)
        gradients[name] = gradient
        np.save(args.output / (name + '_gradient.npy'), gradient)
    difference = gradients['native_alphas'] - gradients['fnit_alphas']
    report = {'schema': 'fnit.gems.alpha-first-state-objective.v1',
              'scope': 'Same real mesh/image/Gaussians and native calculator; change only supplied first-stage smoothed alpha arrays. No mesh update or complete recipe',
              'costs': results, 'native_alpha_minus_fnit_alpha_cost': results['native_alphas'] - results['fnit_alphas'],
              'complete_gradient_relative_l2_delta': float(np.linalg.norm(difference) / np.linalg.norm(gradients['fnit_alphas'])),
              'complete_gradient_max_absolute_delta': float(np.max(np.abs(difference))),
              'native_alpha_getter_values_exact_float32': True,
              'inputs_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in [('shared', args.shared), ('native_alpha', args.native_alpha), ('mesh', args.mesh)]},
              'program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'native_binding_sha256': hashlib.sha256(Path(gems.__file__).read_bytes()).hexdigest()}
    (args.output / 'report.public.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
