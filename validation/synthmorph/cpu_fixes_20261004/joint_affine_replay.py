"""Replay CPU joint affine stages on frozen real features without a CNN."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from fnit.synthmorph import models


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, reference):
    return {'different_values': int(np.count_nonzero(actual != reference)),
            'max_abs': float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'reference', 'output'): parser.add_argument('--' + name, required=True)
    parser.add_argument('--extent', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    features = np.load(args.features); reference = np.load(args.reference)
    started = time.perf_counter()
    pairs = [models._cpu_joint_barycenter(torch.from_numpy(features[f'feature_{i}']).permute(0, 4, 1, 2, 3),
                                         (args.extent,) * 3) for i in range(2)]
    normalized = [mass / models._cpu_inner_sum(mass).unsqueeze(-1) for _, mass in pairs]
    weights = normalized[0] * normalized[1]
    fits = [models._cpu_joint_fit_affine(pairs[i][0], pairs[1-i][0], weights) for i in range(2)]
    average = (fits[0] + models._cpu_joint_inverse(fits[1])) * .5
    inverse = models._cpu_joint_inverse(average)
    halves = [models._cpu_joint_matrix_sqrt(value) for value in (average, inverse)]
    arrays = {f'center_{i}':pair[0].numpy() for i,pair in enumerate(pairs)}
    arrays.update({f'mass_{i}':pair[1].numpy() for i,pair in enumerate(pairs)})
    arrays.update({f'fit_{i}':value.numpy() for i,value in enumerate(fits)})
    arrays.update({f'half_{i}':value.numpy() for i,value in enumerate(halves)})
    arrays.update(weights=weights.numpy(),average=average.numpy(),inverse=inverse.numpy())
    rows = {name:metrics(value,reference[name]) for name,value in arrays.items()}
    residuals = {f'half_{i}':float(torch.linalg.matrix_norm(value.double()@value.double()-matrix.double()).max())
                 for i,(value,matrix) in enumerate(zip(halves,(average,inverse)))}
    output = Path(args.output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'stages.npz',**arrays)
    report = {'scope':'source-bound real CPU joint affine feature replay; no CNN or image resampling',
              'extent':args.extent,'rows':rows,'half_square_root_residuals':residuals,
              'stage_seconds':time.perf_counter()-started,'models_sha256':digest(models.__file__),
              'worker_sha256':digest(__file__),'features_sha256':digest(args.features),
              'reference_sha256':digest(args.reference),'stages_sha256':digest(output/'stages.npz')}
    (output/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
