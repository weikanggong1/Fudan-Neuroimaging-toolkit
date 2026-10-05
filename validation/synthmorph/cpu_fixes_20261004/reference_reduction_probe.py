"""Replay original CPU reduction ordering on bound real feature maps."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from fnit.synthmorph import models


def packet_sum(values, width=8):
    count = values.shape[-1]
    packets = [np.zeros((*values.shape[:-1], width), np.float32) for _ in range(4)]
    end = count // (4 * width) * (4 * width)
    for position in range(0, end, 4 * width):
        for group in range(4):
            packets[group] += values[..., position + group * width:position + (group + 1) * width]
    merged = ((packets[0] + packets[1]) + packets[2]) + packets[3]
    position = end
    while position + width <= count:
        merged += values[..., position:position + width]
        position += width
    tail = np.zeros(values.shape[:-1], np.float32)
    for index in range(position, count):
        tail += values[..., index]
    while merged.shape[-1] > 1:
        middle = merged.shape[-1] // 2
        merged = merged[..., :middle] + merged[..., middle:]
    return tail + merged[..., 0]


def reduce_features(features, extent, width):
    flat = features.reshape(features.shape[0], -1, features.shape[-1])
    count = flat.shape[1]
    streams = [np.zeros((flat.shape[0], flat.shape[-1]), np.float32) for _ in range(4)]
    end = count // 4 * 4
    for position in range(0, end, 4):
        for group in range(4):
            streams[group] += flat[:, position + group]
    mass = ((streams[0] + streams[1]) + streams[2]) + streams[3]
    for position in range(end, count):
        mass += flat[:, position]
    coordinates = [(np.arange(size, dtype=np.float32) - (size - 1) / 2) / size
                   for size in features.shape[1:4]]
    grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), -1).reshape(-1, 3)
    moment = np.zeros((flat.shape[0], flat.shape[-1], 3), np.float32)
    for position in range(count):
        moment += flat[:, position, :, None] * grid[position]
    denominator = packet_sum(flat.transpose(0, 2, 1), width)[..., None]
    centers = np.divide(moment, denominator, out=np.zeros_like(moment), where=denominator != 0) * extent
    return centers, mass


def metrics(actual, reference):
    return {'different_values': int(np.count_nonzero(actual != reference)),
            'max_abs': float(np.max(np.abs(actual - reference)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'reference', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--extent', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    features = np.load(args.features)
    reference = np.load(args.reference)
    rows = {}
    production = [models._cpu_joint_barycenter(
        torch.from_numpy(features[f'feature_{index}']).permute(0, 4, 1, 2, 3),
        (args.extent,) * 3) for index in range(2)]
    for index, (centers, mass) in enumerate(production):
        rows[f'production_feature{index}'] = {
            'center': metrics(centers.numpy(), reference[f'center_{index}']),
            'mass': metrics(mass.numpy(), reference[f'mass_{index}'])}
    normalized = [mass / models._cpu_inner_sum(mass).unsqueeze(-1)
                  for _, mass in production]
    rows['production_confidence_weights'] = metrics(
        (normalized[0] * normalized[1]).numpy(), reference['weights'])
    timing = []
    tensors = [torch.from_numpy(features[f'feature_{index}']).permute(0, 4, 1, 2, 3)
               for index in range(2)]
    for _ in range(5):
        started = time.perf_counter()
        pairs = [models._cpu_joint_barycenter(value, (args.extent,) * 3) for value in tensors]
        confidence = [mass / models._cpu_inner_sum(mass).unsqueeze(-1) for _, mass in pairs]
        weights = confidence[0] * confidence[1]
        timing.append(time.perf_counter() - started)
    for width in (4, 8, 16):
        centers, masses = zip(*(reduce_features(features[f'feature_{index}'], args.extent, width)
                               for index in range(2)))
        for index in range(2):
            if not np.array_equal(features[f'feature_{index}'], reference[f'feature_{index}']):
                raise ValueError('reference replay did not consume identical features')
            rows[f'packet{width}_feature{index}'] = {
                'center': metrics(centers[index], reference[f'center_{index}']),
                'mass': metrics(masses[index], reference[f'mass_{index}'])}
        if width == 8:
            normalized = [mass / packet_sum(mass)[..., None] for mass in masses]
            weights = normalized[0] * normalized[1]
            rows['packet8_confidence_weights'] = metrics(weights, reference['weights'])
            fits = [models.fit_affine(torch.from_numpy(centers[0]), torch.from_numpy(centers[1]), torch.from_numpy(weights)),
                    models.fit_affine(torch.from_numpy(centers[1]), torch.from_numpy(centers[0]), torch.from_numpy(weights))]
            rows['same_centers_and_weights_torch_fits'] = {
                str(index): metrics(fit.numpy(), reference[f'fit_{index}'])
                for index, fit in enumerate(fits)}
    report = {'scope': 'source-derived Eigen CPU four-stream, sequential moment and packet denominator replay; no CNN',
              'extent': args.extent, 'rows': rows,
              'features_sha256': hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
              'production_models_sha256': hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest(),
              'helper_pair_seconds': timing, 'helper_spatial_points': list(tensors[0].shape[2:]),
              'helper_scope': 'two CPU barycenters plus both normalized confidence masses; five calls after numerical probe; input loading excluded',
              'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
