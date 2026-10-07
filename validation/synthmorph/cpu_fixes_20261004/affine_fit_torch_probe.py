"""Replay actual Torch fit strides and small LU variants on saved real matrices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from fnit.synthmorph import models


def metrics(actual, reference):
    return {'different_values': int(np.count_nonzero(actual != reference)),
            'max_abs': float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))}


def lu_inverse(value, fused=False, reciprocal=False, columns=False, solve_reciprocal=False, aggregate=False):
    # Independent four-by-four partial-pivot elimination and triangular solve.
    matrix = value.copy()
    result = np.eye(4, dtype=np.float32)
    def subtract(a, b, c):
        return (a.astype(np.float64) - b.astype(np.float64) * c.astype(np.float64)).astype(np.float32) if fused else a - b*c
    for column in range(4):
        pivot = column + np.argmax(np.abs(matrix[column:, column]))
        matrix[[column, pivot]] = matrix[[pivot, column]]
        result[[column, pivot]] = result[[pivot, column]]
        if reciprocal:
            matrix[column+1:, column] *= np.float32(1) / matrix[column, column]
        else:
            matrix[column+1:, column] /= matrix[column, column]
        for row in range(column + 1, 4):
            matrix[row, column+1:] = subtract(matrix[row, column+1:], matrix[row, column], matrix[column, column+1:])
    # Solve unit-lower triangular L against the row-permuted identity.
    for row in range(4):
        if aggregate:
            total = np.zeros(4, np.float32)
            for column in range(row): total += matrix[row, column] * result[column]
            result[row] -= total
        else:
            for column in range(row):
                result[row] = subtract(result[row], matrix[row, column], result[column])
    if columns:
        for row in range(3, -1, -1):
            result[row] = result[row] * (np.float32(1) / matrix[row, row]) if solve_reciprocal else result[row] / matrix[row, row]
            for column in range(row):
                result[column] = subtract(result[column], matrix[column, row], result[row])
    else:
        for row in range(3, -1, -1):
            if aggregate:
                total = np.zeros(4, np.float32)
                for column in range(row + 1, 4): total += matrix[row, column] * result[column]
                result[row] -= total
            else:
                for column in range(row + 1, 4):
                    result[row] = subtract(result[row], matrix[row, column], result[column])
            result[row] = result[row] * (np.float32(1) / matrix[row, row]) if solve_reciprocal else result[row] / matrix[row, row]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    data = np.load(args.stages); rows = {}
    for key in data.files:
        if not key.endswith('_gram'): continue
        prefix = key[:-5]
        reference = {name: data[prefix + '_' + name] for name in ('x', 'xt', 'gram', 'inverse', 'projection', 'beta')}
        x = torch.from_numpy(reference['x'])
        # Source values recovered independently from the saved original beta path
        # are stored in the original center input when building the TF probe.
        source = torch.from_numpy(data[prefix + '_source']) if prefix + '_source' in data else None
        weights = torch.from_numpy(data[prefix + '_weights']) if prefix + '_weights' in data else None
        if weights is None:
            # Saved weighted transpose divided by corresponding x entries;
            # the homogeneous row is exactly one, so this recovers weights exactly.
            weights = torch.from_numpy(reference['xt'][..., 3, :].copy())
        xt = x.transpose(-1, -2) * weights.unsqueeze(-2)
        inverse_reference = torch.from_numpy(reference['inverse'])
        actual = {}
        for label, transpose in [('production_strides', xt), ('contiguous_transpose', xt.contiguous())]:
            gram = transpose @ x
            inverse = torch.linalg.inv(gram)
            projection = inverse @ transpose
            actual[label] = {name: metrics(tensor.numpy(), reference[name]) for name, tensor in
                             [('xt', transpose), ('gram', gram), ('inverse', inverse), ('projection', projection)]}
            actual[label]['strides'] = {'xt':list(transpose.stride()),'gram':list(gram.stride()),'inverse':list(inverse.stride())}
        production_inverse = models._cpu_joint_inverse(torch.from_numpy(reference['gram']))
        actual['production_inverse'] = metrics(production_inverse.numpy(), reference['inverse'])
        actual['production_projection'] = metrics((production_inverse @ xt.contiguous()).numpy(), reference['projection'])
        variants = {}
        for fused in (False, True):
            for reciprocal in (False, True):
                for columns in (False, True):
                    for solve_reciprocal in (False, True):
                        for aggregate in (False, True):
                            label = f'fused{fused}_reciprocal{reciprocal}_columns{columns}_solverecip{solve_reciprocal}_aggregate{aggregate}'
                            variants[label] = metrics(lu_inverse(reference['gram'][0], fused, reciprocal, columns, solve_reciprocal, aggregate), reference['inverse'][0])
        rows[prefix] = {'actual_torch': actual, 'independent_lu': variants}
    report = {'scope':'actual production weighted-transpose layout and source-derived four-by-four LU variants; no CNN',
              'rows':rows,'stages_sha256':hashlib.sha256(Path(args.stages).read_bytes()).hexdigest(),
              'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__': main()
