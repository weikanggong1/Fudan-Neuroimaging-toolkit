"""Check a converted RAS pull field through independent native FSL applywarp."""
import argparse
import json
from pathlib import Path
from compare_registration import image_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = image_pair(args.candidate, args.reference, mask=args.mask)
    result['acceptance'] = all(
        result[region]['nrmse_reference_p99_minus_p1'] <= 1e-3
        for region in ('whole_grid', 'official_synthstrip_brain')
    )
    report = {
        'scope': 'same FNIT-converted intent-2006 field, actual full T1 grids; independent native FSL applywarp trilinear vs TorchApplyWarp; this tests field consumption and is separate from SynthMorph registration',
        'image': result,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
