"""Compare actual normalized network input images saved by both debug APIs."""
import argparse
import json
from pathlib import Path
from compare_registration import image_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = {name: image_pair(Path(args.candidate)/(name+'.nii.gz'),
                             Path(args.reference)/(name+'.nii.gz'))
            for name in ('inp_1', 'inp_2')}
    Path(args.output).write_text(json.dumps({
        'scope': 'debug normalized network inputs on real 192-grid; the FNIT documented debug contract stores network_transforms.npz while official adds tra/out volumes; compare their shared input images and report geometry independently',
        'images': rows,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
