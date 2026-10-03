"""仅为复现已拒绝的真实 benchmark 生成旧候选；不修改产品源码。"""
import argparse
import hashlib
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if sha(args.baseline)!='d8901752cda4e5b8007f7c318613b343ad3edbd7072d0e0a05539a5d42ac6313':raise ValueError('exact frozen baseline required')
    if args.output.exists():raise ValueError('new candidate filename required')
    source=args.baseline.read_text()
    old='''    Computation runs on ``endpoints.device``. Float32 is used for coordinates
    and metrics; sums use float64 and the streamline count matrix uses int64.
'''
    new='''    Computation runs on ``endpoints.device``. Coordinates, length and FA use
    float32. SIFT2 weights retain float64 precision, as do accumulated sums;
    the streamline count matrix uses int64. Floating output matrices remain
    float32.
'''
    if source.count(old)!=1:raise ValueError('frozen docstring changed')
    source=source.replace(old,new,1)
    old='        value = torch.as_tensor(value, device=device, dtype=torch.float32)'
    new='        dtype = torch.float64 if name == "weights" else torch.float32\n        value = torch.as_tensor(value, device=device, dtype=dtype)'
    if source.count(old)!=1:raise ValueError('frozen weight anchor changed')
    args.output.write_text(source.replace(old,new,1))
    print('rejected_candidate_sha256='+sha(args.output))


if __name__=='__main__':main()
