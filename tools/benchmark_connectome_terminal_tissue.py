"""按真实接受种子与流线端点的 5TT 组织类别比较两份 TCK。"""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tracking import _five_tissue_mrtrix


def _sha(path: Path) -> str:
    """读取一个文件，返回小写 SHA-256 十六进制字符串。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tissue(volume: torch.Tensor, inverse: torch.Tensor,
            points: np.ndarray) -> np.ndarray:
    """在世界毫米位置 `[N,3]` 取五组织分数，返回 `[N,5]`。"""
    return _five_tissue_mrtrix(
        volume, torch.as_tensor(points, dtype=torch.float32), inverse,
    ).numpy()


def _classify(tissue: np.ndarray) -> np.ndarray:
    """把 `[N,5]` 五组织分数归为皮层、皮层下、白质或其他组织。"""
    cgm, sgm, wm, csf, pathology = tissue.T
    dominant = np.argmax(tissue, axis=1)
    result = np.full(len(tissue), "wm", dtype="<U12")
    result[(cgm + sgm >= wm) & (cgm + sgm > csf) &
           (cgm + sgm > pathology)] = "cgm"
    result[(cgm + sgm >= wm) & (cgm + sgm > csf) &
           (cgm + sgm > pathology) & (sgm > cgm)] = "sgm"
    result[dominant == 3] = "csf"
    result[dominant == 4] = "pathology"
    result[tissue.sum(1) < .5] = "outside"
    return result


def _summarize(tracks_path: Path, seeds: np.ndarray, volume: torch.Tensor,
               inverse: torch.Tensor) -> dict:
    """读取按顺序配对的 TCK 和种子，返回长度分位及非种子端点类别。"""
    tracks = nib.streamlines.load(str(tracks_path)).streamlines
    if len(tracks) != len(seeds):
        raise ValueError("accepted seed and TCK track counts differ")
    first = np.asarray([track[0] for track in tracks], dtype=np.float32)
    last = np.asarray([track[-1] for track in tracks], dtype=np.float32)
    starts_at_seed = np.linalg.norm(first - seeds, axis=1) < .05
    lengths = np.asarray([
        np.linalg.norm(np.diff(track, axis=0), axis=1).sum() for track in tracks
    ])
    seed_class = _classify(_tissue(volume, inverse, seeds))
    first_class = _classify(_tissue(volume, inverse, first))
    last_class = _classify(_tissue(volume, inverse, last))
    result = {"tracks": len(tracks), "starts_at_seed": int(starts_at_seed.sum()),
              "seed_class": {}, "terminal_class": {}}
    for label in ("cgm", "sgm", "wm", "csf", "pathology", "outside"):
        selected = lengths[seed_class == label]
        if len(selected):
            result["seed_class"][label] = {
                "count": len(selected),
                "length_quantiles_mm": np.quantile(selected, [.1, .5, .75, .9]).tolist(),
                "last_endpoint_classes": {
                    name: int(np.sum(last_class[seed_class == label] == name))
                    for name in ("cgm", "sgm", "wm", "csf", "pathology", "outside")
                },
            }
    terminal = np.concatenate((last_class, first_class[~starts_at_seed]))
    result["terminal_class"] = {
        label: int(np.sum(terminal == label))
        for label in ("cgm", "sgm", "wm", "csf", "pathology", "outside")
    }
    return result


def main() -> None:
    """解析真实 TCK、种子和 5TT 路径，写带输入哈希的比较 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-tracks", "official-seeds", "fnit-tracks", "fnit-seeds",
                 "five-tissue", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    image = nib.load(str(args.five_tissue))
    volume = torch.as_tensor(np.asarray(image.dataobj), dtype=torch.float32)
    inverse = torch.as_tensor(np.linalg.inv(image.affine), dtype=torch.float64)
    official_seeds = np.loadtxt(args.official_seeds, delimiter=",", comments="#",
                                usecols=(2, 3, 4))
    fnit_seeds = np.load(args.fnit_seeds)
    paths = {name: getattr(args, name.replace("-", "_")) for name in
             ("official-tracks", "official-seeds", "fnit-tracks", "fnit-seeds",
              "five-tissue")}
    report = {
        "input_sha256": {name: _sha(path) for name, path in paths.items()},
        "classes": {
            "official": _summarize(args.official_tracks, official_seeds, volume, inverse),
            "fnit": _summarize(args.fnit_tracks, fnit_seeds, volume, inverse),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: data["terminal_class"]
                      for name, data in report["classes"].items()}))


if __name__ == "__main__":
    main()
