"""矩阵重复性 benchmark 的读入与验收；不运行追踪或重新构造矩阵。"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from .compare_connectome_matrices import NAMES, _compare, _pearson
except ImportError:  # python tools/<script>.py
    from compare_connectome_matrices import NAMES, _compare, _pearson


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrices(directory: Path) -> tuple[dict[str, np.ndarray], dict]:
    """兼容 connectome_、candidate_ 和官方无前缀的四个 CSV。"""
    directory = Path(directory)
    for prefix in ("connectome_", "candidate_", ""):
        paths = {name: directory / f"{prefix}{name}.csv" for name in NAMES}
        if all(path.is_file() for path in paths.values()):
            break
    else:
        if (directory / "matrices").is_dir():
            return load_matrices(directory / "matrices")
        raise FileNotFoundError(f"{directory}: four connectome CSVs not found")
    arrays = {name: np.loadtxt(path, delimiter=",", ndmin=2)
              for name, path in paths.items()}
    validate_matrices(arrays, str(directory))
    metadata = load_metadata(directory, arrays["count"].shape[0])
    hashes = {name: sha256(path) for name, path in paths.items()}
    alignment = metadata.get("canonical_matrix_alignment")
    if alignment is not None and alignment.get("canonical_matrix_sha256") != hashes:
        raise ValueError(f"{directory}: canonical aligned matrix hashes differ from alignment record")
    return arrays, {"directory": str(directory.resolve()),
                    "matrix_sha256": hashes,
                    **metadata}


def validate_matrices(arrays: dict, label: str, nodes: int | None = None) -> None:
    shape = arrays["count"].shape
    if (len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1 or
            (nodes is not None and shape != (nodes, nodes)) or any(
                value.shape != shape or not np.isfinite(value).all()
                for value in arrays.values())):
        raise ValueError(f"{label}: expected four finite square matrices with matching nodes")


def load_metadata(directory: Path, nodes: int, declared: dict | None = None) -> dict:
    """保留 atlas 文件摘要及节点行顺序；缺少来源信息时明确记录 null。"""
    declared = declared or {}
    if declared.get("nodes", nodes) != nodes:
        raise ValueError(f"{directory}: declared node count differs from matrix shape")
    node_rows, node_hash = None, None
    node_path = directory / "nodes.tsv"
    if node_path.is_file():
        with node_path.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            required = ("index", "original_label", "hemisphere", "name")
            if reader.fieldnames != list(required):
                raise ValueError(f"{node_path}: expected columns {required}")
            rows = list(reader)
        if len(rows) != nodes or [int(row["index"]) for row in rows] != list(range(1, nodes + 1)):
            raise ValueError(f"{node_path}: nodes must be ordered 1..K and match matrix shape")
        node_rows = [[row[key] for key in required] for row in rows]
        node_hash = sha256(node_path)
    count_path = directory / "nodes.txt"
    if count_path.is_file() and int(count_path.read_text().strip()) != nodes:
        raise ValueError(f"{count_path}: node count differs from matrix shape")
    labels_path = directory / "region_labels.csv"
    if labels_path.is_file():
        labels = np.loadtxt(labels_path, delimiter=",", ndmin=1)
        if not np.array_equal(labels, np.arange(1, nodes + 1)):
            raise ValueError(f"{labels_path}: expected ordered canonical node indices 1..K")
    atlas_hash = declared.get("atlas_sha256")
    atlas_path = directory / "atlas_dwi.nii.gz"
    if atlas_path.is_file():
        actual = sha256(atlas_path)
        if atlas_hash is not None and atlas_hash != actual:
            raise ValueError(f"{directory}: declared atlas hash differs from actual file")
        atlas_hash = actual
    digest_path = directory / "atlas.sha256"
    if digest_path.is_file():
        actual = digest_path.read_text().split()[0]
        if len(actual) != 64 or any(c not in "0123456789abcdef" for c in actual):
            raise ValueError(f"{digest_path}: invalid SHA-256")
        if atlas_hash is not None and actual != atlas_hash:
            raise ValueError(f"{directory}: atlas hashes disagree")
        atlas_hash = actual
    export_path = directory / "reference_atlas_export.json"
    exported = json.loads(export_path.read_text()) if export_path.is_file() else None
    if exported is not None and exported.get("original_sha256") != atlas_hash:
        raise ValueError(f"{export_path}: source atlas hash differs from atlas.sha256")
    alignment_path = directory / "canonical_matrix_alignment.json"
    alignment = json.loads(alignment_path.read_text()) if alignment_path.is_file() else None
    if alignment is not None and alignment.get("canonical_nodes") != nodes:
        raise ValueError(f"{alignment_path}: canonical node count differs from matrices")
    return {"nodes": nodes, "atlas_sha256": atlas_hash, "node_rows": node_rows,
            "nodes_tsv_sha256": node_hash, "reference_atlas_export": exported,
            "canonical_matrix_alignment": alignment,
            "atlas_hash_scope": "source CLI atlas file" if exported else "supplied atlas file"}


def load_profiles(directory: Path) -> dict[str, tuple[dict, dict]]:
    """读取当前 CLI 的 atlases/*/CSV 或旧 report.json + profile.npz。"""
    directory = Path(directory)
    profile_root = directory / "atlases" if (directory / "atlases").is_dir() else directory
    if (profile_root / "matrices").is_dir():
        profile_root = profile_root / "matrices"
    found = {}
    for child in sorted(profile_root.iterdir()):
        if child.is_dir() and any((child / f"{prefix}count.csv").is_file()
                                  for prefix in ("connectome_", "candidate_", "")):
            found[child.name] = load_matrices(child)
    if found:
        return found
    report_path = directory / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        for profile, info in report.get("profiles", {}).items():
            path = directory / f"{profile}.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in NAMES}
            validate_matrices(arrays, profile, info["nodes"])
            found[profile] = arrays, {"directory": str(directory.resolve()),
                                      "matrix_npz_sha256": sha256(path),
                                      **load_metadata(directory / profile, info["nodes"], info)}
    if not found:
        raise FileNotFoundError(f"{directory}: no atlas matrix profiles found")
    return found


def check_metadata(items: list[tuple[dict, dict]], profile: str) -> dict:
    """有节点表/atlas SHA 时逐值校验；不把缺失的身份验证说成通过。"""
    shape = items[0][0]["count"].shape
    directories = [str(Path(meta["directory"]).resolve()) for _, meta in items]
    if len(set(directories)) != len(directories):
        raise ValueError(f"{profile}: duplicate resolved matrix directories cannot count as independent repeats")
    if any(item[0]["count"].shape != shape for item in items):
        raise ValueError(f"{profile}: matrix shapes differ between runs")
    result = {}
    for field in ("atlas_sha256", "node_rows"):
        present = [meta[field] for _, meta in items if meta.get(field) is not None]
        if present and any(value != present[0] for value in present[1:]):
            raise ValueError(f"{profile}: {field} differs between runs")
        result[field] = {"status": "verified_equal" if len(present) == len(items) else "not_fully_available",
                         "available_runs": len(present), "total_runs": len(items)}
        if field == "atlas_sha256":
            result[field]["scope"] = "source atlas identity; exported reference representation is recorded separately"
    return result


def compact_metrics(first: dict, second: dict) -> dict:
    """保留既有单 atlas 工具的数值定义和严格上三角边集。"""
    full = _compare(first, second)
    upper = np.triu_indices(first["count"].shape[0], k=1)
    result = {}
    for name in NAMES:
        reference, candidate = first[name][upper], second[name][upper]
        common = (reference != 0) & (candidate != 0)
        error = np.abs(candidate - reference)
        reference_mass, common_mass = np.abs(reference).sum(), np.abs(reference[common]).sum()
        result[name] = {
            "pearson": full[name]["values"]["pearson"],
            "nonzero_mean_scaled_mae": full[name]["values"]["normalized_mae"],
            "relative_l1_full_upper": float(error.sum() / reference_mass) if reference_mass else None,
            "relative_l1_common_nonzero": float(error[common].sum() / common_mass) if common_mass else None,
            "support_dice": full[name]["nonzero_support"]["dice"],
            "evaluated_edges": full[name]["values"]["evaluated_edges"],
            "shared_support_edges": full[name]["nonzero_support"]["shared_edges"],
            "diagonal": full[name]["diagonal"],
        }
    return result


def profile_metrics(first: dict, second: dict) -> dict:
    """保留七 atlas 工具定义；FA/长度误差仅在共有 count 边上计算。"""
    upper = np.triu_indices(first["count"].shape[0], k=1)
    count_a, count_b = first["count"][upper], second["count"][upper]
    support_a, support_b = count_a > 0, count_b > 0
    common = support_a & support_b
    total_support = int(support_a.sum() + support_b.sum())
    diagonal = {}
    for name in NAMES:
        a, b = np.diag(first[name]), np.diag(second[name])
        error = b - a
        diagonal[name] = {"reference_nonzero_nodes": int(np.count_nonzero(a)),
                          "candidate_nonzero_nodes": int(np.count_nonzero(b)),
                          "mae": float(np.abs(error).mean()),
                          "rmse": float(np.sqrt((error * error).mean()))}
    result = {"count_support_dice": float(2 * common.sum() / total_support) if total_support else 1.0,
              "count_pearson": _pearson(count_a, count_b), "common_edges": int(common.sum()),
              "diagonal": diagonal}
    for name in ("count", "sift2_fbc"):
        a, b = first[name][upper], second[name][upper]
        mass = np.abs(a).sum()
        result[f"{name}_relative_l1"] = float(np.abs(a - b).sum() / mass) if mass else None
    for name in ("mean_length", "mean_fa"):
        a, b = first[name][upper][common], second[name][upper][common]
        result[f"{name}_common_normalized_mae"] = (
            float(np.abs(a - b).mean() / np.abs(a).mean()) if len(a) and np.abs(a).sum() else None)
    return result


def pairwise(official: list, fnit: list, metric_function) -> dict:
    if len(official) < 3 or not fnit:
        raise ValueError("at least three official repeats and one FNIT run are required")
    return {
        "official": [{"first_run": i, "second_run": j, "metrics": metric_function(official[i], official[j])}
                     for i, j in itertools.combinations(range(len(official)), 2)],
        "fnit": [{"first_run": i, "second_run": j, "metrics": metric_function(fnit[i], fnit[j])}
                 for i, j in itertools.combinations(range(len(fnit)), 2)],
        "cross": [{"official_run": i, "fnit_run": j, "metrics": metric_function(a, b)}
                  for i, a in enumerate(official) for j, b in enumerate(fnit)],
    }


def envelope(within: list, cross: list, *, similarity: bool = False) -> dict:
    """错误 ≤ 官方最大值 / 相似性 ≥ 官方最小值；完整范围仍用于展示。"""
    finite = [float(value) for value in within if value is not None and np.isfinite(value)]
    complete = len(finite) == len(within)
    limits = [min(finite), max(finite)] if finite else None
    boundary = limits[0 if similarity else 1] if limits else None
    decisions = [(float(value) >= boundary if similarity else float(value) <= boundary)
                 if complete and boundary is not None and value is not None and np.isfinite(value)
                 else None for value in cross]
    status = ("not_assessed" if not cross or any(value is None for value in decisions) else
              "passed" if all(decisions) else "failed")
    return {"official_min_max": limits, "official_values": within,
            "fnit_vs_official": cross, "official_defined_count": len(finite),
            "official_comparison_count": len(within),
            "inside_count": (sum(limits[0] <= value <= limits[1] for value in cross if value is not None)
                             if limits else 0),
            "criterion": ">= official_min" if similarity else "<= official_max",
            "threshold": boundary, "comparison_accepted": decisions,
            "accepted_count": sum(value is True for value in decisions),
            "status": status}


def fnit_repeat_envelope(within: list, fnit_values: list, *, similarity: bool = False) -> dict:
    """单列 FNIT 自身重复，只有至少一对时才有可评估结果。"""
    result = envelope(within, fnit_values, similarity=similarity)
    result["fnit_within"] = result.pop("fnit_vs_official")
    return result


def overall_status(ranges: list[dict]) -> str:
    if any(item["status"] == "failed" for item in ranges):
        return "failed"
    return "passed" if ranges and all(item["status"] == "passed" for item in ranges) else "not_assessed"


def seed_labels(values: list[int] | None, count: int, label: str) -> list[int] | None:
    if values is not None and len(values) != count:
        raise ValueError(f"{label}: one seed label per directory is required")
    if values is not None and len(set(values)) != len(values):
        raise ValueError(f"{label}: duplicate seed labels cannot count as independent repeats")
    return values


METRIC_POLICY = {
    "edges": "strict upper triangle, excluding diagonal; diagonal errors reported separately",
    "count_and_fbc": "all off-diagonal edges",
    "length_and_fa": "value metrics only on common nonzero count edges",
    "normalization": "first matrix is the reference; cross pairs start with official, unordered within-arm pairs use supplied directory order",
    "acceptance": "errors <= maximum observed official-pair error; similarities >= minimum observed official-pair similarity",
    "undefined": "null and not_assessed, never zero or a pass",
    "scope": "matrix repeat envelope; not a proof of identical random streams or anatomical pipeline equivalence",
    "sampling": "finite observed repeat range, not a population confidence interval; criterion is fixed before examining real cross results",
    "independence": "caller-declared repeats; unique resolved directories and supplied seed labels are checked, but different directories alone do not prove independent execution",
}
