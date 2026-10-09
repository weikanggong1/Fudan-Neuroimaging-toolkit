"""离散K/H/K1/K2实验内核：有序面环及几何在Torch执行。

对应固定源码 MRIScomputeSecondFundamentalFormDiscrete 内部步骤，不调用原程序。
不是 continuous quadric 拟合，也不生成占位 curv.stats；生产默认尚未切换。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from .curvature_stats_torch import curvature_derivatives_tensor
from .mris_register_objective import _atan_table
from .place_surface_normals import ordered_face_csr


def _dot(first, second):
    return (first[..., 0] * second[..., 0] + first[..., 1] * second[..., 1]) + first[..., 2] * second[..., 2]


def _cross(first, second):
    return torch.stack((first[..., 1] * second[..., 2] - first[..., 2] * second[..., 1],
                        first[..., 2] * second[..., 0] - first[..., 0] * second[..., 2],
                        first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]), dim=-1)


def _length(vector):
    # V3_LEN先以float计算平方和，再调用double sqrt。
    return _dot(vector, vector).double().sqrt().float()


def _triangle_angle(y, x):
    # 复用已有查表。源函数halfPi-table的表达式先在float完成，再赋给double r。
    ax, ay = x.abs(), y.abs()
    major = ax >= ay
    low, high = torch.where(major, ay, ax), torch.where(major, ax, ay)
    index = torch.where(high == 0, 0, (100000 * low / high).long()).clamp(0, 100000)
    sample = _atan_table(str(x.device))[index]
    half_pi = torch.tensor(math.pi / 2, dtype=torch.float32, device=x.device)
    pi = float(np.float32(math.pi))
    value = torch.where(major, sample, half_pi - sample).double()
    value = torch.where((x >= 0) & (y < 0), -value, value)
    value = torch.where((x < 0) & (y >= 0), pi - value, value)
    value = torch.where((x < 0) & (y < 0), -pi + value, value)
    return value.float()


class DiscreteCurvatureTopology:
    """同一有序闭合三角网格的显式整数缓存，坐标/法向/曲率不缓存。

    faces:整数(F,3)，nvertices:正整数N，device默认cuda:0。复用成熟CSR准备，
    按源码从首面开始、随环位置旋转的候选顺序建立面环；Torch批量整数运算。
    孤立/重复角点/无法闭环网格抛ValueError；该实验接口暂不覆盖源码marked扩张分支。
    缓存只对该顶点数和有序面有效，换拓扑必须新建；没有全局持久缓存。
    """
    def __init__(self, *, faces: np.ndarray, nvertices: int, device: str = "cuda:0"):
        offsets, ids, corners = ordered_face_csr(faces, nvertices)
        if nvertices < 1 or not len(ids):
            raise ValueError("a nonempty surface is required")
        degree = np.diff(offsets)
        if np.any(degree < 3) or np.any(np.diff(np.sort(np.asarray(faces), axis=1), axis=1) == 0):
            raise ValueError("isolated/boundary vertices or repeated face corners are unsupported")
        self.faces_np = np.array(faces, dtype=np.int64, copy=True)
        self.faces_np.flags.writeable = False
        self.nvertices = int(nvertices)
        self.device = torch.device(device)
        self.faces = torch.as_tensor(self.faces_np.copy(), device=self.device)
        width = int(degree.max())
        padded = np.zeros((nvertices, width), dtype=np.int64)
        corner_padded = np.zeros_like(padded)
        rows = np.repeat(np.arange(nvertices), degree)
        columns = np.arange(len(ids)) - np.repeat(offsets[:-1], degree)
        padded[rows, columns], corner_padded[rows, columns] = ids, corners
        face_ids = torch.as_tensor(padded, device=self.device)
        corner_ids = torch.as_tensor(corner_padded, device=self.device)
        self.degree = torch.as_tensor(degree, device=self.device)
        slots = torch.arange(width, device=self.device)[None, :]
        rows_t = torch.arange(nvertices, device=self.device)
        used = slots == 0
        used = used.expand(nvertices, -1).clone()
        order = torch.zeros_like(face_ids)
        current = torch.zeros(nvertices, dtype=torch.int64, device=self.device)
        for step in range(width - 1):
            active = self.degree > step + 1
            candidates = (slots + step) % self.degree[:, None]
            current_face = self.faces[face_ids[rows_t, current]]
            candidate_faces = self.faces[face_ids.gather(1, candidates)]
            shared = (current_face[:, None, :, None] == candidate_faces[:, :, None, :]).any(-1).sum(-1)
            available = (shared == 2) & ~used.gather(1, candidates) & active[:, None]
            if bool((active & ~available.any(1)).any()):
                raise ValueError("source face-ring packing failed; marked-vertex branch is not supported")
            picked = available.to(torch.int8).argmax(1)
            current = candidates[rows_t, picked]
            order[:, step + 1] = current
            used[rows_t, current] |= active
        self.face_ids = face_ids.gather(1, order)
        self.corners = corner_ids.gather(1, order)
        self.valid = slots < self.degree[:, None]
        next_slots = (slots + 1) % self.degree[:, None]
        self.next_faces = self.face_ids.gather(1, next_slots)
        first, second = self.faces[self.face_ids], self.faces[self.next_faces]
        matches = (first[..., :, None] == second[..., None, :])
        common_first, common_second = matches.any(-1), matches.any(-2)
        if bool((self.valid & (common_first.sum(-1) != 2)).any()):
            raise ValueError("faces do not close to a common-edge ring")
        index = torch.arange(3, device=self.device)
        common_index = torch.where(common_first, index, 3).sort(-1).values[..., :2]
        # degree不同造成的padding槽不参与统计；先赋安全索引，不能截断有效槽错误。
        common_index = torch.where(self.valid[..., None], common_index, 0)
        self.edge_vertices = first.gather(-1, common_index)
        self.outer_vertices = first.gather(-1, (~common_first).to(torch.int8).argmax(-1)[..., None]).squeeze(-1)
        self.inner_vertices = second.gather(-1, (~common_second).to(torch.int8).argmax(-1)[..., None]).squeeze(-1)
        self.width = width

    @torch.inference_mode()
    def evaluate(self, *, vertices: torch.Tensor, signed_principals: bool = False) -> dict[str, torch.Tensor]:
        """surface RAS/mm的FP32(N,3)→K(mm⁻²)、H/K1/K2(mm⁻¹)及逐面几何。

        vertices必须在缓存目标设备、同N且有限；signed_principals=False按绝对值排K1/K2，
        True按带符号值。使用源码float累计、double sqrt/最后表达式，不改全局TF32。
        面积退化/非有限值抛ValueError，不截断面积、不改变网格、不静默回退。
        返回均为同设备Tensor，调用者负责同步/传输；此内部算子没有独立官方CLI。
        """
        if (not isinstance(vertices, torch.Tensor) or vertices.dtype != torch.float32
                or vertices.shape != (self.nvertices, 3) or vertices.device != self.faces.device):
            raise ValueError("vertices must be same-device float32 (N,3) matching cached topology")
        if not isinstance(signed_principals, bool) or not bool(torch.isfinite(vertices).all()):
            raise ValueError("finite coordinates and a bool signed_principals are required")
        triangle = vertices[self.faces]
        cross = _cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        length = _length(cross)
        if bool((length < 1e-7).any()):
            raise ValueError("degenerate face or source FZERO face-normal branch is unsupported")
        area = (length.double() * .5).float()
        normal = cross * length.reciprocal()[:, None]
        face_angles = []
        for corner in range(3):
            origin = triangle[:, corner]
            first = triangle[:, (corner + 2) % 3] - origin
            second = triangle[:, (corner + 1) % 3] - origin
            face_angles.append(_triangle_angle(_dot(_cross(second, first), normal), _dot(first, second)))
        angles = torch.stack(face_angles, dim=1)
        ni, nj = normal[self.face_ids], normal[self.next_faces]
        ratio = (_dot(ni, nj) / (_length(ni) * _length(nj))).clamp(-1, 1)
        normal_angle = torch.acos(ratio)
        outer, inner = vertices[self.outer_vertices], vertices[self.inner_vertices]
        positive = _length((outer + ni) - (inner + nj))
        negative = _length((outer - ni) - (inner - nj))
        signed_angle = normal_angle * torch.where(positive < negative, 1, -1)
        edge_length = _length(vertices[self.edge_vertices[..., 1]] - vertices[self.edge_vertices[..., 0]])
        area_sum = torch.zeros(self.nvertices, dtype=torch.float32, device=vertices.device)
        angle_sum = torch.zeros_like(area_sum)
        normal_sum = torch.zeros_like(area_sum)
        for slot in range(self.width):
            valid = self.valid[:, slot]
            area_sum = area_sum + torch.where(valid, area[self.face_ids[:, slot]], 0)
            angle_sum = angle_sum + torch.where(valid, angles[self.face_ids[:, slot], self.corners[:, slot]], 0)
            normal_sum = normal_sum + torch.where(valid, signed_angle[:, slot] * edge_length[:, slot], 0)
        numerator = torch.tensor(3, dtype=torch.float32, device=vertices.device)
        gaussian = ((numerator / area_sum).double() * (2 * math.pi - angle_sum.double())).float()
        mean_numerator = torch.tensor(.75, dtype=torch.float64, device=vertices.device)
        mean = (mean_numerator / area_sum.double() * normal_sum.double()).float()
        delta = mean * mean - gaussian
        delta_negative = delta < 0
        root = delta.clamp_min(0).double().sqrt()
        a, b = (mean.double() + root).float(), (mean.double() - root).float()
        first_larger = a >= b if signed_principals else a.abs() >= b.abs()
        first_smaller = a <= b if signed_principals else a.abs() <= b.abs()
        k1, k2 = torch.where(first_larger, a, b), torch.where(first_smaller, a, b)
        result = {"K": gaussian, "H": mean, "K1": k1, "K2": k2,
                  "face_area": area, "face_normal": normal, "face_angles": angles,
                  "delta_negative": delta_negative}
        if not all(bool(torch.isfinite(result[name]).all()) for name in ("K", "H", "K1", "K2")):
            raise ValueError("nonfinite curvature output")
        return result


def write_discrete_curvature(*, surface_file: str | Path, output_prefix: str | Path,
                             device: str = "cuda:0", signed_principals: bool = False) -> dict:
    """nibabel三角表面→八张同网格FP32 morph；完整加载/传输/写出墙钟。

    surface_file为有序三角网格surface RAS/mm，output_prefix为输出前缀，生成
    PREFIX.{K,H,K1,K2,BE,C,FI,S}.crv，返回outputs/vertices/device/seconds/范围。
    device默认cuda:0；signed_principals默认False。输入/拓扑/CUDA错误均抛异常，
    不自动回退，不写curv.stats或平滑/归一化后的占位输出。
    """
    tick = time.perf_counter()
    vertices, faces = fsio.read_geometry(str(surface_file))
    topology = DiscreteCurvatureTopology(faces=faces, nvertices=len(vertices), device=device)
    tensor = torch.as_tensor(np.array(vertices, dtype=np.float32, copy=True), device=device)
    maps = topology.evaluate(vertices=tensor, signed_principals=signed_principals)
    derived = curvature_derivatives_tensor(maps["K1"], maps["K2"])
    maps.update(derived)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in ("K", "H", "K1", "K2", "BE", "C", "FI", "S"):
        output = Path(str(prefix) + f".{name}.crv")
        fsio.write_morph_data(str(output), maps[name].cpu().numpy())
        outputs[name] = str(output)
    return {"outputs": outputs, "vertices": len(vertices), "device": str(device),
            "seconds": time.perf_counter() - tick, "signed_principals": signed_principals,
            "scope": "discrete principal and eight unprocessed maps; not full curv.stats",
            "delta_violations": int(maps["delta_negative"].sum().item())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--signed-principals", action="store_true")
    args = parser.parse_args()
    print(write_discrete_curvature(surface_file=args.surface, output_prefix=args.output_prefix,
                                   device=args.device, signed_principals=args.signed_principals))


if __name__ == "__main__":
    main()
