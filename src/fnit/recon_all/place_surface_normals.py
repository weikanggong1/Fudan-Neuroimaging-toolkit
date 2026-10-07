"""Source-order vertex normals for the initial placement surface."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _unit(vector: np.ndarray) -> None:
    length = np.float32(np.sqrt(np.float32(
        np.float32(vector[0] * vector[0] + vector[1] * vector[1]) + vector[2] * vector[2]
    )))
    if length > 0:
        vector[0] = np.float32(vector[0] / length)
        vector[1] = np.float32(vector[1] / length)
        vector[2] = np.float32(vector[2] / length)


@njit(parallel=True, cache=True, fastmath=False)
def _normals(xyz: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, corners: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in prange(len(xyz)):
        normal = np.zeros(3, dtype=np.float32)
        for entry in range(offsets[vertex], offsets[vertex + 1]):
            face = faces[face_ids[entry]]
            corner = corners[entry]
            previous = face[(corner + 2) % 3]
            following = face[(corner + 1) % 3]
            v0 = np.empty(3, dtype=np.float32)
            v1 = np.empty(3, dtype=np.float32)
            for axis in range(3):
                v0[axis] = np.float32(xyz[vertex, axis] - xyz[previous, axis])
                v1[axis] = np.float32(xyz[following, axis] - xyz[vertex, axis])
            _unit(v0)
            _unit(v1)
            face_normal = np.array([
                np.float32(-v1[1] * v0[2] + v0[1] * v1[2]),
                np.float32(v1[0] * v0[2] - v0[0] * v1[2]),
                np.float32(-v1[0] * v0[1] + v0[0] * v1[1]),
            ], dtype=np.float32)
            _unit(face_normal)
            normal[0] = np.float32(normal[0] + face_normal[0])
            normal[1] = np.float32(normal[1] + face_normal[1])
            normal[2] = np.float32(normal[2] + face_normal[2])
        _unit(normal)
        result[vertex] = normal
    return result


def ordered_face_csr(faces: np.ndarray, nvertices: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """建立每顶点关联面的有序 CSR，只计算整数索引。

    ``faces`` 为 (F, 3) 整数三角面数组，行顺序为面顺序，列为角点顺序；
    ``nvertices`` 为非负顶点总数，包含孤立顶点。返回 int64 数组
    ``offsets`` (N+1,)、``face_ids`` (3F,)、``corners`` (3F,)；第 v 行
    位于 [offsets[v], offsets[v+1])，对应原面号及 0/1/2 角点号。
    稳定排序保留原 face/corner 次序和所有重复项；零 degree 行为空。
    不读取坐标，不改变 RAS 或单位，没有算法默认参数。形状、整数类型、
    顶点数量或索引范围无效时抛出 ValueError，不截断或修补网格。
    属于 mris_sphere/mris_register/mris_place_surface 的内部拓扑准备，
    没有独立官方 CLI；成熟法向内核及真实验证见对应阶段报告。
    """
    if (not isinstance(nvertices, (int, np.integer))
            or isinstance(nvertices, (bool, np.bool_)) or nvertices < 0):
        raise ValueError("nvertices must be a nonnegative integer")
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must have an integer dtype")
    if np.any(faces < 0) or np.any(faces >= nvertices):
        raise ValueError("face indices must be within [0, nvertices)")
    indices = np.asarray(faces, dtype=np.int64).ravel()
    counts = np.bincount(indices, minlength=nvertices)
    offsets = np.empty(nvertices + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(indices, kind="stable")
    return offsets, np.asarray(order // 3, dtype=np.int64), np.asarray(order % 3, dtype=np.int64)


def initial_vertex_normals(vertices: np.ndarray, triangles: np.ndarray, *,
                           topology: FaceNormalTopology | None = None) -> np.ndarray:
    """按原有面顺序计算每顶点单位法向，复用已有单精度数值内核。

    vertices 为 (N, 3) 坐标，triangles 为 (F, 3) 有序顶点索引；输入转换为
    float32/int64；整数索引由共享 ordered_face_csr 建立。坐标单位不影响
    单位法向；返回 (N, 3) float32，无关联面的顶点返回零向量。稳定排序
    只构造关联索引，不重排面或更改累计顺序，重复角点保留。
    属于 mris_sphere / mris_place_surface 的内部几何步骤，无独立 CLI。
    非法三角形形状、非整数索引、负值或越界索引抛出 ValueError。
    vertices须为(N,3)；topology=None可传本次网格FaceNormalTopology，
    复用整数CSR，坐标/法向每次重算；不兼容的缓存抛ValueError。
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if topology is not None:
        topology.validate(triangles, len(xyz))
        return topology.evaluate(xyz)
    offsets, face_ids, corners = ordered_face_csr(triangles, nvertices=len(xyz))
    faces = np.asarray(triangles, dtype=np.int64)
    return _normals(xyz, faces, face_ids, corners, offsets)


NORMAL_TOPOLOGY_API_VERSION = 1


class FaceNormalTopology:
    """同一有序面网格的显式法向拓扑缓存；坐标和法向从不缓存。

    triangles为整数(F,3)，nvertices为N；构造时验证索引并复制/冻结面、
    offsets(N+1)、face_ids/corners(3F)，保留面序/角点序。
    evaluate(vertices)输入float32(N,3)surface RAS/mm，返回原内核
    float32单位法向；shape不符抛ValueError。没有全局驻留/坐标近似。
    换拓扑需新建，white.preaparc/final white/pial可各自持有上下文。
    """
    def __init__(self, triangles: np.ndarray, nvertices: int):
        self.offsets, self.face_ids, self.corners = ordered_face_csr(triangles, nvertices)
        self.faces = np.array(triangles, dtype=np.int64, copy=True)
        self.nvertices = nvertices
        for array in (self.offsets, self.face_ids, self.corners, self.faces):
            array.flags.writeable = False

    def evaluate(self, vertices: np.ndarray) -> np.ndarray:
        """仅重算法向：同N有序坐标→(N,3)float32，孤立顶点为零。"""
        xyz = np.asarray(vertices, np.float32)
        if xyz.shape != (self.nvertices, 3):
            raise ValueError("coordinates differ from cached topology shape")
        return _normals(xyz, self.faces, self.face_ids, self.corners, self.offsets)

    def validate(self, triangles: np.ndarray, nvertices: int) -> None:
        """验证外部面/顶点数与冻结拓扑一致，防止跨拓扑误用。"""
        triangles = np.asarray(triangles)
        if nvertices != self.nvertices or not np.issubdtype(triangles.dtype, np.integer) or not np.array_equal(triangles, self.faces):
            raise ValueError("cached normals require identical ordered topology")


class CoordinateNormalCache:
    """缓存同一优化上下文中同一坐标数组的法向。

    ``FaceNormalTopology`` 只缓存有序面拓扑，连续放置循环仍会对同一个
    ``current`` 数组在目标函数和梯度之间重复计算法向。本类按 NumPy 数组
    对象身份复用一次计算结果；坐标数组一旦换成新对象就重新计算，因此不
    使用近似坐标、哈希或容差，也不改变 float32 累加顺序。调用方必须把
    坐标视为只读；若原地修改坐标，应调用 ``clear`` 后再取法向。

    这是 ``mris_place_surface`` 内部优化，没有独立 CLI。缓存只绑定一个
    ``FaceNormalTopology``，不会跨 white.preaparc、final white 和 pial 网格
    共享法向。
    """

    def __init__(self, topology: FaceNormalTopology):
        if not isinstance(topology, FaceNormalTopology):
            raise TypeError("topology must be a FaceNormalTopology")
        self.topology = topology
        self._vertices: np.ndarray | None = None
        self._normals: np.ndarray | None = None

    def evaluate(self, vertices: np.ndarray) -> np.ndarray:
        """返回同一坐标对象的缓存法向，换对象时按原内核重算。"""
        xyz = np.asarray(vertices, dtype=np.float32)
        if self._vertices is xyz and self._normals is not None:
            return self._normals
        normals = self.topology.evaluate(xyz)
        self._vertices = xyz
        self._normals = normals
        return normals

    def clear(self) -> None:
        """丢弃坐标引用和法向，供调用方在原地更新坐标后显式失效。"""
        self._vertices = None
        self._normals = None
