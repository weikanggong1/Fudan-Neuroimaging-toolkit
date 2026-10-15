"""阶段内固定面拓扑缓存；当前坐标、rip、候选与接受状态不缓存。"""
from __future__ import annotations
import hashlib
import numpy as np


class PlacementTopologyCache:
    """以顶点数和完整int32有序面摘要绑定CSR及设备面索引。

    faces为(F,3)顶点编号，vertex_count为N。validate在每次试步核对完整
    拓扑版本，原数组原位改变也拒绝复用；新网格须建立新缓存。实例只属于
    一个阶段，不跨线程共享。device_faces只缓存静态面，坐标始终由调用方
    提供当前值。没有全局缓存或新增依赖。
    """
    def __init__(self, faces, vertex_count):
        triangles = np.asarray(faces, dtype=np.int32)
        if triangles.ndim != 2 or triangles.shape[1] != 3 or vertex_count < 0:
            raise ValueError('faces must be (F,3) and vertex_count nonnegative')
        if np.any(triangles < 0) or np.any(triangles >= vertex_count):
            raise ValueError('faces exceed vertex domain')
        self.vertex_count = int(vertex_count)
        self.faces = np.ascontiguousarray(triangles).copy()
        self.version = hashlib.sha256(self.faces.tobytes()).hexdigest()
        flat = self.faces.ravel()
        self.incident = np.argsort(flat, kind='stable') // 3
        self.incident_offsets = np.zeros(vertex_count + 1, np.int64)
        self.incident_offsets[1:] = np.cumsum(np.bincount(flat, minlength=vertex_count))
        self._device_faces = {}

    def validate(self, faces, vertex_count):
        triangles = np.asarray(faces, dtype=np.int32)
        if (vertex_count != self.vertex_count or triangles.shape != self.faces.shape
                or hashlib.sha256(np.ascontiguousarray(triangles).tobytes()).hexdigest() != self.version):
            raise ValueError('placement topology cache version changed; create a new stage cache')

    def device_faces(self, device):
        import torch
        target = str(torch.device(device))
        if target not in self._device_faces:
            self._device_faces[target] = torch.as_tensor(self.faces.copy(), device=target)
        return self._device_faces[target]
