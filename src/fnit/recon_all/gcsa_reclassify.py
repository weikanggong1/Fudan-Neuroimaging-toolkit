"""Seeded asynchronous GCSA Gibbs label reclassification."""

from __future__ import annotations

import numpy as np

from .gcsa_gibbs import GibbsModel
from .gcsa_initial import InitialAtlas
from .gcsa_permutation import VnlRandom, vertex_permutation


def reclassify_gibbs(model: GibbsModel, atlas: InitialAtlas, *, seed: int = 1234,
                     max_iterations: int | None = None,
                     snapshot=None, backend: str = "python") -> list[dict]:
    """按固定种子/顶点顺序运行GCSAreclassifyUsingGibbsPriors。

    model为同网格GibbsModel，atlas为其InitialAtlas；seed默认1234，
    max_iterations默认None沿原收敛条件，snapshot接收每轮标签原视图。
    backend默认python，numba显式使用CSR有序CPU内核，无并行更新。
    返回iteration/changed/examined整数列表，原位改变model.labels；标签为
    int32打包RGB，无空间变换。非法后端、概率或算子失败抛异常。
    """
    labels = model.labels
    mark = np.ones(len(labels), np.uint8)
    random = VnlRandom(seed)
    random.open_ran1()  # setRandomSeed primes OpenRan1 once.
    if backend not in ("python", "numba"):
        raise ValueError("Gibbs backend must be python or numba")
    if backend == "numba":
        from .gcsa_gibbs_numba import pack_model, ordered_sweep
        packed = pack_model(model, atlas)
    history = []
    iteration = 0
    while True:
        changed = 0
        examined = 0
        permutation = vertex_permutation(random, len(labels))
        if backend == "numba":
            changed, examined = ordered_sweep(packed=packed, permutation=permutation,
                mark=mark, feature=model.feature, labels=labels)
        for vertex in (() if backend == "numba" else permutation):
            if mark[vertex] == 0:
                continue
            mark[vertex] = 0
            examined += 1
            choices = atlas.prior_nodes[int(model.prior_indices[vertex])]
            if len(choices) <= 1:
                continue
            old = int(labels[vertex])
            best = old
            maximum = model.neighborhood_log_likelihood(int(vertex), old)
            for candidate, _ in choices:
                likelihood = model.neighborhood_log_likelihood(int(vertex), candidate)
                if likelihood > maximum:
                    maximum = likelihood
                    best = candidate
            if best != old:
                labels[vertex] = best
                mark[vertex] = 1
                changed += 1
        iteration += 1
        history.append({"iteration": iteration - 1, "changed": changed,
                        "examined": examined})
        if snapshot is not None:
            snapshot(iteration, labels)
        if changed == 0 or (max_iterations is not None and iteration >= max_iterations):
            break
        for vertex in np.flatnonzero(mark == 1):
            for neighbor in model.neighbors[vertex]:
                if mark[neighbor] != 1:
                    mark[neighbor] = 2
    return history
