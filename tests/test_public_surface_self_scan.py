"""候选枚举的有相交单元合同；真实 MRI benchmark 另存独立报告。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fnit.recon_all.mris_remove_intersection_python import (
    _triangles_intersect,
    mark_intersections,
)


_SCANNER_PATH = Path(__file__).resolve().parents[1] / (
    "validation/fmri/public_ten_20261003/scan_surface_self_intersections.py"
)
_SPEC = importlib.util.spec_from_file_location("public_self_scan_contract", _SCANNER_PATH)
_SCANNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCANNER)


@pytest.mark.parametrize("case_index", range(12))
def test_radius_groups_chunks_and_original_pair_order(case_index):
    # 不同半径、原全局编号与组顺序不一致、共享顶点和有相交的候选。
    generator = np.random.default_rng(120311)
    for _ in range(case_index + 1):
        vertices = generator.normal(size=(24, 3)) * np.repeat([.1, 1., 10., 100.], 6)[:, None]
        faces = np.asarray([generator.choice(len(vertices), 3, replace=False)
                            for _ in range(25)], dtype=np.int32)
    expected_marks, expected_faces = mark_intersections(vertices, faces)
    marks, face_marks, _ = _SCANNER.scan(
        vertices, faces, _triangles_intersect, threads=2, block_faces=3,
        maximum_pairs_per_block=4, timeout_seconds=30,
    )
    assert expected_faces > 0  # 正分支必须真正进入原三角判定。
    assert np.array_equal(marks, expected_marks)
    assert int(face_marks.sum()) == expected_faces
