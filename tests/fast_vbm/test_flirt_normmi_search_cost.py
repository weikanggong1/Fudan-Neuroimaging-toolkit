"""验证仅指定 ``-cost normmi`` 时 FSL 的两阶段代价函数。"""

import numpy as np

from fnit.flirt.core import (
    FSLCorrelationRatio,
    FSLNormalizedMutualInformation,
    _RigidNMIEngine,
)


def test_rigid_nmi_keeps_default_correlation_ratio_for_angular_search():
    image = np.arange(12 * 12 * 12, dtype=np.float32).reshape(12, 12, 12)
    engine = _RigidNMIEngine(
        moving=image,
        reference=image,
        moving_vox2world=np.eye(4),
        reference_vox2world=np.eye(4),
        moving_voxel_sizes=(2.0, 2.0, 2.0),
        reference_voxel_sizes=(2.0, 2.0, 2.0),
        device="cpu",
    )
    assert type(engine.level.cost) is FSLCorrelationRatio
    engine.set_scale(4.0)
    assert type(engine.level.cost) is FSLNormalizedMutualInformation
