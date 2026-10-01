"""兼容旧 nuclei 导入名；计算与图谱准备统一使用 subregions 实现。

``segment_nuclei`` 与 ``segment_subregions`` 是同一函数对象，返回
``SubregionResult``。旧 Python 参数须迁移为 ``t1``、``atlas_root``、
``coarse_segmentation``、``wmparc`` 和 ``output_dir``；旧命令行参数由
``fnit-nuclei`` 转换。``prepare_nuclei_atlas`` 使用统一图谱布局和
``output_root`` 参数，不再建立旧 ``average`` 目录或下载全局 LUT。
"""

from .pipeline import segment_subregions as segment_nuclei
from .setup import prepare_subregion_atlases as prepare_nuclei_atlas
