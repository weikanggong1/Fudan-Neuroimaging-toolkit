"""WM/aseg 编辑共用的整数标签与原始存储校验，无CUDA初始化。"""
from __future__ import annotations

import nibabel as nib
import numpy as np


def integer_label_array(image: nib.MGHImage) -> np.ndarray:
    """同网格MGH整数语义标签→XYZ int32数组；拒绝非有限或小数标签。

    允许float32存储精确整数，不改变毫米affine。空数据、超int32范围、
    小数或非有限标签抛ValueError。只转换数据，不写文件、不调用原软件。
    """
    source = np.asarray(image.dataobj)
    if (source.size == 0 or not np.isfinite(source).all()
            or source.min() < np.iinfo(np.int32).min
            or source.max() > np.iinfo(np.int32).max):
        raise ValueError("label volume is not finite int32-compatible data")
    result = np.asarray(source, dtype=np.int32, order="C")
    if not np.array_equal(source, result):
        raise ValueError("label volume contains noninteger values")
    return result


def validate_aseg_storage(image: nib.MGHImage) -> None:
    """完整WM编辑的byte-access证明仅覆盖原始4字节int32/float32存储。

    float32还须integer_label_array精确整数检查。拒绝int16/uint8等其他
    原存储；事后转int32不等价于源程序的原存储字节索引，无隐式回退。
    """
    storage = image.get_data_dtype().newbyteorder("=")
    if storage not in (np.dtype(np.int32), np.dtype(np.float32)):
        raise ValueError("complete hybrid byte-access proof requires original int32 or float32 aseg storage")
