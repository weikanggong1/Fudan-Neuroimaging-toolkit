# T1 脑图像去噪

`denoise_volume` 对 `brain.mgz` 执行 FreeSurfer `recon-all` 固定参数的自适应非局部均值去噪，生成 `antsdn.brain.mgz`，供后续白质分割使用。仓库内 Numba 核心在 CPU 上计算 3×3×3 局部均值/方差、5×5×5 搜索邻域、3×3×3 图像块权重和重叠块平均。输出按 FreeSurfer 的正数半进位规则写为 `uint8`，用 NiBabel 读取图像并保留源文件的 MGH 头部及尾部标签。无需 ANTsPy 或已安装的 FreeSurfer。

## 输入、输出和使用方法

| 项目 | 含义 |
| --- | --- |
| `input_file` | 三维 `uint8` `.mgh`/`.mgz` 脑图像路径；当前 recon-all 使用 `mri/brain.mgz`。不是原始 T1 NIfTI。 |
| `output_file` | 输出 `.mgh`/`.mgz` 路径；通常为同一被试目录下的 `mri/antsdn.brain.mgz`。 |
| 返回值 | 字典：`voxels` 为写出的体素数，`total_seconds` 为函数从读入到写盘的墙钟秒数。 |
| 输出结构 | 单幅三维 `uint8` MGH/MGZ；维度、仿射和 MGH 元数据取自输入，仅更换体素值。 |

安装仓库的 recon-all Python 阶段依赖后，在 Python 中调用：

```python
from fnit.recon_all.ants_denoise_python import denoise_volume

result = denoise_volume(
    input_file="/data/sub01/mri/brain.mgz",  # 输入：已生成的三维 uint8 脑图像
    output_file="/data/sub01/mri/antsdn.brain.mgz",  # 输出：去噪后的 MGH/MGZ 文件
)
# result["voxels"] 是输出体素数；result["total_seconds"] 是本次函数墙钟秒数。
```

独立命令行：

```bash
python -m fnit.recon_all.ants_denoise_python \
  --i /data/sub01/mri/brain.mgz \
  --o /data/sub01/mri/antsdn.brain.mgz
```

`--i` 和 `--o` 分别对应上述 `input_file` 和 `output_file`。完整重建由 `fnit-recon-all` 自动调用同一个 `denoise_volume` 函数，无须单独执行本命令。

官方 FreeSurfer 对应命令仅用于配对验证：

```bash
AntsDenoiseImageFs -i /data/sub01/mri/brain.mgz -o /data/sub01/mri/antsdn.brain.mgz
```

该官方调用不传 `--rician`，使用 Gaussian 模型；固定搜索半径为 2，图像块及局部统计半径为 1。仓库函数输入只接受这条固定的 `uint8`、三维 recon-all 路径，不提供未验收的其他半径或噪声模型开关。

## 同输入验收

[真实 T1 的逐体素和耗时报告](../../validation/recon_all/python_gpu_port/ANTS_DENOISE_STATUS.md)使用官方同名命令输出作参照。当前 Numba CLI 输出与官方 **16,777,216/16,777,216** 个体素相同；单次观测为 Python CLI **19.72 秒**、官方 **27.58 秒**。官方写盘的尾部标签比输入多 1 字节，故压缩文件 SHA-256 不相同。该阶段使用 CPU，整例的表面、配准和脑区指标仍需独立验收。
