# N4 校正单步重放

FreeSurfer 8.2 的 `mri_nu_correct.mni` 会运行 `AntsN4BiasFieldCorrectionFs -i orig.mgz -o nu0.mgz --dtype uchar`。其 ITK N4 滤波器使用四级拟合、每级 50 次迭代、收敛阈值 0、缩小倍数 4；未提供掩膜时使用所有体素，并以一个线程运行。这些设置对照了已完成被试的日志和 [FreeSurfer 源码提交 d932c45](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/AntsN4BiasFieldCorrectionFs/AntsN4BiasFieldCorrectionFs.cpp)。Python [`n4_sitk.py`](../../src/fnit/recon_all/n4_sitk.py) 使用 SimpleITK 2.5.6 重放**单独的 `orig.mgz` → `nu0.mgz` 步骤**，沿用 FreeSurfer 的 uchar 截断及 `floor(x + 0.5)` 取整。

## 输入、输出与调用

`correct_volume(input_file, output_file) -> None` 读取一张三维 MGH/MGZ `orig.mgz`，写出 uchar `nu0.mgz`。它在 CPU 上调用 SimpleITK/ITK，不生成后续 `nu.mgz`；当前 recon-all 入口将其用于 N4 阶段，再由 Python wrapper 生成 `nu.mgz`。

```python
from fnit.recon_all.n4_sitk import correct_volume

correct_volume(
    input_file="orig.mgz",  # 输入的原始 MGH/MGZ 影像
    output_file="nu0.sitk.mgz",  # 输出的 N4 校正后 uchar 影像
)
```

函数返回 `None`；校正图写入 `output_file`。安装 `SimpleITK==2.5.6`、`nibabel` 和 NumPy 后，也可运行：

```bash
python -m fnit.recon_all.n4_sitk --i orig.mgz --o nu0.sitk.mgz
```

`--i` 对应 `input_file`，`--o` 对应 `output_file`；两项都必填。对应的 FreeSurfer 单步命令为 `AntsN4BiasFieldCorrectionFs -i orig.mgz -o nu0.mgz --dtype uchar`。

## 真实输入的同阶段对照

测试读取 headcw 上已完成被试 sub01 的 256³ uint8 `orig.mgz`；仓库不含这张 MRI。官方可执行程序为 FreeSurfer 8.2.0-1，SHA-256 为 `9d77c96f6462e0f0f46a42b3ed59d7ed7364318d2006fb31dff96a953929d239`。官方和 Python 都以一个 CPU 线程读取同一输入并写出 `nu0.mgz`。Python 将 SimpleITK 除法结果转回 float32 后再取整为 uchar，与 FreeSurfer 的 ITK 图像类型一致。

| 项目 | 官方 FreeSurfer | Python/SimpleITK |
| --- | ---: | ---: |
| 墙钟耗时 | 168.26 s | 96.14 s |
| 输出 MGZ 文件 SHA-256 | `4db513d79a7f64cc72a86b93d093f332f2fc13f63d4f3602d5ca18ca1a4c5b3c` | `5b4ff78c1629dfb4f20abd18dfda4c4dbfe462f185e5f328552fa9a4d05aa299` |
| 解码体素 SHA-256（C 顺序 uint8） | `ef6bd8aff016332d3e317a743358dce8f92bc63d26d82a5498dbd689647cf982` | `ef6bd8aff016332d3e317a743358dce8f92bc63d26d82a5498dbd689647cf982` |
| 解压 MGH SHA-256 | `8fe2938cf0d94b38019758c1c25c79b27783da4f71ca89fc28a4465c2d6e49aa` | `e4510816373b619b01b4e2f0b3cb94c69a02b8957c771bc06598bbbc1cbe88a4` |

输入 MGZ 文件 SHA-256 为 `d79723f94bfc149ff36c89094a3d734b888a03cecbc32dc57a22992e8a5e817f`（2,038,466 字节）；解码体素 SHA-256 为 `84da8a990ef60c6ba30a4ddfbaba597a90e32bca720f96c6d741fcb08c504825`。

两份输出均为 256³ uint8，仿射相同；**16,777,216 个解码体素全部一致**，MGH 头的前 284 字节也相同。完整解压文件仍不同：官方尾部为 1,345 字节，Python 通过 `mgh_compat.py` 保留输入的 1,346 字节尾部。N4 尾部带 XFORM 标签，不能套用本包其他阶段的 SynthSeg 色表规范化规则。因此本次体素与仿射一致不代表 MGH 元数据逐字节一致，也不验收下游皮层指标。较早的 SimpleITK 重放耗时 124.65–125.58 秒；同一主机上最终一次为 96.14 秒，尚无受控试验解释时间波动。

下载的 SimpleITK wheel 为 52.8 MB，临时安装占 265 MB。它调用 CPU 编译版 ITK，替代了此步的 FreeSurfer N4 可执行程序，但不是 GPU N4。另一个实验性 Torch 近似实现 `n4_gpu.py` 未通过本项检查（相对官方 `nu0.mgz` 的前景 MAE 5.434、RMSE 6.801），未用于 recon-all。

上游 `mri_nu_correct.mni` 在 `nu0.mgz` 后还会恢复全局平均强度、按原始几何转换，并利用 Talairach 变换调用 `mri_make_uchar`。后续 Python wrapper 和完整阶段的逐字节对照见 [N4_WRAPPER_VALIDATION.md](N4_WRAPPER_VALIDATION.md)。当前 recon-all 入口调用本 CPU N4 阶段；这份单步对照本身不检验后续分割和表面指标。

headcw 的 `/tmp/reconall_n4_sub01_20260925/` 保存验证日志 `official.log`、`final_sitk.log`、`final_compare.log`、`pytest_all.log`（5 项测试通过）。本次重放所用模块源码 SHA-256 为 `7d864006979ffe456d1fad8245f9dc1fc3cdaead7f435fd4beef4d853b2eae20`。
