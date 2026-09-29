# 拓扑缺陷体积图

`native_free._run_defects_volume(binary, subject, hemi, assets)` 将拓扑修复阶段写出的顶点缺陷标记投射到被试 conform 体积网格。标准单 T1 流程先生成 `label/H.nofix.cortex.label`，再调用 Conda 源码构建的 `mri_label2vol --defects`。`H` 为 `lh` 或 `rh`。

| 参数 | 输入含义 |
| --- | --- |
| `binary` | 已核验的 Conda `mri_label2vol` 可执行文件路径。 |
| `subject` | 被试目录，包含 `mri/`、`surf/`、`label/`、`scripts/`。 |
| `hemi` | `lh` 或 `rh`；必须先处理左侧，再处理右侧。 |
| `assets` | 经过校验的 FNIT 模板目录，作为子进程的 `FREESURFER_HOME`。 |

每侧需要 `surf/H.orig.nofix`、`surf/H.defect_labels` 和 `label/H.nofix.cortex.label`。左侧以 `mri/orig.mgz` 为模板写 `mri/surface.defects.mgz`；右侧读同一输出并合并右侧缺陷。输出是与 `mri/orig.mgz` 同 shape、同 conform 仿射的 MGH 体积，编码包含左右侧的 1000/2000 偏移。函数返回 `None`；缺少前置文件、原生命令失败或输出缺失时抛异常。

标准流程内部调用示例：

```python
from pathlib import Path
from fnit.recon_all.native_free import _run_defects_volume

_run_defects_volume(
    binary=Path("/conda/envs/fnit/bin/mri_label2vol"),  # Conda 编译程序
    subject=Path("/data/subjects/sub01"),  # 已生成拓扑缺陷文件的被试目录
    hemi="lh",  # 先写左侧缺陷；随后以 "rh" 合并右侧
    assets=Path("/data/fnit-assets"),  # 固定源码版本的数据资产目录
)
```

官方 FreeSurfer 8.2 的对应命令为：

```bash
mri_label2vol --defects surf/lh.orig.nofix surf/lh.defect_labels \
  mri/orig.mgz 1000 0 mri/surface.defects.mgz label/lh.nofix.cortex.label
mri_label2vol --defects surf/rh.orig.nofix surf/rh.defect_labels \
  mri/surface.defects.mgz 2000 1 mri/surface.defects.mgz label/rh.nofix.cortex.label
```

## 真实数据配对

输入为仓库去标识的 `examples/data/sub-01_T1w.nii.gz` 所对应的已归档官方 FreeSurfer 8.2 被试。候选和官方程序各自读取**相同的**双侧 `orig.nofix`、`defect_labels`、皮层标签与 `orig.mgz`，在 headcw 顺序运行，两个输出的 256 × 256 × 256 个体素差异为 **0**，仿射最大差为 **0**，dtype 同为 `int32`。压缩文件哈希不同，因此没有声称完整文件字节一致。候选左右侧单次墙钟为 1.11/1.03 秒，官方为 1.81/1.07 秒；共享节点单次计时不作为加速结论。此结果验证投射命令和当前源码构建程序，不验证 FNIT 自产拓扑上游的整例数值精度；非零自相交修复是另一阶段。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码中的 mri_label2vol](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mri_label2vol)。
