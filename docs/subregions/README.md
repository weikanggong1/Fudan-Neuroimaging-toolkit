# 皮下亚区：TorchGEMS 实验接口

[返回首页](../../README.md) · [SynthSeg](../synthseg/README.md)

`TorchGEMS` 读取 FreeSurfer GEMS `AtlasMesh.gz` 和 `compressionLookupTable.txt`，在 PyTorch 上计算四面体插值的标签先验、Gaussian EM 后验和可选网格变形。`segment_subregions` 把一个或多个图谱包放到 T1 个体体素网格，输出单幅标签图。此接口目前是通用引擎，**尚未复现** FreeSurfer brainstem、thalamic nuclei、hippocampal subfields 的结构专用流程；其输出不应当作官方核团分割的等价结果。

## 输入图谱

每个结构需一个目录，例如 `/absolute/path/atlases/brainstem/`，其中有 `AtlasMesh.gz`、`compressionLookupTable.txt`。也可由 `GEMSAtlas.save_npz()` 保存为 `atlas.npz`。可选 `atlas_to_native_voxel.npy` 是 4×4 仿射，把图谱网格顶点变换到本次 T1 **体素坐标**；缺少时会尝试用 SynthSeg 与图谱共享标签的质心拟合仿射。这个初始化不是 FreeSurfer 的专用配准，且共享标签不足四个时会报错。用户必须确认图谱的来源、许可、方向及目标结构标签；图谱不随 wheel 分发。

每个图谱包必须有 `config.json` 并显式写出 `include_label_ids`；否则周围的脑组织会被当作核团输出。可选 `support_coarse_label_ids` 把输出限制到粗标签区域。以下仅演示 brainstem 四个细分标签的输入契约，不是已经验收的官方参数：

```json
{"include_label_ids": [173, 174, 175, 178], "support_coarse_label_ids": [16]}
```

其他可选项有 `mesh_index`、`label_classes`、`alignment_label_map`、`output_label_map`、`output_name_prefix`、`em_iterations`、`deform_iterations`、`deform_lr` 和 `deformation_weight`。其中 `label_classes` 每项对应图谱压缩标签的 Gaussian 类别；若省略，每个标签各拟合一类。官方核团脚本使用专用分组及超先验；默认配置不能替代它们。

## Python 用法

```python
from fnit import segment_subregions

result = segment_subregions(
    t1="sub-01_T1w.nii.gz",                         # 输入：单幅 3D T1 路径
    atlas_root="/absolute/path/atlases",           # 图谱根目录，子目录为结构名
    structures=["brainstem"],                      # 只运行指定图谱包；"all" 为全部
    coarse_segmentation="sub-01_33class.nii.gz",  # 同一 T1 网格的粗结构标签
    synthseg_weights=None,                          # 无粗标签时自动运行 SynthSeg 的权重
    auto_initialize=True,                           # 无显式 4×4 仿射时拟合标签质心
    device="cuda:0",                               # PyTorch GPU；也可用 "cpu"
    em_iterations=8,                                # Gaussian EM 迭代次数
    deform_iterations=0,                            # 网格变形迭代次数；0 为不变形
)
result.labels.save("sub-01_subregions.nii.gz")
roi = result.mask("Medulla")
```

`t1`、`coarse_segmentation` 可为路径或 Nibabel image；粗标签也可直接给三维 NumPy 数组，必须与 T1 同网格。`atlas_root` 下的每个结构目录只用一个图谱。返回 `labels` 为原 T1 shape/affine 的 `int32` NIfTI，`label_table` 为 `{编号: 名称}`，`confidence` 为原网格上已选标签的最大后验，`initialization` 记录图谱到 T1 的仿射与处理裁剪范围，`structure_results` 保留每个结构裁剪区的后验、先验、顶点、Gaussian 参数、目标函数历史和最小 Jacobian。多图谱同体素竞争时取后验置信度最高者；重复的输出标签编号若名称冲突则报错。

CLI 单结构命令：

```bash
fnit subregions --i sub-01_T1w.nii.gz --o sub-01_subregions.nii.gz \
  --atlas-root /absolute/path/atlases --structure brainstem \
  --coarse-segmentation sub-01_33class.nii.gz --device cuda:0 \
  --em-iterations 8 --deform-iterations 0
```

与之比较的 FreeSurfer 命令依赖事先完成的 recon-all subject，而 FNIT 入口接收原始 T1 和显式图谱；例如 `segmentThalamicNuclei.sh sub-01 "$SUBJECTS_DIR"`、`segmentBS.sh sub-01 "$SUBJECTS_DIR"`、`segmentHA_T1.sh sub-01 "$SUBJECTS_DIR"`。三者的输入、网格及算法步骤不同，当前不能把标签值或运行时间直接视作同方法对比。实际图谱解析规模、测试范围及缺失的数值对照见[验证记录](../../validation/subregions/README.md)。
