# 六张 BA/VPnl 注释的同输入验证

`label2annot_python.py` 对应双侧 `BA_exvivo`、`BA_exvivo.thresh` 和 `mpm.vpnl` 六次 `mris_label2annot --maxstatwinner`。输入为同侧 `orig` 的有序顶点、已映射到被试的逐顶点 `.label` 文件，以及对应的颜色表；输出为 `.annot` 文件。相同最大统计值由命令中靠后的标签胜出，未命中的顶点使用颜色表中的 unknown 编码。输出编码的顶点序号与 `orig` 相同。

2026-09-24 的冻结真实 T1 `fs_sub01` 上，以官方已生成的标签为输入，六张文件与归档官方文件逐字节一致。把 72 张输入标签换成 FNIT 映射器先前逐字节验证的文件后，六张注释仍逐字节一致。这只验证冻结表面上的连续标签链，不是从原始 T1 开始的整例。

2026-09-29 的主流程接线测试发现，按资产清单顺序组装 BA 标签会使 `lh.BA_exvivo`、左右阈值 BA 分别有 5、67、199 个顶点编码不同。原因是同统计值时的标签优先级随命令顺序变化。主流程改为官方命令中的顺序后，在冻结真实 T1 的双侧 `sphere.reg`、`white`、`orig` 上，六张注释的全部顶点编码与官方一致。[接线记录](exvivo_wiring_20260929.json)。测试目录名称和颜色表路径不同，故此次比较顶点编码，不以文件哈希作为标准。

旧版同主机的一次左 BA 单命令观察耗时：官方 0.564 秒，FNIT 0.263 秒。短阶段单次观察不能推出整例速度差异。

```python
from pathlib import Path
from fnit.recon_all.label2annot_python import write_label_annotation

write_label_annotation(
    surface_file=Path("/data/subjects/sub01/surf/lh.orig"),  # 左半球有序表面
    color_table_file=Path("/data/fnit-assets/average/colortable_BA.txt"),  # BA 颜色表
    label_files=[Path("/data/subjects/sub01/label/lh.BA1_exvivo.label")],  # 已映射标签；完整 BA 应按官方顺序提供全部 14 张
    output_file=Path("/data/subjects/sub01/label/lh.BA_exvivo.annot"),  # 输出逐顶点注释
)
```

上述单标签例子只展示接口，完整 `BA_exvivo` 使用 14 张标签，入口已固定顺序。对应官方命令形式为 `mris_label2annot --s sub01 --hemi lh --ctab colortable_BA.txt --l lh.BA1_exvivo.label ... --a BA_exvivo --maxstatwinner --noverbose`。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
