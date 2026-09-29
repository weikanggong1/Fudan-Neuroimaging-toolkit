# fsaverage 表面标签映射

`SurfaceLabelMapper` 对应固定单 T1 流程中的 `mri_label2label --srcsubject fsaverage --trgsubject sub01 --regmethod surface`。输入为 fsaverage 的双侧 `sphere.reg` 和 72 张 `.label`，以及被试的双侧 `sphere.reg` 与 `white`；输出为被试 `label/H.*.label`，每行保存目标顶点编号、surface RAS 坐标和统计值。每侧缓存一次源/目标球面最近顶点关系，再处理该侧的 36 张标签。

对 2026-09-24 的一例真实 T1 冻结官方表面，FNIT 的 72/72 张标签与官方文件逐字节一致，包括顶点顺序、坐标、统计文本和文件头。代表性 `lh.FG1.mpm.vpnl.label` 的 SHA-256 为 `02df9ccb88212f18f9e5be7e9b8ed2e0351804465a661481686c2c422e3b8054`，`rh.BA1_exvivo.label` 为 `0bc04fd8fcdf94d3a9e858421b14fde584664a63541f51414282eb4e442557b3`。输入表面来自官方冻结整例，此结果不能替代 FNIT 自产表面的连续验证。2026-09-29 已将映射器接入标准入口，并核对六张最终注释的全部顶点编码；见[接线记录](exvivo_wiring_20260929.json)。

| 旧版同输入计时 | 官方 | FNIT |
| --- | ---: | ---: |
| 左侧 FG1，headcw 单次调用 | 3.17 s | 0.53 s |
| 右侧 BA1，headcw 单次调用 | 2.94 s | 0.61 s |
| 72 张标签 | gpucw1 旧日志合计 236.24 s | headcw FNIT 旧观察 7.90 s |

最后一行不是同机配对，不能用于速度比。映射器建立双侧缓存分别观察到 0.264 s 和 0.287 s。所用 fsaverage 资产是两个球面和 72 张标签，合计 21,196,583 字节；FNIT 运行时不需要系统 FreeSurfer 程序。

```python
from pathlib import Path
from fnit.recon_all.label2label_surface_python import SurfaceLabelMapper

mapper = SurfaceLabelMapper(
    source_sphere=Path("/data/fnit-assets/subjects/fsaverage/surf/lh.sphere.reg"),  # fsaverage 左侧配准球面
    target_sphere=Path("/data/subjects/sub01/surf/lh.sphere.reg"),  # 被试左侧配准球面
    target_white=Path("/data/subjects/sub01/surf/lh.white"),  # 被试左侧 white，提供目标坐标
    target_subject="sub01",  # 输出 label 文件头中的被试名
)
target_vertex_ids = mapper.map_label(
    source_label=Path("/data/fnit-assets/subjects/fsaverage/label/lh.BA1_exvivo.label"),  # 源标签
    output_label=Path("/data/subjects/sub01/label/lh.BA1_exvivo.label"),  # 目标标签
)
# target_vertex_ids 为输出文件中的目标顶点编号数组。
```

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
