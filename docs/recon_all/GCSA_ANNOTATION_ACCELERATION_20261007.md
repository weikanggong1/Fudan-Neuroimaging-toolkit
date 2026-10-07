# GCSA 注释阶段的共享曲率计算

## 功能

`gcsa_label_python.label_surface()` 对同一半球的 `smoothwm` 和 `sphere.reg` 运行固定图谱分类。它需要五次邻域平均曲率作为 GCS 输入，同时需要主方向供 Gibbs 邻域边使用。原实现分别构造坐标张量、面法向、二阶邻域和局部二次曲面拟合，导致同一表面重复传输和重复执行批量 SVD。

当前实现通过 `mean_curvature_and_principal_directions()` 共享这些只读中间量，一次批量 SVD 同时产生两组结果；`GCSAFeatureCache` 再把这组几何结果、邻接、aseg、皮层标签和 sphere-to-ico 映射复用于同一半球的三套图谱。曲率拟合公式、SVD 奇异值阈值、五次平均顺序、float32 输入和 surface-RAS 坐标约定保持不变。旧的 `mean_curvature_five()` 与 `principal_directions()` 接口保留，供独立诊断使用。

## Python 调用

```python
from fnit.recon_all.gcsa_label_python import GCSAFeatureCache, label_surface

cache = GCSAFeatureCache(subject, "lh", device="cuda:0")  # 每个半球 worker 建一次
result = label_surface(
    subject=subject,                    # 含 surf/、label/、mri/ 的自产被试目录
    hemi="lh",                          # 半球；只能为 lh 或 rh
    atlas_file=dk_atlas,                 # 本次独立 GCS 图谱
    ico4_file=ico4, ico7_file=ico7,      # sphere.reg 到图谱节点的模板
    output_file=output_annot,            # 输出 .annot
    device="cuda:0",                    # 显式目标设备；CPU 诊断可写 "cpu"
    prepared=cache,                      # 同半球后续图谱复用只读几何缓存
)
```

缓存中的 `feature` 为 `(N,) float32`，单位为 `mm^-1`；`principal` 为 `(N, 2, 3) float32`，每个顶点的两个 surface-RAS 切向方向。输入面拓扑必须与 `smoothwm`、`sphere.reg` 使用的顶点顺序一致。非法 shape、非有限坐标、不同半球或不同设备复用由缓存抛出异常。

## 命令行和原软件对应关系

该函数属于 `mris_ca_label` 的内部特征步骤，没有独立命令行。完整阶段由 FNIT 的 `gcsa_label_python` 调用；FreeSurfer 对应命令是 `mris_ca_label`。本优化没有调用或复制 FreeSurfer 可执行文件，只复用 FNIT 自有 PyTorch 几何算子。

## 验证和性能边界

新增回归测试把共享函数分别与原有两个独立函数在四面体表面上比较，要求曲率和方向逐元素绝对误差不超过 `1e-6`。当前提交已完成 Python 语法检查；真实 gpucw1 端到端耗时仍需在恢复远端控制链后测量。理论上该阶段减少一次法向/邻域准备和一次批量 SVD，实际收益应以包含 GPU 数据传输、Gibbs 分类和文件写出的整段 `annotation_hemisphere_group` 墙钟为准。

本改动不改变标签语义、颜色表、Gibbs 随机种子、顶点顺序、输出注释格式或默认 TF32/FP32 策略。当前真实 annotation 阶段约 119–190 s，单图谱约 34–52 s；缓存收益必须用包含准备、GPU 同步、Gibbs、写出和双侧 worker 墙钟的真实测评确认，不能按图谱数线性外推。
