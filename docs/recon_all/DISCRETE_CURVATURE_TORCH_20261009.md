# PyTorch 离散曲率：同网格八张原始顶点图

## 1. 功能简介

`discrete_curvature_torch.py` 实现固定源码的有序面环、离散 K/H、主曲率
K1/K2，并复用已有 `curvature_derivatives_tensor` 生成 BE/C/FI/S。
整数拓扑只准备一次；坐标、面法向、面积和曲率在指定 PyTorch 设备计算。
该文件提供离散核心与单独八图接口；完整自有 `curv.stats` 已由
[完整曲率统计入口](FULL_CURVATURE_STATS_PYTHON_20261010.md) 复用接入。

```mermaid
flowchart LR
  A[有序三角表面] --> B[面环整数缓存]
  A --> C[面积 / 法向 / 源码查表角]
  B --> D[顺序累计 K / H]
  C --> D
  D --> E[K1 / K2]
  E --> F[复用 BE / C / FI / S]
  F --> G[nibabel 八张 morph 图]
```

## 2. Python 调用及输入输出

```python
from fnit.recon_all.discrete_curvature_torch import write_discrete_curvature

report = write_discrete_curvature(
    surface_file="/data/subject/surf/lh.smoothwm",  # 有序闭合三角表面，surface RAS/mm
    output_prefix="/data/diagnosis/lh.smoothwm",  # 输出文件前缀，不覆盖生产输入
    device="cuda:0",  # 明确目标GPU；CPU也可用于同输入排错
    signed_principals=False,  # 保留固定命令默认的绝对值主曲率排序
)
```

| 接口/参数 | 类型、默认值 | 含义 |
|---|---|---|
| `surface_file` | 必填 `str/Path` | nibabel 可读三角网格；原有顶点顺序和 surface RAS/mm 保留 |
| `output_prefix` | 必填 `str/Path` | 输出 `PREFIX.{K,H,K1,K2,BE,C,FI,S}.crv`；父目录自动创建 |
| `device` | `str='cuda:0'` | 指定 PyTorch 设备；不自动回退 CPU |
| `signed_principals` | `bool=False` | False 按绝对值排序，True 按带符号值排序；保留源码相等时的选择 |

返回字典：`outputs` 为图名到文件路径的映射；`vertices` 为 N；`device`、
`signed_principals` 为实际设置；`seconds` 包含读入、拓扑准备、传输、计算和写出；
`delta_violations` 为源码 `H²-K<0` 分支的顶点数；`scope` 明确不是完整统计命令。
每张 morph 为同序 FP32(N,)；K、BE、FI、S 为 mm⁻²，H/K1/K2/C 为 mm⁻¹。

驻留计算可直接使用以下两个接口：

- `DiscreteCurvatureTopology(faces=..., nvertices=..., device='cuda:0')`：
  `faces` 为整数(F,3)，`nvertices` 为正整数 N，建立只属于该有序拓扑的整数缓存。
  换拓扑必须新建；不缓存旧坐标、法向或曲率。
- `context.evaluate(vertices=..., signed_principals=False, angle_backend="torch")`：`vertices` 必须为缓存设备上
  的有限 FP32(N,3)、surface RAS/mm；返回同设备 K/H/K1/K2，及 `face_area`(F,)、
  `face_normal`(F,3)、`face_angles`(F,3)、`delta_negative`(N,) 张量。
  面积 mm²，法向无单位，角度弧度。调用者负责计时同步和传输。

`angle_backend="source_libm"` 显式复用源acosf，并在CPU采用源sqrtf；
返回另含法向夹角和比值 `(N,width)`，padding槽不参与积分。默认torch角度保持兼容。
计算保留 FP32 有序累计、固定Conda的float主曲率sqrt及源码其他double表达式；不关闭 TF32，
不启用半精度，也不改变网格或截断小面积。非法 dtype/shape、非有限坐标、
孤立/边界/不闭环网格、退化面或未覆盖的 marked 顶点分支会抛异常。
此接口不执行平滑、归一化、皮层掩膜过滤、直方图和标准 `curv.stats` 输出。

## 3. 命令行

```bash
python -m fnit.recon_all.discrete_curvature_torch \
  --surface /data/subject/surf/lh.smoothwm \
  --output-prefix /data/diagnosis/lh.smoothwm \
  --device cuda:0
```

`--signed-principals` 可显式启用带符号排序，默认关闭。完整同输入脚本为
`tools/benchmark_recon_discrete_curvature.py`：`--subject` 指定一个或多个冻结被试目录，
`--output` 必须不存在，`--reference-program` 指定独立参考程序，`--assets` 指定参考资产，
`--device` 默认 cuda:0，`--threads` 默认 4，`--code-version` 必须填写实际版本。

## 4. 原软件对应关系

固定源码 `MRIScomputeSecondFundamentalFormDiscrete`、`MRIS_discreteKH_compute`
和 `MRIS_discretek1k2_compute` 属于原命令内部步骤，没有独立 CLI。
本次参考命令在隔离的诊断被试副本执行：

```bash
mris_curvature_stats -m --writeCurvatureFiles -G \
  -o /data/reference/stats/lh.curv.stats -F smoothwm reference_subject lh curv sulc
```

该原生命令还计算统计和其他图，工作范围大于候选八图 API；不能用两者时间相除
宣称完整 `curv.stats` 提速。参考输入/程序、每张图和候选源码 SHA 见机器报告。

## 5. 最新真实数据结果

2026-10-10，完整自有入口在A100、4线程上验证公开ds000114 sub-06/sub-07
双侧自产smoothwm。采用显式source_libm，八图在CPU/GPU和当前同机固定
Conda双参考间全部值及文件字节一致，旧max≤10⁻⁵/P99≤10⁻⁶门全部通过。
完整十节curv.stats也逐字节一致，详见
[完整精度/时间/显存](FULL_CURVATURE_STATS_PYTHON_20261010.md#5-本版真实精度完整阶段时间与显存)。
这里的同输入阶段通过不代替原始T1整例或整体指标等效。

旧2026-10-09 H100不同冻结网格的失败记录用于定位：K一致而H局部1ULP
被小面积网格放大。此后分离了source acosf、float sqrt与float fabs重载，
没有修改阈值或特定顶点。旧证据见
[原四侧报告](../../validation/recon_all/optimizations/20261009_discrete_curvature/report_v2.json)、
[局部诊断](../../validation/recon_all/optimizations/20261009_discrete_curvature/diagnostics/report.json)、
[CPU acos隔离](../../validation/recon_all/optimizations/20261009_discrete_curvature/diagnostics/acos_cpu_v2.json)。

## 6. 更新和验证记录

- v1 真实不同度数网格暴露无效 padding 槽索引越界；v2 只给无效槽安全索引，
  有效面仍按源码环顺序检查。保留失败报告，不修补特定顶点。
- 2026-10-10修正主曲率sqrt(float)表达式，补独立float/double差别契约；
  完整统计与派生图测试共17项CPU通过，CUDA真实四侧完整阶段另测通过。
- CPU acos隔离的首版位图统计未统一 nibabel 大端 dtype；第二版统一字节序后重测，
  首版报告明确保留为诊断错误，数值输出未变。
- 没有新增依赖。PyTorch、NumPy、Numba、nibabel、matplotlib 已列入主页 Conda 环境。
  全新安装及无预装脑影像软件的隔离整例尚未验证。

## 7. 参考文献与源码

- [FNIT 离散实现](../../src/fnit/recon_all/discrete_curvature_torch.py)
- [FreeSurfer固定源码及离散曲率内部函数](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp)
- Fischl B. FreeSurfer. *NeuroImage* 62:774–781, 2012.
