# PyTorch 表面放置正则梯度

## 1. 功能与流程

`PlacementRegularizationTorch` 将现有 Python pial 的固定网格正则梯度迁移到
PyTorch：带方向筛选的一跳梯度平均、normal/tangent spring、两跳二次曲率。
它按原邻接顺序逐项累加，并保留现有 5×5 float32 QR 运算次序。它复用
placement 的公式；脑区统计的主曲率 SVD 是另一项算法，不能代替这里的 QR。

完整 `place_pial_t1()` 新增 `regularization_backend="torch"` 显式选项。
四轮边界重新估计、目标函数、动态候选、异步逐顶点碰撞接受、缩步拒绝、
内侧壁固定和最终相交修复保留原行为。默认仍为 `"cpu"`，本页不把固定
网格梯度测试当作完整 pial、最终 white 或原始 T1 整例验收。

```mermaid
flowchart LR
    A[当前坐标与有序法线] --> B[强度与原表面斥力]
    B --> C[PyTorch signed averaging]
    C --> D[normal spring]
    D --> E[两跳 quadratic curvature]
    E --> F[tangent spring]
    F --> G[原异步碰撞与试步接受]
    G --> H[原目标函数和四轮停止规则]
    H --> I[内侧壁固定与完整相交清理]
```

没有新增依赖。PyTorch、NumPy、Numba、SciPy 和 nibabel 已由主页 Conda
环境提供。MRI 和网格读写继续使用 nibabel；本模块不启动外部可执行程序。

## 2. Python 调用和输入输出

### 完整 Python pial 的调用

```python
from pathlib import Path
from fnit.recon_all.place_pial_python import place_pial_t1

pial_report = place_pial_t1(
    subject=Path("/data/fnit/sub-01"),  # 已有 FNIT 自产 white、MRI、标签和灰白质阈值
    hemisphere="lh",  # 左半球；右半球为 rh
    output=Path("/data/diagnostic/sub-01/lh.pial.T1"),  # 独立诊断表面，避免覆盖已验收结果
    max_steps=200,  # 四轮总迭代保护上限；不收敛时抛异常
    sampling_backend="cpu",  # MRI 强度采样；torch/triton 也可显式选择
    regularization_backend="torch",  # 本页新增的 PyTorch 固定网格正则梯度
    candidate_backend="tree",  # 保留原动态候选完整扩展规则
    device="cuda:0",  # 当前进程内明确的目标 GPU
    trace_callback=None,  # 可选每一步只读诊断回调
)
```

七项前置文件、空间、返回结构和失败行为沿用
[完整 Python pial 说明](PYTHON_PIAL_PLACEMENT.md)。返回字典增加
`regularization_backend`，明确实际选择。`"torch"` 必须显式给出 `device`；
CUDA 不可用或内存不足传播 PyTorch 异常，不静默退回 CPU。

### 内部固定网格上下文

| 参数 | 格式、默认值和含义 |
|---|---|
| `neighbors` | `(N,D)` 整数一跳邻接；保留 placement 的原次序 |
| `valid` | `(N,D)` bool；有效项必须在每行 padding 前面 |
| `offsets` | `(N+1,)` 整数 CSR 偏移，从 0 开始、单调、末项等于 candidates 长度 |
| `candidates` | `(M,)` 整数两跳顶点索引；顺序沿用 `two_ring_neighbors()` |
| `ripped` | `(N,)` bool 固定 rip 掩膜；改变掩膜须新建上下文 |
| `device` | 默认 `"cuda:0"`；可用显式 `"cpu"` 做算法诊断，裸 `"cuda"` 被拒绝 |
| `chunk_size` | 默认 32768，正整数；只限制曲率临时内存，不截断邻接 |

`regularize()` 的全部参数如下。

| 参数 | 格式、默认值和含义 |
|---|---|
| `vertices` | `(N,3)` float32，当前 surface RAS 坐标，单位 mm |
| `normals` | `(N,3)` float32，对应当前坐标及同顶点顺序的单位法线 |
| `gradient` | `(N,3)` float32，pial 强度＋原表面斥力梯度 |
| `iterations` | 非负整数，signed averaging 次数；四轮 pial 依次使用 16/8/4/2 |
| `spring_weight` | 默认 0.3，有限浮点数，保留 float32 权重转换 |
| `after_average` | 默认 None；white 诊断可传 `(N,3)` 自斥力，在 averaging 后加入 |

返回 `(N,3)` float32 NumPy 梯度，与输入顶点一一对应；它是优化器的位移
方向项，不能作为独立表面输出。该上下文只拥有固定邻接与 rip 的独立副本，
每次重新上传当前坐标和法线；white.preaparc、final white、pial 不共享上下文。
索引越界、非前缀有效邻接、形状/非有限值、非法设备或参数会抛异常。
若有效顶点的 QR 退化产生非有限梯度，回传后抛 `FloatingPointError`，
不把 NaN 送入试步接受，也不静默换成另一曲率算法。

CUDA 上逐顶点并行，单顶点邻接仍依次累加。signed averaging 的累加使用
float64，曲率 normal equations 和 QR 使用 float32，QR 的 SNRM2 平方根
保留 float64 的显式步骤。它不启用 FP16/BF16，也不更改进程的 TF32 策略。
这里没有矩阵乘法，周围其他 GPU 阶段仍可使用默认 TF32。

### white 首轮诊断接口

`place_white_preaparc_prefix()` 和 `first_white_preaparc_step()` 也增加同名
显式选项。它们仍只是第一轮的 1–17 步，不能替代完整 white.preaparc
或最终 white。白质自斥力在 signed averaging **之后**加入，不能使用
pial 的斥力顺序。

```python
from fnit.recon_all.place_white_preaparc_python import place_white_preaparc_prefix

white_prefix_report = place_white_preaparc_prefix(
    subject_dir="/data/fnit/sub-01",  # 含 orig、阈值及 brain.finalsurfs/wm/aseg.presurf
    hemi="lh",  # 当前半球
    output="/data/diagnostic/lh.white-first-step",  # 非最终white文件名的独立诊断网格
    steps=1,  # 默认1；仅允许第一轮的1–17步
    diagnostics=None,  # 可选.npz原CPU分项数组；开启后另有诊断计算成本
    regularization_backend="torch",  # 固定网格正则梯度
    device="cuda:0",  # 显式目标设备；CPU默认可为None
)
```

`subject_dir` 包含 `surf/H.orig`、`surf/autodet.gw.stats.H.dat` 与
`mri/{brain.finalsurfs,wm,aseg.presurf}.mgz`；表面是 surface RAS/mm，
MRI 是原 conform 体素网格。`output` 保留 orig 的有序三角面和完整尾部。
返回原逐步 SSE/RMS、步数和分项秒数，并增加实际
`regularization_backend/device`。`steps`、半球、后端或输入无效、拒绝所有
试步时抛异常。`first_white_preaparc_step()` 参数相同但没有 `steps`，
固定调用第一步。

## 3. 命令行

该正则梯度是内部函数，没有独立 FNIT 生产 CLI。将上面的具名 Python 调用
保存为 `run_pial_candidate.py` 后，用实际 FNIT Conda 环境执行：

```bash
python run_pial_candidate.py
```

white 首轮接口提供已有 CLI 的相应选项：

```bash
python -m fnit.recon_all.place_white_preaparc_python \
  /data/fnit/sub-01 lh /data/diagnostic/lh.white-first-step \
  --steps 1 --regularization-backend torch --device cuda:0
```

真实固定输入配对脚本提供显式 CLI：

```bash
python validation/recon_all/python_gpu_port/benchmark_placement_regularization.py \
  --subject /data/fnit/sub-01 \
  --candidate-directory /data/candidate/src/fnit/recon_all \
  --output /data/benchmark/pial_regularizer.json \
  --device cuda:0 \
  --threads 4
```

`subject` 是同一真实已生成被试；`candidate-directory` 只覆盖本次候选模块；
`output` 保存完整 JSON；`device` 是显式目标 GPU；`threads` 是双方 PyTorch
线程预算。Numba/BLAS/OpenMP 总线程应同时由外层配置，不自动扩展资源。

## 4. 原软件对应调用

原公式属于 `mris_place_surface --pial` 的内部梯度，没有独立官方 CLI。
独立 benchmark 参考命令是：

```bash
mris_place_surface --adgws-in surf/autodet.gw.stats.lh.dat \
  --seg mri/aseg.presurf.mgz --threads 4 --wm mri/wm.mgz \
  --invol mri/brain.finalsurfs.mgz --lh --i surf/lh.white \
  --o diagnostic/lh.pial.T1 --pial --nsmooth 0 \
  --rip-label label/lh.cortex+hipamyg.label \
  --pin-medial-wall label/lh.cortex.label --aparc label/lh.aparc.annot \
  --repulse-surf surf/lh.white --white-surf surf/lh.white --restore-255
```

官方参考仅在隔离诊断路径运行，不是 FNIT 生产依赖。本轮固定输入梯度测试
首先比较现有自有 CPU 函数；完整官方几何差异与历史 pial 差异分开记录。

## 5. 本轮验证

2026-10-09 在 gpucw1 现有 FNIT 环境完成三项 CPU 算法回归：批量 5×5 QR
与原 Numba 每元素一致；含 ripped 顶点和 white `after_average` 顺序的完整
regularizer 与原 CPU 梯度每元素一致；非法设备/分块配置被拒绝。
这些人工小网格仅是单元测试，不是性能或脑区等效 benchmark。
本地 FNIT 环境随后完成 15 项针对性回归（10.74 s），覆盖新 regularizer、
原 pial 终止拒绝规则、静态索引及统计缓存；默认 CPU 控制流未退化。

真实 ds000114 sub-07 固定输入正则梯度在 gpucw1 H100、四线程、CUDA0
完成 ABBA；每次计时包括当前坐标/法线上传、计算和完整梯度回传，边界
同步明确目标 GPU。32768 分块的结果如下。

| 半球 | 顶点/面数 | CPU 中位数 (s) | Torch 中位数 (s) | 该内核速度比 | 不同元素/P99/max | allocated/reserved 峰值 (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| LH | 114342/228680 | 0.217123 | 0.097106 | 2.236× | 0/0/0 | 140668416/197132288 |
| RH | 114824/229644 | 0.208376 | 0.090175 | 2.311× | 0/0/0 | 127419904/197132288 |

固定邻接上下文建立分别为 0.164/0.151 s，单列而不计入每次梯度。双方
四次配对的结果哈希均相同。GPU第一次调用 LH 0.241 s、RH 0.092 s；
冷设备/JIT与热循环不能混报。上述分配器峰值只覆盖本次内核，不是
整个进程或 recon-all 的显存峰值。初版 4096 分块的 LH/RH Torch
0.453/0.446 s 比 CPU 慢：重复小块 QR 的 kernel 调度是瓶颈，增大块
后消除了大部分调度开销，公式、候选范围和精度不变。

完整机器报告保留[4096 初版](../../validation/recon_all/optimizations/20261009_placement_torch/regularizer_sub07_chunk4096.json)
与[32768 候选](../../validation/recon_all/optimizations/20261009_placement_torch/regularizer_sub07_chunk32768.json)。
候选 regularizer 源码 SHA-256 为
`3db2cca686387f3c1a38cf7cee656a7b92e779f162050b50ca7e9903fed7482b`；
报告中的 CPU 源模块哈希绑定实际旧函数，而不是用本地后续提交替换。
之后只去掉 SNRM2 比值中无必要的最小正数钳制，以保留原公式对 subnormal
值的处理，并对非有限 QR 梯度明确报错；15 项 CPU 回归包括该最终源码。上述 GPU 表和后台完整 v1
测试仍绑定表中 SHA，未改写运行中的冻结文件。

完整左侧 pial CPU→Torch 同输入配对已后台启动，保留全部试步、四轮
边界和最终清理；它尚未完成。本页只发布已完成的内核数字。

真实输入测试使用已有公开 ds000114 sub-07 的自产 `white`、MRI 与标签，
左右半球同输入 ABBA 配对。脚本记录输入/源码 SHA-256、明确设备、线程、
冷 JIT、上下文建立、坐标上传/梯度回传、GPU同步边界和 allocated/reserved。
固定正则梯度的预先容差为 max absolute ≤ 1e-5；完整 pial 必须另检查
逐轮试步、最终几何和网格质量，不能只靠此容差准入默认流程。

完整 pial、最终 white、原始 T1 整例、官方重复性和父子进程同时显存尚未
由本模块重新验证；整体指标等效保持 `not_assessed`。本页没有把历史整例
耗时改标为本轮结果，也不以局部提速推算 10 分钟目标。

## 6. 更新记录

- 2026-10-09：新增固定网格 PyTorch regularizer，完整 Python pial 增加
  显式 `regularization_backend`；CPU 默认和原完整清理保留。
- 更早的 MRI Torch/Triton 采样与原完整 Python pial 记录见
  [Python pial](PYTHON_PIAL_PLACEMENT.md) 和
  [placement sampling](../../src/fnit/recon_all/PLACEMENT_SAMPLING_20261002.md)。

## 7. 源码和参考文献

- [FNIT regularizer](../../src/fnit/recon_all/place_surface_regularization_torch.py)
- [固定 FreeSurfer d932c45 mrisurf.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf.cpp)
- [mris_place_surface](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_place_surface/mris_place_surface.cpp)
- Dale, Fischl & Sereno. *NeuroImage* 9, 179–194 (1999),
  [doi:10.1006/nimg.1998.0395](https://doi.org/10.1006/nimg.1998.0395)。
