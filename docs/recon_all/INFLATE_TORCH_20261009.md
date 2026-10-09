# 标准 inflation 与 sulc 的 PyTorch 实验后端

## 1．功能和范围

`run_standard_inflate()` 从 FNIT 自产 `smoothwm` 生成 `inflated` 和 `sulc`。它复用已有 `inflate_python.py` 的完整 CPU 配方，并新增显式 Torch GPU 后端。GPU 法向复用 `TorchFaceNormalTopology`，梯度平均复用 `RegistrationGradientAverager`；距离、归一化弹簧、动量、RMS、面积缩放和 sulc 用 Torch 张量计算。当前是独立实验接口，recon-all 默认仍为 Conda 源码构建的 `mris_inflate`。

```mermaid
flowchart LR
    A[smoothwm / 有序三角网格] --> B[缓存一环和完整二环]
    B --> C[距离力 / 有序平均]
    C --> D[归一化弹簧 / 动量更新]
    D --> E[sulc累积更新前法向投影]
    E --> F[重算法向 / 面积 / 距离 / RMS]
    F -->|继续原六档与停止规则| C
    F --> G[包围盒居中 / 原面积缩放]
    G --> H[inflated与零均值sulc]
```

本接口对应固定默认配方：邻域二环，平均档次 16/8/4/2/1/0，每档默认 10 步，`dt=momentum=0.9`，距离权重 `0.1×sqrt(averages)`，归一化弹簧权重 1，RMS 目标 0.015。高分辨率体积头的 x 体素尺寸在 `(0,0.8)` mm 时，按原 CLI 调整每档步数。暂不支持 patch/ripped 顶点、explode、非默认力权重、sphere projection 或其他 CLI 配方。

现有几何函数以前只写 `inflated`，没有输出 `sulc`。完整接口补充固定源码的实际定义：每一步用动量位移与**更新前**单位法向的点积累积 FP32 `curv`，最后以 FP64 均值居中，再写顶点图。不会用径向距离或位移模长替代。原几何 API 仍保留兼容行为。

## 2．Python 调用、全部输入和输出

```python
from fnit.recon_all.inflate_standard_run import run_standard_inflate

report = run_standard_inflate(
    input_surface="subject/surf/lh.smoothwm",  # FNIT自产三角表面，surface RAS，单位mm
    inflated_output="diagnostic/lh.inflated",  # 同顶点顺序和有序面的膨胀表面
    sulc_output="diagnostic/lh.sulc",  # 同顶点顺序的FP32有符号累计深度，单位mm
    backend="torch",  # 显式使用Torch；默认numpy用于既有CPU回归
    device="cuda:0",  # 显式GPU；默认cpu，不自动回退设备
    profile=True,  # 默认False；测量时逐子段同步目标GPU，生产不需过度同步
)
```

| 输入参数 | 类型、默认值与意义 |
|---|---|
| `input_surface` | 必填 str/Path；FreeSurfer 三角表面格式，坐标 `(N,3)`、面 `(F,3)`，surface RAS/mm，有体积几何头时保留 |
| `inflated_output` | 必填 str/Path；写 `(N,3)` FP32 坐标和原有序面、体积几何文本，中心平移及面积缩放后的表面 |
| `sulc_output` | 必填 str/Path；写 `(N,)` FP32 morph 格式，和输入顶点一一对应，单位 mm |
| `backend` | str，默认 `numpy`；`numpy` 复用 CPU NumPy/Numba，`torch` 选择新增张量实现 |
| `device` | str，默认 `cpu`；Torch 可用 CPU 做诊断或显式 `cuda:N`；NumPy 必须为 CPU |
| `profile` | bool，默认 False；True 对 Torch 子段前后同步，用于定位，返回时间包括该开销 |

三个路径必须不同。输入须有限、有合法索引、正总面积、无孤立顶点，体积 x 体素尺寸须为正有限数。数据没有重采样；体积 header 只用于声明几何和高分辨率步数。缓存仅属于本次有序面拓扑，不跨网格阶段或坐标版本复用法向。

返回字典包含路径、后端、设备、顶点/面数、每档步数、读取校验时间、初始化/传输时间、计算至最终输出下载时间、写出时间、`integration` 的实际步数/RMS/停止原因/子段计时，以及 `total_seconds_including_io`。Torch `integration_wall_seconds` 含必要停止同步；最终文件总墙钟包含加载、验证、拓扑、搬运、计算和两份文件写出。子段时间嵌套在总墙钟中，不能重复相加。

`TorchInflationContext(faces=..., nvertices=..., device="cuda:0")` 只缓存一/二环整数索引、有效掩膜及成熟法向/平均上下文。`integrate(vertices=..., niterations=10, rms_target=0.015, profile=False, callback=None)` 输入同设备 FP32 `(N,3)`，返回同设备新坐标、未去均值 sulc、步数/RMS/时间。可选只读 callback 接收 `(step,coordinates)`；不得原地改候选。`finalize(coordinates=..., original_vertices=..., sulc=...)` 返回居中/面积缩放坐标及零均值 sulc。坐标与原网格单位都是 mm，sulc 的符号是原累积投影约定。

输入非法、CUDA 不可用、编译失败或非有限计算会抛异常，可能留部分文件；没有参考复制、占位文件、近似回退或 CPU 静默回退。API 不修改全局 TF32/精度策略，不启用 FP16/BF16。

## 3．命令行与复现

```bash
python -m fnit.recon_all.inflate_standard_run \
  --input-surface subject/surf/lh.smoothwm \
  --inflated-output diagnostic/lh.inflated \
  --sulc-output diagnostic/lh.sulc \
  --backend torch \
  --device cuda:0 \
  --threads 4 \
  --profile \
  --report diagnostic/lh.inflate.json
```

输入、输出、后端和设备与 Python 参数相同；`--threads` 默认 4，固定本进程 Torch/Numba 预算；`--profile` 默认关闭；`--report` 是必填 JSON 路径。CLI 只在自己进程开启 TF32，记录实际策略、CUDA 是否预初始化、源码哈希和张量峰值。API 总时间不含 Python 进程导入，完整冷进程墙钟还需外部测量；自报进程时间含本模块导入、API及哈希。已有主页 Conda 的 PyTorch、Triton、NumPy、Numba 和 nibabel 已覆盖这条路径，无新增生产依赖；Pytest 仅是独立测试工具。

真实阶段比较使用 `validation/recon_all/optimizations/20261009_inflate_torch/benchmark_complete.py`：`--data` 是逐 SHA 的公开 smoothwm manifest 目录，`--native` 是 FNIT 独立 Conda 源码构建的参考程序，`--output` 必须不存在。`--device` 默认 `cuda:0`，`--threads` 默认 4，`--repeats` 默认 2；可重复 `--case case/lh` 筛选。配对顺序为 native/NumPy/Torch，再 Torch/NumPy/native，GPU 每次完整 API 前后同步。输入、算法模块和参考程序均记录实际 SHA。参考只在候选全部计算后用于诊断比较。

`prepare_inputs.py` 的 `--config` 为 JSON 数组，每项明确指定 `case`、`hemisphere`、`surface`、`public_source_url` 和 `source_recipe`；`--output`/`--archive` 都必须是新路径。只复制显式公开表面和清单，不递归复制被试、权重或许可证，不修改源影像。

冷进程与缓存策略分别用 `benchmark_cli.py` 和 `benchmark_allocator.py` 检查。两者均要求 `--source-root` 是冻结候选源码、`--data` 是公开输入包、`--pair` 是状态为 `complete_stage_pair` 的同输入对照、`--output` 是新目录，`--device cuda:0 --threads 4` 显式选择资源。CLI 检查从空 NumBa/Triton 缓存开始，每张表面新建 Python 进程，后续子进程可复用编译缓存；它记录外部完整墙钟与子进程 API 时间。缓存关闭检查须在进程启动前设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`，另指定 `--profiling-module` 为带 SHA 的 FNIT 同期显存采样模块；不会在已初始化进程中更改分配器。

`plot_receipts.py --data INPUT_PACKAGE --pair COMPLETE_PAIR --output NEW_FIGURES` 绘制全部顶点的矢状投影和同索引误差，并记录图、输入和报告 SHA；仅用于展示，不重采样候选、不代替三维网格检查。

## 4．对应原软件与源码

固定 FreeSurfer 8.2 单 T1 表面阶段为：

```bash
mris_inflate -threads 4 subject/surf/lh.smoothwm diagnostic/lh.inflated
```

默认同时写输出目录下的 `lh.sulc`，半球来自表面身份；Torch 接口明确指定两个路径。官方程序只属于独立 benchmark，生产不调用系统 FreeSurfer。本轮参考为固定源码独立 Conda 构建、迁移重定位后的程序。

源码固定为 `d932c45b7941662ea380a05efef580568b98d41a`：[CLI](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_inflate/mris_inflate.cpp)、[MRISinflateBrain](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_integrate.cpp)、[sulc tracking / zeroMeanCurvature](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_metricProperties.cpp)。这些是该 CLI 的内部步骤，没有独立官方命令；只在临时诊断目录审查，不把无关上游源码复制发布。

## 5．本版真实精度、耗时和资源

本轮选择公开 ds000114 sub-06/sub-07 的 FNIT 自产冻结双侧 smoothwm；四个表面已逐 SHA 核验，12,886,865 字节的小包只迁移到获授权私有目录。这是固定同输入完整阶段验证，不是原始 T1 空目录整例。A100-SXM4-80GB、Xeon Platinum 8369B，同主机 CPU affinity 0–3、4 线程；Torch 2.5.1/CUDA 11.8，TF32 开启、无半精度。参考程序为相同固定源码的独立 Conda 构建产物，SHA `8c3e5f688a635a4bf7e1c49efec9737bef2310a2fcc5b221332fffc54667be17`。

首个 sub-07 LH 的 v1 已完成：114342 顶点、228680 面，两次 CPU/Torch 的有序坐标、sulc 和体积几何头均与同输入 native 精确一致；原生重复输出也精确。native 墙钟 7.476/7.495s，NumPy 32.628/32.732s，GPU 8.471/6.873s；首轮 GPU 含子段剖析和首次 kernel 编译，第二轮不开逐段同步。实测 GPU 第二次的初始化/传输为 5.827s，完整 60 步积分仅 1.015s，确认 Python list/set 整数拓扑是新瓶颈。

v2 复用成熟法向的有序 face CSR，用 Numba 构建完全同序一环和二环，取代重复 list/set 遍历；不改变整数候选、不截断邻域，也不改变浮点公式。五项结构测试已通过，完整两例双侧每侧 60 步、各两次配对已完成。CPU 与 GPU 的 16 次候选比较全部满足：有序面相同、所有 FP32 坐标和 sulc 元素差异数 0、最大/P99/RMSE 0、九项体积几何头字段相同。参考程序两次重跑的解码几何和 sulc 字节均相同。表面文件注释/附加写出信息不同，因此不以整个 inflated 文件哈希不同认定几何变化。

| 冻结表面 | 顶点 / 面 | native 第1/2次，s | CPU 第1/2次，s | Torch 第1/2次，s |
|---|---:|---:|---:|---:|
| sub-06 LH | 130346 / 260688 | 12.261 / 8.304 | 37.575 / 36.401 | 1.953 / 0.957 |
| sub-06 RH | 132837 / 265670 | 8.407 / 8.138 | 37.634 / 35.862 | 1.247 / 1.057 |
| sub-07 LH | 114342 / 228680 | 7.223 / 7.082 | 29.554 / 30.270 | 0.988 / 0.958 |
| sub-07 RH | 114824 / 229644 | 7.265 / 7.533 | 28.721 / 28.501 | 0.999 / 0.854 |

这些 API 时间包含读取、校验、初始化、传输、完整积分、最终处理和两个输出写入；已运行结构测试且 CUDA 初始化，不含 Python 导入。第一次 Torch 启用逐段剖析，第二次关闭剖析；共享负载保留在机器收据中，不能据此给出稳定吞吐承诺。仅相加四个第二次阶段观察，native 31.058s、CPU 131.033s、Torch 3.826s，Torch 相对 native 8.12 倍、相对已有 CPU 34.24 倍；这不是 recon-all 整例提速。

v3 仅补 CLI 线程、实际精度、源码哈希与进程边界报告，API 数值核心 SHA 与 v2 一致。独立冷 CLI 四个输出也均与 native/v2 逐元素一致，父进程和四个子进程调用前均未初始化 CUDA。

| 冻结表面 | 新 Python 进程外部墙钟，s | 完整 API，s | 初始化/传输，s |
|---|---:|---:|---:|
| sub-06 LH | 7.341 | 4.928 | 2.754 |
| sub-06 RH | 7.770 | 5.341 | 3.663 |
| sub-07 LH | 13.768 | 9.898 | 7.854 |
| sub-07 RH | 8.918 | 5.229 | 3.670 |

四次新进程合计 37.799s；每侧新建 CUDA/Python 进程会抵消 GPU 运算收益，不能将热 API 时间作为冷进程成绩。首个子进程使用空编译缓存，其后三个复用同一编译缓存，但仍为新进程且受共享 CPU/GPU 负载影响。

另外按生产入口实际低显存策略，在**新进程启动前关闭 CUDA 分配缓存**，完整四网格再次计算，坐标、sulc 与 native/缓存开启 v2 仍然全部零差异；时间变为 55.581/69.894/57.661/43.437s，合计 226.573s。相同算法与数据下，大量 eager 张量的临时分配在该策略中成为瓶颈。当前不能直接在关闭缓存的生产路径选择此 Torch 实现；后续优先复用缓冲区或验证独立执行策略，不全局取消既有低显存措施。

![两例双侧 smoothwm、inflated 和同索引误差](../../validation/recon_all/optimizations/20261009_inflate_torch/reports/a100_20261009_v5/figures/smoothwm_inflated_error.png)

图使用全部顶点的矢状投影，误差色标固定 0–0.001 mm，四张表面最大误差均为 0。它展示形状与本次差异，不承担自相交、局部翻折或 T1 边界叠加的验收。

保留已有 geometry 开发门：先证明相同顶点数和有序面，再判断最大同索引顶点距离 ≤0.001 mm。sulc 保留不同元素数、最大/P99/RMSE及方向、逐字节严格诊断；新的指标等效门尚未建立，不根据结果事后设门。网格连通性/非流形的有序面保持不变，完整自相交检查与最终脑区指标另列，不能只凭面数或平均相关性宣布等效。

GPU 张量 allocated/reserved 显式选择目标设备：v2 最大分别为 378,390,528/612,368,384 字节；冷 CLI 最大分别为 380,396,544/494,927,872 字节。它们是张量分配器计数，不是父子进程同期占用。关闭分配缓存时张量峰值不可用，报告为 null；0.5 秒采样观测目标卡峰值 848,297,984 字节、全部计算进程和的上界 884,998,144 字节。两类查询时刻不同，前者含卡开销与共享负载，后者含该卡全部进程；容器 PID 与驱动 PID 归属未解决，因此父子树同期峰值为 null，不把未知占用记为零。不能据本段孤立网格计数宣布 recon-all 整例满足 20,000,000,000 字节。

完整 [v2 同输入阶段报告](../../validation/recon_all/optimizations/20261009_inflate_torch/reports/a100_20261009_v5/v2_fourmesh/summary.json)、[冷 CLI](../../validation/recon_all/optimizations/20261009_inflate_torch/reports/a100_20261009_v5/v3_cold_cli/summary.json)、[缓存关闭反例](../../validation/recon_all/optimizations/20261009_inflate_torch/reports/a100_20261009_v5/v4_cache_disabled/summary.json) 及源码/输入/程序/动态库 SHA 均保留。冻结基底为 `a756fffb`，候选以逐模块 SHA 和三版 patch manifest 绑定；未把基底 commit 等同于含未提交补丁的全部实现。40 份公开收据仅替换私有目录与主机名，3476 个数字/布尔/null 字段保持原样；[映射清单](../../validation/recon_all/optimizations/20261009_inflate_torch/reports/a100_20261009_v5/public_export_manifest.json) 记录原始与公开 SHA。原始私有包 SHA 为 `7cec9a18f7b4e70e385069727e26d530bd336c084232aa8de1f89226e37f1c22`。依赖检查属于已声明 Conda 运行时，未验证物理上无预装软件的干净环境隔离。

## 6．更新与验证记录

2026-10-09 v1 补已有积分的可选 sulc 和子段诊断，新增完整双输出 runner 和 Torch 全积分实验后端；完整 sub-07 LH 对照通过后发现 5.8 秒整数拓扑瓶颈。v2 用有序 CSR/Numba 缓存完整邻域，四网格两次配对通过。v3 保留同一数值算法，补独立冷 CLI 边界与显存报告。v4 独立诊断生产缓存关闭策略，确认同输出仍明显慢，保存启动参数/最小包缺辅助导入的失败日志。v5 收据导出在相同运行时核对库哈希，并去除私有路径。成熟 normals/averaging 内核复用，保留原 API、生产原生默认、原严格诊断和既有球面实现。五项结构测试检查整数邻域顺序/重复/空行、完整档次、更新前 sulc、输入不变、非法参数和 CPU/GPU 对照；模拟小网格只属于测试，真实证据另保留。

尚未声称原始 T1 整例达到十分钟，也未用局部 GPU 时间减去历史整例时间估算提速。两例双侧同输入已通过；生产候选仍需按实际分配缓存策略、后续 sphere/配准及自产连续链完成回归。标准 inflation 以外的配方和三维自相交质量尚未验证。

## 7．参考

Dale AM、Fischl B、Sereno MI，Cortical surface-based analysis I: segmentation and surface reconstruction，NeuroImage，1999。[DOI](https://doi.org/10.1006/nimg.1998.0395)。

Fischl B、Sereno MI、Dale AM，Cortical surface-based analysis II: inflation, flattening, and a surface-based coordinate system，NeuroImage，1999。[DOI](https://doi.org/10.1006/nimg.1998.0396)。原实现：[FreeSurfer](https://github.com/freesurfer/freesurfer)。
