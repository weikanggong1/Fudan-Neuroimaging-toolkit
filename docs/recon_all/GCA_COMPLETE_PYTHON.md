# 完整 Python GCA EM 配准

## 1. 功能简介

`mri_em_register_python.register_t1` 将同网格 T1 强度和脑掩膜配准到固定 GCA，输出 voxel-to-voxel LTA。候选搜索复用 FNIT 已有实现；稳定样本和全样本均运行完整 EM、L-BFGS 及步长搜索，不启动 FreeSurfer 程序。

本次现场核对发现基线 Python 入口仅在全样本上执行首次方向的步长搜索，缺少稳定样本 EM 及后续准牛顿方向；已有梯度还丢弃缺少分类器的样本。现在补齐优化协议，并恢复这类样本的 mean=0、inverse_variance=1 梯度贡献；它们仍不计入平均协方差行列式。已有候选搜索、GCA 读取和文件写出继续复用。六项协议单元检查已通过。真实同输入检查暴露已有强度预处理缺少宽分布禁用缩放分支；共享子函数现已补齐稳健 Gaussian 宽度检查及有序 FP32 直方图平滑。修正后两例完整注册的输入 SHA 与独立原版相同，输出 LTA 矩阵逐元素一致。CA normalization 同样复用该子函数；两例各两次同输入原版回归的 norm 与六帧 ctrl_pts 均零差，详见 CA normalization 文档。

```mermaid
flowchart LR
 A[自产nu和brainmask] --> B[GCA读取与强度缩放]
 B --> C[稳定样本平移和线性搜索]
 C --> D[稳定样本完整EM clamp=0]
 D --> E[全样本完整EM clamp=6]
 E --> F[VOX_TO_VOX LTA]
```

## 2. Python 调用与输入输出

```python
from pathlib import Path
from fnit.recon_all.mri_em_register_python import register_t1

# nu和brainmask来自同一条FNIT生产链；atlas_dir含本地校验的资源。
nu_file = Path("subject/mri/nu.mgz")
brainmask_file = Path("subject/mri/brainmask.mgz")
gca_file = Path("resources/average/RB_all_2020-01-02.gca")
output_lta_file = Path("subject/mri/transforms/talairach.lta")
registration_report = register_t1(
    nu_path=nu_file, atlas_path=gca_file, mask_path=brainmask_file,
    output_path=output_lta_file, device="cuda:0", search_backend="torch",
    candidate_chunk=64, sample_chunk=8192,
    reduce_on_device=True, inverse_backend="torch",
)
```

- `nu_path`：三维 MGZ/MGH 强度图，固定单 T1 conform 网格、毫米 affine。
- `atlas_path`：固定 `RB_all_2020-01-02.gca`。资源大小和 SHA-256 沿用 `assets.ASSET_FILES`，验证工具先校验；函数不下载资源。
- `mask_path`：同网格脑掩膜；掩膜外强度置零。
- `output_path`：新 LTA 文件，type=0，4×4 FP32 voxel-to-voxel 矩阵及源/目标几何。路径 TAG 和文本 provenance 不作为数组精度指标。
- `device`：计算设备；Torch 搜索要求明确 CUDA 设备。
- `search_backend`：默认 `cpu`，有序 NumPy/Numba 搜索；`torch` 使用已有 CUDA 候选评分。
- `candidate_chunk`：默认 64，每批变换候选数；`sample_chunk`：默认 8192，每批样本数。两者只控制分批，不能减少候选或样本。
- `reduce_on_device`：默认 `False`；显式 `True` 在 CUDA 上累加评分，须同输入核对搜索结果。
- `inverse_backend`：默认 `cpu`，源序余子式逆矩阵；`torch` 为同公式批量 CUDA 实现，要求 Torch 搜索。

返回矩阵、分步秒数、搜索记录和 `em_complete_protocol`。后者分别给出稳定/全样本数、每次准牛顿 pass、L-BFGS 重启、目标评估和终止原因。稳定阶段 tolerance=1e-5、全样本 1e-7，保持源 FP32 外层停止算术；内部 12 参数 L-BFGS 使用 FP64、7 对历史和 More–Thuente 0.1 Wolfe 条件。参数、非有限状态或无进展错误明确传播，不回退外部程序。

## 3. 命令行调用

```bash
python -m fnit.recon_all.mri_em_register_python --help
python tools/benchmark_recon_volume_gca_complete.py \
  --nu subject/mri/nu.mgz --mask subject/mri/brainmask.mgz \
  --atlas resources/average/RB_all_2020-01-02.gca \
  --output validation/gca-python-new --mode python --device cuda:0
```

验证工具输出 typed JSON、矩阵和 SHA，不将影像、权重或私有服务器路径放进公开指标。`output` 必须是不存在的新目录。Python 阶段计时包含加载、传输、JIT、优化和写出，排除解释器和顶部导入；历史字段 `seconds_including_spawn_load_IO` 在 Python 模式没有涵盖 spawn，不能误读为全进程耗时。原版模式从子进程启动计时。GPU allocator 峰值与整卡 NVML 内存分别记录，容器进程归属未建立时 own NVML 为未知。

## 4. 原软件调用

仅在隔离 benchmark 目录运行独立参考，输入与 FNIT 阶段完全相同：

```bash
mri_em_register -uns 3 -mask subject/mri/brainmask.mgz \
  subject/mri/nu.mgz resources/average/RB_all_2020-01-02.gca \
  reference-only/talairach.lta
```

生产函数不调用该命令，也不读取它生成的文件。许可证留在本地；原版代码只用于核对算法，没有复制发布其实现。资源获取遵守项目固定 Release 清单与原作者再分发条款，未获许可的资源保留原站来源。

## 5. 真实数据精度与耗时

2026-10-11，基线 `aeabc5aba9341f0358772eb00b1f07eb2cbaf643`；最新数值实现与 typed 回执为 `24a643af`。两例公开 T1 的冻结自产 `nu/brainmask` 在共享 A100 上串行运行，CPU affinity 四核。两例执行均完成，参考与候选输入文件 SHA 相同，输出 type=0 LTA 的 4×4 矩阵逐元素相同，严格最大误差为 0。参考源固定为 `d932c45`；这一验证范围是完整 GCA 注册，不代表完整 recon-all 等效。

| 输入 | Python文件API/s | 独立原版进程/s | 准备/s | 平移/s | 线性搜索/s | 完整EM/s | 写出/s | LTA最大误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 公共例A | 77.620 | 227.627 | 14.210 | 2.966 | 48.442 | 11.066 | 0.206 | 0 |
| 公共例B | 86.989 | 255.712 | 14.405 | 2.926 | 57.826 | 11.208 | 0.150 | 0 |

Python 时间包括数据读取、传输、JIT 和输出，排除解释器与顶部导入；原版包括进程启动，因此不能把比值解释为严格同边界的内核加速比。共享 GPU 负载与缓存状态使这些数值属于本轮观测。

例A稳健均值108、sigma 10.800，高于源阈值10.093，故禁用缩放并使用峰值107；例B均值109、sigma 7.300，低于源阈值10.187，保留峰值109。这两个输入覆盖共享预处理分支。稳定和全样本完整 EM 均保留源终止行为，本轮未接受最佳参数更新；六项协议单元检查另覆盖后续准牛顿方向和非有限拒绝试探，不能从本轮无更新推论其他输入的行为。

旧 `8c79f320` 例A 105.885 s 与原版矩阵不一致，失败记录保留。修正后曾发生最终 NumPy 标量 JSON 序列化错误，该轮仍为失败；最新两例具备完整 typed 成功回执。脑图由自产 prefix 的强度/分割对比生成并保存在私有目录，影像不进入公开仓库。整体等效尚未判定。

## 6. 更新与 benchmark 记录

- `aeabc5a`：继承完整候选搜索，但 Python EM 仅首次方向；作为冻结历史基线保留。
- `c937db14`：补齐稳定和全样本 EM、源步长搜索及完整轨迹。
- `322adb8d`：修复缺失分类器梯度贡献和目标评估预算；不丢弃源体素。
- `8c79f320`：保持 FP32 外循环停止条件及累计最佳更新记账；四项单元检查通过。第一例严格矩阵与原版未通过，历史失败保留。
- `51b09f03`：补齐宽分布缩放禁用分支及有序 FP32 平滑，CA normalization 复用共享修复。
- `a6a78895`：保留源非有限拒绝试探与步长收缩；六项协议检查通过。
- `24a643af`：修复 NumPy 标量报告序列化；两例完整 GCA 同输入严格矩阵检查通过。

新结果按源码 SHA、输入 SHA、资源 SHA 和完成回执记录。旧耗时仅作为旧环境观测保留，不混写为本次性能提升。

## 7. 原实现与参考文献

算法协议核对的固定 FreeSurfer 源版本为 `d932c45b7941662ea380a05efef580568b98d41a`：

- [mri_em_register 主流程](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_em_register/mri_em_register.cpp)
- [MRIemAlign 与准牛顿外层](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrimorph.cpp)
- [GCA 目标与梯度](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/gca.cpp)
- [直方图宽度拟合协议](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/histo.cpp)
- [OpenDFPMin 包装](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/numerics.cpp)
- [VXL 1.17 L-BFGS 协议](https://github.com/vxl/vxl/blob/v1.17.0/v3p/netlib/opt/lbfgs.c)
- Fischl et al. Whole brain segmentation: automated labeling of neuroanatomical structures in the human brain. *Neuron* 33, 341–355 (2002). DOI: [10.1016/S0896-6273(02)00569-X](https://doi.org/10.1016/S0896-6273(02)00569-X).
- Liu & Nocedal. On the limited memory BFGS method for large scale optimization. *Mathematical Programming* 45, 503–528 (1989). DOI: [10.1007/BF01589116](https://doi.org/10.1007/BF01589116).


2026-10-11 后续协议检查还覆盖无有效分类器试探：源实现产生非有限梯度后收缩步长，不能把梯度改为零，也不能提前跳过试探。初始非有限梯度和非有限目标仍明确失败；试探的非有限梯度以 `gradient_finite=False`、`gradient_norm=null` 记录，保留完整源步长协议。六项协议检查在新快照通过。第一例修复后的残留 LTA 与独立原版逐元素相同，但该轮报告序列化失败，保留失败状态，不以残留文件作为完成 gate。`24a643af` 修正新增搜索轨迹中的 NumPy 标量为 JSON 数值并补齐报告错误回执；最新新快照两例严格对比均通过。

新增输入合同：目标求值前核对 canonical affine 底行 `[0,0,0,1]`，并要求容差转换为 FP32 后仍有限且为正；非法输入不访问 objective。宽分布禁用时直接返回 atlas 峰值，令缩放为1，只在非禁用分支验证局部选峰索引。固定源 `HISTOfindBin` 先映射区域 bin，越界会置峰0并报错；FNIT 此保护保留禁用缩放的意图，扩展低强度区域的处理范围。低强度宽分布的合成数组只用于合同检查，不作为原版或真实数据 benchmark。既有例A禁用返回107、例B未禁用返回109，两例冻结成功证据不重标为新源码整版重跑。

交付前 `b19a212c` 的八项 GCA 协议与两项 API 边界合同检查共10项通过，覆盖调用者线程和精度状态恢复、输入校验及失败回执。另在固定 `60d4a1c7` 快照上使用实际 A100 GPU 运行两项候选评分回归：默认 dtype 改变不影响坐标、设备归约保留顺序和并列最优；两项通过且调用者状态恢复。对应 [合同记录](../../benchmark/recon_volume_python_20261011/contracts_final.public.json)、[GPU记录](../../benchmark/recon_volume_python_20261011/gpu_tests.public.json) 与 [JUnit XML](../../benchmark/recon_volume_python_20261011/gpu_tests.xml)。这些检查不重标既有真实数据 benchmark 的源码版本。


### 2026-10-11 安装包 guarded GCA 同输入回归

协调者源码 `9292a0a7be63f6bc96f2dfa358261cb1587e0f23` 的0.16.0 wheel（5309664 bytes，SHA256 `3e5e2eb0332d328cd92a968a17ae8cc2c6d872a961d103bca4a9d0d65df39e51`）通过私有 `pip --target` 安装。全部包内 Python 文件与 wheel 逐文件核验，六个数学模块与已交付核心源码相同。新增生产 worker 守卫不改变数学函数。

输入沿用既有两例 FNIT 自产 `nu.mgz`、同网格 `brainmask.mgz` 及固定 GCA，分别核对原成功回执的 SHA。LTA 是本轮输出；旧自产与隔离原版 LTA 的文件字节先读取以核对各自成功回执 SHA，4×4矩阵仅在 worker 退出后解析比较；参考 LTA 不传入生产 worker。每例新 exec 运行完整 stable/all EM，`candidate_chunk=64`、`sample_chunk=8192` 与原成功配置相同。CLI 冷启动指新进程、新 CUDA context、独立空 NumBa 缓存；文件系统和设备硬件缓存未控制。

| 输入 | 冷CLI含退出/s | 对旧自产LTA最大差 | 对隔离原版LTA最大差 | 完整过程指标 | stable/all接受更新 |
|---|---:|---:|---:|---|---|
| 公共例A | 173.706 | 0 | 0 | 全部一致 | 0 / 0 |
| 公共例B | 173.714 | 0 | 0 | 全部一致 | 0 / 0 |

过程一致性逐项覆盖直方图预处理、平移/线性搜索轨迹、完整 EM 记录、代价、试探次数和线性迭代次数。两例本来均未接受 EM 更新，保留此真实输入限制，不扰动输入制造更新，也不将合同视为 MRI 证据。本轮只确认该安装包的 guarded GCA 计算可执行及此范围的复现。与隔离原版参考的LTA比较是16个矩阵元素零差，两例文件SHA均不同。

实际注册入口守卫启用且禁令环境为1，直接 Python 执行记录无禁用程序尝试。该守卫范围为 Python subprocess/exec/posix_spawn，不是外部 shell 子孙的完整系统调用沙箱。两例 TF32 均为True、autocast=False；CPU目标矩阵和梯度实际为float32，优化向量为float64，保留原有有序求值例外。Torch/NumBa 各4线程、interop=1、CPU affinity为4核。目标为共享A100；准备时GPU利用率两次为0%，安装后启动采样为9%。因此耗时是含探针开销的共享设备观测，不能直接与旧排除导入的API时间构成加速比。父子同时显存归属未知，仍为null；子进程allocator峰值单列，20 GB门和整体等效均未判定。

[范围摘要](../../benchmark/recon_guarded_gca_9292a0a7_20261011/guarded_gca_summary.public.json)与[完整typed回执](../../benchmark/recon_guarded_gca_9292a0a7_20261011/guarded_gca.full.public.json)绑定 wheel、生产源码、输入、旧回执、LTA、比较/探针源码哈希及实际线程、精度和退出结果。未启动新前段或完整重建；已有脑图仍属于原冻结prefix证据。

独立诊断参数：`--root` 为含固定索引、自产输入及原成功回执的FNIT根目录；`--workspace` 为含冻结wheel、私有installed与audit/sitecustomize.py的新任务源码目录；`--output` 为尚不存在的新benchmark目录，产生每例LTA、私有worker日志和JSON、typed比较回执。固定生产版本、案例和资源哈希由脚本核验，任何失败保留失败报告并以非零退出。

```bash
# 冻结wheel由协调者交付；安装不会修改共享Conda环境。
FNIT_PRIVATE_ROOT="/path/to/FNIT"
FNIT_GCA_WORKSPACE="/path/to/private/gca-workspace"
FNIT_GCA_BENCHMARK_OUTPUT="/path/to/private/new-run/benchmark"
FNIT_GCA_GPU_UUID="GPU-your-selected-device-uuid"
mkdir -p "$FNIT_GCA_WORKSPACE/audit"
cp tools/probe_recon_guarded_gca_worker.py "$FNIT_GCA_WORKSPACE/audit/sitecustomize.py"
python -m pip install --no-deps --no-index --no-compile \
  --target "$FNIT_GCA_WORKSPACE/installed" \
  "$FNIT_GCA_WORKSPACE/fudan_neuroimaging_toolkit-0.16.0-py3-none-any.whl"
export PYTHONPATH="$FNIT_GCA_WORKSPACE/installed"
export CUDA_VISIBLE_DEVICES="$FNIT_GCA_GPU_UUID"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export FNIT_RECON_FORBID_ORIGINAL_STAGES=1
# 96-99是本次4核例子；现场按许可亲和性选择，不与其他任务争用。
taskset -c 96-99 python tools/benchmark_recon_guarded_gca_worker.py \
  --root "$FNIT_PRIVATE_ROOT" --workspace "$FNIT_GCA_WORKSPACE" \
  --output "$FNIT_GCA_BENCHMARK_OUTPUT"
```

原软件对应命令与文献沿用第4、7节；本轮读取既有隔离同输入原版成功结果，不启动新的原版参考程序。

导出后的文案修订只澄清 LTA 字节预检与矩阵解析的时序，未重新执行；旧报告中的 benchmark/probe SHA 继续指向实际运行的冻结脚本，不改写为文案修订后的文件SHA。
