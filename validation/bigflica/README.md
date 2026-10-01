# BigFLICA：真实 UKB 输入、GPU 对照与内存测试

## 当前DicL效果匹配

同一1000人完整VBM/FA/MD掩膜R500/D200投影，对照服务器sklearn1.7.1的 `MiniBatchDictionaryLearning`。已移除额外ridge，匹配LARS节点停止，并修复统计顺序、SVD驱动及整模态GPU缓存读取。独立初始化的三模态字典、原子余弦及LASSO指标通过预设容差；FA/MD的OMP30重建仍未通过。[最新报告、误差轨迹、因果控制和复现命令](dicl_match_real1000_20261001.md)记录实际来源、耗时与87项定向回归。本轮仅验收DicL，未重新运行FLICA或30,000人全链。

## mMIGP 500维测试

DicL本轮修复前，同1000人完整VBM/FA/MD掩膜的R500/D200/C20测试已完成。mMIGP约25.57秒；当时CPU/GPU两种字典来源及o/R两种噪声的八个拟合均由初始20成分收缩为零，未通过C20。列数超过字典原子数使当前自动DD走取1分支，不能把结果只解释为增加PC；改善后的字典未重跑FLICA。[历史测试配置、阶段耗时与限制](mmigp500_real1000_20261001.md)。

## 初始化与逐模态先验修复

当前源码修正了 PCA 的 MATLAB SVD 尺度、逐模态 W 先验、零影像判定、归一化统计保存精度和脑图回归设计检查。相关 BigFLICA/SuperBigFLICA 共137项测试通过。真实1000人新先验原始体素R控制为20个成分；同字典新初始化的R控制为7、当前拟合函数默认o为2，CPU/GPU一致并拒绝C20输出。[本轮完整控制与剩余问题](flica_initialization_prior_fix_real1000_20261001.md)记录其整体RMS输入、拟合阶段时间及限制；下文保留旧源码的基线，便于追溯。

## 原始体素先行的定位试验

固定1000人和完整三模态掩膜，使用 MATLAB 整体 RMS/SVD 设置，原始体素 FLICA 在 GPU1000次更新后仍保留20个成分；同初态 CPU/GPU100次参数相对差最大 `1.65e-10`。[原始体素基线](raw_flica_real1000_20261001.md)记录 z-stat、残差、阶段时间和显存。随后同输入 mMIGP 原投影/归一化投影为1/19个成分，同投影 CPU/GPU DicL 后均为7。[压缩对照](compression_after_raw_real1000_20261001.md)区分数值误差与尺度、DD、噪声坐标变化。[数学修复](flica_math_fixes_20261001.md)及这些定位试验不替代下面独立30,000人默认流程的失败结果。

## 30,000 人独立 CPU/GPU 对比（当前结论）

按用户后续指令启动的 30,000 人试验已结束，**CPU 和 GPU 均未通过 C20 验收**。CPU 保留 17 个有效成分，GPU 保留 13 个；两端在 FLICA 诊断后停止，没有生成最终 C20 course、z-stat 脑图、阈值图或可用于新被试的完整模型。对比脚本完成不代表算法验收完成。[不含被试 ID 的结果](bigflica_real30000_cpu_gpu_20260930.json)记录状态 `scientific_acceptance_incomplete`。

本次运行冻结生产源码为 `6f8dcea1eb7a0f19ac91df38495a4640d385a1d0`。两端独立从原始 NIfTI 建库，使用同一有序被试清单、模态和完整掩膜；输入签名相同，清单 SHA-256 为 `18fc46f3ed9d28660d04a19d8207a51977520f9f01cbd2daf61bbe6fc3a4be88`。未使用 task 模态，也没有读取 CPU 投影作为 GPU 输入。

| 模态 | 每名被试的文件 | 掩膜体素数 |
|---|---|---:|
| VBM | `VBM_2mm.nii.gz` | 157,901 |
| FA | `dti_FA_2mm_mmorf.nii.gz` | 222,257 |
| MD | `dti_MD_2mm_mmorf.nii.gz` | 222,261 |

三种影像均为 `91×109×91`、2 mm 标准空间网格。固定参数为 C20/R100/D200、DicL 最多 1000 epoch、batch32、LARS 最多120个事件、FLICA `max_iter=1000`、噪声模式 `R`、seed0、feature_block2048、top_voxels1000。CPU 使用 float64 标准化和 mMIGP、sklearn DicL；GPU 使用 float32 标准化和严格 float32 mMIGP，DicL 内部沿用 float64。mMIGP 的 TF32 关闭，未使用 float16。

### 实测耗时与资源

| 阶段 | CPU 秒 | GPU 秒 |
|---|---:|---:|
| 读取与标准化：90,000 幅影像 | 5,478.73 | 4,358.81 |
| mMIGP | 3,541.38 | 2,176.20 |
| DicL | 79.85 | 145.19 |
| FLICA，至有效秩检查失败 | 23.60 | 46.58 |
| **至失败的总墙钟** | **9,182.28（153.04 分钟）** | **6,753.04（112.55 分钟）** |

CPU 峰值 RSS 为 8.48 GiB；GPU 流程主机峰值 RSS 为 1.80 GiB，分配显存峰值3.66 GiB、保留显存峰值3.85 GiB。原始 `N×P` 矩阵逐被试写入 HDF5，后续分块读取。运行器每5秒监测 RSS，到32 GiB退出；CUDA分配器上限20 GiB，mMIGP预算19 GiB。

两端在同一 gpucw1 顺序运行，CPU 线程均为8。GPU 首次初始化显存不足，在读入影像前失败；保留该尝试后，固定 H100 UUID，由 CPU 先运行、GPU 后运行。表中 GPU 时间仅为第二次运行，不含首次失败和等待。共享 GPU 持续有外部负载，未清空操作系统页缓存，且 CPU/GPU 的 mMIGP 精度不同。**这些是单次观测，不能作为等价重建或受控硬件加速比。**本次 DicL 和 FLICA 的 GPU 耗时均高于 CPU。

### 精度与未通过项

两端 mMIGP 收敛检查均通过：CPU 63次子空间迭代，整体相对残差 `7.84e-9`；GPU 90次，残差 `2.96e-7`。`U` 的相对 Frobenius 误差为 `6.07e-6`，最低同序绝对相关为 `0.9999999575`。VBM/FA/MD 投影的相对误差分别为 `1.47e-6/1.03e-6/7.09e-7`。

DicL 字典经过 Hungarian 原子匹配和符号校正后，相对 Frobenius 误差仍为 VBM `0.3039`、FA `0.1618`、MD `0.2937`；对应绝对余弦中位数为 `0.9905/0.9911/0.9881`，最小值为 `0.2580/0.8250/0.1390`。少数原子的差异明显。此前2,050人同投影控制支持“上游微小扰动可在非凸训练中放大”的解释，但本次未做30,000人同投影控制，不能仅据此排除 DicL 实现差异。

| FLICA 诊断 | CPU | GPU |
|---|---:|---:|
| 有效成分 / 请求成分 | 17 / 20 | 13 / 20 |
| VBM 重建范数比 | 0.001091 | 6.54e-17 |
| FA 重建范数比 | 0.3695 | 0.2860 |
| MD 重建范数比 | 0.3044 | 0.2831 |

重建范数比是 `||重建字典|| / ||输入字典||`，不是解释方差。GPU 的 VBM 重建几乎为零；CPU/GPU 分别有3/7个 course 成分被置零。保留原有效秩和模态检查，没有改为 C3、删列或降低阈值。剩余工作包括：固定同一投影定位 DicL 差异与停止轨迹；检查 FLICA 的剪枝、噪声估计及 VBM 抑制；恢复有效 C20 后核验 course、各模态脑图和留出被试投影；再测受控阶段耗时。

### 复现输入、脚本与输出

[运行脚本](benchmark_public_real30000.py)接受两个位置参数：实验目录，以及 `cpu` 或 `cuda:0`。实验目录事先包含 `subjects.txt`（恰好30,000个不同训练被试目录名）和 `input.json`。每个后端的 `cpu/`、`gpu/` 子目录须尚不存在。安装沿用项目 `environment.yml`，没有新增依赖；本次实测环境是 PyTorch2.5.1、NumPy1.26.4、sklearn1.7.1、nibabel5.4.0，不能写成固定sklearn1.5.2的Conda环境重建验收。

准备好被试清单及三个掩膜后，在实际执行的源码检出中生成配置：

```python
import hashlib
import json
import subprocess
from pathlib import Path

benchmark_directory = Path("/absolute/private/benchmark_30000")
subjects_root = Path("/absolute/private/ukb_multimodal_new")
mask_directory = Path("/absolute/private/masks")
subjects_file = benchmark_directory / "subjects.txt"  # 已准备的30,000人训练名单
source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
benchmark_config = {
    "subjects_root": str(subjects_root),
    "modalities": {
        "vbm": {"image": "VBM_2mm.nii.gz", "mask": str(mask_directory / "vbm.nii.gz")},
        "fa": {"image": "dti_FA_2mm_mmorf.nii.gz", "mask": str(mask_directory / "fa.nii.gz")},
        "md": {"image": "dti_MD_2mm_mmorf.nii.gz", "mask": str(mask_directory / "md.nii.gz")},
    },
    "held_out_subject": "NEW_SUBJECT",  # 不在训练名单内；只有模型验收通过才调用
    "subjects_sha256": hashlib.sha256(subjects_file.read_bytes()).hexdigest(),
    "source_commit": source_commit,  # 记录实际源码；本报告的历史源码是6f8dcea
    "execution_order": ["cpu", "gpu"],
}
(benchmark_directory / "input.json").write_text(json.dumps(benchmark_config, indent=2))
```

以下命令从仓库根目录顺序执行。`benchmark_directory` 指上述私密目录，`selected_gpu_uuid` 指有至少20 GiB空闲显存的GPU；本次自动等待使用22 GiB余量。状态码2表示科学验收失败，应保留诊断并继续另一端对照；状态码1表示执行异常。配置中的提交号必须对应实际运行源码，复现历史测量还须使用该冻结版本的 `src`。

```bash
benchmark_directory=/absolute/private/benchmark_30000
selected_gpu_uuid=GPU-REPLACE-WITH-YOUR-UUID
# CPU和GPU分别从原始影像建库；不要复用另一端的归一化或投影缓存。
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  PYTHONPATH=src python validation/bigflica/benchmark_public_real30000.py "$benchmark_directory" cpu
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$selected_gpu_uuid" \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  PYTHONPATH=src python validation/bigflica/benchmark_public_real30000.py "$benchmark_directory" cuda:0
python validation/bigflica/compare_public_real30000.py "$benchmark_directory"
```

每端保存 `progress.json`、`resources.jsonl`、`aggregate.json` 和 `output/` 中的归一化、mMIGP、DicL缓存及FLICA诊断。[对比脚本](compare_public_real30000.py)按体素块读取投影，生成 `comparison.json`；只有两端完整模型都通过时才比较 C20 course 和脑图。目录中的被试清单、配置、影像、缓存、日志及可能含私密路径的错误信息留在私密环境；本仓库只发布审查后的无被试ID汇总。下面各节保留较早2,050人及小样本的测试条件，不替代本次结果。

## 当前实现所需的补充证据

以下只保留用于核对原实现、固定投影精度、输出和新被试接口的对照；18人/C3结果不替代上面的30,000人/C20验收。

| 用途 | 真实数据与结果 | 记录 |
|---|---|---|
| CPU与原notebook输出比较 | 18人、四个小掩膜、C3/R10/D40；course绝对相关约1，z图相关最低0.9999999999999993 | [原软件输出对照](map_compare.json) |
| FLICA原源码排查 | 同一2,050人四模态R100/D200字典，原Python与FNIT在10/30/100轮的H、X、W一致；有效秩同为20→2→2 | [原源码对照](upstream_flica_parity.json)、[维度和噪声诊断](flica_dimensionality_audit.json) |
| 固定相同投影的DicL/FLICA控制 | 2,050人VBM/FA/MD完整掩膜，R100/D200；字典相对差约1e-6–7e-5，C3 course及九张z图相关均超过0.9999999984 | [同投影隔离](three_structural_f32_same_projected_control_real2050.json) |
| 此前ADMM/LARS运行时 | 同一2,050人投影，内部float64；沿用已测CPU基线，GPU各模态字典误差1.06e-5/1.54e-6/6.56e-5；CPU/GPU C20均为11/20 | [历史阶段报告](dicl_bpdn_real2050.json) |
| 当前缓存与输出 | 2,050×3 course、九张z NIfTI、阈值图和PNG；掩膜内NIfTI与数组相同，留出被试返回有限course；复用缓存耗时22.19秒 | [缓存及输出检查](dicl_bpdn_cache_api_real2050.json) |
| 单被试接口示例 | 18人三个小掩膜，C3/R10/D40、噪声R；API与CLI输出检查及独立留出投影 | [接口检查](three_structural_public_R_smoke_real18.json)、[留出投影](three_structural_public_R_heldout_apply_real1.json) |

此前2,050人DicL阶段耗时为CPU165.84秒（同输入实测后复用）和GPU144.23秒，GPU峰值分配显存1.29 GiB；共享负载与测量时窗不同，不能作受控加速比。这个固定投影结果与30,000人独立全链的CPU DicL79.85秒、GPU145.19秒属于不同输入和比较条件。

[阶段脚本](benchmark_dicl_real2050.py)接受三个位置参数：含R100 float32投影及U的目录、已完成的同输入CPU基线目录（填 `-` 时重新拟合）、新的私密输出目录。[缓存检查脚本](check_cache_api_real2050.py)接受已有完整C3公开模型的父目录、阶段脚本输出目录、新的私密模型输出目录，验证输入哈希后复用mMIGP/DicL。[同投影定位脚本](diagnose_public_gpu_projected_dicl_cpu_real2050.py)读取已有GPU投影、字典和C3模型，隔离CPU sklearn与GPU DicL的差异。上述脚本仍对应2,050人阶段核验，不作为30,000人冷启动模板。

```bash
projected_directory=/absolute/private/mmigp_100
stage_output_directory=/absolute/private/dicl_stage
existing_model_directory=/absolute/private/public_C3_output
cache_check_directory=/absolute/private/cache_check
# 在同一投影上重新测CPU，并执行当前GPU DicL。
CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-YOUR-UUID PYTHONPATH=src \
  python validation/bigflica/benchmark_dicl_real2050.py \
  "$projected_directory" - "$stage_output_directory"
# 用经过输入哈希核对的阶段字典检查公开输出及留出被试调用。
CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-YOUR-UUID PYTHONPATH=src \
  python validation/bigflica/check_cache_api_real2050.py \
  "$existing_model_directory" "$stage_output_directory" "$cache_check_directory"
```

历史阶段脚本的C3只用于数值与接口检查；C20检查失败也会在报告中保留，`status=complete`仅表示阶段测试结束。该历史DicL实现SHA-256为 `fa6f41c3ca7b2a477b986ec2c72482cf748f3e8002318e883fc56b1eb4364b52`；其余元数据保留在所链接JSON中。当前源码及验收见页首最新效果报告；早期候选、部分运行结果和重复版本仍可从提交历史追溯。
