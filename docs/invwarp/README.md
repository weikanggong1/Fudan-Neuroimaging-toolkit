# TorchInvWarp：在指定网格反转FSL位移场

| 项目 | 内容 |
|---|---|
| 输入 | 3D反场参考网格、前向dense场或FNIRT系数 |
| 输出 | 定义在原始网格、指向标准空间的三通道反场 |
| 对应原软件 | FSL invwarp；absolute输出另加convertwarp |
| Python / CLI | TorchInvWarp / fnit invwarp、fnit-invwarp |
| CPU / GPU | CPU或CUDA；FP64几何/求逆、FP32保存 |

## 1. 功能简介

`TorchInvWarp` 将一个 FSL pull 场求逆，并在指定原始影像网格上保存反场。例如，DWI→MNI 的场定义在 MNI 网格、指向 DWI；求逆后的 MNI→DWI 场定义在 DWI 网格、指向 MNI。反场可交给 `TorchApplyWarp`，将标准空间掩膜或标量图重采样到个体空间。

支持 FSL dense 相对/绝对场及 intent-2007 FNIRT 三次样条系数。计算使用 PyTorch；生产 API 不启动 FSL。先从前向场估计全局仿射，再执行固定点修正。坐标与关键计算保持 float64，保存的 NIfTI 是 float32，不使用 FP16/BF16。CUDA 分支保持既有运算，遵循 FNIT 的 TF32 设置。

## 2. Python 调用

```python
from fnit import TorchInvWarp

native_reference_path = "/absolute/path/native_T1w.nii.gz"  # 输入：决定反场的完整 3D 网格
forward_warp_path = "/absolute/path/T1w_to_MNI_coeff.nii.gz"  # 输入：MNI 网格上的前向 pull 场/系数
inverse_warp_path = "/absolute/path/MNI_to_T1w_warp.nii.gz"  # 输出：native 网格上的反场

inverse_result = TorchInvWarp(device="cuda:0").run(
    reference=native_reference_path,
    warp=forward_warp_path,
    output=inverse_warp_path,
    warp_convention="auto",  # FNIRT 系数自动解释为相对位移；dense 场建议显式指定
    output_convention="relative",  # 保存相对位移，单位为 FSL scaled-mm
    iterations=30,  # 固定点修正的最大次数
    tolerance_mm=0.01,  # 全网格修正分量的最大绝对值小于该值时提前停止
)
print(inverse_result.image.shape, inverse_result.valid_fraction, inverse_result.qc)
```

仅计算、不保存时调用 `TorchInvWarp(...)(reference, warp, ...)`；也可 `from fnit.invwarp import invwarp`，再用 `invwarp(reference, warp, device="cpu", ...)` 调用同样的函数接口。

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `device` | 否 | str / torch.device / None | `'cpu'` | 计算设备；显式 CUDA 不可用时报错。 |
| `reference` | 是 | 路径 / NIfTI / None | 无 | 参考图；决定输出空间、shape、affine。 |
| `warp` | 是 | 路径 / NIfTI | 无 | FSL dense 场或 intent-2007 FNIRT 系数。 |
| `warp_convention` | 否 | str | `'auto'` | auto/relative/absolute；系数由 intent 解释。 |
| `output_convention` | 否 | str | `'relative'` | relative位移或absolute pull坐标，单位 FSL scaled-mm。 |
| `iterations` | 否 | int | `30` | 固定点修正最大次数，正整数。 |
| `tolerance_mm` | 否 | float | `0.01` | 最大修正分量停止阈值，正有限数，mm。 |
| `output` | 是 | 路径 / None | 无 | 输出文件或前缀；父目录按接口创建。 |

### 输出

```text
/data/
└── MNI_to_T1w_warp.nii.gz
```

结果为 `WarpFieldResult`：

| 字段 | 输出结构与含义 |
|---|---|
| `image` | float32 NIfTI，shape=`reference.shape + (3,)`；affine、qform、sform 沿用 reference。相对场 intent=2006，绝对场 intent=0。 |
| `valid_fraction` | 全输出网格中，最终采样坐标落在前向场有效网格内的比例。 |
| `qc` | 设备、输入/输出约定、最大次数 `iterations`、实际次数 `iterations_used`、`converged`、阈值及有效场内中位残差。 |

场外采样使用边界值；场外反场不具有保证的解剖意义。固定点法对大变形或折叠场可能不收敛，需同时查看 `converged`、场内残差和有效覆盖，不能只检查文件成功保存。

### 输入数据格式

reference为3D原始空间影像；warp为目标网格上的(X,Y,Z,3)场或intent-2007系数。两个输入可为路径或nibabel对象。矩阵/场使用FSL scaled-mm；反场定义在reference网格，指向前向场的目标空间。

### 只计算及绝对输出

```python
from fnit.invwarp import invwarp

native_reference_path = "/data/native_T1w.nii.gz"  # 反场目标网格
forward_coefficient_path = "/data/T1w_to_MNI_coeff.nii.gz"  # 前向FNIRT系数
absolute_inverse_output_path = "/data/MNI_to_T1w_absolute.nii.gz"  # 绝对pull坐标
absolute_inverse_result = invwarp(
    reference=native_reference_path, warp=forward_coefficient_path,
    device="cpu", output_convention="absolute",  # 只计算，不自动保存
    iterations=30, tolerance_mm=0.01,
)
absolute_inverse_result.save(absolute_inverse_output_path)  # 明确保存请求
```

对系数输入，warp_convention不改变其相对位移解释；只有输出约定改变保存值。absolute仍是FSL scaled-mm，不是scanner RAS。

### 检查求逆结果

qc的converged只表示修正步达到停止阈值。valid_fraction反映前向场网格覆盖；两者都不能替代解剖配准评估。

- 对标准掩膜回个体空间，下一步以native参考网格调用ApplyWarp并使用nearest。
- 对连续图，选择与实际分析一致的插值并保留同一反场。
- 大变形或折叠输入可能不收敛；iterations_used、场内残差和有效覆盖应一起保留。
- Python保存会替换同名文件；命令行默认保护已有文件。

## 3. 命令行调用

```bash
# --ref：反场的完整 native 输出网格；--warp：前向 pull 场或 FNIRT 系数。
# --rel：dense 输入及输出均使用相对位移；--out：保存 4D 反场。
# --niter：FNIT 固定点修正最大次数；--device：运行设备。
fnit invwarp --ref /absolute/path/native_T1w.nii.gz \
  --warp /absolute/path/T1w_to_MNI_coeff.nii.gz \
  --out /absolute/path/MNI_to_T1w_warp.nii.gz \
  --rel --niter 30 --device cpu
```

独立入口 `fnit-invwarp` 接受相同参数。省略 `--rel/--abs` 时输入约定为 `auto`、输出为 relative；`--abs` 选择 absolute dense 输入和输出，系数输入仍按 relative 解释。`--overwrite` 允许替换已有输出；CLI 只暴露默认 `tolerance_mm=0.01`，需要更改阈值或混合输入/输出约定时用 Python。

| CLI 参数 | Python 参数 | 含义 |
|---|---|---|
| `--ref / --warp / --out` | `reference / warp / output` | 参考/前向场/保存路径 |
| `--rel / --abs` | `warp_convention / output_convention` | dense输入和输出同约定 |
| `--niter` | `iterations` | 最大修正次数；CLI不暴露tolerance_mm |
| `--device / --overwrite` | `device / 写盘策略` | 设备与覆盖 |

## 4. 原软件调用

```bash
# FSL 仅用于独立 benchmark：保持相同前向场及输出网格。
invwarp --ref=/absolute/path/native_T1w.nii.gz \
  --warp=/absolute/path/T1w_to_MNI_coeff.nii.gz \
  --out=/absolute/path/fsl_MNI_to_T1w_warp.nii.gz --rel
```

2026-10-04 在 CPU评测节点 实测的 FSL 6.0.7.4 `invwarp` 不接受 `--niter`，也没有显式 CPU 线程选项；FNIT 的 `--niter` 只控制自己的固定点求解。官方 benchmark 保留原程序内部默认停止设置。双方使用相同 CPU 亲和性和线程上限，实际 CPU 利用率另列；不能称原程序实际使用了 8 线程。

FSL 使用其原生求逆算法和 Jacobian 约束；FNIT 尚未提供 `regularise/jmin/jmax/noconstraint` 对应选项。现场 FSL 6.0.7.4 的 `--rel/--abs` 指定 dense 输入约定，保存的反场始终为 relative；系数输入忽略 dense 约定。部分官方网页将这些 flag 描述为输入与输出约定，本页采用实际源码和输出验证的行为。

需要 absolute 输出时，原版参照追加以下转换，两个完整进程及读写均计时；absolute 输入、relative 输出直接使用 `invwarp --abs`。相同输入和输出契约不表示两套求解器数值等价。

```bash
# invwarp 先保存 relative 反场；明确转换为 absolute 输出。
convertwarp --ref=/absolute/path/native_T1w.nii.gz \
  --warp1=/absolute/path/fsl_MNI_to_T1w_warp.nii.gz \
  --out=/absolute/path/fsl_MNI_to_T1w_absolute.nii.gz --rel --absout
```

## 5. 最新精度和运行时间

CPU prepared FP64 系数取样使用 fresh-base 两步加法，停止检查减少整卷临时数组；公开参数和 CUDA 运算保持既有路径。[实现范围、CPU评测节点 完整 1/8 核门槛及独立阶段时钟](CPU_ALLOCATION_20261004.md)记录全部文件、QC 和 30 次实际停止一致。CPU评测节点 新官方配对的完整进程中位数为 35.878/50.899 s（原版/FNIT，1 核）与 36.679/21.661 s（8 核）；单核目标仍未达到。

当前 Inv/Convert/Apply/normalizer 的 runtime SHA 与这份正式配对相同。CPU 完整进程包含启动、导入和全量读写；完整 pair warmup 后保留三次交替配对，API 另做完整 warmup 和三次全量求逆及保存。

| CPU 上限 | FSL 完整进程中位数 | FNIT 完整进程中位数 | FNIT 热 API 中位数（含读写） |
|---|---:|---:|---:|
| 1 | 35.878 s | 50.899 s | 48.571 s |
| 8 | 36.679 s | 21.661 s | 18.441 s |

原版实际平均约一核，8 表示双方的资源上限；CPU1 仍慢于官方，CPU8 达本病例速度目标。脑掩膜内向量差 mean/median/p95/max 为 0.01723/0.01008/0.03361/7.31428 mm，全网格分量 max 为 7.02610 mm；两套求解器尚不数值等价。[正式记录](assets/cpu-allocation-v2-node8-official-20261004.public.json)保留全部重复、精度与源码核对。

| H100 完整操作，源 v28 | 起始 FNIT API 两组中位数 | v28 API 两组中位数 | 对应调用 peak allocation（B） |
|---|---:|---:|---:|
| 本例完整系数反场 | 4.298 / 4.094 s | 3.870 / 3.993 s | 1,275,725,824；16 次均相同。 |
| 同期 FNIRT 完整六层、三阶段 TBSS | 14.866 / 13.365 s | 14.854 / 14.102 s | warmup 3,564,715,520；测量 3,565,562,368；旧新对应调用相同。 |

[源 v28 H100 公报](../../validation/multimodal_cpu_20261004/gpu_cpu_final_v28_ready_retry_20261004.public.json)包含两例各四进程，每进程完整 warmup 一次及三次测量，共 32 次保存。反场全部 18,808,200 值，TBSS 的 cout、iout、jout 与完整 pull Jacobian，以及所有 header/extensions/affine，均与起始 FNIT `1d31e7b` 逐位一致。GPU API 包含读取、计算和保存，排除启动与入口导入，函数内惰性导入计入；同 UUID、TF32、20 GB 上限及前后源码/输入 SHA 均核对。共享时钟不作稳定速度结论；TBSS 的 CPU 对照见 [FNIRT 功能页](../fnirt/README.md)。

### 分步骤 benchmark

| 阶段 | FNIT | FSL |
|---|---|---|
| 求逆迭代/取样及停止检查 | [独立阶段记录](CPU_ALLOCATION_20261004.md)，未与官方同阶段配对 | 未记录 |

测试绑定源v28，基线FNIT `1d31e7b`，官方FSL6.0.7.4；CPU Intel Xeon Gold6418H、1/8核预算，GPU H100、TF32、20GB上限；1例完整真实系数反场。CPU完整进程含启动导入读写，GPU完整API含读写、不含启动/入口导入。内部FP64、保存FP32。

另用真实 T1/TBSS 完成 14 个功能分支、各 1/8 线程，共 28 个完整记录，其中 18 项有新官方配对、10 项仅复用已核验的精度参照：[逐项精度与时间表](CPU_BENCHMARK.md#14-项真实功能的完整-18-线程结果) · [聚合报告](assets/cpu-functional-v17-corrected-v2-20261004.public.json)。系数分支的指定脑区 p95 向量差约 0.03361 / 0.00659 mm；T1 dense 分支约 1.78 mm，最大 311.44 mm，尚不能称数值等价。CPU1 relative 系数及未压缩输出仍存在速度差距；复用精度参照的分支没有新原版时钟，未计算速度比。

![公开 T1 的原版与 FNIT 反场模长及向量差](assets/invwarp_displacement.png)

公开脑图绑定ds000114 T1的v17 CPU8，覆盖256×156×256×3；不是v28新脑图。[图示SHA及范围](../fnirt/assets/figures-v17-20261004.public.json)。

## 6. 最近版本和 benchmark

| 日期 | 更新与真实验证范围 |
|---|---|
| 2026-10-04，源 v28 | prepared CPU FP64 两步 fresh-base 加法和标量停止减少临时分配；CPU评测节点 完整官方 CPU1 仍慢于原版，CPU8 更快。H100 完整反场与三阶段 TBSS 共 32 次保存的全部输出及对应显存峰值一致。 |
| 2026-10-04，源 v26 | CPU prepared-source 的严格静态对角坐标分支保持 FP64；1/8 线程的完整文件、30 次实际停止和 QC 与旧版一致。新官方 warm+3 配对的 CPU1 仍慢于原版，CPU8 更快；完整 GPU 16 次保存及峰值显存一致。 |
| 2026-10-04 | CPU 复用 FP64 残差/矩阵乘缓冲并就地更新；CUDA 分支保留原算式。拒绝非正整数 iterations 和非有限/非正 tolerance，新增实际停止次数和收敛字段。完整旧循环回归逐位核对 dense、系数、输出约定及停止状态；CPU 官方配对结果独立记录。 |
| 2026-10-04 参照修正 | 现场发现原版保存 relative 反场；修正 absolute 输出的隔离参照为完整 `invwarp + convertwarp --rel --absout` 链。旧错误约定比较保留失败/无效原因，不作为误差或速度结论。生产求逆算法未因此改动。 |

## 7. 参考文献、原软件和资源

源码位置：[原软件 `fnirt-2203.0/invwarp.cc`](../../src/fnit/_vendor_fsl/sources/fnirt-2203.0/invwarp.cc)；[FNIT `invwarp/core.py`](../../src/fnit/invwarp/core.py)。固定 tag/commit、Git tree 及每文件 SHA-256 见[来源清单](../../src/fnit/_vendor_fsl/manifest.json)。

代码改写沿用 [FSL 6.0 非商业许可证](../../licenses/FSL-6.0.txt)，完整来源与再分发要求见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

- Andersson, Jenkinson & Smith, *Non-linear registration, aka spatial normalisation*, FMRIB Technical Report TR07JA2 (2007)，[原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。`invwarp` 无单独的方法论文。
- [FSL FNIRT 原实现（含 invwarp）](https://git.fmrib.ox.ac.uk/fsl/fnirt)及[官方参数说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html#invwarp)。具体可接受选项以实测二进制解析为准。
- 派生实现受 [FSL Software Licence 6.0](../../licenses/FSL-6.0.txt)约束；官方程序只用于隔离参考，不是 FNIT 生产依赖。
- 公开示例来源、CC0 元数据、去面部处理及逐文件 SHA-256 见 [SOURCES.json](../../examples/data/SOURCES.json)。公开 brain figure 不使用本轮私有病例。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 本功能无模型权重；用户自备参考影像、掩膜或变换 | 定义目标网格/变换 | 各文件原作者 | 按实际文件 | 按实际文件 | 不随本功能发布用户数据 |

[完整历史说明与调试证据](../../validation/invwarp/readme_archive_20261005.md) · [返回主页](../../README.md)

<!-- 旧版文档锚点兼容 -->
<a id="1-功能简介"></a> <a id="2-python-调用输入输出与参数"></a> <a id="3-命令行调用"></a> <a id="4-原软件调用与对应范围"></a> <a id="5-最新真实数据精度耗时与脑图"></a> <a id="源-v28最新-cpu-与-gpu-完整对照"></a> <a id="源-v26cpu-采样优化与历史官方配对"></a> <a id="公开-t1-示例与完整功能矩阵"></a> <a id="6-最近版本与-benchmark-记录"></a> <a id="7-参考文献原实现与许可"></a>
