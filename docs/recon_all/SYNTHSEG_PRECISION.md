# SynthSeg 的实际前向精度

## 修复内容

旧 recon-all 调度先关闭 cuDNN TF32，再构造 SynthSeg。两个模型构造函数都会
调用 `configure_device`，它在 CUDA 上重新开启 matmul 和 cuDNN TF32，因此
调度器的关闭设置没有覆盖实际前向。这是可由代码调用顺序确认的错误，不能
依据旧报告的 `fp32_exceptions` 字段推断当时模型实际使用了什么精度。

现在设备选择和精度设置分开。`configure_device(device=None,
configure_precision=True)` 保留其他功能的原 CUDA 默认；设置
`configure_precision=False` 时只检查并返回设备，不改变 TF32。
SynthSeg 的两个构造函数使用这个选项，实际推理时再应用模型精度策略。

阶段 1 修复的是实际 cuDNN 策略：独立 API 仍默认 `cudnn_tf32=True`，
recon-all 显式传入 `False`，在构造完成后和两次真实前向的作用域内关闭
cuDNN TF32，保留 matmul TF32 默认。不能把这项 TF32 → FP32 卷积修复
当作只有计时变化的性能替换；标签、体积和耗时的变化须单独比较。

本次[冻结真实 T1 的旧有效策略诊断](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/old_effective_tf32/actual-forward.json)
绑定 `3d9856c...+stage1tar0444db72`：original/flipped 两次前向都实测
`cudnn_tf32=True`、`matmul_tf32=True`，输入、权重和输出为 float32，
autocast 关闭。它复现了旧整例的标签数据（不同体素 0，affine 差异 0），
但该比较记录的内存 dtype 不同，不能写成严格逐文件通过。这份记录证明
旧的有效 CUDA 策略；历史 `fp32_exceptions` 声明不等于真实 FP32 执行证据。

## 输入、参数与输出

`SynthSeg(weights=None, device="cpu", threads=None, cudnn_tf32=True)` 接收
权重文件或目录、设备名和 PyTorch 线程数。`threads=None` 保留线程设置，负数
使用 CPU 核数。权重、标签、名称和拓扑数组的查找方式不变。

`cudnn_tf32` 是 CUDA 卷积的具名策略：

| 值 | 实际前向行为 |
|---|---|
| `True`（默认） | 开启 cuDNN TF32，保持原 CUDA 模型默认 |
| `False` | 关闭 cuDNN TF32；float32 输入和权重、且未开启 autocast 时为 FP32 卷积 |
| `None` | 继承调用方进入前向时的 cuDNN 设置 |

CUDA matmul 在模型作用域内仍保持 TF32 默认。两个 TF32 开关在返回或异常时
恢复为调用方原值。CPU 推理不更改这些开关，其记录中的 cuDNN/matmul 值
只是全局状态，不表示 CPU 使用 TF32。

这些选项不会主动开启 FP16/BF16，也不会取消调用方主动开启的 autocast。
因此 `cudnn_tf32=False` 只规定 cuDNN 的 TF32 策略，不能单独证明前向全部
使用 float32。实际精度必须查看下面的记录。后端开关是进程全局状态，具有
不同精度要求的推理不应在同一进程中重叠执行。

`SynthSeg.__call__(image, keep_geometry=False, color_lut=None)` 的输入为单幅
3-D T1 路径或 nibabel 影像。默认输出约 1-mm RAS 对齐网格；
`keep_geometry=True` 用最近邻恢复输入网格。`color_lut` 可指定存在的颜色表。
返回 `SynthSegResult`：

| 字段 | 结构和单位 |
|---|---|
| `segmentation` | int32 NIfTI 分割；affine 将输出体素映射到世界坐标/mm |
| `volumes_mm3` | 标签整数 → 软体积/mm³ |
| `total_intracranial_mm3` | 总颅内容积/mm³ |
| `label_names` | 标签整数 → 名称字符串 |
| `near_tie_voxels` | 近并列概率导致的标签调整体素数 |
| `precision` | 请求的 cuDNN 策略及实际每次前向记录，见下表 |

`SynthSegSegmenter(weights, labels, device="cpu", cudnn_tf32=True)` 也支持相同
策略。它接收官方 HDF5 权重及 55 项标签数组；
`posterior(image, flip=True, smooth=True)` 接收预处理 3-D 张量，返回留在模型
设备上的 `(33,D,H,W)` 无量纲概率。左右翻转集成和平滑默认开启；关闭平滑
时必须同时关闭翻转，否则抛出 `ValueError`。

`precision["forwards"]` 按实际执行顺序记录 original 和 flipped：

| 字段 | 含义 |
|---|---|
| `pass`、`device` | 前向名称和输入张量实际设备 |
| `matmul_tf32`、`cudnn_tf32` | 进入模型前向时的实际开关 |
| `input_dtype`、`model_dtypes`、`output_dtype` | 输入、模型参数集合及输出实际 dtype |
| `autocast.cpu/cuda.enabled` | 调用方是否启用相应设备的 autocast |
| `autocast.cpu/cuda.dtype` | autocast 目标 dtype；关闭时仍会记录配置值 |

`precision["posterior_buffer_reused"]` 记录翻转集成是否完成内部缓冲复用；
完成时 `posterior_ensemble_dtype` 记录集成张量的 dtype。原式
`0.5 * (original + flipped)` 会分别产生相加和缩放的大后验临时张量；
现在复用仅供本次调用使用的 `flipped`，执行
`flipped.add_(original).mul_(0.5)` 并返回第零批次视图。
保持先加后乘的运算顺序，未更改模型、概率平滑、翻转、标签规则或精度策略。
33 通道 FP32 后验每份为 `33×D×H×W×4` 字节；减少这两次临时分配
不等于已证明整阶段显存或耗时下降，峰值还受到卷积和后处理影响。
无翻转、无平滑或前向失败时复用字段为 false。返回概率沿用输入网格，
留在模型设备，不与原始输入张量共享存储。

模型失败时抛出原异常，并保留 `segmenter.precision` 中已经进入的前向记录；
失败前向没有 `output_dtype`。无效设备、缺少文件、标签/权重形状不符、无效
精度参数或不存在的颜色表继续抛出异常，不返回成功结果。后处理、标签语义、
坐标约定和软体积算法保持原流程。

## recon-all 示例

```python
from fnit import SynthSeg

synthseg_model = SynthSeg(
    weights="/path/to/verified/weights",  # 已按固定清单校验的权重目录
    device="cuda:0",  # 当前进程中明确指定的目标 GPU
    threads=4,  # PyTorch 线程预算，不等同整个进程只产生四条线程
    cudnn_tf32=False,  # recon-all 已声明的 cuDNN FP32 卷积例外
)
synthseg_result = synthseg_model(
    image="/path/to/fnit-subject/mri/orig.mgz",  # FNIT 自产 conform T1
    keep_geometry=True,  # 将分割恢复到该 conform 输入网格
    color_lut="/path/to/verified/assets/FreeSurferColorLUT.txt",  # 已校验的颜色表
)
synthseg_result.segmentation.save(
    path="/path/to/fnit-subject/mri/aseg.synthseg.mgz",  # 实际分割输出路径
)
print(synthseg_result.precision)  # 保存实际前向开关、dtype 和 autocast 状态
```

对应独立参考命令为 `mri_synthseg --i T1 --o aseg --threads 4`，CPU 参考额外
指定 `--cpu`。cuDNN 策略是 FNIT 的 PyTorch 参数，官方命令没有同名开关。
官方软件只用于独立 benchmark 路径。

## 本次验证范围

2026-10-01 在 headcw 的既有 Conda 环境执行八项 CPU/fake-model 回归，
`CUDA_VISIBLE_DEVICES` 为空，未初始化或占用 GPU。测试总计 0.060 秒，全部
通过，覆盖两个构造函数不重置精度、三种策略在两次前向中生效、异常恢复、
默认设备选择兼容、无效参数及调用方主动 CPU autocast 的真实 dtype 记录。
新增的非对称 FP32 输入、非平凡通道重排测试中，两次模型前向的缓冲复用
结果与旧表达式逐值相同，返回存储地址就是内部相加目标；第二轮失败也
正确恢复设置且不将复用标记为完成。实测 `segment.py` SHA-256 为
`5dce81225289fdfb62bfa5a7787e26737caa41ff74c33e0885304b239d732f33`。
测试中的 BF16 由测试明确开启，用来检验记录，并非生产默认。

```bash
# 使用主页 Conda 环境；该回归不需要 GPU、权重下载或 pytest。
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover \
  --start-directory tests/synthseg_parc \
  --pattern test_precision_policy.py \
  --verbose
```

### 冻结真实 T1：精度修复与缓冲复用分别比较

以下均为 sub-01 的相同 FNIT `orig.mgz`，在 gpucw1 的同一 GPU
`GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`、PyTorch 2.5.1、四线程运行。
阶段墙钟包含模型加载、传输、前向、后处理、分割和体积 CSV 写出，并在
目标 GPU 上同步；外层命令墙钟另包含进程启动和诊断报告等开销。

| 真实单阶段 | 代码版本及调用方式 | 阶段墙钟（s） | 外层命令墙钟（s） | 父子同时占用的采样峰值（bytes） |
|---|---|---:|---:|---:|
| 旧有效 cuDNN TF32，分配缓存关闭 | 阶段 1 快照，CLI | 32.033 | 46.721 | 19,411,238,912 |
| 修复为 cuDNN FP32，无缓冲复用，缓存关闭 | 阶段 1 快照，已初始化 CUDA API | 70.258 | 85.570 | 12,897,484,800 |
| 修复为 cuDNN FP32，无缓冲复用，缓存开启 | 阶段 1 快照，CLI | 59.367 | 73.869 | 20,352,860,160 |
| cuDNN FP32，有缓冲复用，缓存关闭 | 1b8c36d，已初始化 CUDA API | 80.795 | 96.692 | 14,508,097,536 |
| cuDNN FP32，有缓冲复用，缓存开启 | 1b8c36d，CLI | 64.597 | 71.573 | 18,138,267,648 |
| cuDNN FP32，有缓冲复用，缓存开启 | 1b8c36d，已初始化 CUDA API | 42.089 | 50.164 | 18,138,267,648 |

阶段 1 快照是 `3d9856c9dc659a50b685dbc8c0b9b6c695461fe7` 上的冻结工作区，
归档 SHA-256 为
`0444db72c248fac01bf9385c615da4f0e9fbade94ebbe2ccd2a70942f0c2180f`。
缓冲复用测试绑定完整提交 `1b8c36d25a68e253a1e59b6d02114890afa467de`。
不能把前一快照标记写成干净 Git 提交。

精度修复单独改变了本例 148 个标签体素，相对旧有效 TF32 的最低分区
Dice 为 0.999668325；总颅内软体积从 1,280,259.125 变为
1,280,294.250 mm³，差 35.125 mm³。这是卷积精度策略的真实变化，
不能归为缓冲复用误差或随机性。原始报告为
[修复后缓存关闭的 API](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/corrected_uncached_api/actual-forward.json)。

缓冲复用则以相同 cuDNN FP32 策略的阶段 1 结果为参考。
[缓存关闭 API](../../validation/recon_all/python_gpu_port/performance_20261001/final_1b8c36d/buffer_uncached_api/actual-forward.json)
、[缓存开启 CLI](../../validation/recon_all/python_gpu_port/performance_20261001/final_1b8c36d/buffer_cached_cli/actual-forward.json)
和[缓存开启 API](../../validation/recon_all/python_gpu_port/performance_20261001/final_1b8c36d/buffer_cached_api/actual-forward.json)
均记录 original/flipped 两次真实前向：cuDNN TF32=False、matmul TF32=True，
输入/权重/输出都是 float32，CPU/CUDA autocast 均关闭；
`posterior_buffer_reused=True`，集成张量 float32。三者相对 FP32 参考的
不同标签体素为 0，所有分区 Dice 为 1，affine 差为 0，体积 CSV 完全相同。
只读核对还确认分割文件本身逐字节相同。

这些是单次阶段观察，执行方式、时刻和资源占用不完全相同。
此前两次缓冲复用观察的阶段时间没有比相应无复用运行更短；新增缓存 API
观察为 42.089 s。没有相同启动方式和运行顺序的交替重复配对，不能据此
将耗时变化归于缓存或缓冲复用。缓存开启的本次采样峰值约 18.138 GB，阶段 1 曾为
20.353 GB，超过 20,000,000,000 字节；缓存关闭的本次峰值约 14.508 GB。
不能据此将缓存改成默认开启，也不能以孤立 SynthSeg 证明整例满足预算。

三份最终监测的请求间隔都是 2.0 s、查询超时 5.0 s；缓存关闭 API、
缓存开启 CLI/API 的最大实际间隔分别为 2.751/2.307/2.292 s，进程查询
失败均为 0。它们没有连续峰值证明。
缓存关闭时 PyTorch allocated/reserved 不可用；缓存开启 CLI 实测
allocated=15,621,712,896、reserved=17,574,133,760 字节，和进程占用是
不同统计范围。见
[API 监测](../../validation/recon_all/python_gpu_port/performance_20261001/final_1b8c36d/buffer_uncached_api_monitor/monitor.json)
和[CLI 监测](../../validation/recon_all/python_gpu_port/performance_20261001/final_1b8c36d/buffer_cached_cli_monitor/monitor.json)。

缓存开启 API 先调用 allocator 选择，再保留一个 float32 CUDA 标量并同步，
随后进入 SynthSeg；`allocator.cuda_initialized_at_entry=False` 表示选择策略时
CUDA 尚未初始化，`initialized_api=True` 表示模型调用前已经初始化，二者
没有冲突。选择时移除继承的 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`，实际策略
为 enabled、Torch 统计可用。它与缓存 CLI 的输入、权重、源码、GPU UUID、
四线程、cuDNN benchmark/deterministic 和实际前向设置完全相同；区别是
显式初始化及保留标量，运行时刻也不同。API 实测 allocated 为
15,621,713,408 字节（比 CLI 多 512 字节），reserved 同为
17,574,133,760 字节；不由这个小差值推断性能原因。
见[缓存 API 监测](../../validation/recon_all/python_gpu_port/performance_20261001/final_1b8c36d/buffer_cached_api_monitor/monitor.json)。

### dtype 失败项的只读审计

原 API 报告仍保留 `same_dtype=False`。实际执行的旧探针把落盘 MGH
参考与 `result.segmentation` 的内存 int32 NIfTI 对象比较；本次保存为
MGH 后实际存储 dtype 为 `>f4`（float32），所以两者表示不同。
这不只是整数的字节序区别，不能把原 False 改写成通过。

后续 CLI 探针改为重新加载已写出的候选 MGH，在相同文件表示上比较，
得到 `same_dtype=True`。本次另在 headcw 只读核对三份既有文件，没有重新
运行模型：FP32 参考、buffer API 和 buffer CLI 都是 `>f4`，形状相同，
affine 和标签值差为 0，且三份 MGH 的 SHA-256 完全相同。标签值仍为
离散分割 ID；返回对象的 int32 类型不等同于此次 .mgz 存储类型。
本次未改变该写出转换，也未据此宣布官方逐文件通过。

完整证据在
[落盘 dtype 与缓冲回归审计](../../validation/recon_all/python_gpu_port/performance_20261001/final_1b8c36d/buffer_stored_dtype_audit.json)：

| 实际文件/程序 | SHA-256 |
|---|---|
| 三份相同的 segmentation.mgz | `50f9e58f7bcc64a99594f9f81b81c65870ab90c7d5731371344bba6b8daa7230` |
| 三份相同的 volumes.csv | `aef612478a7e908c4f0aaa29a52db31806520e12a5c8e53637df3816b2f3b3da` |
| 原内存候选探针 | `35458da8d4264ad10ae055ecc18a81cf4feb91c2d2965eeb1b5fa125e1ca7de9` |
| 后续落盘候选探针 | `5168f615ce8591d5473bed05cfd7f1c159bedd095dfba2503fbd690880c93477` |
| buffer API actual-forward.json | `70b0a1e1f727ed580aff38ac359e888c7ea0f08afd25f1ec49637a2df815790b` |
| buffer CLI actual-forward.json | `cb3fe251b82d51b28df8a9fb2f542dbe287930f6b13762c0fca0144f413bf822` |
| buffer cached API actual-forward.json | `db23fe176d8271429a6cc75c8e58b65f9c3e2c185f8a662dddf5588db4e903c1` |
| buffer cached API monitor.json | `5a32505e7e559ac5cc661bee543ac716f0d296072dc38d337166f557d7b8d159` |

新增缓存 API 使用后续落盘候选探针，`same_dtype=True`；只读核对其保存 MGH
也是 `>f4`，与 FP32 参考逐字节相同，SHA-256 沿用表中 `50f9e58f…`；CSV
也沿用 `aef61247…`。原三份文件审计保持不变，新增证据与 CLI 参数逐项核对
见[缓存 API 独立汇总](../../validation/recon_all/python_gpu_port/performance_20261001/buffer_cached_api_summary.json)。

最终模型模块 `segment.py` 的 SHA-256 为前述 `5dce8122...d732f33`；
三份前向报告分别保留其完整值，以及实际 `_dmri.py`、`synthseg.py` 和
权重/标签数组摘要。共同的原始阶段输入 SHA-256 为
`646cfe39c550fd1626c4a9bfbc3d065a9afdd9cddbd849585465a822b425bc09`，
模型 HDF5 为
`f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e`。

本页报告本例冻结阶段验证。两例候选原始 T1 整例已执行完成，受控同精度
基线及官方比较尚未结束，状态见[当前验证目录](../../validation/recon_all/python_gpu_port/performance_20261001/README.md)。
第二例完整 SynthSeg 缓冲配对、连续显存峰值和整体指标等效尚未完成；
本页不从单阶段时间推断整例加速。
计时范围、allocator 和统计可用性见[剖析说明](PROFILING.md)。

源码与参考：[FNIT 实现](../../src/fnit/synthseg_parc/segment.py)、
[focused tests](../../tests/synthseg_parc/test_precision_policy.py)、
[官方 SynthSeg](https://github.com/BBillot/SynthSeg)、
[SynthSeg 论文](https://doi.org/10.1016/j.media.2023.102789)。
