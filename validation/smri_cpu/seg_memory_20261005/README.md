# SynthSeg CPU decoder 拼接候选

## 1. 功能简介

本候选只减少普通 SynthSeg 2.0 分割与皮层分区 U-Net 的 CPU 临时缓冲。
最后四级 decoder 的原操作先将特征最近邻放大两倍，再与 encoder skip 拼接；
候选直接构造同样连续的 NCDHW 拼接张量。先写 skip 通道，再通过整数索引
复制低分辨率特征。卷积、ELU、BN、后验、标签、软体积及精度策略不变。

适用条件是 CPU、FP32、eval、无梯度、无调用者 autocast、全网络无观察 hooks、
连续 NCDHW 输入及精确两倍空间尺寸。其他调用保留原路径，包括 CUDA、训练、
有梯度、channels-last、容器/叶子/global hooks。`cpu_conv.py` 的原后端分块与
大 1×1×1 oneDNN 投影防崩分支保留。没有新增依赖。

```mermaid
flowchart LR
    A[encoder skip 与 decoder 特征] --> B{CPU 推理资格}
    B -->|符合| C[一次连续目标分配和两次 copy]
    B -->|其他| D[原 nearest 和 cat]
    C --> E[原卷积 ELU BN 与后续输出]
    D --> E
```

本次完整原始T1的旧新输出门和同节点官方参考均已完成。
协调者已将候选提交 `585bf181` 合入为 `7e0890a5`，整合回归2288 passed、
11 skipped、3 subtests、0 failed；三个候选源码SHA在回归前后不变。
本仓库 `SynthSegPlus` 对应普通 `--parc`，仍未实现 robust SynthSeg+。

## 2. Python 调用与输入输出

公开 API 与参数没有变化，见 [SynthSeg](../../../docs/synthseg/README.md)
和 [皮层分区](../../../docs/synthseg_plus/README.md)。例如：

```python
from fnit import SynthSeg

segmentation_model = SynthSeg(
    weights="/path/to/verified_weights",  # 已校验的 H5 与配套标签目录
    device="cpu",  # 本候选的适用设备
    threads=8,  # 相同官方对照的计算线程预算
)
segmentation_result = segmentation_model(
    image="case02_T1w.nii.gz",  # 单幅真实3D T1，不读取参考标签
    keep_geometry=False,  # 返回约1 mm的RAS网格
    color_lut=None,  # 沿用原保存接口
)
segmentation_result.segmentation.save("case02_synthseg.nii.gz")
segmentation_result.write_volumes_csv(
    source="case02_T1w.nii.gz",  # CSV的输入标识
    path="case02_synthseg.vol.csv",  # 32前景结构与颅内软体积
)
```

内部 `join_nearest_cpu(skip, value)` 输入均为 CPU FP32 五维张量：
`skip=(B,C_skip,2D,2H,2W)`、`value=(B,C_value,D,H,W)`。调用方先完成上述资格
检查。输出为新的连续 `(B,C_skip+C_value,2D,2H,2W)` 张量；前部通道来自
skip，后部是 factor-two nearest，输出与两输入不共享存储。内部函数不修改
线程、oneDNN、TF32或autocast状态，不提供公共参数。

## 3. 命令行与阶段复现

公开命令仍为：

```bash
fnit synthseg --i case02_T1w.nii.gz --o case02_synthseg.nii.gz \
  --weights /path/to/verified_weights --csv-vols case02_synthseg.vol.csv \
  --device cpu --threads 8
# 皮层普通模式加 --parc；皮层快速模式再加 --fast。
```

有限阶段脚本 `stage.py` 的参数如下。数组与 checkpoint 保留在私密运行目录，
不随本报告分发。

| 参数 | 含义 |
|---|---|
| `--mode capture` | 用已验收 CPU CNN 产生最后级 skip/nearest source，在尚未消费的最后卷积之前声明停止；不输出完整预测 |
| `--mode baseline/candidate` | 对同一 checkpoint 比较原 nearest+cat 与候选 broadcast copy |
| `--source` | 已冻结、六个实际源码 SHA 符合门槛的 FNIT src 目录 |
| `--checkpoint` | 新的 capture 目录；后续阶段只读取并校验文件 SHA |
| `--report` | 本阶段独立 JSON 路径 |
| `--array` | 已绑定 case02 的正确192×224×256预处理 FP32数组；仅 capture 读取 |
| `--input` | 同一公开原始T1；capture核对其SHA |
| `--weights` | 已核验的模型资源目录；仅capture使用 |
| `--helper` | 冻结候选 `cpu_join.py`；candidate记录其SHA |
| `--repeats` | 进程内操作次数，默认4；各次保留耗时和输出SHA |

`run_stages.py` 配置上述 Python/worker/source/helper/input/array/weights，
另指定 `--run` 新运行目录、`--lock` 同节点共享锁和 `--affinity` 同一八物理核。
按 capture→A1→B1→B2→A2 串行；每阶段 timeout 600秒、10秒后强制结束。
冷进程/GNU time/RSS与内部copy秒数分别记录，checkpoint读写/hash不计为copy。

完整回归脚本不是新的公共推理入口；它们固定本次公开case02输入SHA及旧新源码
SHA，避免无意测试另一冻结版本。`full.py`用于CPU，`full_gpu.py`另外记录GPU
完整API的全局精度状态、实际前向dtype和分配器峰值。参数如下：

| 脚本/参数 | 含义 |
|---|---|
| worker `--source` | 实际FNIT `src`目录；`--arm baseline/candidate`决定必须匹配哪组源码SHA |
| worker `--mode seg33/parc/parc-fast` | 33类分割、普通皮层分区、快速皮层分区；均从原始T1开始 |
| worker `--device cpu/cuda:0` | 只运行所选设备；CPU固定8线程，GPU读取所选可见设备 |
| worker `--input`、`--weights` | case02原始NIfTI与五项已核验模型资源；推理不读官方输出 |
| worker `--output` | 必须不存在的新目录；保存API图、体积CSV及私密完整worker记录 |
| worker `--cudnn-tf32 default/false` | 默认沿用公开API；False实际仅测试33类recon-all策略，两政策分别旧新配对 |
| worker `--memory-budget-bytes` | GPU分配器上限，默认20,000,000,000字节；不是进程树物理显存测量 |
| `run_full.py --python/--worker` | 现有Conda Python与冻结worker；每arm独立冷进程，timeout 600秒 |
| controller `--baseline/--candidate` | 两个源码冻结目录，不覆盖正在运行的目录 |
| controller `--run/--lock/--affinity` | 新运行目录、同节点共用锁、同一八物理核；每arm单独取锁 |
| controller `--input/--weights/--device/--cudnn-tf32` | 转交worker的同名参数；`--visible-gpu`默认1，仅用于CUDA可见设备映射 |
| controller `--modes`、`--candidate-first` | 默认三功能AB；只对fast补测设定单功能BA，保留首组与补组记录 |
| `collect_full.py --run/--comparator/--official-manifest/--output` | 对已保存结果进行数值和几何比较；`--require-complete`要求所有计划arm完成，不做推理 |
| `collect_final.py --root/--output` | 读取固定任务索引下的新官方三模式与fast BA；重新核验原软件/资源源码，不做拟合 |
| `collect_native_repeat.py --root/--input/--output` | 向不可变首份公开报告追加声明33复测；两个输出文件均保留，不覆盖首轮 |

例如为两个已冻结目录运行CPU三功能对照：

```bash
python validation/smri_cpu/seg_memory_20261005/run_full.py \
  --python /path/to/conda/environment/bin/python \
  --worker validation/smri_cpu/seg_memory_20261005/full.py \
  --baseline /path/to/qualified_old/src --candidate /path/to/frozen_new/src \
  --input /path/to/ds003138/case02_T1w.nii.gz \
  --weights /path/to/verified_weights --device cpu \
  --affinity 32,36,40,44,48,52,56,60 \
  --run /path/to/new_complete_run --lock /path/to/shared_cpu8.lock
```

实际首组controller SHA为`4dbea740…`；随后新增BA排序参数的controller SHA
为`fe86b89b…`。各完整JSON绑定实际执行版本，未将后改脚本SHA回填到首组。

## 4. 原软件调用

```bash
mri_synthseg --i case02_T1w.nii.gz --o official_synthseg.nii.gz \
  --cpu --threads 8 --vol official_synthseg.vol.csv --noaddctab
# --parc 与 --fast 使用相应普通2.0模式。
```

nearest+cat 是 CNN 内部操作，没有独立官方命令。原软件只用于隔离参考；
候选推理不调用 FreeSurfer 或 TensorFlow。

## 5. 本版精度、时间与验证范围

基线提交 `6763958572d577e33258f5a2e83bb796e451cae3`。CPU卷积分块文件 SHA
`78a0fac2300a45a01ff6790d24218f825ebc21a21f2033a8e6b8c58ebb60dc47` 不变，
实际阶段 producer 六个 SynthSeg 文件与已验收 `candidate_large_pointwise_v3`
逐个核对。预先声明门见 [acceptance.json](acceptance.json)。

本地关键合同：**31 passed**，包括两网络32³完整输出逐值对照、四级实际分派、
单体素轴/多batch/通道顺序、signed zero/NaN bits、stride、独立storage、
autocast、训练/梯度及各类hooks回退。这些不是真实速度 benchmark。

完整 `tests/synthseg_parc` 回归为 **70 passed、6 skipped、3 subtests**。
其中四个WMH mock缺少成熟pipeline已要求的状态属性，使用协调者原源码也会失败；
本候选未修改WMH生产代码。test-only 提交 `be1cc745` 给TinyModel补入该属性后通过，
不将原有失败归因于本候选。

公开CC0 OpenNeuro ds003138 case02的正确模型网格为192×224×256。
有限阶段先由合格原CNN捕获最后一级decoder输入，在尚未消费的最后卷积之前
声明停止；这是partial capture，没有完整预测。v3复用同一不可变checkpoint，
候选资格检查与实际全网络no-hooks遍历包含在每次计时内。

| 有限阶段v3（四个ABBA冷进程、每个4次） | 原 nearest+cat | 候选 |
|---|---:|---:|
| 16次操作的中位耗时 | 0.9723 s | 0.5294 s |
| 两个进程的最大RSS | 6,773,888 / 6,773,828 KiB | 4,804,300 / 4,804,240 KiB |
| 两个进程墙钟（包含读取/hash） | 35.916 / 33.612 s | 35.662 / 31.762 s |

每次输出含792,723,456个FP32值，全部bits、shape、stride、skip-first通道顺序
与独立storage通过；输入值未改。局部中位提速 **1.84×**；copy/hash/io的冷进程墙钟
单列。全部门及SHA见 [CPU_JOIN_STAGE.public.json](CPU_JOIN_STAGE.public.json)。
旧小尺寸profile不用于本版占比。完整原始T1 CPU与实际H100 GPU分别验收，
此阶段通过不代表完整网络提速。完整CPU/GPU结果如下。

### 完整原始T1的CPU对照

同一nodecw7、相同八物理核与8线程，冷进程按33类、普通parc、fast逐对运行；
每个arm单独取得共享锁。完整API包含原始T1预处理、全部模型、后处理、体积
统计；文件保存、导入/资源SHA核验及进程墙钟单列。

| CPU功能 | 原/新API（s） | 原/新冷进程（s） | 原/新最大RSS（KiB） |
|---|---:|---:|---:|
| 33类，单次AB | 112.681 / 109.777 | 115.886 / 112.952 | 14,981,148 / 13,191,188 |
| 普通parc，单次AB | 58.704 / 52.843 | 61.648 / 55.684 | 14,199,436 / 14,193,744 |
| fast，首组AB | 40.082 / 40.411 | 42.718 / 43.369 | 14,199,116 / 14,193,788 |
| fast，补组BA | 37.691 / 37.549 | 40.363 / 40.464 | 14,198,848 / 14,193,720 |

fast两组API差的符号相反；合并中位数38.887 / 38.980秒，约0.24%差异。
本例33类和普通parc观察到减少；fast完整耗时近似不变，不称其整体加速。
33类进程RSS约15.34→13.51GB；parc/fast的其他缓冲占据峰值，整体RSS未有
类似下降。局部节省不能逐级相加推断全流程收益。

全部七张图（33类一张、两种parc各三张）的硬数组、header binaryblock、
affine、pixdim、dtype及extensions旧新精确相同。33/101/101个numeric CSV
均完全相同，实际CPU join调用为8/12/8。fast补组四个进程的三图与CSV文件SHA
也相同。见 [CPU_FULL.public.json](CPU_FULL.public.json) 与
[官方和fast补组记录](OFFICIAL_NODE7_FAST_BA_REPEAT.public.json)。

同nodecw7新官方输出与原nodecw10参考的三模式硬标签均0个不同体素。
新官方计时只与同nodecw7的FNIT记录并列，旧nodecw10时间不参与速度比较。
本次新官方参考的剩余差异如下；它们均为旧版已有误差，没有新增或移除：

| CPU功能 | 官方不同体素 | 最小前景Dice | CSV最大绝对差（mm³） |
|---|---:|---:|---:|
| 33类 | 1 | 0.99999857 | 0.800 |
| 普通parc | 5 | 0.99982140 | 0.657 |
| fast | 1 | 0.99997630 | 0.220 |

每个结构的Dice、硬体积、CSV软体积与新增/移除/保留误差都保存在完整JSON；
不把平均Dice或旧新精确相同称为完全官方一致。33类官方重复与首轮硬数组
也0个不同体素，重复相对候选仍保留同一个差异体素。

### 同节点官方CPU冷进程计时

官方FreeSurfer 8.2隔离参考与FNIT均在nodecw7固定八物理核、相同8线程预算，
隐藏CUDA，每个进程独立取得同一共享锁。官方launcher、实际embedded Python
脚本、五项模型资源、环境脚本、原始T1与任务调度器均绑定实际SHA。资源监控
约0.25秒一次；官方采样峰值39个进程树线程包含空闲线程池，实际affinity始终
限定上述八核，不能将39解释为39个物理计算核。

| 功能 | 官方冷进程（s） | FNIT候选冷进程（s） | FNIT候选完整API（s） | 官方采样进程树RSS峰值（GB） |
|---|---:|---:|---:|---:|
| 33类，首次 | 373.087 | 112.952 | 109.777 | 15.776 |
| 33类，声明复测 | 55.046 | 同上 | 同上 | 17.550 |
| 普通parc | 48.296 | 55.684 | 52.843 | 26.304 |
| fast | 33.784 | 43.369（AB）/40.464（BA） | 40.411/37.549 | 19.663 |

首次33类官方冷进程明显异常，记录显示CPU占用30%、260次major fault、
较大的文件读取和调度等待，但没有确定其详细原因。仅增加一次同source/输入/
线程/核绑定的声明复测，保留首轮记录；不使用373秒作为加速分母，不把两次
中位数当作典型官方速度。官方三模式各只有一次常规冷进程观察，不能称稳定
benchmark。

官方墙钟覆盖模块/CLI导入、全部推理和主图/CSV保存；FNIT冷worker还包含
资源SHA核验，并保存API返回的全部图（parc三张）和CSV。API与冷进程分别
列出，不能把不同计时边界直接变成精确加速比。本候选仍未达到全部CPU功能
等于或快于官方的目标。长时卷积需另以真实layer oracle和实际backend分析；
此次不会重复已拒oneDNN slab或放宽硬标签门。完整来源及各结构数值见
[同节点官方与fast ABBA报告](OFFICIAL_NODE7_FAST_BA_REPEAT.public.json)。

本例33类的真实保存标签如下。上排为同节点声明复测官方结果，中排为本次
CPU候选；下排圈出仍保留的一个差异体素所在切片。上两排采用各轴前景中位
切片，下排使用差异所在切片，因此不将三排当作同一切片。图只读取CC0标签，
没有重采样或输入MRI强度；来源和文件SHA见
[图像记录](case02_cpu_labels.public.json)。渲染使用原有隔离参考Python，
没有给FNIT运行环境新增依赖或重新运行模型。

![当前case02官方与CPU候选33类标签](case02_cpu_labels.png)

### 实际H100旧新对照

GPU1的八个完整原始T1进程覆盖公开默认33类/普通parc/fast，以及recon-all
实际使用的33类 `cudnn_tf32=False`。两种政策分别配对，所有输出图、header与
numeric CSV旧新精确相同。CUDA仍会调用廉价的资格检查并立即返回False；
实际CPU copy调用为0，原CUDA nearest/cat执行体保留。

| 政策/功能 | 原/新API（s） | 原/新Torch allocated峰值（GB） | 原/新reserved峰值（GB） |
|---|---:|---:|---:|
| 默认33类 | 5.287 / 4.951 | 10.712 / 10.712 | 14.615 / 14.615 |
| 默认普通parc | 9.310 / 9.199 | 12.568 / 12.568 | 18.207 / 18.207 |
| 默认fast | 6.764 / 6.512 | 12.568 / 12.568 | 18.900 / 18.900 |
| 33类显式False | 5.586 / 5.589 | 8.598 / 8.598 | 9.326 / 9.326 |

每次实际前向均为FP32，无CPU/CUDA autocast；前两次33前向的cuDNN TF32
按相应True/False配置记录。分配器上限为20,000,000,000字节，以上allocated与
reserved均低于该上限。采样器约0.5秒一次，实际最大间隔0.648秒，失败0次。

这是共享H100时间观察：正式arm开始前已有45,512MiB外部占用，GPU利用率
较高；前后占用相同，整体GPU peak含其他作业。不能把整体`nvidia-smi`数值
当作本FNIT进程树峰值，也不能用这些shared wall/API差声称GPU提速。证据只
支持此次原始T1旧新输出及Torch allocated/reserved一致、CUDA张量数学未改。
见 [默认GPU完整记录](GPU_DEFAULT_FULL.public.json) 和
[False策略记录](GPU_FP32_FULL.public.json)。

默认GPU相对官方还保留158 / 328 / 498个不同体素，CSV最大绝对差约118.28 /
118.28 / 121.60mm³；本候选没有改善或增加这些默认TF32差异。显式False的33
类单例相对官方0个不同体素、全部区域Dice=1，但CSV仍有最大0.20mm³尾差；
该单例不能推广为所有输入和参数都与官方逐值相同。

### 当前成熟子函数的精度上下文问题

此次实际记录确认：普通parc/fast的完整API进入时matmul TF32=False，退出时
True；旧新两arm相同。原因是现有 `SynthSegParc.__init__` 调用
`configure_device(device)`，在 `SynthSegPlus` 首次调用的lazy模型构造时
沿用其全局TF32设置。此构造副作用会覆盖调用者的显式精度上下文。
33类独立接口与显式False前向的作用域恢复门均通过。

本lossless候选未修改这一已有政策，不能称所有完整API的flags均恢复。
后续须独立修复构造不改global、默认前向仍保留TF32、允许False/None声明，
并验证正常/异常退出的恢复、所有实际分割/分区网络及现有GPU回归；新数学
结果与此次只改变CPUcopy的报告分开。

原33类 CPU已完成的严格标签与CSV门和历史脑图见
[已有实测](../../smri_cpu_20261004/t2_seg/README.md)；这些是历史基线，
不能改标为本候选结果。

## 6. 本次记录

- v1：CPU限定一次目标分配、split spatial axes broadcast copy；卷积和精度未改。
- harness首次将controller命名 `queue.py`，遮蔽标准库queue，Torch导入前失败。
  没有产生CNN checkpoint；失败receipt/log保留。更名 `run_stages.py` 后在
  全新目录派发，不覆盖失败记录或在跑源码。
- 完整原始T1：CPU六arm、GPU默认六arm和显式False两arm全部执行并通过旧新保存输出门；
  fast另补BA两arm，共16个FNIT完整进程。隔离的同节点官方三模式完整完成，
  异常33首轮另补一次声明复测，共4个原软件进程；没有再拟合parc/fast。
- v2：同一真实checkpoint的helper-only计时及逐值结果通过，但其资格检查未在timer内，
  未作正式性能验收；v3在同一checkpoint上加入实际网络资格检查，16次再次通过。
- 全oneDNN slab和中层oneDNN混合的case02新标签误差保持在原报告；不复跑
  已拒路线，不放宽tie或硬标签门。

## 7. 参考文献与源码

原实现：[SynthSeg](https://github.com/BBillot/SynthSeg)；Billot et al.,
Medical Image Analysis 2023；普通皮层入口及原权重定义见
[FreeSurfer mri_synthseg](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthseg)。
nearest与cat的候选是仓库自有张量拷贝实现；现有 WMH生命周期helper提供
资格与hook保护设计参考，没有复制无关原软件代码。
