# SynthSeg CPU 剩余耗时审计（2026-10-06）

本页只读分析已保存的真实数据计时与当前源码；没有再次执行 CNN、原软件、拟合或阶段 benchmark，也没有修改生产代码。机器可读来源、SHA-256、计时与缺测项见 [AUDIT.public.json](AUDIT.public.json)。当前审计源码是已验收 TF32 修复 `46eead65` 的实际文件；报告工作树为 `a3936ea4`，协调者已接入的工作树读到 `a5a21bdc`。按文件 SHA 绑定，不把服务器已发布的前一版 `6f624040` 当成这次源码。

## 1. 目前能确定的时间范围

同一公开 CC0 T1（`case02`，输入 SHA `73e3866d…`），预处理后的网络网格为 **192×224×256**；正式 CPU 对照均限制 8 线程、同 8 个物理核。原软件计时是独立 CLI 冷进程；FNIT 计时含验证工具的预校验、导入及保存。两条记录保存的输出数量也不同：原软件为合并图和 CSV，FNIT `--parc` 为分割、parcel、合并图和 CSV。因此下面并列实际墙钟，不给精确加速倍率。

| 模式与记录 | FNIT 冷进程墙钟 / API（s） | 官方冷进程墙钟（s） | 本次解读 |
| --- | ---: | ---: | --- |
| 33 类，已验收 CPU join 版本 | 112.952 / 109.777 | 55.046 | 官方复测正常值；首次 373.087 异常另保留，不作速度分母 |
| 普通 `--parc`，CPU join 版本 | 55.684 / 52.843 | 48.296 | 速度目标未通过 |
| `--parc --fast`，CPU join 版本 | 43.369 / 40.411 | 33.784 | 后续 BA 与原 AB 合并未显示实质旧新退化；仍未通过官方速度目标 |
| 普通 `--parc`，最新 TF32 作用域版本 | 53.577 / 51.018 | 48.296（复用前批） | 默认 CPU 输出旧新严格一致；不把跨批噪声算作优化收益 |
| `--parc --fast`，最新 TF32 作用域版本 | 38.160 / 35.347 | 33.784（复用前批） | 同上；后批共享负载不同，不能替代配对正式速度门 |

33 类源与 CPU 数学未被后续 TF32 修复改变，故复用现有完整33类结果；没有为了报告再次跑它。正式正常官方33类复测与首条官方33类输出、CSV一致。

### 加载、I/O 与网络分别知道多少

| 33 类候选当前完整记录 | 秒 | 占112.952s墙钟 | 范围 |
| --- | ---: | ---: | --- |
| preflight | 2.463 | 2.18% | 哈希、导入、运行环境准备；不是影像预处理 |
| 模型构造 | 0.189 | 0.17% | 普通33类 eager 权重/模型加载 |
| 完整 API | 109.777 | 97.19% | T1预处理、两次CNN、两次blur、集成、后处理及体积；内部份额缺测 |
| 保存 | 0.153 | 0.14% | 单分割图和CSV保存 |
| 其余工具/进程边界 | 0.371 | 0.33% | 外层时钟剩余；不是一个已定位的科学步骤 |

外层非 API 一共 **3.175 秒**。即使完全消除这些开销，仍解释不了与正常官方55.046秒相比的约58秒墙钟差距。完整 API 的97.19%不能写成“CNN占97.19%”，因为它包含预处理与后处理。

Plus 构造约0.001秒是**延迟加载**：权重及子模型构造发生在首次 API 内，不能因此说其模型加载比普通33类快约200倍。最新Plus/fast保存分别0.439/0.454秒；这部分不是主要差距。最新API内部的T1读取/百分位归一化/1mm重采样、模型加载、CNN、blur、后处理和体积占比分别还没有可靠计时。

## 2. 最有依据的热点是普通33类的卷积后端

源码路径与数学区别已经核对：

- [`synthseg.py:149–154`](../../../src/fnit/synthseg_parc/synthseg.py)：普通33类CPU在整个 `posterior` 作用域禁用oneDNN，保留之前已验证的CPU崩溃规避。
- [`cpu_conv.py:105–117`](../../../src/fnit/synthseg_parc/cpu_conv.py)：该作用域的CPU FP32、eval、无梯度、无autocast卷积调用保序 `convolution_slabs`；Plus沿用oneDNN，只对超过安全大小的1×1末层投影分块。
- [`cpu_conv.py:74–91`](../../../src/fnit/synthseg_parc/cpu_conv.py)：slab最大深度32、带完整halo，首末深度边界才补零，原始输入切片后调用 `F.conv3d` 并复制到输出；当前没有实际ATen算子trace，不能把每层底层算法或im2col份额写成已测。
- [`segment.py:92–101,199–217`](../../../src/fnit/synthseg_parc/segment.py)：普通33类的两个33通道3×3×3高斯blur也处在禁用oneDNN的作用域，调用group33 slabs。Plus同一blur在oneDNN开启时调用普通分组卷积。69通道parcel blur在 [`pipeline.py:72–85`](../../../src/fnit/synthseg_parc/pipeline.py)。**这三个blur的实际总时间缺测**。
- [`model.py:28–31`](../../../src/fnit/synthseg_parc/model.py)：`conv→ELU→conv→ELU→BN` 顺序与跳连位置不能为提速改动。

普通33类只有两次Seg网络；非fast Plus有相同两次Seg网络，再加一次parcel网络，却在同CPU完整记录中更快。这个事实与源码的后端差异一致，支持优先查卷积与blur；它不能单独证明差距有多少秒来自CNN或blur。官方TensorFlow实际卷积后端、编译选项与逐算子时钟未附在现有正式报告中，不能宣称已证明“官方就是使用oneDNN”。

### 已测的保序/误差边界

真实 `case02` 前两层诊断来自 [reduction_diagnostic.public.json](../../smri_cpu_20261004/t2_seg/reduction_diagnostic.public.json)。原后端第二层slab6.065秒与旧后端逐值一致；oneDNN slab1.655秒，但229,059,640 /264,241,152值改变，最大绝对差1.431e-5。完整oneDNN与混合中层路线各产生2个新增错误标签（候选相对官方3个不同体素，原CPU1个）。**这两条已拒路线不再重跑**，也不移除现有低内存或末层防崩分支。

## 3. 已保存profile能支持什么，不能支持什么

旧真实profile是 `case01`、网格**192×224×224**、旧源码、另一CPU节点。普通33类API147.199秒，其中Module卷积hook累计96.123秒；最后decoder的conv0 47.214秒、conv1 12.325秒、首个encoder conv1 11.538秒。旧fast parcel API59.238秒，Module卷积累计23.994秒。它们仅支持“高分辨率decoder/encoder卷积值得优先查”的次序；不能把这些秒数或比例套到当前 `case02` 的109.777秒。

旧普通profile中的LCC11.863秒包含在postprocess15.087秒中；当前CPU已使用与官方相同的SciPy六邻域连通分量（[`postprocess.py:17–30`](../../../src/fnit/synthseg_parc/postprocess.py)），不能以旧图传播LCC计时重复报当前瓶颈。旧profile还预先实例化Plus延迟子模型，并用Module hooks：当前 `cpu_join_allowed` 明确不允许hooks，该诊断方式会改变当前的CPU路径；Gaussian `F.conv3d` 也不被其Module卷积计数包含。

真实最后一级nearest+cat阶段旧0.972秒、新0.529秒、局部1.836倍，保存输入/输出逐值及stride一致。它是**一个join操作**；30.397秒capture停在末级conv前，没有预测输出。不能将1.836写成整网络提速，不能按8次join简单相乘。完整33类旧新115.886→112.952秒只是相应完整实测观察。

## 4. 接下来的最小profiling方案（尚未执行）

先只做**一例当前33类完整CPU诊断**，复用相同公开T1、五项权重及当前14文件SHA；同8核/线程预算、共用CPU锁、一个新进程。无需重跑官方。

1. 先静态短合同证明观察器不设置Module hooks、不改 `cpu_join_allowed`，只在Python函数/forward调用边界加 `perf_counter`；不改输入、内存格式、dtype、autocast、slab策略、线程或数学。记录是否仍实际调用8次CPU join。
2. 分别计时预校验/导入、构造、T1读取/重采样/方向/归一化、original CNN、original blur、flipped CNN、flipped blur、翻转集成、后处理、tie argmax、软体积、保存。LCC是后处理子项，不能重复累加；卷积层是CNN子项，BN/ELU/pool/softmax/join另记exclusive残余。
3. 每个卷积只记实际shape、后端flag、slab数/深度；函数内部可以累计原有补零、卷积、结果copy边界时钟。先确认观察器没有改变分配或分块；若需要ATen证据，再用现存真实层检查点做有限operator profiler，不在整个大网格开启重型shape/memory trace。
4. 输出与已保存候选逐体素/每标签、dtype/header、CSV数值及文件SHA一致，源码前后不变才接受分步诊断。记录计时观测开销；带观察器总时间不是新的正式性能门。保留正式112.952/55.046秒，并只用诊断分辨份额。
5. 若这一个profile无法定位Plus共有剩余，再各补一个parc/fast profile；不预先启动三模态新队列。它们必须把延迟权重加载计入API，另记一次parcel网络、group69 blur、parc归一化/soft-volume输出。

## 5. 保序优化的优先顺序

| 优先项 | 具体位置与可能做法 | 必须先证明 | 当前结论 |
| --- | --- | --- | --- |
| 高分辨率卷积的数据移动 | `cpu_conv.py:78–91`；在**原slab几何和后端不变**的前提下评估临时缓冲复用/首尾补零与copy开销 | 当前真实profile先证明copy/pad占比；原输入、halo、通道顺序、数值逐值一致，CUDA路径不变 | 未实现；可能只带来有限收益，不能承诺填平全部差距 |
| 禁用oneDNN时的布局探针 | 仅现存真实第二层及末decoder输入，`channels_last_3d`输入/权重与mkldnn=False有限比较 | 布局可改变ATen路径/归约顺序，必须先实测oracle逐值与官方最终门；不能称默认lossless，也不能照搬SynthSR结论 | 后续备选研究；与已拒oneDNN slab路线分开 |
| 概率张量扫描/临时分配 | `postprocess.py:83–102`、`synthseg.py:29–31,55–65`、`pipeline.py:99–106`；只有确认瓶颈后才评估buffer生命周期 | 保留归一化、near-tie与拓扑顺序、NumPy官方软体积求和及方向；禁止直接换Torch不同归约 | 未测占比；不得按旧LCC11.863秒承诺收益 |
| blur常量/临时缓冲 | 两个33-channel及一个69-channel blur；先区分kernel构造与卷积 | 原3³kernel数值/归一化、零填充、累加顺序逐值一致；不将3D卷积改成三轴可分离卷积 | 小kernel缓存只省极小构造；真正group卷积开销缺测 |
| warm调用的模型复用 | 已有对象/权重缓存 | 缓存精度声明与设备不混用；说明warm与cold范围 | 33eager加载仅0.189秒，不能解决本次cold主要差距 |

优先排除**重复的大体积copy与无必要分配**，但不先改网络数学。GPU不加计时hooks或布局变化，不更改TF32/autocast，也不撤回已通过的CPU join、原后端低内存slab和1×1防崩措施。整体CPU速度目标仍未通过。

## 6. 来源与核验

- [完整CPU join旧新记录](../seg_memory_20261005/CPU_FULL.public.json)：各arm原始T1、source/weight SHA、全部输出、preflight/construct/API/save/worker/outerclock。
- [最新TF32作用域CPU完整记录](../seg_tf32_20261005/CPU_FULL.public.json)：当前parc/fast默认完整旧新门；CPU源码数学不变。
- [同节点官方计时与fast BA](../seg_memory_20261005/OFFICIAL_NODE7_FAST_BA_REPEAT.public.json)：正常33复测、373秒首条异常、parc/fast原生计时，以及fast AB/BA。
- [真实join阶段](../seg_memory_20261005/CPU_JOIN_STAGE.public.json)：局部join、真实输入checkpoint、partial capture边界。
- 旧profile只读取索引指向的两份真实记录；完整私密路径不发布，公开JSON保留FNIT相对路径、文件SHA、版本/网格限制和已读数值。
- 本次仅做报告来源和当前源码SHA验证；没有新增速度或精度实测。独立官方benchmark与FNIT运行隔离，生产不调用官方软件。
