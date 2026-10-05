# MSM、HCP 特征与表面 CPU 官方对照

## 1. 功能与本轮范围

本轮基线为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。完整功能清单见 [功能矩阵](FEATURE_MATRIX.md)：MSMSulc、MSMAll、VN、DR、WRN、配准准备、表面几何、体积到表面投影、CIFTI 和 surface pipeline。

此前的 10 项完整基线回执见 [聚合检查点](report.checkpoint.public.json)。2026-10-05 已取回候选配准 CPU1/8 六项、HCP 特征 CPU1/8 八项的完整数值、实际源码与输入 SHA，见 [最新精度与耗时报告](completion_status_20261005.public.json)。全部 14 项输入在配对前后不变；保留完整帧数、顶点、d7–d21 组数和原配准停止条件。原 nodes/spectra 六项与新旧 FNIT 十二项的完整数值核对已完成，见下方逐项结果。固定投影与完整 surface 的历史初次尝试在原版 command_0、候选启动前因参考源码元数据部署缺失而失败，原失败目录保留；隔离修复后的 CPU1/8 原版与候选完整重试已收齐科学聚合：固定投影逐值一致，完整 180 帧 surface 仍有差异。

```mermaid
flowchart LR
    I[真实完整输入] --> O[固定原程序 CPU 1 / 8]
    I --> B[冻结 FNIT 基线 CPU 1 / 8]
    I --> C[优化 FNIT CPU 1 / 8]
    O --> V[全部输出数值 / 几何 / 轴 / CPU 使用]
    B --> V
    C --> V
    C --> G[同输入完整旧 / 新 GPU 配对]
    classDef mono fill:#fff,stroke:#000,color:#000;
    class I,O,B,C,V,G mono;
```

## 2. Python 调用与输入输出

每项 API 的完整变量示例、逐参数说明和输出结构分别见 [MSMSulc](../../../docs/msm/README.md)、[MSMAll](../../../docs/msm/msmall.md)、[HCP 特征](../../../docs/msm/features.md)和 [surface](../../../docs/fmri/surface.md)。本目录的 `adapter.py` 仅用于统一记录完整 API，不作为新的运行时入口。

本轮真实输入：

- MSMSulc：左右 native 球面分别 120,035 / 122,950 顶点，参考球面 163,842 顶点；原四级配置完整执行。
- MSMAll：左右各 32,492 顶点的真实 WRN C 特征，33 列；原一级 coarse 和三级 refine 分别完整执行。
- VN/DR/WRN：完整 490 帧，90,568 个有效 grayordinates，d40 参考、32 个显式选中组件；ICA mixing 为 106 列，其中 84 noise / 22 signal。原 91,282 轴中 714 个常数位置已在输入准备时排除，双方读取同一已声明 BrainModelAxis。
- WRN：匹配 59,380 个皮层 grayordinates 的正均值 1 area、全部 d7–d21 参考、同一双侧 midthickness 与 14 mm 原 Workbench 平滑。
- 固定几何投影：490 帧 T1w / MNI BOLD、TR 0.735 s，完整双侧 native 几何；输出 32k GIFTI 与 91,282 列 CIFTI。
- 完整 surface：已公开病例的完整 180 帧、TR 2.1 s，以同一已完成 volume 和 recon-all 为起点，surface 输出目录为空；计入几何准备、MSMSulc、投影、CIFTI、QC 与保存。volume 和 recon-all 计算另属各自整链测试。

原软件产生全部参考输出；FNIT 调用只读原始声明输入，不读取参考输出。逐顶点数组、个体路径和新派生影像保留在私有运行目录。

## 3. 测试命令

协调者用同一 harness 配对运行，各参数含义如下：

```bash
python tools/benchmark_multimodal_cpu.py run \
  --manifest /private/task04/registration.manifest.private.json \
  --baseline-root /path/to/baseline_cc940273 \
  --candidate-root /path/to/candidate \
  --output-dir /private/task04/registration_cpu1_cpu8 \
  --python /path/to/fnit-environment/bin/python \
  --threads 1,8 --cpuset 3,7,19,27,35,43,47,51 \
  --lock-file /private/locks/task04_msm_surface.lock \
  --backends official,candidate --device cpu \
  --single-observation --repetitions 1 --api-repetitions 0
```

`manifest` 声明全部输入、原程序、配置和输出；两 source roots 固定源码版本；`threads` 是进程及子进程总预算；`cpuset` 指定八个不同物理核，单线程使用首核；`lock-file` 使同组任务串行；`single-observation` 表明仅各一次整例；`api-repetitions=0` 不追加热调用。原程序、输入及许可按私有 manifest 现场核验，路径占位符需替换为自己的授权文件。

## 4. 原软件调用

MSMSulc 原版每侧使用当前整个 CPU 预算，两侧依次运行；FNIT 8 核 API 可将预算分给两侧，但进程及子进程始终在同一八核内。

```bash
newmsm --inmesh="$ROTATED_NATIVE_SPHERE" --refmesh="$REFERENCE_SPHERE" \
  --indata="$NATIVE_SULC" --refdata="$REFERENCE_SULC" \
  --conf="$ORIGINAL_CONFIGURATION_WITH_THREAD_BUDGET" --out="$OUTPUT_PREFIX"

newmsm --inmesh="$SOURCE_SPHERE" --refmesh="$REFERENCE_SPHERE" \
  --indata="$SOURCE_FEATURES" --refdata="$REFERENCE_FEATURES" \
  --trans="$INITIAL_SPHERE" --inweight="$SOURCE_WEIGHTS" \
  --refweight="$REFERENCE_WEIGHTS" \
  --conf="$ORIGINAL_MSMALL_CONFIGURATION_WITH_THREAD_BUDGET" --out="$OUTPUT_PREFIX"
```

HCP 使用固定 v4.7 的原 `ComputeVN.m` 与 `MSMregression.m`，连同匹配的原 CIFTI/GIFTI/FSLnets 读写依赖，在 MATLAB R2018b 实际调用。DR+VN/WRN 的中间归一化使用原 `SingleSubjectConcat.sh` 的 Workbench MEAN 与除法步骤。主表使用 `nTPsForSpectra=0`，输出原 maps/weights；原 nodes 的 `nTPsForSpectra=490` 支路还包含 spectra 与绘图，精度和时间单列。

surface 参照使用用户指定 fMRIPrep 25.2.4 镜像内的原工作流与 NiWorkflows CIFTI 实现。原程序仅在隔离参照运行中使用。

## 5. 已完成真实对照

### 完整 CPU1 配准基线

下表是 nodecw8 的 fresh process 墙钟，包含读入、计算、所有输出和进程启动。每项各一次，不能推断稳定中位数。

| 完整双侧功能 | 原版 / s | 冻结 FNIT / s | 本轮精度 |
| --- | ---: | ---: | --- |
| HCP 四级 MSMSulc | 1550.168 | 2185.466 | 有序 faces、顶点对应相同；L/R 角差 mean 0.587 / 0.554°，p99 2.724 / 1.915°；未达到严格参照 |
| WRN C 一级 MSMAll | 96.764 | 93.945 | 全双侧球面坐标逐位相同，0 相对翻面 |
| WRN C 三级 MSMAll | 2034.212 | 2608.629 | 全双侧球面坐标逐位相同，0 相对翻面 |

MSMSulc 的 CPU 不一致已定位到 Point 运算中向量除法的末位舍入，进而改变共享边三角面归属。CPU literal double 修复先通过完整 affine 与首轮成本检查，随后完整四级配准达到双侧坐标和有序 faces 逐位相同。原 FastPD/WLS 固定源码构建与原完整 WLS 包逐位通过。

本次 CPU 移植另发现单点几何路径的标量除法缺陷：NumPy 返回零维标量，直接交给 `torch.from_numpy` 会报错。已用 `np.asarray` 保持标量 Tensor 返回；数组结果不新增运算或复制，原数组及 CUDA 分支保持相同。标量、单三角形距离与旧 Tensor 差分、既有 Point/SphereMap 定向检查共 29 项通过，见[标量修复回执](point_scalar_regression_20261006.public.json)。完整批量 benchmark 未触发此缺陷，其源码范围与最终修复的关系单列在[源码语义核对](surface_final_source_semantics_20261005.public.json)。

### 最新完整 CPU1/8 配准

以下在 nodecw8 完成，每项各一次。fresh process 包含程序启动、读入、计算和全部保存；API 是 FNIT 函数的完整读写调用。所有 CPU1 候选双侧坐标与有序 faces 都和原版严格单线程逐位相同；候选 CPU8 与 CPU1 的全部球面文件 SHA 也相同。

| 功能 | CPU 预算 | 原版 fresh / s | FNIT fresh / s | FNIT API / s |
| --- | ---: | ---: | ---: | ---: |
| HCP 四级 MSMSulc | 1 | 1537.903 | 455.084 | 452.696 |
| HCP 四级 MSMSulc | 8 | 466.986 | 173.393 | 171.057 |
| WRN C 一级 MSMAll | 1 | 96.724 | 58.948 | 56.162 |
| WRN C 一级 MSMAll | 8 | 28.814 | 26.671 | 24.196 |
| WRN C 三级 MSMAll | 1 | 2020.025 | 1512.800 | 1510.518 |
| WRN C 三级 MSMAll | 8 | 600.420 | 360.841 | 358.670 |

原版 MSMSulc 的 CPU8 与 CPU1 也逐位相同。原版 MSMAll 多线程存在变化：coarse 的右侧相对单线程平均角差 0.706°、p99 4.266°；refine 的左/右平均角差 0.251/0.277°、p99 0.980/1.419°。FNIT CPU1/8 保持严格单线程参照，不能将这种原版线程差异标成 FNIT 精度回退。

四级 MSMSulc 左侧参照和候选都存在 1 个相对取向改变面，右侧为 0；MSMAll coarse/refine 双侧为 0。数值逐位匹配和几何零翻面是两项独立检查。

实际锁内候选全部 FNIT Python 源码树 SHA 为 `72059515e650f7db02084fd41816ab78abf61bfbc1286e8f398a2ebdfd5b294f`，逐模块与输入 SHA 见报告。最终快照将 selector 的 CPU 分支增加 float64 条件。实际读回旧源码后核对，公开入口先转换 points 与缓存几何为 float64，CPU 完整调用保持同一分支；CUDA 保持原 Tensor 分支。四个变更文件的 SHA 与 ready-path 语义见 [最终表面源码核对](surface_final_source_semantics_20261005.public.json)。

### 完整 490 帧 HCP CPU1 基线

以下在 nodecw10 同一 CPU1 预算完成。该节点共享负载约 2,450–2,510，列出的时间为实测观测。原函数与 fresh process 分列：MATLAB 启动/退出开销不是算法时间。

| 功能 | 原函数 / s | 原 fresh process / s | FNIT 完整 API / s | FNIT fresh process / s | maps 最大误差 / RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| VN | 30.391 | 222.734 | 42.421 | 67.636 | 2.44e-4 / 1.84e-5 |
| DR | 29.700 | 111.389 | 61.820 | 85.235 | 5.05e-5 / 8.05e-7 |
| DR+VN | 34.819 | 143.223 | 77.818 | 100.794 | 6.91e-6 / 5.09e-7 |
| WRN | 1781.215 | 1888.675 | 954.113 | 977.446 | 3.28e-6 / 2.35e-7 |

原 DR+VN 还包含 31.704 s 的 Workbench 归一化准备，原 wrapper 合计 69.928 s。所有表项均核对完整数组、有限性、保存 float32 与 BrainModelAxis；DR/WRN maps 的全部轴一致，40 组件 weights 逐位相同。VN 的 ScalarAxis 标签名称不同，类型、形状、BrainModelAxis 和全部数值均单独记录。此表未将主参照没有写出的 nodes 标为通过。

已保存的 CPU8 基线使用 nodecw10 同一组八个物理核。旧基线 WRN CPU8 的完整 API 已保存，原队列在后续候选阶段没有追加第八条记录。现已独立核对实际输入、全源码、原输出 SHA 与 CPU 预算，恢复该已完成结果；只变化的参考 launcher 在 FNIT 调用之外，默认原命令不变。原队列和旧尝试保留原字节，详见下方完整节点报告。最新候选完成全部八项。

| CPU8 功能 | 原函数 / s | 原 fresh process / s | FNIT 完整 API / s | FNIT fresh process / s | maps 最大误差 / RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| VN | 23.207 | 260.850 | 31.984 | 56.685 | 2.44e-4 / 1.84e-5 |
| DR | 21.724 | 82.336 | 54.316 | 75.814 | 2.10e-5 / 6.81e-7 |
| DR+VN | 29.141 | 129.319 | 67.716 | 88.363 | 8.26e-6 / 4.88e-7 |

CPU8 原 DR+VN 的 Workbench 准备为 33.662 s，原 wrapper 合计 64.867 s。fresh process 包含各自程序启动、读写和结束；FNIT API 与原函数/准备另列。当前 VN、DR 的完整 FNIT API 没有在这些观测中快于原函数，不能只根据 MATLAB 启动开销宣布达到算法速度目标。各组共享负载和实际 user/system CPU 记录保留在聚合检查点，线程预算不是实际持续用满八核的证明。

### 最新完整 490 帧 HCP CPU1/8

最新候选已在 nodecw10 完成全部 maps/weights 配对。以下原版与 FNIT fresh 是同轮进程墙钟；原版 MATLAB 函数和 Workbench 准备的分项仍需导出。不能用 MATLAB 启动差异证明 VN/DR 计算已经快于原函数。

| 功能 | CPU 预算 | 原版 fresh / s | FNIT fresh / s | FNIT API / s | maps 最大误差 / RMSE |
| --- | ---: | ---: | ---: | ---: | --- |
| VN | 1 | 107.621 | 63.373 | 39.671 | 2.44e-4 / 1.84e-5 |
| VN | 8 | 131.837 | 55.529 | 31.964 | 2.44e-4 / 1.84e-5 |
| DR | 1 | 112.120 | 86.136 | 60.982 | 5.05e-5 / 8.05e-7 |
| DR | 8 | 95.747 | 74.894 | 52.172 | 2.10e-5 / 6.81e-7 |
| DR+VN | 1 | 381.639 | 103.930 | 78.871 | 6.91e-6 / 5.09e-7 |
| DR+VN | 8 | 184.906 | 99.782 | 74.660 | 8.26e-6 / 4.88e-7 |
| WRN d7–d21 | 1 | 1943.838 | 723.399 | 699.610 | 3.28e-6 / 2.35e-7 |
| WRN d7–d21 | 8 | 1373.877 | 533.785 | 509.053 | 2.28e-6 / 2.00e-7 |

全部 maps 有限、形状与保存 float32 正确；DR/WRN 轴一致，所有 40 列 weights 逐位相同。VN 的 BrainModelAxis 一致，ScalarAxis 名称仍不同，报告明确 `axes_equal=false`。本轮保留节点负载、实际 user/system CPU 和 affinity，数字是共享节点的一次观测。WRN 缓存保持数学结果；VN/DR 的完整 CPU profile 用于判断仍慢于原函数的热点。

### 完整节点与振幅谱核对

[全矩阵报告](full_nodes_spectra_v3.public.json)逐项核对 DR、DR+VN、WRN 的 6 个原版输出和 12 个旧／新 FNIT 输出。每项 nodes 为 490×40，振幅谱为 245×40；组件顺序固定，不调整符号或重新排列。全部数组有限，旧／新 FNIT nodes 文件 SHA 在每种配置和线程预算下相同。

| 功能 | CPU | nodes 最大差 / RMSE | 振幅谱最大差 / RMSE |
| --- | ---: | --- | --- |
| DR | 1 | 5.288e-3 / 7.998e-4 | 4.949e-1 / 1.455e-2 |
| DR | 8 | 5.087e-3 / 7.981e-4 | 4.949e-1 / 1.450e-2 |
| DR+VN | 1 | 5.424e-5 / 1.056e-5 | 4.830e-3 / 1.932e-4 |
| DR+VN | 8 | 5.169e-5 / 1.055e-5 | 4.830e-3 / 1.928e-4 |
| WRN | 1 | 9.570e-5 / 1.611e-5 | 6.511e-3 / 3.504e-4 |
| WRN | 8 | 5.746e-5 / 1.431e-5 | 4.998e-3 / 3.125e-4 |

原脚本以 5 位有效数字保存 nodes 和 spectra，FNIT nodes 保留 17 位。报告另外用原已保存 nodes 重建振幅谱，记录相同文本量化造成的误差量级；上表仍为实际误差，不称为逐位一致。nodes 和 spectra 的组件平均相关均大于 0.9999999998。谱核对按原 `nets_demean` → `abs(FFT)` → 前 245 点执行；这是保存节点的离线核对，生产 API 没有新增谱输出，也没有将这次 FFT 的耗时当作生产速度结果。

### 固定投影与完整 surface 实际门槛

[完整原版／候选报告](projection_surface_metadata_v1.public.json)已核对 CPU1/8 保存输出。固定 490 帧几何投影的左右 GIFTI 全帧与 91k CIFTI 逐值相同，轴和 float32 保存类型相同。

| 完整范围 | CPU | 原版 fresh / s | FNIT fresh / s | FNIT 完整 API / s | 实际精度 |
| --- | ---: | ---: | ---: | ---: | --- |
| 固定几何投影，490 帧 | 1 | 466.650 | 414.180 | 412.364 | 全部值及 CIFTI 轴相同 |
| 固定几何投影，490 帧 | 8 | 268.903 | 130.268 | 128.419 | 全部值及 CIFTI 轴相同 |
| fresh surface，180 帧 | 1 | 1855.025 | 783.233 | 781.290 | 球面与时序仍有差异，未通过严格匹配 |
| fresh surface，180 帧 | 8 | 521.294 | 251.735 | 249.809 | 同上；FNIT CPU1/8 输出一致 |

180 帧的严格单线程原版对照中，球面左／右平均角差为 0.302742/0.345475°；CIFTI 最大差 388.922、RMSE 13.8695，逐 grayordinate 时间相关均值 0.977986、p01 0.732025。完整调用和元数据修复已完成；这些误差仍需定位和修复，不能由固定几何零差或其他病例的 MSM 零差推断本例整链一致。原版和候选使用同一已完成 volume、recon-all 和配置。差异定位已排除完整准备数组、归一化网格、刚性和前两轮离散更新；首个粗网格级已完成自然收敛观察，结果如下。

#### 本例完整输入的逐段诊断

[已完成前缀聚合](surface_prefix_aggregate_complete_second_v1.public.json)保留真实输入、源码、配置、原版库与观察程序 SHA。原版观察程序只截取已发生的阶段，正式配置和已有整链输出保持原字节；诊断耗时包含观察开销，不作速度 benchmark。

| 完整边界 | 左半球 | 右半球 |
|---|---|---|
| 准备后的球面、脑沟、参考与仿射数组 | 全部一致 | 全部一致 |
| 归一化网格、面积、刚性网格和最终刚性坐标 | 一致 | 一致 |
| 原版实际查询坐标上的全部刚性成本重放 | 85 次全部一致 | 89 次全部一致 |
| 首离散 DATA、CP、拓扑和初始标签 | 一致 | 一致 |
| 首 alpha 全部三角成本最大差 | 4.97×10⁻¹⁴ | 7.46×10⁻¹⁴ |
| 第一轮 DATA、CP 与标签 | 一致；更新 0 个 CP | 一致；更新 0 个 CP |
| 第二轮 DATA、CP 与标签 | 一致；更新 11 个 CP | 一致；更新 59 个 CP |

原版的实际收敛条件保证前两轮继续执行。两轮完整 DATA 均为 2,562×3、CP 为 162×3，全部坐标与标签误差为零；成本中的末位差异未改变这两轮的选择。完整 180 帧仍未达到严格匹配，后续粗网格及细网格阶段继续定位。第二轮观察程序的历史初次尝试因 `ldd` 绝对链接格式漏解析在算法执行前失败；九库同实体及 SHA 校验已修复，失败目录保留，见[前置校验修复](surface_prefix_preflight_recovery_v1.public.json)。公开报告只含数量、误差和校验值。



#### 粗网格级自然收敛（2026-10-06）

[完整粗级聚合](surface_coarse_level_aggregate_v1.public.json)复用原配置最大 10 次迭代和原版收敛条件，左右实际分别迭代 8/6 次。左侧最终 DATA、CP 与最后暂定标签完全一致；右侧 CP/标签完全一致，DATA 的 3 个 float64 坐标最大差 3.553×10⁻¹⁴，转为 float32 后一致。

右侧第 4/5 次成本的候选减原版分别为 0.0725037/0.0954643；该差异超出末位舍入。收敛末坐标接近不能证明整个成本轨迹一致，180 帧整链仍未通过。下一边界是细网格初始化、权重、旋转和首个 alpha 成本。观察程序初次尝试仅因 `_ico` 回调漏接 `cached_area` 失败；补全观察接口后完成，生产计算和原配置未改，见[观察接口修复](surface_coarse_observer_recovery_v1.public.json)。

#### 首个细网格 setup 与成本（2026-10-06）

[细级聚合](surface_fine_setup_aggregate_v1.public.json)在真实粗级之后核对 10,242 个 DATA、642 个 CP 和首个 alpha 的 1,280×8 个三角成本；没有执行细级 Fusion 或完整新配准。左侧布局、几何、旋转与标签位置精确，首 alpha 最大差 3.98×10⁻¹³。右侧拓扑、初始标签和标签位置精确，几何最大差 4.27×10⁻¹⁴、float32 一致，但源点归属缓存布局不同：12 个 offset 条目、104 个展平缓存槽位不同；这些数量不能当作改变归属的源顶点数。右侧首 alpha 成本最大差 0.120256、RMSE 0.00409368，零标签成本最大差 0.0843831。

原版输入/参考 ROI 权重均为 1，未发现 ROI 丢失；原生源码、9 个实际 SDK 库、输入及候选模块的前后 SHA 不变。分歧已定位到源点→CP 三角面归属，具体边界运算仍需核对，整链仍未宣告匹配。

最终三种配准的 12 次 GPU 旧／新配对完成，全部坐标和有序 faces 相同；详细时间、显存和共享负载见 [完整 GPU 聚合](gpu_registration_abba_final_recovery_v1.public.json)及[图表](../../../docs/msm/README.md#本轮-cpu-优化后的完整-gpu-回归2026-10-05)。coarse 优化版在本次繁忙共享环境观测较慢，不报告稳定加速或性能回退。

### 公开脑图例子

下图是项目已经公开的 HCP 参考 RSN，用于说明空间特征。它不展示本轮私人病例的派生图或差异图。

![公开 HCP RSN 的左右外侧与内侧视角](../../../docs/msm/images/reference_rsn.png)

## 6. 本轮更新和测试记录

- 固定原 native 与独立 FNIT 构建：Conda GCC 11.2、`-O3 -std=c++17 -fno-fast-math -ffp-contract=off`；实际 import 的新 `.so` SHA `69fda883c5022d412172eba2de164b79726d18c068b7c421a200b331b6502ad8`。
- CPU containing-face 查询复用静态三角几何，使用独立行 Numba 运算；保留原候选顺序、有限边距离和面号 tie break。CUDA 和梯度调用保留原路径。
- CPU float64 无梯度 Point 运算使用 literal double normalize/tangent/投影/面积权重；全真实首轮 gate 用于定位，最终配准精度仍由完整输出确定。
- CPU WRN 复用固定 BOLD spatial/temporal demean，完整 d7–d21 的顺序和 pinv 容差不变。约增加 710 MB CPU 内存；CUDA 或输入需要梯度时仍按原序列执行。
- 2026-10-05 本地 Point/selector/WRN 检查 30 项通过，sphere execution 的 CPU/CUDA 检查 11 项通过。新增 float32 查询保留 tensor 分支检查；服务器冻结候选保持其原哈希。见 [逐模块与哈希记录](focused_checks_20261005.public.json)。小型功能控制不替代完整真实 CPU/GPU benchmark。
- CA/CAT 尚缺可证明来源的同一个体真实 T1w/T2w myelin 与对应 bias；当前资源缺口保留在功能矩阵中。

## 7. 原实现、来源与许可

- [newMSM](https://github.com/rbesenczi/newMSM)、[官方 MSM 说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/msm.html)。本轮实际原二进制 SHA `af5c04246cfeea19233232168acbc1f31266f28b1a7cb6779b8f1bfa32eb9615`；相关原运算和 HOCR/FastPD 来源见项目 notices。
- [HCP Pipelines v4.7](https://github.com/Washington-University/HCPpipelines/tree/v4.7.0)，commit `f8cac6892f88bdf889d644711ff038198eb81533`。原特征来源使用 HCP BSD 许可；匹配依赖保留各自许可，原 MATLAB 和二进制不随 FNIT 分发。
- [fMRIPrep 25.2.4](https://github.com/nipreps/fmriprep/tree/25.2.4)、[Connectome Workbench](https://github.com/Washington-University/workbench)、[NiWorkflows](https://github.com/nipreps/niworkflows)。镜像 SHA `8e32238619053c1f9d1739b26f4afd72df809d914f5a5771707bf5da4b1d0f39`；Workflows 与 WB 的实际版本分别记录。
- Robinson et al. (2018), *NeuroImage*, Multimodal surface matching with higher-order smoothness constraints；Glasser et al. (2016), *Nature*, A multi-modal parcellation of human cerebral cortex。
- [第三方来源与许可](../../../THIRD_PARTY_NOTICES.md)。模板大小、SHA 与取得位置由私有 manifest 绑定；未获得再分发授权的资源只从原站获取。
