# recon-all 五阶段串行优化

本轮从代码 0c8ab327c1a4d10eaf6c8066a16d050e09e30c90 开始，在工作分支
recon-all-serial-20261001 顺序完成。既有整例绑定 c248520，不能将其70.715分钟
结果改标成本轮提速。[阶段状态](../../validation/recon_all/optimizations/20261001_serial/status.json)。

## 功能与流程

原始T1 → conform/N4 → SynthStrip/Talairach → SynthSeg → 归一化/GCA → WM/filled
→ MNI辅助/非线性 → 拓扑/remesh → white.preaparc/球面/配准 → white/pial → 指标。

~~~mermaid
flowchart LR
  A[Synth GPU与模型生命周期] --> B[归一化有序GPU内核]
  B --> C[网格与标准球面]
  C --> D[N4重建与GCA审查]
  D --> E[white与pial热点]
  E --> F[原始T1与空目录两例整例验收]
~~~

开发顺序为全部Synth GPU、归一化、网格/球面/拓扑、N4/GCA、white/pial。
每阶段固定同输入先回归，最后从原始T1与空目录跑两例整例；冻结检查点不算整例。
默认TF32，保留已验证FP32例外，不启用半精度。

## 第一阶段接口与数据结构

复用FNIT SynthStrip、SynthMorph、LimbicUNet；SynthSeg已有GPU及实际精度策略保留。
recon-all的EntoWM、MNI152 affine、MCA/dura、vsinus接收主设备，CPU接口保留供显式诊断。

| 接口/新增参数 | 输入、默认值及限制 | 输出 |
| --- | --- | --- |
| SynthStrip构造 configure_precision=True | 原参数不变；False构造不覆盖调用方TF32，不改变模型 | 原模型对象 |
| SynthStrip前向 precision_report=None | 原影像/border/fill不变；列表收集实际前向设置 | 原StripResult的image/mask/distance，同网格 |
| SynthMorph构造 configure_precision=True | 默认保留独立库TF32旧行为；recon-all False并施加本阶段策略 | 原模型对象 |
| SynthMorph调用 compute_inverse=True, precision_report=None | 默认双向；False仅限非线性transform_only=True且无output_dir，否则ValueError | False时inverse=None，正向不变；省去负velocity积分/未消费反向合成 |
| SynthMorphNetwork.forward compute_inverse=True | (1,1,I,J,K)，空间边长为32倍数；反对称速度两次前向保留 | 固定到移动网格pull位移及可选逆向位移 |
| segment_sclimbic_image | source为1mm nibabel影像；model_path为声明H5；rows为(标签,名称)序列；fov=160，device="cpu"，model=None，stats_path=None，etiv=None，precision_report=None | 原网格/affine int32 MGHImage；统计体积/etiv单位mm³ |
| mri_sclimbic_seg/mri_entowm_seg precision_report=None | 原输入/输出路径、权重/ctab、fov、device、stats和Talairach路径不变 | 原MGZ与统计，返回Path |
| _infer_crop model_cache=None, precision_report=None | float32裁剪强度、native几何、start体素起点、H5、rows、fov和device；同次MCA双侧复用 | 裁剪网格int32标签，消除临时MGZ/ctab |
| mri_mcadura_seg/mri_vsinus_seg precision_report=None | 原路径、device、weights_dir、stats/ctxseg/Talairach参数不变 | 原MCA/静脉窦标签；缓存按权重SHA、设备和策略，仅单次调用驻留 |
| register_mni152_affine precision_report=None | 自产orig、affine权重、两张MNI模板，device="cpu"，threads=4 | 原裁剪图与aff.lta/reg.targ_to_invol.lta |
| run_mni_aux_chain | 原被试/权重/资产，device="cpu"，threads=4 | 原lta/mca_dura/vsinus Path字段；新增runtime实际前向列表 |
| register_talairach precision_report=None | 原影像/模板/权重/XFM/LTA，device="cpu"，threads=4 | 原4×4矩阵及LTA/XFM，保留FP32例外 |
| record_network_forward | 已加载module、真实inputs、records列表及可选元信息 | 追加JSON设备/dtype/TF32/autocast；返回None，无额外同步 |
| _run_white_mri_chain | 自产被试、声明资源、三项Conda程序、threads、stage及显式device | 原MRI/warp/检查图，返回辅助网络运行记录；finalsurfs后处理仍CPU |

缺少影像或资源、非1mm输入、模型/标签通道不符、CUDA不可用或原生程序失败抛异常，
无静默CPU回退。scanner RAS、conformed体素和world-mm位移沿用原接口。
MCA/vsinus最终MGZ沿用原写出元数据，内部int32接口不改变公共输出格式。
完整旧参数见[MNI辅助链](MNI_AUX_CHAIN.md)、[非线性链](MNI_NONLINEAR_CHAIN.md)、
[SynthMorph](../synthmorph/README.md)与[SynthStrip](../synthstrip/README.md)。

## Python示例

~~~python
from fnit.recon_all.mni_aux_chain import run_mni_aux_chain
from fnit.recon_all.sclimbic import mri_entowm_seg
actual_forwards = []  # 实际设备、dtype、TF32和autocast
mri_entowm_seg(
    input_path="/bench/candidate/mri/nu.mgz",  # 自产1mm conform强度
    output_path="/bench/candidate/mri/entowm.mgz",  # 同网格标签
    asset_dir="/models",  # 经清单核对的H5与ctab
    device="cuda:0",  # 显式目标GPU
    stats_path="/bench/candidate/stats/entowm.stats",  # 原体积统计
    talairach_lta="/bench/candidate/mri/transforms/talairach.xfm.lta",  # 自产eTIV变换
    precision_report=actual_forwards,  # 前向时的实际策略
)
auxiliary = run_mni_aux_chain(
    subject_dir="/bench/candidate",  # 自产orig、nu、synthseg.rca
    weights_dir="/models",  # affine、MCA/dura、vsinus
    assets_dir="/fnit-assets",  # 两张MNI模板、三张先验
    device="cuda:0",  # 同一GPU，默认TF32
    threads=4,  # 固定CPU线程预算
)
print(auxiliary["runtime"])  # affine、MCA左/右、vsinus四次前向
~~~

## CLI、官方参考与复现

~~~bash
python -m fnit.recon_all.mni_aux_chain /bench/candidate \
  --weights /models --assets /fnit-assets --device cuda:0 --threads 4
# 同输入阶段：output为空，source为带哈希自产检查点。
python validation/recon_all/optimizations/20261001_serial/benchmark_synth_aux.py \
  --source /bench/frozen --output /bench/new-stage --weights /models \
  --assets /fnit-assets --device cuda:0 --threads 4 \
  --commit ACTUAL_CODE_COMMIT --report /bench/stage.json
~~~

内存接口及前向记录属于命令内部步骤，没有独立官方CLI。对应mri_synthstrip、
mri_synthmorph、mri_entowm_seg、mri_mcadura_seg、mri_vsinus_seg；
固定参数及原论文沿用各功能页。官方只在独立参考目录运行。

## 本轮验证与更新

候选8302bdc已保存Git快照。两例第一阶段回归已完成，见 stage1_summary.json。
整例、整体等效和隔离部署仍待验证。33项专项测试通过。
首次真实基准计算结束但JSON序列化失败，计时作废、保留日志，修复后新空目录重跑。

同GPU旧/新标签及几何预先要求不改变；跨CPU/GPU单列误差与每标签Dice，
不将未知整体阈值当已批准。forward-only比较正向，保留原生数值求逆。
计时同步指定GPU，包含加载、传输及读写；NVML记录同时父子占用；
20GB指20,000,000,000字节，采样峰值不等于连续峰值。

无新增生产依赖，PyTorch/nibabel/h5py已在主页Conda声明。pytest在隔离testdeps，
不修改生产环境。干净安装/运行隔离未验证，PATH/ldd不足以宣布通过。
原实现和文献：[FreeSurfer源码](https://github.com/freesurfer/freesurfer)及上述功能页。


## 第二阶段：有序GPU内核与缓冲复用

保留已有normalize_t1/normalize_t1_aseg及控制点规则，优化它们调用的
voronoi_fill_torch、smooth_bias_torch。公开参数与输出不变：
前者输入同shape三维source/control，返回同设备float32偏置及levels/controls/wall_seconds；
后者输入同shape偏置/source/control，sigma默认8体素，返回偏置和核长度/设备/秒数。
shape错误、空控制点等沿用异常，无近似传播或窗口缩小。

传播一层一个CUDA kernel，按原dz/dy/dx顺序，仅读更早层，边界夹取的重复
邻居仍计数。距离、稳定索引排序继续使用原SciPy/NumPy实现；排序索引只搬入一次。
每个浮点和及除法保留float32，不启用FMA融合或半精度。
平滑每轴一次kernel，按原65项核顺序累加、nearest边界，两份缓冲交替使用，
控制点原强度恢复不变。CPU显式接口保留原实现。Triton已在主页Conda环境声明。

新增内部ordered_wavefront参数是连续同设备float32 field、int32 distance、
int64 ordered和CPU层边界；原位返回field。ordered_smoothing输入三维float32
CUDA偏置及一维同设备核，返回同shape偏置；调用方负责控制点恢复与计时。
两者属于mri_normalize内部步骤，没有独立官方CLI，不改变空间或TF32配置。

15项专项回归通过，包含边界重复、非连续输入、sigma=1/2/8及逐元素严格相等。
首例冻结自产输入、同GPU4线程且保留关闭分配缓存时，两轮含加载/传输/读写：
236.794→45.345秒、257.361→57.177秒。T1/brain与旧版本均零差异体素、
几何/dtype一致、保存SHA相同。第二例154.839→45.231秒、210.151→63.332秒，同样两张输出零差异、SHA相同。两例是同输入阶段实测，不是整例提速。
同输入未修改实现打开缓存时45.758/58.844秒，证明分配策略对当前实现开销很大。
本次保留低显存策略，通过减少临时张量和CUDA启动消除开销。
关闭缓存的PyTorch统计标为unavailable，不将零值当零显存；NVML报告另列。


## 第三阶段：标准球面

复用已有 RegistrationGradientAverager 的完整邻接有序平均；CUDA 只承担
这一算子，目标函数与顺序归约继续用 NumPy/Numba。原邻居顺序、所有平均
轮数、float32 中间值、逐顶点 double 总和、接受规则和末尾清理保留。
独立顶点行可以四线程计算；总和按原顶点顺序累计。只缓存固定面的 CSR
和原 smoothwm 的面积，当前坐标与法向每次重新计算。

run_standard_sphere 的完整输入为 inflated、smoothwm、output 三个表面路径；
finish_device="cpu" 选择末尾清理设备，averaging_device="cpu" 选择平均设备。
三角面必须有同一有序面与顶点对应关系，坐标采用 surface RAS/mm；sphere
输出半径100mm的float32(N,3)坐标及原有序面。返回路径、两个设备、初始负面积
比例、投影/metric/拓扑准备/finish/总秒数、完整 updates 和负面计数的字典。
3000次更新未收敛抛 RuntimeError；不兼容面、设备或数组抛异常，无CPU静默回退。
计时包括读取、JIT、搬运和输出写入；CUDA平均回到CPU时已等待结果。

FaceNormalTopology(triangles, nvertices) 保存只读整数有序面及 CSR；
evaluate(vertices) 接收 float32(N,3) 坐标并返回本轮单位法向，无关联面为零；
validate(triangles, nvertices) 检查缓存兼容性。不同表面版本分别建缓存，
非法形状、整数类型或索引抛 ValueError，不复用过期坐标。
first_epoch_gradient 与 nonlinear_epoch_gradient 新增仅关键字 normal_topology=None、
original_metric=None；后者为原面绝对面积(F,)mm²及float32总面积mm²，调用者必须
提供本次smoothwm的量，None仍按原路径计算。它们为内部算子，无独立官方CLI。

~~~python
from fnit.recon_all.sphere_standard_run import run_standard_sphere
sphere_report = run_standard_sphere(
    inflated="/data/self/surf/lh.inflated",  # 自产完整 inflated，surface RAS/mm
    smoothwm="/data/self/surf/lh.smoothwm",  # 同有序面和顶点，提供距离与原面积
    output="/data/diagnostic/lh.sphere",  # 已创建的新诊断目录
    finish_device="cpu",  # 保留当前末尾相交清理
    averaging_device="cuda:0",  # 显式目标GPU，完整有序平均轮数
)
~~~

~~~bash
# 三个位置参数分别为同序 inflated、smoothwm 和输出；线程预算由调用环境固定。
python -m fnit.recon_all.sphere_standard_run \
  /data/self/surf/lh.inflated /data/self/surf/lh.smoothwm /data/diagnostic/lh.sphere \
  --finish-device cpu --averaging-device cuda:0 --report /data/diagnostic/report.json
~~~

对应官方独立参考命令 mris_sphere INPUT OUTPUT；原代码、论文与前置要求见
[网格与球面说明](CPU_GEOMETRY_PERFORMANCE.md)。生产不执行官方命令。
24项专项测试通过，包括有序SSE与法向缓存更新；真实四侧结果随后列在本页。
拓扑GA与动态remesh保留已有实现：共享defects文件需按顺序写出，动态缩边后续
读取前一顶点更新，不能直接替换成同时更新的GPU算法。未声称这两个阶段提速。

四半球同输入实测（74ae022，H100、4线程、分配缓存关闭，含JIT/IO）：

|输入|优化前秒|优化后秒|坐标最大误差mm|
|---|---:|---:|---:|
|sub01 lh|245.118|154.903|0|
|sub01 rh|184.147|137.240|0|
|sub02 lh|177.995|121.193|0|
|sub02 rh|162.809|110.374|0|

全部有序面与坐标相同；GPU采样峰值约0.49GB，仅为本阶段，整体等效仍未判定。完整程序/输入/源码哈希与监控边界见 [stage3_summary.json](../../validation/recon_all/optimizations/20261001_serial/stage3_summary.json)。

## 第四阶段：N4/GCA

两例原生N4量化前与量化后零差异，新增完整参数与实际分段时间见
[N4说明](N4_ITK_CONDA.md)。实际拟合占约98%，空间重建四线程没有整段收益，
生产保留1线程；GCA现有Python只有第一EM方向，保留成熟Conda完整优化器。
修复主页构建时间戳循环，记录编译器、源文件、ITKConfig和二进制SHA。
这阶段没有宣称提速，整例记录实际N4子段，旧程序能力缺失时明确标记。

## 第五阶段：white/pial

完整pial的CPU剖析定位到逐顶点异步更新及其碰撞查询：约2463万次候选判定，
不是原始white桶构造占主要时间。61926c7复用静态CSR及原始坐标桶索引，将
同一顶点关联面的KD查询和碰撞计算批量化；整个顶点仍按原顺序接受更新。
拒绝试步产生的旧MHT状态、完整候选扩展、边界强度、四轮和最终清理均保留。
接口、具名示例和全部内部量见[Python pial说明](PYTHON_PIAL_PLACEMENT.md)。

32项专项测试通过，四半球真实法向和候选CSR完全一致。sub01左侧完整41步
新旧有序坐标及保存SHA相同。不启用剖析、分别使用空JIT缓存，含加载/写出
1453.060→1232.634秒，单次新/旧顺序配对减少15.17%；共享负载下尚未验证稳定
吞吐，不能与1687.49秒的cProfile运行直接计算提速。官方、Conda、
Python三方同输入运行在隔离目录；Python与官方完整坐标完全一致，Conda相对官方P99为0.225255mm、最大1.482209mm；冻结Conda与本次Conda几何相同。该构建差异单列，不归因于随机性。
Conda176.787秒、官方133.863秒，Python虽已提速仍较慢；当前保留Conda完整white/pial。

整例首次尝试在Talairach子进程加载模型时CUDA OOM，28.709秒退出，不计入
完成整例或提速。采样未显示目标GPU耗尽；同设备与原精度/allocator下的独立
Talairach及已初始化CUDA完整输入链复跑成功，原因未确定。完整失败日志、采样
和新空目录retry1配置保留；不使用检查点拼接成整例。

随后sub02 CLI整例也在相同位置失败。六次原始T1输入链监控对照（两例、原入口
和提前初始化）均成功，orig/SynthStrip数组与几何、LTA矩阵零差异，未定位根因。
没有将不具备失败/成功配对证据的初始化改动作为修复发布。候选源码保持61926c7，
第二例在新的空目录retry2运行。最新完成范围、失败日志与复现配置见
[当前整例结果](../../validation/recon_all/optimizations/20261001_serial/WHOLE_RESULTS.md)。

## 整例复现与只读比较

候选61926c7从原始T1与新空目录运行。sub01使用已初始化CUDA的Python API；
sub02使用CLI，优化前后均为同一H100 UUID、4线程和关闭分配缓存。sub01基线
复用c248520的已有GPU完整记录：其src树与本轮起点0c8ab32均为
7304eef4cf14d79baa87f49c98dc8a3b7c06181d。它是已有记录复用，不是本轮重跑，
共享负载及加载缓存可能不同；sub02另测当前GPU基线，不能拿旧CPU耗时配对。

capture_runtime_manifest.py接收runtime-root、已声明expected-manifest、
build-manifest、code-commit和新的output路径；native-bin-dir可显式选择本轮
独立程序束，默认环境bin。程序束只连接源码构建产物，N4使用新构建、其余13项
保留已核对的Conda产物。脚本重新核对两例输入、权重、资产及程序大小/SHA，
输出JSON包含主机/CPU/GPU/库版本及不符列表；文件缺失或不符即失败。
不读取或复制许可证内容；硬件快照和哈希核验不等于干净隔离部署通过。

collect_hotspot_whole_comparison.py的config含hot、comparison_root、python、
label_table、code_commit及两例cases。每例指定candidate、baseline、official、
launch；只读已成功完成的subject与completion.json，不写回生产。
可选scripts_dir固定比较器目录、code_root固定其导入的FNIT版本，默认兼容
历史c248520布局；additional_quality可含baseline、official，为各自另写质量报告。
JSON/CSV和脑图位于comparison_root；线程固定4，质量检查的180秒/2000万候选
预算不足会明确记录incomplete，不作为通过。输出已有、执行失败、缺失输入或
比较异常会抛错并保留日志。两脚本属于独立benchmark，没有官方独立等价CLI。

~~~bash
# 三项配置分别声明解释器、原始T1、新空目录、GPU、线程、代码归档和独立程序束。
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/whole_sub01_candidate.json  # 预初始化CUDA API
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/whole_sub02_baseline.json  # 同GPU的旧版CLI
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/whole_sub02_candidate.json  # 同GPU的候选CLI
python validation/recon_all/python_gpu_port/collect_hotspot_whole_comparison.py \
  --config /bench/whole_comparison_config.json  # 三次整例结束后串行只读比较
~~~

整例墙钟采用相同monitor命令边界，包含导入、校验、加载、传输与读写。
分别报告执行、138项输出完整性、严格复现、优化是否引入退化和整体指标。
整体等效未批准阈值，保持not_assessed；保留逐标签Dice、厚度/面积/体积偏差、
双向点到三角面距离及局部穿越/球面翻折。所有计时与统计以完成后实际报告为准。
