# recon-all 五阶段串行优化

本轮从代码 0c8ab327c1a4d10eaf6c8066a16d050e09e30c90 开始，在工作分支
recon-all-serial-20261001 顺序完成。既有整例绑定 c248520，不能将其70.715分钟
结果改标成本轮提速。[阶段状态](../../validation/recon_all/optimizations/20261001_serial/status.json)。

## 功能与流程

原始T1 → conform/N4 → SynthStrip/Talairach → SynthSeg → 归一化/GCA → WM/filled
→ MNI辅助/非线性 → 拓扑/remesh → white.preaparc/球面/配准 → white/pial → 指标。

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

候选8302bdc已保存Git快照。本页建立时真实回归执行中，无本轮整例时间、
整体等效或隔离部署结论。33项专项测试已通过；加入实际设备记录后重新验证。
首次真实基准计算结束但JSON序列化失败，计时作废、保留日志，修复后新空目录重跑。

同GPU旧/新标签及几何预先要求不改变；跨CPU/GPU单列误差与每标签Dice，
不将未知整体阈值当已批准。forward-only比较正向，保留原生数值求逆。
计时同步指定GPU，包含加载、传输及读写；NVML记录同时父子占用；
20GB指20,000,000,000字节，采样峰值不等于连续峰值。

无新增生产依赖，PyTorch/nibabel/h5py已在主页Conda声明。pytest在隔离testdeps，
不修改生产环境。干净安装/运行隔离未验证，PATH/ldd不足以宣布通过。
原实现和文献：[FreeSurfer源码](https://github.com/freesurfer/freesurfer)及上述功能页。
