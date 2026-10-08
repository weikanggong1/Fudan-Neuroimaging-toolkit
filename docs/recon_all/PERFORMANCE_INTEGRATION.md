# recon-all 性能候选的生产接入

本轮整例接口继续读取一份原始 T1，在空被试目录生成标准 MRI、双侧表面、
顶点图、注释和统计。五个专属分支已合并，候选 Python wheel 复用主页 Conda 依赖，安装到私有target目录，已核对171个
recon-all源码文件。两例原始T1空目录整例已完成：3440.429→2248.712秒、3668.963→2322.500秒，分别减少34.64%和36.70%。两例三方比较亦完成：最终几何、分割和脑区统计相对基线无变化，顶点图浮点差通过既有门槛；严格135/138的三个失败仅为MNI头字节。相对官方的既有差异、局部穿越和未判定的整体等效单列。完整报告见[当前结果](../../validation/recon_all/optimizations/20261002_parallel/FINAL_RESULTS.md)。冻结阶段或MRI前缀表面链不代替整例。

## Python 输入、输出和参数

```python
from fnit.recon_all.native_free import run_recon_all_python

report = run_recon_all_python(
    t1="/data/sub-01_T1w.nii.gz",  # 原始单 T1，NIfTI；保持真实原始网格
    subject_dir="/data/fnit/sub-01",  # 新空输出目录；失败保留已产生的诊断
    weights_dir="/data/fnit_weights",  # 已按资源清单验证大小和SHA-256的权重
    assets_dir="/data/fnit_assets",  # GCA、表面图谱和模板，保持授权来源
    native_bin_dir="/data/fnit-native/bin",  # 本轮用已核验的私有native bundle；正常安装可用None自动发现
    device="cuda:0",  # 明确目标设备；不自动启用FP16/BF16
    threads=4,  # 总线程预算；双半球各2线程
    hemisphere_workers=2,  # 独立exec和私有目录，保留共享文件发布顺序
    native_optimizations="auto",  # auto在CUDA上启用FNIT CUDA候选评分；original为Conda GCA；torch强制启用候选评分
    wm_backend="native",  # torch启用FNIT PyTorch/CPU有序混合WM分割
    profile_stages=True,  # 本次配对验证同步计时；日常调用可设False
    cuda_allocator_cache="auto",  # 已初始化API保留实际分配器；新进程保留低显存策略
)
```

`native_optimizations` 默认 `auto`：CUDA 设备优先使用 FNIT PyTorch 分块 GCA
候选评分；CPU 或显式 `original` 保持原生评分；GPU 不可用时不自动切换到
不完整后端。`cpu_cached` 仍只在显式的 CPU 受控选择中按 capability version 2、
`reduction=upstream_ROMP_partials` 和4线程启用。`original` 固定原始原生评分及放置，用作同程序
控制。标准pial始终使用保留的 `mris_place_surface`；独立程序
`mris_place_surface_white_fast` 仅供white.preaparc及最终white。
已安装快速white程序能力或固定源码版本不符时明确失败，不静默混用。

`native_optimizations="torch"` 只替换 GCA 的候选平移/线性搜索：分块
`grid_sample`、似然评分和设备端归约运行在 PyTorch CUDA，EM 精修和 LTA 写出
仍由 FNIT Python FP32 完成。它不读取官方 LTA，也不声称完整 CUDA EM 等价；
真实 sub-07 同输入矩阵与 CPU 路径一致，速度收据需绑定当前硬件和共享负载。

`wm_backend="torch"` 使用仓库已有 `mri_segment.py`：向量化分类可放在 CUDA，
FreeSurfer 扫描顺序的直方图、连通组件和有序填充仍在 CPU，随后交给现有
`mri_edit_wm_with_aseg`。该模式是显式混合迁移，默认仍为 `native`，在当前
真实 T1 完成逐体素比较前不会静默替换生产默认。

CUDA WM 两次后编辑调用成熟 `fix_ento_wm_gpu`，输入/输出仍为对应 conformed
MRI 网格的MGH/MGZ，保存原强度dtype和几何；完整分割及aseg编辑核心保留
Conda程序。此前多字节MGH上传的端序问题已经在子函数修复，不改变体素值。
表面保持surface RAS/mm和有序面的对应关系，面积mm²、体积mm³、厚度mm。

返回字典和 `fnit-native-free-run.json` 提供输出路径、阶段墙钟、实际原生选择、
程序SHA、GPU精度、线程和执行/完整性/网格状态；异常明确抛出，失败信息尽量
写入同一报告。`native_optimizations` 不改变参数、步长、停止规则或输出范围。
GPU后编辑没有静默CPU回退；显式CPU调用继续使用原有CPU后编辑。

CUDA 的 MNI 非线性阶段继续使用已有 SynthMorph GPU 模型，转换、完整逆场和
最近邻检查图接入自有 GPU 后处理；CPU 使用 Conda 路径。输出仍为 1 mm 网格、
NIfTI world RAS 位移（mm），没有用负前向场替代求逆。报告记录实际后处理后端
及外层模型调用的实际精度。内部保留变形网络的两次前向；逐前向 hook 的同输入验证另见任务5报告。模型保留已验证的 FP32 例外，其他阶段 TF32 不变。

## CLI 与原软件

```bash
# 显式2个半球worker，总线程仍为4；剖析包含读写与CUDA同步。
python -m fnit.recon_all.native_free /data/sub-01_T1w.nii.gz /data/fnit/sub-01 \
  --weights-dir /data/fnit_weights --assets-dir /data/fnit_assets \
  --device cuda:0 --threads 4 --hemisphere-workers 2 \
  --native-optimizations auto --wm-backend native --profile-stages
```

完整流程参考为 `recon-all -s SUBJECT -i T1 -all`。优化只是对应完整
`mri_em_register` 和 `mris_place_surface --white` 内部的重复工作消除，
没有独立等价的官方GPU命令；pial的完整命令参数见
[原生pial说明](NATIVE_PIAL_PLACEMENT.md)。系统官方参考程序仅在benchmark路径运行；生产使用FNIT独立源码构建程序。

## 当前已测范围与剩余验证

- MRI前缀表面链测量 `db7af9b`：2530.09→1420.96秒；不是原始T1整例。
  278条已比较脑区面积、no-TH3体积、厚度和顶点数零差异，严格复现失败。
  同算子的未预声明容差诊断与已预声明检查分列，不追认门槛。
- GCA v3同程序完整注册：219.740→187.703秒、185.731→163.217秒；两例
  LTA矩阵零差异。详见本轮task_03版本绑定JSON和RESULTS。
- 单例冻结左半球white两阶段候选与未改重建控制全部坐标/诊断输出一致。
  pial单次比当前程序慢约4%，保留原程序，不以局部GPU算子时间代替完整阶段。
- 两例两轮 MNI 完整阶段（新模型加载、自产变形场、转换、求逆和检查图）提速
  3.25–3.61倍，数组、几何、dtype和位移语义全部一致；显存采样峰值
  13,103,005,696字节。测试来自任务5的固定源码版本，整例完整链耗时另见当前结果，不能将冻结阶段倍数外推。
- 两例原始T1基线已从空目录连续完成，覆盖已初始化CUDA API及CLI：整例
  命令墙钟为3440.429秒、3668.963秒，均有138/138输出；同期父子进程显存
  采样峰值12,027,166,720和6,142,558,208字节。基线绑定`6f67cc0`，
  候选绑定`8d750e2`；两例候选完整命令墙钟2248.712秒、2322.500秒，均138/138输出齐全、生产网格检查通过。候选同期采样峰值8,749,318,144和10,896,801,792字节，原始报告与阶段CSV全部保存。
- 官方归档sub01在gpucw1为6790秒，sub02在nodecw10为4143秒，Synth模型
  使用CPU，`-parallel -openmp 4`可能使双侧总预算达到8线程。这些不是本轮
  同资源配对时间，不用于声称FNIT相对官方的整例提速。
- 整体科学指标等效没有正式门槛，保持`not_assessed`。显存上述数值是采样
  最大值，不是连续峰值保证；新Conda完整安装和物理隔离环境未完成验证。

## 版本与参考

本页说明已验证的接口和算法接入；实际整例、严格诊断与指标依据分别见当前结果。五任务报告位于
`validation/recon_all/optimizations/20261002_parallel/`，各报告保留实际测试提交
和源码/资源/程序SHA。安装路径见
[Conda原生安装](CONDA_CPP_STAGES.md)。

原实现：[FreeSurfer固定源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)、
[FNIT](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit)。
参考：Fischl, FreeSurfer, NeuroImage 62 (2012), 774–781。
