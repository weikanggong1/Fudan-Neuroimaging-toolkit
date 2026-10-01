# recon-all 五阶段优化的真实整例结果

生产候选为 `61926c7dceae2f9097fa296e306cf1efa0a09191`，源码归档 SHA-256 为
`6d231caff2f394b5f40eaf4119bcbc1a45e42a7387c90436c6ef8114ba9fc705`。
后续工作分支提交仅保存说明、诊断脚本和报告，生产源码树不变。
本页只记录实际完成的运行；第二例及数值比较仍在串行验证中。

## 整例墙钟与范围

所有新运行从原始T1和空输出目录开始，主机gpucw1，指定同一H100 PCIe的GPU UUID，
Torch intraop、Numba与OpenMP/BLAS预算为4；API调用方既有interop设置保留。
计时采用相同监控命令边界，包含导入、前置校验、模型、传输、计算和读写。
各内部步骤已包含在父阶段，不能重复累加。

| 输入与调用 | 旧版完整命令 | 候选完整命令 | 时间减少 | 输出 |
| --- | ---: | ---: | ---: | --- |
| sub-01，已初始化CUDA的Python API | 4242.884 s | 3557.969 s | 684.915 s，16.143%，1.193倍 | 66阶段完成，138/138存在 |
| sub-02，GPU CLI | 运行中 | 待运行 | 尚无结果 | 尚无结果 |

sub-01复用此前完成的 `c248520` GPU API基线；其生产src树与本轮起点 `0c8ab32`
完全相同，见[源码身份](baseline_source_identity.json)。这不是本轮新测基线，共享负载
和缓存可能不同。sub-02另测同GPU的旧版CLI，不能用此前CPU运行作为GPU基线。
单次共享主机观察不代表稳定吞吐。首例完整原始记录见[执行摘要](whole/sub01/execution_summary.json)、
[旧运行](whole/sub01/baseline_run.json)、[候选运行](whole/sub01/candidate_run.json)。

## 首例当前瓶颈

以下为当前原始T1整例实测，单位秒。括号中的内部耗时包含在外层阶段。

| 部分 | LH／单阶段 | RH | 实际实现 |
| --- | ---: | ---: | --- |
| N4 | 123.02 | — | 自有C++封装、ITK N4，独立Conda构建，CPU |
| GCA注册 | 215.73 | — | 固定FreeSurfer源码的Conda C++，CPU |
| MNI非线性 | 196.10 | — | 自有PyTorch CUDA网络＋Conda源码构建的warp/数值求逆 |
| 表面生成合计 | 683.58 | 639.51 | 自有Numba/PyTorch与Conda C++混合 |
| 其中remesh | 120.54 | 111.59 | 自有NumPy/Numba，CPU，保持动态拓扑和有序更新 |
| 其中white.preaparc放置 | 204.92 | 187.89 | 固定源码Conda C++，CPU |
| 其中标准球面 | 156.85 | 105.35 | 自有Numba目标函数＋既有PyTorch CUDA平均 |
| 球面配准 | 135.38 | 110.70 | 自有Numba/PyTorch CUDA，未更改已验证算法 |
| 最终表面合计 | 394.27 | 356.26 | Conda white/pial＋自有PyTorch GPU指标 |
| 其中最终white | 198.59 | 170.69 | 固定源码Conda C++，CPU |
| 其中pial | 191.94 | 182.16 | 固定源码Conda C++，CPU |

厚度、面积和曲率继续使用已有GPU实现：LH/RH厚度2.893/2.566 s，单张面积
0.011–0.025 s，单张曲率0.311–0.415 s。本轮没有重复实现它们。

## 阶段回归与数值状态

五阶段的冻结同输入回归见[串行说明](../../../../docs/recon_all/SERIAL_OPTIMIZATION.md)。
这些结果不能代替整例比较：

- 辅助模型旧GPU与新GPU输出相同；CPU迁往GPU的少量标签差异单列。
- 两例T1与第二轮brain归一化保存文件SHA相同；传播和平滑的有序规则不变。
- 两例双侧标准球面的有序坐标、面及完整优化轨迹相同。
- N4浮点和量化输出相同；拟合约120 s，重建约1 s。四线程重建没有实质提速，保留默认1线程。
- 完整Python pial仅sub-01 LH配对：1453.060→1232.634 s，减少15.170%，完整文件及41步/清理轨迹相同。
  同输入Python与官方坐标一致；当前Conda程序相对官方P99为0.2253 mm、最大1.4822 mm，原因尚未定位。
  Python仍慢于Conda的176.787 s，生产保留Conda默认，不将这次Python提速计入整例收益。

| 验收项 | 当前状态 |
| --- | --- |
| 执行完成与138项完整性 | sub-01完成；sub-02待完成 |
| 相对优化前严格138诊断 | 待两例整例后的只读比较 |
| 优化是否引入退化 | 待逐标签、同序表面、区域指标和扩展质量比较 |
| 与官方严格复现 | 待比较；既有精度问题没有声明已解决 |
| 整体指标等效 | not_assessed，尚无正式确认的整例门槛 |
| 连通性、非流形、球面翻折、white/pial穿越 | 标准生产检查已运行；扩展三组配对质量检查待运行 |

官方参考只用于独立比较，不进入生产输入。顶点对应不成立时使用双向点到三角面
距离，不将最近顶点距离称为逐顶点一致。脑区Dice、厚度/面积/体积及局部异常将
随当前版本比较图一并保存，不根据结果调整门槛。

## 显存、安装与失败尝试

sub-01候选父子进程同次查询合计最大采样值为 **16,185,819,136字节**，即16.186 GB、
15.074 GiB；旧基线为14,508,097,536字节。新运行请求1 s，3037个样本，最大间隔
3.557 s，失败查询0。低于20,000,000,000字节的是观察最大值，连续峰值未验证。
[原始采样](whole/sub01/candidate_gpu_samples.csv)、[监控摘要](whole/sub01/candidate_monitor.json)。
已初始化CUDA API的allocator状态明确记录为preserved_preinitialized_unknown；
PyTorch allocated/reserved不可用时不填零。CLI的实际策略另随第二例报告。

TF32默认保留。实际SynthSeg前向在cuda:0、float32、cuDNN TF32关闭、autocast关闭；
其他已验证FP32例外保持。所有实际Synth及辅助网络调用使用FNIT GPU实现，未用半精度。

新N4在现有主页Conda依赖内从源码构建，生产使用14项独立程序束；
[程序清单](native_bundle_manifest.json)和[当前资源指纹](runtime_fingerprints_61926c7.json)
保留大小、SHA及主机版本。无新增生产依赖。全新环境安装和没有预装脑影像软件的
隔离整例未在本轮验证，不能凭PATH或ldd宣布通过。

首个整例尝试在Talairach子进程CUDA模型分配处失败。相同原始T1、配置及初始化CUDA
调用在隔离输入链重放成功；retry1从新的空目录完整运行成功。未复现的根因保持未定位，
没有改用CPU、半精度或更换GPU。失败日志见[保留目录](whole_failed_attempt1/)。

## 复现

配置中的input为原始T1，output和diagnostic_root必须改为新目录。完整入口、空间、
返回值及失败行为见[recon-all说明](../../../../docs/recon_all/README.md)。

```bash
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/whole_sub01_candidate_retry1.json  # GPU API、代码归档、原始T1、资源和4线程
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/whole_sub02_baseline_retry1.json  # 同GPU旧版CLI基线，新的空输出目录
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/whole_sub02_candidate_retry1.json  # 同GPU候选CLI，新的空输出目录
python validation/recon_all/python_gpu_port/collect_hotspot_whole_comparison.py \
  --config /bench/whole_comparison_retry1_config.json  # 仅在整例完成后只读比较
# 两例完整原始报告和比较结果收回后汇总；输出路径必须不存在。
python validation/recon_all/optimizations/20261001_serial/summarize_whole.py \
  --reports validation/recon_all/optimizations/20261001_serial \
  --output /bench/serial_whole_summary.json  # 绑定输入、程序、代码及所有报告SHA
```

脚本为benchmark控制，没有独立官方等价CLI。对应单T1官方流程为
`recon-all -i T1w.nii.gz -s subject -all -openmp 4`，仅在参考路径使用官方程序。
各算法原源码和文献见[串行说明](../../../../docs/recon_all/SERIAL_OPTIMIZATION.md)。

| 版本 | 本轮记录 |
| --- | --- |
| 8302bdc | 全部辅助网络GPU及生命周期，阶段回归 |
| 198c309 | 有序归一化CUDA内核与两例相同文件回归 |
| 74ae022 | 标准球面CSR/SSE及复用GPU平均，四半球完整回归 |
| 3b5b104 | N4内部计时、线程能力及两例浮点/量化回归 |
| 61926c7 | pial完整候选批查询及编译碰撞；当前整例生产快照 |
