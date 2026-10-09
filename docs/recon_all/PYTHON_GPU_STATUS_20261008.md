# recon-all 纯 Python GPU 迁移状态

本页绑定当前工作树提交，说明严格 `backend="python-gpu"` 的开放条件。当前默认入口仍是经过真实 T1 验证的混合流程；它不能因为选择 `cuda:0` 就被称为纯 GPU。

## 已接入且可在 CUDA 上执行的阶段

| 阶段 | 当前实现 | 说明 |
|---|---|---|
| conform、SynthStrip、Talairach affine、SynthSeg、brainmask | FNIT PyTorch CUDA | 保留 SynthStrip/SynthSeg/Talairach 的 FP32 例外，其他矩阵计算遵循默认 TF32 |
| MNI 辅助与非线性后处理 | FNIT PyTorch CUDA | 不读取官方输出；变换写出和逆场检查仍计入阶段墙钟 |
| finalsurfs 体积 mask、EntoWM/ACJ 编辑 | FNIT PyTorch CUDA | 2026-10-08 起由 recon-all 将目标设备传给 `finalsurfs_python`，CPU 入口仍使用原 Python 实现 |
| 表面指标与 Jacobian | FNIT PyTorch CUDA（指标）；Jacobian 默认 CPU | 厚度、面积、曲率和顶点体积保留 GPU；真实 114k 顶点配对中 Jacobian CUDA 1.155 s、CPU 0.079 s，故 GPU 仅作显式实验后端 |
| WM/aseg静态编辑 | FNIT PyTorch CUDA＋有序Numba CPU，显式torch-hybrid | 两例完整同输入0体素差异，34.4→8.2秒、34.6→7.9秒；有序核心仍为CPU，原始T1链未完成 |
| 标准sphere法向 | FNIT有序Torch，显式后端 | 单左侧完整阶段170.63→133.57秒、文件SHA相同；其余优化和finish仍有CPU计算 |
| 缺陷体积投射 | 完整FNIT PyTorch，显式torch后端 | 两例双侧投射标签、Dice及空间一致；约3秒的GPU API无稳定提速，不是拓扑GA |
| 主曲率衍生图 | 固定K1/K2的FNIT PyTorch | BE/C/FI/S已验证，完整离散主曲率和curv.stats仍由原生计算 |
| 完整固定N4配方 | FNIT PyTorch实验接口 | 一例200次自产反馈4.332秒；有方向性强度差，未改默认；第二例及下游尚未验证 |
| WM局部直方图 | FNIT PyTorch显式后端 | 两例两遍完整候选0差异，局部103–125倍；完整WM文件API与整例尚未完成 |
| 离散K/H及八图 | FNIT PyTorch实验接口 | 两例双侧K逐位一致，H/主曲率尾差；原探索门未全通过，标准curv.stats未替换 |

## 仍阻断严格纯 profile 的阶段

N4、完整GCA/EM注册、完整WM segmentation和有序aseg核心、拓扑GA、inflate/remesh/自相交修复、完整white及离散曲率统计仍阻断严格纯profile。
完整Python pial已存在，但有序顶点更新/碰撞仍在CPU；本轮GPU正则项完整pial
1396.651→1483.542秒、慢6.22%，不据局部内核收益设默认。它与官方仍有既有局部差异。
缺陷投射完整Torch已接入显式后端，但自产原始T1整例验证尚未完成。
现有 `n4_gpu.py` 是非ITK近似，不能用作精准替代；新的
`n4_itk_torch_experimental.py` 已运行完整固定配方，见[完整N4实测](N4_COMPLETE_TORCH_20261009.md)，
其中uint8差3808体素均为-1、脑内1932点，仍需定位和下游验证，未当作随机尾差。
`mri_em_register_score_gpu.py`只加速候选打分，Python white是首轮前缀，不能冒充完整阶段。

因此新增 `backend="python-gpu"`（或 CLI `--backend python-gpu`）会在创建被试输出前返回结构化 `PurePythonGpuUnavailable`，列出阻断阶段；它绝不会悄悄回退到 Conda 原生程序，也不会生成近似占位文件。`backend="native"` 保持现有可验收路径。

```python
from fnit.recon_all.native_free import run_recon_all_python

report = run_recon_all_python(
    t1="/data/sub01_T1w.nii.gz",       # 原始单幅 T1w NIfTI
    subject_dir="/data/subjects/sub01", # 必须为空
    weights_dir="/data/fnit-weights",   # 已校验权重
    assets_dir="/data/fnit-assets",     # 已校验模板/图谱
    device="cuda:0",                    # 纯 profile 必须显式 CUDA
    backend="python-gpu",               # 所有阶段完成后才会实际运行
)
```

## 性能边界

冻结三号基线的九例原始 T1 FNIT 中位端到端时间为 **2669.520 s**；官方参考中位数为 **5839.091 s**。超过 100 s 的主要阶段为 surface group（791–1041 s）、finish surface（369–662 s）、sphere/register（233–368 s）、annotation（119–190 s）、N4（125–132 s）和 GCA/EM（120–214 s）。即使表面和注册相关阶段理想化为零，前后段仍约 900–1000 s；达到 600 s 需要全流程至少约 4.45 倍加速。

上述数字来自绑定旧提交的真实收据，不重标为当前提交的新整例。
2026-10-09更新已包括上表同输入阶段转写和回归；原始T1空目录control/candidate
控制保存42阶段后GPU节点SSH不可达，原会话结束、无CLI返回码；候选未启动。
当前没有新整例提速结论，也没有达到10分钟的证据。
完整CUDA及干净环境整例仍需继续测量；显存按同期父子进程采样，预算
20,000,000,000字节。新增后端与复现命令见 [本轮整例说明](TORCH_INTEGRATION_20261009.md)。

## 迁移顺序

先完成 N4/EM 的 GPU 优化器并做体素、LTA、后续 norm 的同输入回归；再处理保持扫描顺序的 WM/aseg 操作；最后迁移动态拓扑、white/pial 和 sphere/register。每一阶段通过冻结输入、FNIT 自产前段和原始 T1 空目录三层验证后，才能从 capability matrix 的 blocked 移入 ready。
