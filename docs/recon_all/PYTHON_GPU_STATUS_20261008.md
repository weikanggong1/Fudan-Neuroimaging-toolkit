# recon-all 纯 Python GPU 迁移状态

本页绑定当前工作树提交，说明严格 `backend="python-gpu"` 的开放条件。当前默认入口仍是经过真实 T1 验证的混合流程；它不能因为选择 `cuda:0` 就被称为纯 GPU。

## 已接入且可在 CUDA 上执行的阶段

| 阶段 | 当前实现 | 说明 |
|---|---|---|
| conform、SynthStrip、Talairach affine、SynthSeg、brainmask | FNIT PyTorch CUDA | 保留 SynthStrip/SynthSeg/Talairach 的 FP32 例外，其他矩阵计算遵循默认 TF32 |
| MNI 辅助与非线性后处理 | FNIT PyTorch CUDA | 不读取官方输出；变换写出和逆场检查仍计入阶段墙钟 |
| finalsurfs 体积 mask、EntoWM/ACJ 编辑 | FNIT PyTorch CUDA | 2026-10-08 起由 recon-all 将目标设备传给 `finalsurfs_python`，CPU 入口仍使用原 Python 实现 |
| 表面指标与 Jacobian | FNIT PyTorch CUDA（指标）；Jacobian 默认 CPU | 厚度、面积、曲率和顶点体积保留 GPU；真实 114k 顶点配对中 Jacobian CUDA 1.155 s、CPU 0.079 s，故 GPU 仅作显式实验后端 |

## 仍阻断严格纯 profile 的阶段

N4、GCA/EM 注册、完整 WM segmentation 和 aseg 编辑、拓扑 GA、inflate/remesh/自相交修复、white.preaparc、最终 white、pial、缺陷投射及曲率统计仍没有经过真实同输入和自产链验证的完整 Python/CUDA 等价实现。现有 `n4_gpu.py` 是非 ITK 近似，`mri_em_register_score_gpu.py` 只加速候选打分，`place_white_preaparc_python.py` 是 1–17 步前缀，不能直接替代生产阶段。

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

上述数字来自绑定旧提交的真实收据，不重标为当前提交的新整例。当前本轮只完成设备接线与严格 profile 守门，没有声称 10 分钟已达到。完整 CUDA kernel 分解和干净环境整例仍需在所有阻断阶段完成后测量；显存继续按同期父子进程采样并以 20,000,000,000 字节为预算。

## 迁移顺序

先完成 N4/EM 的 GPU 优化器并做体素、LTA、后续 norm 的同输入回归；再处理保持扫描顺序的 WM/aseg 操作；最后迁移动态拓扑、white/pial 和 sphere/register。每一阶段通过冻结输入、FNIT 自产前段和原始 T1 空目录三层验证后，才能从 capability matrix 的 blocked 移入 ready。
