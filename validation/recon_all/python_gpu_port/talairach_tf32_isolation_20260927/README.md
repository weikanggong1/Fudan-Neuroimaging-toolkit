# Talairach 仿射精度：同一真实 T1 的局部测试

本报告只检查 Talairach 仿射及有限的下游步骤，不是 recon-all 整例验收。官方参照是同一 T1 的 FreeSurfer 8.2 结果。输入、权重、模板、源码及候选输出的 SHA-256 见 [`precision_report.json`](precision_report.json) 和 [`connected_report.json`](connected_report.json)；后续 GCA 检查见 [`gca_gate_report.json`](gca_gate_report.json)。对应官方命令为 `mri_synthmorph -m affine -t aff.lta synthstrip.mgz mni305.cor.stripped.mgz -j 4`。候选实现调用 `fnit.synthmorph.SynthMorph(..., model="affine", extent=256)`。官方体积图和变换没有作为候选输入。

## 固定输入的仿射比较

各次运行使用同一份已保存的候选 `synthstrip.mgz`、MNI305 模板和 affine 权重，源码哈希见 JSON。cuDNN benchmark 和 deterministic 标志均为 true，与前一步 SynthStrip 的状态一致。每行只计时一次；最后一行是默认设置的热启动重复，因此不能将这些数字当成配对速度基准。

| 运行方式 | matmul TF32 | cuDNN TF32 | 推理秒数 | voxel LTA 最大元素误差 | eTIV 误差 mm³ |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU float32 | 关 | 关 | 17.836 | 0.00012255 | +0.48617 |
| GPU 默认，首次 | 开 | 开 | 2.200 | 0.04733276 | −822.54696 |
| GPU float32，两项均关 | 关 | 关 | 4.025 | 0.00008392 | −0.89466 |
| GPU，仅关 matmul | 关 | 开 | 1.511 | 0.00392628 | −3.52243 |
| GPU，仅关 cuDNN | 开 | 关 | 1.483 | 0.07858276 | −826.74013 |
| GPU 默认，重复 | 开 | 开 | 1.412 | 0.04733276 | −822.54696 |

GPU 默认所得 world affine 与先前 v3 候选完全相同。只在 affine 推理期间关闭两项 TF32 后，eTIV 绝对误差由 822.547 降至 0.895 mm³，约缩小 919 倍；LTA 和 eTIV 仍未与官方逐位相同。上表的隔离试验当时未修改生产代码；生产入口的改动见下文。

## 从原始 T1 连到 brainmask

另一次连通试验从同一原始 T1 出发，运行候选 SynthStrip 和 Talairach；仅在 SynthMorph affine 前向期间关闭两项 TF32，再运行候选 `nu`、`T1` 和 `brainmask`。N4 不读取 Talairach 变换，因此复用此前保存的**候选** `nu0.mgz`。该试验没有运行 GCA 注册或后续步骤。

| 候选体积图 | 相对官方的差异体素 | 仿射 / MGH 头 |
| --- | ---: | --- |
| `orig.mgz` | 0 / 16,777,216 | 相同 / 相同 |
| `synthstrip.mgz` | 0 / 16,777,216 | 相同 / 相同 |
| `nu.mgz` | 0 / 16,777,216 | 相同 / 相同 |
| `T1.mgz` | 0 / 16,777,216 | 相同 / 相同 |
| `brainmask.mgz` | 0 / 16,777,216 | 相同 / 相同 |

该连通试验的 voxel LTA 最大元素误差为 0.00008392；eTIV 为 1,310,265.657875 mm³，官方为 1,310,266.552537 mm³。输入至 Talairach 用时 13.435 秒，其中 SynthStrip 6.540 秒、Talairach affine 3.983 秒；从保存的 `nu0` 生成 `nu` 用时 1.818 秒，`T1` 归一化 62.867 秒，`brainmask` 0.983 秒。MGZ 历史元数据可使整文件哈希不同，故分别比较了体素数组、仿射和 MGH 头前 284 字节。

`probe_precision.py` 是可复用 CLI，位置参数依次提供 moving 图像、模板、权重目录、官方及先前的 affine/voxel LTA、输出目录。固定输入结果使用等价的临时脚本，脚本 SHA-256 记在 `precision_report.json`。`probe_connected.py` 是连通试验实际脚本，输入 T1、空候选被试目录、权重目录、模板目录、保存的候选 `nu0.mgz` 和只读官方被试目录。两者均支持 `--device cuda:0 --threads 4`；官方路径只供比较读取。

## GCA 至 presurf 的有限续跑

`probe_gca_gate.py` 接着运行上述候选被试。它调用与 v3 整例相同的固定 Conda `mri_em_register` 程序（哈希见 JSON）、候选 Python `run_ca_normalize` 和候选 Python 胼胝体分割。为避免重复网络推理，复用同一 T1 已保存的**候选** `synthseg.rca.mgz`；使用前已独立核对它与官方的差异。官方影像、标签、变换和表面均未作为重建输入。等价注册命令为 `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz RB_all_2020-01-02.gca transforms/talairach.lta`。

| 输出 | 对照结果 |
| --- | --- |
| GCA `talairach.lta` | 16 个矩阵元素全部相同 |
| `norm.mgz`、`ctrl_pts.mgz` | 各 0 / 16,777,216 差异体素；仿射和 MGH 头相同 |
| 已保存的候选 `synthseg.rca.mgz` | 0 / 16,777,216 差异体素；仿射和 MGH 头相同 |
| `aseg.auto_noCCseg.mgz`、`aseg.auto.mgz`、`aseg.presurf.mgz` | 各 0 / 16,777,216 差异体素；仿射和 MGH 头相同 |
| `cc_up.lta` | 最大矩阵元素误差 0.00000763，未逐位相同 |

本次共享节点上，GCA 注册、CA 归一化、胼胝体及复制步骤分别用时 329.526、37.143、25.454 秒。v3 旧运行对应为 251.249、27.048、24.438 秒；节点负载不同，不能计算受控速度比。SynthMorph LTA/eTIV 的改进不改变 GCA 注册：GCA 读取 `nu`、`brainmask`，另写 `talairach.lta`。此 T1 的 GCA 到 `aseg.presurf` 数值匹配，`cc_up.lta` 仍有小误差；WM、表面、脑区统计和其他被试均未纳入该试验。

## 生产入口检查

随后修改 `register_talairach`：仅在 affine 前向期间关闭 matmul 和 cuDNN TF32，并在 `finally` 中恢复进入函数时的标志，异常退出亦然。包内其他 SynthMorph 调用仍按原默认设置运行。四个针对性测试通过，覆盖成功推理和异常时的标志恢复。批量入口每例使用独立进程，这两个进程级标志不会跨被试工作进程共享。

修改后的生产函数对**完全相同的已保存 `synthstrip.mgz` 字节**、权重和模板调用一次，affine LTA 的 SHA-256 与隔离试验的 GPU 双关结果相同。相对官方，world/voxel LTA 的最大元素误差分别为 0.00001281 / 0.00008392，eTIV 误差 −0.89466 mm³；调用前后两项 TF32 标志均为 true。构造模型、推理并写入 affine LTA 和 XFM 的一次墙钟为 4.791 秒，PyTorch GPU 峰值分配 4497 MiB。输入和模块哈希见 [`production_register_report.json`](production_register_report.json)。这次没有重复下游步骤，也没有做配对官方耗时测试。
