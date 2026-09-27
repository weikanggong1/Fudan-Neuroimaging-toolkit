# 候选 LH `sphere` 完整单阶段验收

2026-09-27 在 gpucw1 的同一真实 T1 候选前缀上，分别对保存的 FreeSurfer 8.2 原生输出和修复后的 Python 输出按顶点索引比较。没有重跑完整 `recon-all`。输入 `lh.inflated`、`lh.smoothwm` 是同面序的 FreeSurfer 三角表面（均 106,622 顶点、213,240 面），SHA-256 分别为 `f3a2128de469dc3c8d06e2649f85f420b26c84f95aea3c7912f677b4e57afd30` 和 `6d1a5d30639f3678c2e825d69b816f9bc35d5411546145f6c0bcab29a5fd0b91`。[官方输入副本审计](official_input_provenance.json)确认两份复制输入逐字节相同。修复后的度量源 SHA-256 为 `dde4f42db3f8f526c2ccb1c1379db4edc5369b0f41cbb72ed8b84dc14030ca92`；本轮实际载入的 12 个球面模块哈希见[源码清单](implementation.sha256)。 集成到当前分支后，[源码转移核对](integration_source_scope.json)显示 10/12 个模块字节完全相同；另两个差别分别限于函数文档字符串，以及模块文档字符串和独立 CLI 参数。两者参与球面计算的函数体 AST 相同。本完整阶段结果继承的是相同计算路径，不是集成分支重新运行一次。

## 用法与输出

Python API `run_standard_sphere(inflated, smoothwm, output, finish_device="cpu") -> dict` 读取两张同面序表面，写出 `surf/lh.sphere`：`float32` 有序球面坐标 `106622×3`、`int32` 有序三角面 `213240×3`，另返回输入/输出路径、初始负面积比例、目标度量耗时、每次优化的阶段/权重/平均次数/步长/耗时、最终负三角面数轨迹和总耗时。本轮只验证 `sphere`，没有生成 `sphere.reg`、脑区 annotation 或皮层厚度。

```python
from fnit.recon_all.sphere_standard_run import run_standard_sphere

report = run_standard_sphere(
    inflated="surf/lh.inflated",  # 候选膨胀表面，决定初始球面
    smoothwm="surf/lh.smoothwm",  # 同面序白质平滑表面，决定原始距离
    output="surf/lh.sphere",      # 输出有序 FreeSurfer 球面文件
    finish_device="cpu",          # 只控制末尾重叠修复设备
)
```

独立 CLI 为 `python -m fnit.recon_all.sphere_standard_run surf/lh.inflated surf/lh.smoothwm surf/lh.sphere --finish-device cpu --report report.json`；三个位置参数依次是上述两张输入与输出路径，`--finish-device` 仅控制最后修复，`--report` 保存 API 返回字典。等价原生命令为 `mris_sphere -threads 4 -seed 1234 surf/lh.inflated surf/lh.sphere`；原生命令会从同一 subject 的 `surf/lh.smoothwm` 读取目标度量。FNIT 生产运行不调用预装 FreeSurfer。

## 实测精度与时间

| 输出及同索引比较 | 修复后 Python | 同候选输入官方 |
| --- | ---: | ---: |
| 优化更新次数 | **248** | 248（由末尾修复起始编号核对） |
| 有序坐标逐位相同 | **319,866 / 319,866** | 参考 |
| 有序面 | **完全相同** | 参考 |
| 两者之间的均值/P99/最大位移，mm | **0 / 0 / 0** | 0 |
| 对归档官方的均值/P99/最大位移，mm | **2.959344 / 5.437959 / 7.066581** | 2.959344 / 5.437959 / 7.066581 |
| 完整单阶段墙钟时间 | **884.56 s**（内部计时 879.80 s） | **328.61 s** |

修复后 Python 比同输入官方慢 **2.69 倍**，仍是 CPU/Numba 主优化；末尾 Torch CPU 修复耗时 8.70 s，原始度量/JIT 37.99 s。两次运行在共享节点不同时间完成，时间只作观察值。Python `/usr/bin/time -v` 峰值 RSS 773,672 KiB；官方计时文件为 217,164 KiB。修复后 248 次更新及 14 个末尾负面数 `[29,9,9,12,7,8,5,6,5,6,2,4,3,1]` 与[官方日志](candidate_sphere_official_control/mris_sphere.log)一致。精度和哈希见[三方审计 JSON](audit.json)、[Python 逐步报告](report.json)、[Python 计时](time.txt)及[官方计时](candidate_sphere_official_control/time.txt)。

修复后 `lh.sphere` 的文件 SHA-256 为 `c3780d38be3a50cd5e33c021949a9f1712d2921c908fb9cdc5f65def2ececcee`，同输入官方为 `150d1b88b00961069227ae84356950a88b7df330c1a9879a0892f69db62e4feb`。有序坐标与面在文件中的**原始几何载荷 SHA-256 也相同**：`7d9b1339fb7abdf909b1bb377e1c6853c0d38cfe34ddce18e118dca3eb961da2`。全文件差别来自创建行和尾部元数据：Python 的 611 字节尾部与官方尾部开头逐字节相同，官方其后还有 1,779 字节程序来源标签。这个差别不影响表面几何读数。

相对 2026-09-24 归档官方 `lh.sphere` 的 2.959344 mm 均值差异在同候选输入官方与归档官方之间也**完全相同**。因此本轮已把该差异定位到 `sphere` 的上游输入；[上游逐阶段审计](UPSTREAM_FIRST_DIFFERENCE.md)已定位第一张不等的表面为 `white.preaparc`，不能把归档结果的差异归咎于修复后的 Python `sphere`。完整 `sphere.reg`、annotation、最终 white/pial 和皮层统计尚未由本轮验证。

`audit_full_sphere.py` 的必填 CLI 参数为 `--inflated`、`--smoothwm`（候选输入），`--python-sphere`（本轮输出），`--same-input-official`（同候选官方），`--archived-official`（原始归档官方），`--python-report`（API JSON），`--native-time`（官方计时文件），`--report-json`（输出三方比较 JSON）。JSON 含输入/输出 SHA、更新数、三对同索引误差、几何载荷及尾部哈希和时间；不复制影像或表面数据到 Git。
