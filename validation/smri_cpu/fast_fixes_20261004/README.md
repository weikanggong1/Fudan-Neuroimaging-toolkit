# TorchFAST：CPU 标量数学精度修复

[功能、参数与调用](../../../docs/fast/README.md) · [本轮完整记录](results/report.json) · [上一版 CPU 审计](../../smri_cpu_20261004/task04/README.md)

## 功能与原因

本版修复 `execution="fsl"` 的 CPU 指数和对数计算。上一版保留了 FAST 的原位扫描、随机流与 PVE 候选顺序，但 Torch 向量指数与系统 C `expf` 的末位差异会改变校正后的线性强度，再改变组织矩估计和少数离散 PVE 选择。

真实完整输入的 34 次 moments/bias/PVE 跟踪中，前 32 次记录相同；首次分歧是线性强度的 `_fsl_moments` 输入。使用标量 C 数学后，本例默认和已测试非默认配置的八张输出都与官方逐体素相同。该结论对应本输入及配置，其他原软件模式仍按功能说明中的支持范围判断。

CPU 通过 Numba 对独立元素并行调用系统 `expf/logf/exp/log`，保持 FP32/FP64 的原有使用位置，关闭 `fastmath`。不改变扫描依赖和 Torch 全局函数。运行时不调用 FSL、不加载 FSL 库；Numba 仅在实际 CPU 数学调用时导入，已在主页 Conda 配置中，`libm` 来自系统 C 运行库。CUDA 使用原来的 Torch 数学算子，默认 `tensor` 算法也保留原计算规则。

## 真实输入与协议

输入为公开 OpenNeuro ds003138 一例 T1 的 SynthStrip brain，224×288×288、2,843,038 个正脑体素。两臂使用同一脑图，SHA-256 为 `fd8c3253a6960777715c85844a38e760959b6bf80849bd28b7e5b906dee0dea4`。FNIT 显式 mask 与正输入取交集；未使用官方分割初始化。

CPU 主机为 nodecw7，Intel Xeon Gold 6418H。原版 FSL 6.0.7.4 与 FNIT 均限制在相同八个物理核心 `3,7,11,15,19,23,27,31`，各库线程预算为 8，同锁串行。默认顺序为官方→FNIT→FNIT→官方；随后一组非默认官方→FNIT。墙钟含新进程启动、读写和八张 gzip 输出，排除排队等待和事后评分。原 FAST 主要单线程，此处比较的是相同八核资源预算。

节点是共享的，测量前后 load 约 82–123；每次完整时钟、RSS 和负载分别保存在 JSON。旧 nodecw10 的时间没有与本次混算。

## CPU 完整命令行结果

| 配置 | 官方冷进程 | FNIT 冷进程 | 精度 |
|---|---:|---:|---|
| 默认，ABBA 两次 | 411.291 / 368.198 s | 110.173 / 100.174 s | 八图全部逐体素相同，affine 相同 |
| `-N -W 5 -I 2 -O 2 -f 0 -H 0 -R 0` | 119.226 s | 53.592 s | 八图全部逐体素相同，affine 相同 |

默认两侧中位数比为 **3.706 倍**，非默认单组为 **2.225 倍**。默认 FNIT 采样进程树 RSS 为 3.104–3.290 GB，官方为 2.240–2.282 GB；不把更快计算说成更低内存。

八图为 CSF/GM/WM PVE、seg、pveseg、mixeltype、bias、restore。四组逐图比较的不同体素、最大绝对差和 RMSE 均为 0。修复前默认 PVE 分别有 5/27/22 点差异、pveseg 有两点差异，这些差异已在本输入上消除。

![同一真实脑图的 GM PVE 与零差图](results/gm_pve_match.png)

图为固定中央切面；全部体素的结论来自逐图评分。原始 MRI 不随报告发布。图由已有本地绘图环境生成，服务器冻结前缀缺少 matplotlib；主页 Conda 和 Python 依赖已包含它，运行中的前缀未修改。

## GPU 回归

在同一 H100、FP32、20 GB allocator 预算下，完整真实脑图的 `tensor` 和 `fsl` 各八图，在旧/新版本之间全部数组 SHA、完整 NIfTI header 和 affine 相同。峰值 allocated/reserved 也相同：`tensor` 为 3.836/5.505 GB，`fsl` 为 2.603/5.505 GB。

最终代码还将 Numba 导入延迟到实际 CPU 标量调用；数学规则没有改变。该源码另做一次完整 CPU 默认 CLI，105.840 s，八图仍与已保存官方参考逐值相同。此前六次 CLI 的时间仍绑定早期候选，不把更新后的源码身份写到旧测量上。

| 完整脑 GPU API | 第一组旧→新 | 最终源码新→旧→旧→新 |
|---|---:|---:|
| `tensor` | 3.850 / 3.840 s | 3.838 / 3.446 / 3.751 / 3.849 s |
| `fsl` | 21.205 / 23.298 s | 21.713 / 22.989 / 23.275 / 22.333 s |

共享 GPU 的耗时有波动，两组 `fsl` 的快慢方向不同，不能据单组宣布稳定提速或减速。生产 CUDA 内核及数学规则保持原样；默认 `tensor` 计算路径未改。完整科学输出和内存回归已通过，计时按原值保留。

中央 64³ 子区域也完成额外回归：两个后端的八图 SHA 相同，allocated/reserved 为 `tensor` 57.433/81.789 MB、`fsl` 40.937/81.789 MB。其编译缓存建立后的旧/新 API 为 `tensor` 1.015/1.014 s、`fsl` 2.974/2.842 s；旧首次 `fsl` 24.534 s 含首次编译，不用它计算提速。子区域不代替完整脑验收。

## 复现与源码

单被试 CPU 调用仍使用公开接口：

```python
from fnit.fast import TorchFAST

fast_model = TorchFAST(device="cpu", threads=8, execution="fsl")
fast_result = fast_model(image="T1_brain.nii.gz", mask="brain_mask.nii.gz")
fast_result.pve_gm.save(path="results/T1_brain_pve_1.nii.gz")
```

```bash
fnit fast -i T1_brain.nii.gz --mask brain_mask.nii.gz \
  -o results/T1_brain --device cpu --threads 8 --execution fsl -b -B
# 独立的原软件对照；FNIT 本身不执行这条命令
fast -t 1 -n 3 -b -B -o original/T1_brain T1_brain.nii.gz
```

[标量数学桥](../../../src/fnit/fast/_fsl_math.py)与[组织分割](../../../src/fnit/fast/algorithm.py)的实际 SHA 绑定在记录中。[诊断脚本](scalar_math_diagnostic.py)在冻结 `f1cbdab1` 源码的独立进程内比较 Torch/libm；它不用于产品运行，也不全局修改已安装 FNIT。[GPU 回归脚本](gpu_regression.py)的 `--size 0` 保留完整输入，`--size 64` 明确使用真实空间子区域。

[评分脚本](collect_results.py)接收完整运行目录；`--skip-figure` 可仅收集数值记录。最终代码的 FAST 单元测试共 38 项通过。

## 最近版本与参考

| 版本 | 实测变化 |
|---|---|
| 本版 CPU 标量数学修复 | 上述默认/非默认八图与官方逐体素相同；同节点八核计时另列 |
| `f1cbdab1` | 原序 Numba 扫描已接入；默认仍有少数 PVE/bias/restore 差异 |
| 较早原序 PyTorch 版本 | 算法依赖已对齐，但 CPU 小张量派发占主要时间 |

- [FAST 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fast.html)、[FAST4 代码库](https://git.fmrib.ox.ac.uk/fsl/fast4)。
- Zhang, Brady & Smith (2001), IEEE TMI, [doi:10.1109/42.906424](https://doi.org/10.1109/42.906424)。
