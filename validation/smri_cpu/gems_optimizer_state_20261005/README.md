# GEMS CPU：已保存 Double 与 FP32 优化状态的只读审计

## 1. 功能

复用[raw prior 37 步诊断](../gems_cpu_objective_20261005/README.md)保存的真实阶段 points／gradients，定位方向、曲率历史与 alpha 最早分叉。此次 **没有求 mesh objective，没有执行 native optimizer、EM 或新拟合**；生产和 GPU 完全未改。

从官方保存的 Double gradients 按固定原源码定义重建 L-BFGS direction／history，再以返回最大移动推断 alpha，并检查该 direction × alpha 能否产生实际保存的下一点。**这是 source-defined 重构和 alpha 推断，不是 native 内部 trace。** 官方 37 步的点增量最大坐标残差仅 `1.8474e-13` voxel；每步 private direction 同样逐值产生原已存 FP32 点。因此该证据支持方向公式一致，尚不证明两种内部精度／closure 的轨迹等价。

## 2. 输入、输出与调用

输入均为已存在的私密 checkpoint，格式与空间沿用上份报告。官方 initial／accepted points 和 gradients 为 FP64 数组；private points 为 FP32，gradients 由 FP32 leaf backward／投影后提升为 FP64 optimizer state。原点和原梯度数组不发布；公开结果为 `summary.public.json` 中的误差、历史标量与 SHA。

Python 内调用同一入口：

```python
import subprocess
import sys

# source_dictionary 只包含本次已冻结的 checkpoint/source 路径，示例不含影像数组。
subprocess.run([
    sys.executable, "trace_saved_directions.py",
    "--candidate", "../gems_cpu_objective_20261005/optimizer_candidate_v2.py",
    "--raw-first", "/private/completed-raw3/python",
    "--raw-continue", "/private/completed-raw37/python",
    "--native-run", "/private/completed-native37/native",
    "--native-first", "/private/validated-native-v5",
    "--capture", "/private/capture-hippo-amygdala-right",
    "--output", "/private/new-saved-state-audit",
], check=True)
```

命令行使用相同参数：

```bash
python trace_saved_directions.py \
  --candidate ../gems_cpu_objective_20261005/optimizer_candidate_v2.py \
  --raw-first /private/completed-raw3/python \
  --raw-continue /private/completed-raw37/python \
  --native-run /private/completed-native37/native \
  --native-first /private/validated-native-v5 \
  --capture /private/capture-hippo-amygdala-right \
  --output /private/new-saved-state-audit
```

全部参数必填。`--candidate` 是前次实际 v2 optimizer source；`--raw-first`、`--raw-continue` 分别提供原3步和续34步；`--native-run` 提供已完成官方37步；`--native-first` 提供官方原初态 gradient；`--capture` 绑定同一影像／alpha／几何状态；`--output` 必须新建为实际0700目录。SHA 或 private direction 复现点门失败立即退出。输出的 gradient 差若来自两条不同轨迹，字段明确标为 `at_distinct_accepted_states`，不能当作 same-point gradient 门。

## 3. 对应原软件与实测结果

这个只读审计没有独立官方命令。原公式为固定 [samseg L-BFGS](https://github.com/freesurfer/samseg/blob/2ce2b6be69f2954ea704e593a5be79c284a3a8c3/gems/kvlAtlasMeshDeformationLBFGSOptimizer.cxx)，依赖安装扩展／recipe 和阶段 source 的完整 SHA 见本次 JSON 及上份 [source bindings](../gems_cpu_objective_20261005/source_bindings.public.json)。native 点／gradient 保存文件逐项核对原 run SHA。此次只读累计的 Python 重构不是拟合耗时，不发布新的速度比或分割脑图。

| 更新 | private alpha | inferred native alpha | 两不同状态的方向 rel L2 | 下一 accepted points 最大坐标差，voxel |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | `3.3024e-8` | `3.8145e-6` |
| 2 | 1 | 约1 | 0.00109908 | 0.000352450 |
| 3 | 0.5 | 约0.5 | 0.00648489 | 0.00185940 |
| 4 | 0.0625 | 约0.0078125 | 见完整 JSON | 见完整 JSON |

step1 官方 Double points 舍入 FP32 后虽与 private points 全相等，二者原始几何仍有数微 voxel 差。在这两个不同几何上，accepted gradient relative L2 已为 **0.00123330，即0.1233%**，下一轮 `y=g_new−g_old` 差约0.0463%。到 step2 direction 差约0.1099%；step3约0.6485%；step4首次出现实质 alpha 分叉。官方 points／gradient 不都可逐值表示为 FP32。

这些结果与已通过的同点 gradient `≈3e-8` 门同时成立：**前者比较不同几何，后者比较相同 FP32 coordinates**。不能用0.1233%否定同点目标修复，也不能因同点通过而忽略内部 Double state。初始很小的差异如何经局部几何／梯度历史传播，还需控制变量探针。

另只读确认118个 alpha mass 偏离1超过1e−3的行全部为0；其余非零行质量在 `0.9999997488–1.0000002607`，符合上一份归一化触发原因。

## 4. 更新与下一步

这一版新增已保存轨迹的方向／历史／推断 alpha 审计，不替换任何生产代码。下一步只在新私密目录跑 bounded1／3：CPU FP64 points、gradient／projection／solver state，独立保留 FP32 ownership；先源码／数学合同及官方初点门，再逐项观察 Double accepted coordinates。GPU与当前公开 rasterizer 默认策略保持原路径。不重新执行 native37，不启动 fresh37 或完整 recipe。

完整核团 Dice≥0.95、体积差≤5%及 CPU 同线程提速仍待验收。原论文和全功能参考见[主诊断报告](../gems_cpu_objective_20261005/README.md)。
