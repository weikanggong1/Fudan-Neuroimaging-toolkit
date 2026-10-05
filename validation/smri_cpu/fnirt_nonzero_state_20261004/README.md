# FNIRT 共享非零系数与正则权重状态探针（2026-10-04）

## 范围与结论

固定[首差诊断](../fnirt_first_diff_20261004/README.md)已经保存的官方第一个 accepted **FP64 参数向量**，在同一真实 GM、官方 FLIRT、template/mask、最粗层 grid 上比较 cost、gradient、Hessian diagonal 和固定向量乘积。本轮不求解新的系数，不运行完整 pipeline，不改生产 FNIRT 或 GPU。

**未发现非零 regularizer 或 Gram 的量级错误。** 双方有效 mask 都为 14,820，bending energy 相差 `1.4e-15`；将 image 项消去后的 regularizer 梯度 action 相对 L2 差约 `3.2e-15`。梯度及 Hessian 的剩余差异为末位级，不能据此宣布完整 nonlinear estimator 等价，或作为新的生产修复。

## 实际比较

同一参数向量的系数形状 `7×8×7×3`，另有一个 global intensity scale。recipe 保留完整四层 schedule，初始化后截获首层 `_LevelSystem`；后续层没有优化或执行。原生 oracle 同样只建立 cost object，读取上述保存参数。

| 同一非零参数上的指标 | 官方 native | FNIT baseline | CPU flip 候选 |
| --- | --- | --- | --- |
| 有效 mask count | `14820` | `14820`（数组 exact） | `14820`（数组 exact） |
| bending energy | `0.4557489432611880` | `0.4557489432611866` | 同 baseline |
| SSD | `73.0927213271` | `73.0927197273` | `73.0927200461` |
| cost | `73.4298866156` | `73.4298850084` | `73.4298853287` |
| LM gradient 相对 L2 差 | 基准 | `5.3422e-9` | `2.7652e-8` |
| direct gradient 相对 L2 差 | 基准 | `4.4806e-8` | `2.1967e-8` |
| Hessian diagonal 相对 L2 差 | 基准 | `9.1279e-12` | `4.6022e-12` |
| Hessian 固定向量乘积相对 L2 差 | 基准 | `6.2054e-10` | `7.0096e-10` |
| 消去 image 项并除去 lambda 差后的 regularizer action 相对 L2 | 基准 | `3.2047e-15` | `3.2121e-15` |

warped、derivative、全部 vector 指标及数组 SHA 在[完整比较](comparison.public.json)。CPU flip 是[完整真实阶段已拒绝的候选](../fnirt_cpu_orientation_20261004/README.md)，本表不恢复默认候选；SM 图更近不保证 LM gradient 或完整估计更近。

## Lambda 与 LM 调度的现场源码核对

实际安装 FSL 6.0.7.4 / FNIRT 2203.0：

- `fnirt_costfunctions.cpp:897–898` 将当前 mean SSD 写入 `latest_ssd`；随后 `929` 使用 `Lambda() × BendEnergy / mask_count`。所以 cost 没有先使用前一轮 SSD 再更新的情况。
- `fnirt_costfunctions.h:297` 的 `Lambda()` 返回 `latest_ssd × lambda`；grad/hess 本身不更新 `latest_ssd`。
- `miscmaths/nonlin.cpp:409–413` 的 LM 仅在前一 trial 成功时更新 gradient/Hessian；`450–453` 的拒绝分支复用原 gradient/Hessian 并增加 damping。它不会在这条标准 LM 重试路径中调用“accepted 参数 + rejected trial 最新 SSD”的新 gradient。
- 完整 CPU 配对的各层 `attempts == accepted_iterations`，本例没有 rejected trial；最早离散分叉是第一级第三次 PCG `80 / 49` 次。上述 rejected-state 假说不能解释这个实际分叉。

另做一个明确的**人工状态顺序隔离**：保持非零系数，把 scale 增加 3 后调用 `cf(trial)`，再直接调用 `grad(shared)`。trial cost `23088.964366`，lambda 从 `10963.908199` 变为 `3447442.143604`；FNIT 已有 `gradient(..., effective_lambda=...)` 能重现这个权重状态。两次 gradient 相减、再各自除去 lambda 差，隔离出的正则 action 如上表。这项测试只证明 stateful 权重和子算子，**不是 native LM 实际拒绝 trial 调度**。

原生重置后重复 `cf(shared)` 自身有 `4.1651e-7` 的末位差（`73.4298866156 → 73.4298870321`）；对应 gradient 相对 L2 `4.2134e-13`。原记录与 repeat 指标都保留，没有按 bitwise 一致描述原生缓存状态。

## 运行与复现

固定输入及 actual source/object/library/program SHA 在 [binding](binding.public.json)，各 Python 臂的输入、实际 registration SHA 和 affinity 在 [baseline](baseline.public.json)、[candidate](candidate.public.json)。源码基点 `b4a55d8ff2738d9cfd9744f9759f62dead9bc4fc`；候选独立 freeze 的 SHA 与上一份完整阶段报告一致。

nodecw7、同一 `nodecw7.gems.cpu8.lock`、核 `32,36,40,44,48,52,56,60`、8 threads；三项探针顺序运行，exit 0。运行前后 loadavg 为 `20.79,21.20,22.08` / `23.43,21.78,22.22`，留存在文本记录。整个队列从 20:37:24 到 20:38:34；包含脚本、读取、落盘和有限探针，不是估计速度 benchmark。

[official_nonzero.cpp](official_nonzero.cpp) 只在隔离官方 benchmark 环境调用已安装 FNIRT cost；不随 FNIT 生产加载。headcw 以 C++17/O0 编译，并复用此前已绑定、未改动的 installed-source objects；没有复制原软件源文件或 binary 到 FNIT 生产。[fnit_nonzero.py](fnit_nonzero.py) 初始化后只读共享参数，保存状态；[compare_nonzero.py](compare_nonzero.py) 计算原生/FNIT 差异和纯正则 action。

[run_nonzero.sh](run_nonzero.sh) 保存实际路径、锁和原生 command-line 输入。Python 参数 `--gm/--template/--mask/--affine` 对应已校验 NIfTI 和 FLIRT 矩阵；`--parameters` 为保存的真实 FP64 text；`--output` 必须为新目录。

```bash
PYTHONPATH="$FNIT_FROZEN_SOURCE/src" python fnit_nonzero.py \
  --gm "$OFFICIAL_GM" --template "$GM_TEMPLATE" --mask "$REFERENCE_MASK" \
  --affine "$OFFICIAL_FLIRT" --parameters "$SHARED_OFFICIAL_PARAMETERS" \
  --output "$NEW_PROBE_OUTPUT"
python compare_nonzero.py --run "$NEW_PROBE_RUN" --output "$NEW_COMPARISON_JSON"
```

首个前台 build 的本地 SSH relay 150 s 超时；远端编译继续并完成，随后确认 binary/object 存在、object/link 日志为空，未重复构建，实际 oracle 运行成功。远端 `rg` 缺失、首次源码目录假设错误及 `nonlin.cpp` 非 UTF-8 源码视图读取也在 binding 记录；实际 source hash 使用原始字节。

## 有限下一步与参考

下一步可以在共享第二个 accepted 非零参数上保存同一 Hessian/RHS，逐轮比较 native/FNIT PCG residual 和 accepted delta，以检查求解累计顺序/停止边界。该 probe 尚未运行；当前证据不支持 regularizer、Gram 或 rejected-lambda 生产修复，也没有解决完整 estimator 偏差。

[FNIRT 官方源码](https://git.fmrib.ox.ac.uk/fsl/fnirt)、[MISCMATHS](https://git.fmrib.ox.ac.uk/fsl/miscmaths)、[basisfield](https://git.fmrib.ox.ac.uk/fsl/basisfield)。参考：Andersson、Jenkinson、Smith，*Non-linear registration, aka spatial normalisation*，FMRIB TR07JA2（2007）。
