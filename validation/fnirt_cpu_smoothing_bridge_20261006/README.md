# 当前 CPU 平滑来源桥：整图逐位通过，私有 adapter 未接入（2026-10-06）

## 1. 功能与结论

本次只核验保存的真实 normalized GM 在**当前 Numba CPU helper**上的来源一致性：当前 plain 与旧保存 plain 图，以及当前私有方向 adapter 与既有官方 smoothed 图，两项各 **18,579,456 个 FP32 值全部逐位相同**。一次科学 worker、三份 exit均0，没有新 normalization、RHS、H、PCG、官方程序或完整配准。[验收摘要](ACCEPTANCE.public.json)、[原始 summary](run/smoothing/summary.public.json)。

旧 `e67c71a6` 在CPU调用PyTorch offset reference；current `caab8ffc` 调用 `c31cef1b` Numba。旧冻结树没有新helper，本次补上这一真实保存图的compiled-helper来源桥。[源码/上下文审计](source/SOURCE_RUNTIME_GAP.public.json)。它没有重新发现已知的X flip整图等价，也没有证明完整配准改善。

**旧完整orientation候选仍被拒绝，生产和GPU源未改。** 旧候选整图平滑exact却使coefficients RMSE `.084011→.127315`、warped brain RMSE `.020061→.031622`。不能用本次两个平滑门恢复它为默认；[旧完整负结果](../smri_cpu/fnirt_cpu_orientation_20261004/README.md)与[脑图](../smri_cpu/fnirt_cpu_orientation_20261004/cpu_brain.png)保留。

## 2. Python接口、输入和输出

[smoothing_adapter.py](source/smoothing_adapter.py)是validation私有adapter，不导出到FNIT公共API：

```python
# moving_tensor: CPU Float32 [N,C,X,Y,Z]，本例[1,1,224,288,288]，requires_grad=False。
# header_flip_x: 原GM的qform/sform解析结果，不能从官方像素或结果估计。
# moving_pixdim_mm: header pixdim，本例(.8000000119,.7777777910,.7777777910)。
# mature_blur: 当前FNIT _fsl_masked_gaussian_blur，函数body保持原样。
adapted_moving_tensor = smooth_original_storage(
    moving_tensor,
    fwhm_mm=6.0,                   # GM最粗层保存的input FWHM，mm
    voxel_sizes=moving_pixdim_mm,  # 不用affine列范数替代pixdim
    flip_x=header_flip_x,          # 声明的storage orientation
    blur=mature_blur,              # 当前已绑定的成熟CPU函数
)
```

- `volume`：CPU FP32、5D、no-grad输入；不支持此trial中的mask、CUDA或F64/训练输入。
- `fwhm_mm`：平滑FWHM；≤0保留成熟no-op，测试未搜索该参数。
- `voxel_sizes`：三轴pixdim/mm。`flip_x`必须显式bool；neurological时X flip→blur→flip回原索引，radiological直接blur。
- `blur`：worker传入当前成熟函数的计数代理，数学body未改。输出仍为FP32原网格Tensor；不写影像文件。
- `classify_header(header, np)`：从声明header解qform/sform。只接受双form已知、同符号、abs(det)>1e-6的保守诊断范围；未知/冲突/近奇异明确拒绝。这不是完整NEWIMAGE header支持，也不是NEWMAT determinant逐位证明。

唯一运算输入是旧保存的 `normalized_moving.npy`（Float32 `[224,288,288]`，Fortran order，74317952B，SHA `8531ea82…`）。原GM只读348字节header和全文件SHA；不读原GM像素。9文件、5header与17 FNIRT源码+5import依赖全SHA绑定见[expected](source/expected.public.json)。原GM、官方normalized和smoothed三份header348 SHA相同；code为1/1，两form均正sign，pixdim一致。

整图reference仅供事后评分：adapter计算结束后才decode官方saved smoothed图，actual dtype为Float32，proxy slope1/intercept0，**没有reference cast**。归一化不重算；旧producer e67与新caab使用范围分别记录。

输出仅JSON/日志/exit、shape/dtype/stride与SHA摘要；MRI数组、原header内容、Numba缓存和官方源码/DSO未发布。[全部原metadata来源](COLLECTION_BINDINGS.public.json)。

## 3. 实际CLI与资源门

[run_smoothing.sh](source/run_smoothing.sh)保存实际命令；[worker](source/smoothing_bridge.py)、[IO](source/smoothing_io.py)及[preflight](source/preflight_smoothing.py)固定到freeze `6efcc5d5…`。实际统一W/R leaf为 `smri_cpu_20261004/remaining_20261006/fnirt-cpu-smoothing-bridge-v1`。[完整PLAN](source/PLAN.md)、[单次派发receipt](run/launch.public.json)。

```bash
# 原已批准的单次派发；此leaf已有输出，脚本拒绝重复enqueue。
python lifecycle_smoothing.py --phase enqueue   --freeze-sha256 6efcc5d57568e26431dbb21c6054a3ccd6abde15be6e86ff28cbddb790eb9bbd   --approved-lifecycle

# 本地只读stdlib报告核验：不import数值栈或回放计算。
python verify_report.py
```

nodecw7，CPU `32,36,40,44,48,52,56,60`；common CPU8锁后module锁，Torch/OMP/BLAS/Numba8、interop1；child180s、AS20,000,000,000B，controller19600s/锁等待19000s。CUDA_VISIBLE_DEVICES为空，CUDA未初始化；flags前后相同。source payload600、W/R700，worker也umask077/output700。[现场上传验证](source/UPLOAD_SERVER_VERIFY.public.json)。

单次controller PID51502/startticks535257672；prepared/queued/completed通过sorted六INDEX锁登记、重读保留其它字段、atomic replace保留permissions。三个exit原文件为[overall](run/smoothing.exitcode)、[science](run/smoothing.science.exitcode)、[postflight](run/preflight_after.exitcode)。canonical main前后均`db61cebc…`。

[index_completed](run/index_completed.public.json)记录本任务完成登记时的INDEX哈希；`FINAL_CONTEXT`是稍后只读现场快照，其INDEX哈希可随其它任务登记变化。没有把整个共享INDEX声明为前后不变。`collect_metadata.py`保留实际私有工作目录中的收集代码及相对relay约定；本地可直接运行的报告入口是`verify_report.py`。

## 4. 对应原实现

这是FNIRT内部 `SmoothObj`→`masked_smoothing`→NEWIMAGE separable smooth，没有独立官方平滑CLI。旧隔离官方 `fnirt --config=GM_2_MNI152GM_2mm.cnf` 的图已经保存，本次没有再调用fnirt、FSL库或其它原程序。

核沿用旧/当前相同AST：sigma先Float，exp后权重Float，offset升序Float总和，Double reciprocal归一化；每轴X→Y→Z、每offset Double product/add后写回Float。方向adapter只改变CPU内部X的存储累积顺序，随后恢复原网格。当前核、mean和公共默认body未修补。[完整来源审计与边界](source/SOURCE_RUNTIME_GAP.public.json)。

## 5. 最新真实结果、运行上下文和计时

| 整图门 | 值数 / dtype | bits不同 / max / RMSE |
| --- | --- | --- |
| current compiled plain vs旧saved plain | 18,579,456 / FP32 | 0 / 0 / 0 |
| current compiled adapter vs既有native saved图 | 18,579,456 / FP32 | 0 / 0 / 0 |

实际调用为mature blur2、compiled CPU helper2、adapter1；normalization/RHS/H/PCG/evaluate/linearize/native/GPU均0。plain首个bit差会停止adapter；本次三gates全true，输入/source/harness/header及所有可用operand/flags前后exact。[完整解释](INTERPRETATION.public.json)。

实际Python3.11.16/Torch2.5.1/NumPy1.26.4/Numba0.61.2/nibabel5.4.2、llvmlite0.44.0；Numba parallel、OMP、FP32 Fortran 5D输入/输出及FP64 C-order权重签名，新run-local空cache。compiled-cache文件只绑定SHA，未发布其内容。[上下文](FINAL_CONTEXT.public.json)。machine assembly/FMA未审；没有官方每轴arrays，因此只声称这幅完整图逐位门。FSL DSO前后快照为空，不扩大为全程系统调用监控。

| 时钟 | 本次秒数 | 范围 |
| --- | --- | --- |
| import/bind/header/restore | 2.525261 | 当前worker前置 |
| plain cold | 2.927571 | 单次成熟blur，含新JIT |
| adapter warm | .729019 | 单次blur+两次flip，使用该进程暖JIT |
| worker全部 | 8.044317 | 上述步骤、参考decode、整图比较和后置checks均包含 |

self peak RSS943660KiB；AS cap20GB。单次cold→warm顺序没有速度倍率含义，不与官方/完整注册时钟比较，嵌套时钟不相加。本次无需新脑图；旧完整阶段脑图与退化指标另页保留。

## 6. 更新与失败历史

- 2026-10-04：真实normalized数组已exact，旧plain moving有1,296,806值不同；旧flip平滑整图exact，但完整orientation候选恶化、未采纳。
- 2026-10-06：moving-only RHS控制总LMg误差下降71.8%，主要由mixed scale主导，三个系数块仅约21–30%；[独立RHS报告](../fnirt_cpu_rhs_moving_control_20261006/README.md)。本次未重复它。
- 本次：先源/保存图/header/Conda门，再唯一2-arm平滑来源桥通过，producer不改production。12payload prepared文件仍保留当时0worker状态；后续launch/summary单独记真实执行，未改旧JSON SHA。

原log包含OMP弃用信息，science/post均exit0；没有科学失败或重试。[状态历史](STATUS_FAILURES.public.json)、[metadata传输](METADATA_TRANSPORT_RECEIPTS.public.json)。已有v27/nodecw8其它真实验收保留自身范围。一般radius Float/Double字面量边界和mean threshold promotion未在本例改动，也未据此解释剩余RHS或完整注册差异。

## 7. 参考与许可

[FNIRT](https://git.fmrib.ox.ac.uk/fsl/fnirt)、[NEWIMAGE](https://git.fmrib.ox.ac.uk/fsl/newimage)、[FSL 6许可](../../licenses/FSL-6.0.txt)。Andersson、Jenkinson、Smith，*Non-linear registration, aka spatial normalisation*，FMRIB TR07JA2（2007）。GM数据沿用既有ds003138 v1.0.1/CC0来源与资源核验；本leaf只公开scalars/哈希/code，不发布完整数据、许可证或原软件代码。

所有计算复用主页Conda已有PyTorch/Numba/NumPy/nibabel，无新依赖。官方仅用于旧保存benchmark评分，FNIT runtime没有安装FSL程序/库依赖。
