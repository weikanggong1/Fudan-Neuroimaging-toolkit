# FNIRT 当前 CPU 平滑来源桥：准备版，未运行

## 1. 目的和当前状态

本 leaf 只准备一次 finite 保存图诊断，**没有上传、注册 INDEX、启动 worker 或执行数值**。目标是把旧 PyTorch 平滑的真实逐位结果与当前 Numba CPU helper 建立来源桥；没有新 normalization、RHS、H、PCG、原生程序或完整配准。

旧 `e67c71a6` 实际源码在 CPU 上直接进入 PyTorch offset reference，冻结树没有 `_smoothing_cpu.py`；current `caab8ffc` 在 CPU FP32/5D/no-grad 输入上调用 `c31cef1b` Numba helper。因此旧 whole-image 证据不能自动当成当前 compiled helper 在该保存图上的实测。已有v27/nodecw8真实功能和组件合同保持其原有效范围，但未提供本node7首差保存moving的current/old整图bit checkpoint。这一差别由现场源码和旧 reference/optimized JSON 确认，记录在 `SOURCE_RUNTIME_GAP.public.json`。核、reference blur、均值和 masked blur 的旧/当前 AST 一致；本轮不改它们。

## 2. 输入、私有 adapter 和输出

唯一运算输入为已保存 `normalized_moving.npy`，Float32 `[224,288,288]`、Fortran order、74317952B、SHA `8531ea82…`。它由旧 e67产生；producer report 和原GM identity一并绑定。shape是本次验证门，不进入 adapter 的算法选择。

原GM仅读取348字节 header及文件哈希，不读其像素。原GM、官方 normalized、官方 smoothed header348 SHA均 `fd2f1bfa…`，qform/sform code为1/1，pixdim为 `.800000011920929 / .7777777910232544 / .7777777910232544`。worker用 nibabel 从这348字节重建 header；两form在FP64/FP32 determinant诊断下需同符号且远离零（abs>1e-6），不猜未知、冲突或近奇异form。此保守guard不是NEWIMAGE ZERODET阈值，也不声称NumPy determinant和NEWMAT逐位相等。

```python
# 以下只展示拟执行接口；本准备阶段没有调用。
# moving_tensor: Float32 CPU [1,1,X,Y,Z]，requires_grad=False。
# resolved_header_flip_x: 由已绑定的原GM双form header确定。
# mature_cpu_blur: 当前FNIT _fsl_masked_gaussian_blur，原函数body未改。
adapted_moving_tensor = smooth_original_storage(
    moving_tensor,
    fwhm_mm=6.0,                   # 已保存GM最粗层 input FWHM，mm
    voxel_sizes=moving_pixdim_mm,  # header pixdim，不能用sform列范数
    flip_x=resolved_header_flip_x,
    blur=mature_cpu_blur,
)
```

adapter只接受CPU FP32、5D、no-grad，当前trial仅None mask。neurological storage先X flip，调用当前成熟blur，再flip回原索引；radiological直接blur，FWHM<=0保留成熟no-op。它不读官方图、系数、轨迹、被试或形状特例来决定算法。公共默认、CUDA/Triton、F64/训练fallback、TF32/autocast和生产源均未改。

输出只有JSON：两次整图比较的bits/count/max/RMSE/relativeL2、shape/dtype/strides摘要、source/input/header和operand前后SHA、实际versions/prefix identity/Numba dispatcher签名、flags、时间与原exit；不写影像/array文件或原header内容。

## 3. 唯一拟执行顺序和调用界限

1. stdlib验证freeze、17 FNIRT源码+5实际import依赖、9文件和5header绑定；环境/8核亲和性/interpreter/prefix必须匹配。
2. 在run-local空Numba cache中导入当前源码，绑定实际import文件，记录flags/CUDA未初始化及FSL DSO快照；解析header guard。
3. 加载保存normalized和旧plain保存图。**仅一次**当前CPU mature plain blur；全图FP32 bits必须与旧plain保存图相同，否则立刻停止，adapter不运行。
4. **仅一次**私有header adapter，仍调用同一成熟CPU helper。计算完成后才读取官方saved smoothed图的像素；禁止根据reference修改输出。
5. 官方decoded reference保持其实际dtype，要求Float32，**不cast**；整图全部值bits必须相等，失败即停，不重试、不加核/radius搜索。
6. finally无论成功或数值失败都恢复计数spy，记录所有可用operand、source/header/freeze和flags后置门，保留原异常；所有门通过才accepted。runner无论science成功/失败都跑after preflight并保留science RC优先级。

上限：mature blur2、实际compiled CPU helper2、adapter1；normalization/RHS/H/PCG/evaluate/linearize/native/GPU均0。三个axis由原helper源码loop绑定，但不单独插桩count或声称官方each-axis bit门。官方per-axis arrays没有保存，本trial只做完整图bit门。

## 4. CLI、调度和官方对应

准备代码在本地独立private目录；canonical新leaf拟为 `smri_cpu_20261004/remaining_20261006/fnirt-cpu-smoothing-bridge-v1`。执行仍待root审核整份freeze后的单次授权；不直接运行下面命令。

```bash
# 将来只运行经过审核的冻结run_smoothing.sh；当前未执行。
# controller:19600s；shared lock wait:19000s；science child:180s。
python lifecycle_smoothing.py --phase prepared --freeze-sha256 "$REVIEWED_FREEZE_SHA256" --approved-lifecycle
python lifecycle_smoothing.py --phase enqueue --freeze-sha256 "$REVIEWED_FREEZE_SHA256" --approved-lifecycle
```

nodecw7，同CPU8物理核 `32,36,40,44,48,52,56,60`，共用 `nodecw7.gems.cpu8.lock`再module锁；Torch/OMP/BLAS/Numba8、interop1；科学child20,000,000,000B AS与180s TERM/5s KILL；CUDA_VISIBLE_DEVICES为空，无loader/JIT override。登记时sorted六锁并重读/保留其余INDEX字段，atomic replace保留permissions。所有新目录700/文件600，worker自己也umask077/output mode700，防止绕runner直调产生宽权限数据。

原软件内部对应 `fnirt_CF::SmoothObj`→`masked_smoothing`→NEWIMAGE separable smooth；没有独立官方CPU平滑CLI。旧 `fnirt --config=GM_2_MNI152GM_2mm.cnf` 完整输出已保存。本轮不会再调用fnirt或其它原软件。

## 5. 已有真实结果与解释边界

旧真实 normalized moving/fixed已经与native逐值相等。旧plain moving有1,296,806值不同、max `3.814697265625e-5`；旧X flip→blur→flip back已经整图逐位相同。当前准备trial不会把这件早已证明的事实标成新发现。

**旧完整orientation候选已拒绝**：它只改CPU平滑，但完整coeff RMSE `.084011→.127315`、warped brain RMSE `.020061→.031622`，Jacobian brain RMSE `.013270→.020135`，更差。当前两次平滑门即使通过，也不能恢复它为默认或声称完整精度改善。

独立moving RHS控制已完成：LM totalg relL2 `8.3642e-7→2.3589e-7`，约71.8%下降由mixed scale主导，三个coefficient blocks仅约21–30%。此背景说明预处理有贡献，不证明唯一cause或warp/分割改善。新trial不再次算RHS。

plain冷JIT与adapter暖cache仅各记录一次，不形成速度倍率。旧runtime只在这些JSON记录了主机/8核/source，未完整绑定其LLVM/DSO；当前记录真实版本和Numba签名，但不作machine assembly断言。DSO快照只是记录时状态，不是全程系统调用audit。

## 6. 准备/失败历史及停止门

2026-10-04首差与旧orientation完整负结果保持原样。2026-10-06来源策略审查后，确认current编译helper实际不同；只读确认三份header348相同、NPY header的dtype/layout和actual extra imports。没有再次flip计算。

任何文件/header/runtime/dtype/layout/首plain bit门失败都不运行后续adapter；adapter门失败停止，不改阈值。源码body、输入SHA和case参数不能热改。停止后只回传metadata，由root决定下一项有限诊断。

radius Double/Float字面量边界和一般mean threshold promotion仍为另行general scalar合同缺口；本例旧normalized已exact，不在本trial混入修补或搜索。unknown/单form/mixed/near-zero header属于本诊断明确拒绝的范围，未声称完整NEWIMAGE header覆盖。

## 7. 来源、依赖和许可

来源：[FNIRT](https://git.fmrib.ox.ac.uk/fsl/fnirt)、[NEWIMAGE](https://git.fmrib.ox.ac.uk/fsl/newimage)，installed source定义的SHA和行号见先前策略audit。FNIT当前mature PyTorch/Numba代码和已有Conda Python/Torch/NumPy/nibabel/Numba复用，无新依赖、FSL DSO linkage或原程序runtime调用。官方仅作为既有保存benchmark参考，未复制发布官方source/DSO、原MRI数组或许可证。
