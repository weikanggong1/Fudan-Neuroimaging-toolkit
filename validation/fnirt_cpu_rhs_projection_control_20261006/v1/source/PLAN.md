# FNIRT 单处导数投影控制（准备版）

本轮只准备源码，尚未运行科学 worker、上传、登记或派发。只有根代理独立审查冻结清单并授权后，执行唯一一次有限控制。

## 1. 固定输入与唯一改动

复用 accepted stage1 的 68 项 schema，只恢复必要 29 项逻辑值与原 stride；同一 solve3 第二 accepted 点，参数 SHA `21d339a4…`。使用上一 moving-only 控制已绑定的官方保存平滑 moving，固定 `lambda=9049.463427795125`、field、coords、mask、count、fixed、scale、三个 basis 和 bending Gram。

一次成熟采样恢复 raw voxel derivatives，先用当前 FP32 逆矩阵投影，并要求完整 FSL-order RHS SHA `b17ee043…`、count `14341`、SSD `60.329755786510965`、cost `61.81924340344455` 与上一控制逐位相同。只有全部通过，才将 `state.gradient_fsl` 替换为同一 raw derivative 的 signed-axis FP32 除法。

原软件正 pixdim 定义来自 `ObjVxs` 和 NEWIMAGE sampling matrix；默认读取转换到 radiological 存储，X 翻转的链式导数符号由实际双 form header 与当前保存 diagonal Jacobian 同时确定。正 determinant 对原存储坐标给 `(-1,1,1)`，负 determinant 给 `(1,1,1)`。没有猜测矩阵转置或从 native gradient 选择符号。未知、冲突、近奇异 form、旋转或 shear 直接停止。

## 2. 停止门顺序

1. 23 源码文件、9 输入文件、冻结 payload、当前 main HEAD 和实际 Conda/CPU8/无 loader override 先门。
2. checkpoint schema、逻辑值、stride、独立 storage、packed point 和 raw fixed 位门；moving header identity/form 门。
3. 唯一 sampler 的 coordinates、warped、valid mask、raw voxel gradient 四个实际 SHA/shape/dtype/stride 门，以及 mask/count 门。
4. 几何轴符号和 diagonal reciprocal 位门在两个 projection 前；原 projection SHA 和 residual/count/SSD/cost/lambda 位门。
5. 原 FSL-order 完整 RHS/scale 位门在 candidate 前。只做一次 bending.normal，最多两次只读缓存使用。
6. 候选明确 `aten.div.Tensor`：除数是 FP32 三元素 Tensor，broadcast shape `[3,1,1,1]`，没有 Python scalar、0D Tensor 或显式 reciprocal。候选输出复制到与原 projection 相同 stride。
7. 若 projected values 位相同，跳过重复 RHS；否则候选只运行一次原 FSL-order 前缀。scale 最后一个元素必须 bits0，随后才读取 native 总 g 作事后比较。
8. finally 在失败和成功均保存已有 operands/source/flags/CUDA/DSO snapshot 后门；首个差异即停止，无数值重试。

## 3. 计数与调度

最多 coordinates1、sampler1、current projection1、candidate projection1、state scalar prefix1、FSL-order prefix2、bending.normal1、cached bending reads2。0 evaluate/linearize/LM prefix/H/diag/normal cache/PCG/SCG/native/raw MRI/GPU/blur/normalization/完整配准。

nodecw7 八物理核 `32,36,40,44,48,52,56,60`，四个线程环境变量均8，实际 canonical Conda。共用 CPU8 锁等待120s，module 锁等待20s，science child180s/kill-after5s，controller360s，20,000,000,000 bytes AS 上限。共锁未得只留0科学调用 receipt，并自然非0结束，不自动 retry。 prepared/queued/completed 的六个索引锁按统一排序，以 `LOCK_EX|LOCK_NB` 在共同25秒获取截止内等待；未全部取得即释放全部已开/已得锁并停止，该元数据阶段科学调用0、INDEX读取/覆盖0，不自动重试或再次enqueue。科学失败后仍做 after preflight，并保留原 science exitcode。输出目录700、文件600；只返回 scalar/hash/schema，不下载或公开数组。

## 4. 解释边界

源码和已安装 ELF 符号里的 divps/divss 证明该 arithmetic 定义；未观察本次或旧官方运行实际 partial。没有官方同点 partial/warp/Jte cache。native SSD 未单独保存，本轮控制 lambda 不由新 SSD 更新。

scale 不依赖 derivative projection，本轮其误差必须完全保留。总 g 的 mixed scale 成分占主导，不能由 coefficient 改善推断完整 warp/segmentation 改善。FNIT separable adjoint 和 scale reduction 的其他顺序差仍在。旧 orientation 候选完整配准恶化结果保留并继续拒绝。平滑桥的整图 bitmatch 不等于配准验收。本轮不改生产默认、GPU、容差或显存策略，也不运行 solver/native/全配准。
