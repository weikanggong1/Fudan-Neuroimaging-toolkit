# FNIRT 单 moving 输入 RHS 控制：准备版，0 科学调用

本轮仅完成 worker、输入/源码门和有限调度器，等待 root 独立复核及唯一执行授权。没有上传、INDEX 更新或 worker 派发；不修改生产源码、GPU 路径或阈值。

## 1. 固定状态与目的

从已验收的 stage1 checkpoint 恢复 solve3 第二 accepted 参数点：参数 SHA `21d339a4…`、NPZ SHA `78abee85…`，保存 count14341、SSD60.329756040354546、弯曲能量2.3604429240341207。当前 production registration `caab8ffc…`、optimizer `903d5031…`；17个FNIRT包生产文件、原 assembly 控制源码与3个实际导入依赖（_nib/_transforms/flirt.coordinates）共21项绑定。完整哈希见 expected.public.json。

目的仅为隔离已保存平滑 moving 的贡献。基线复现当前已冻结 FNIT 输入产生的 g；它不是官方同预处理输入的证明。旧 e67 输入来源、当前 caab 的 AST 范围及已拒绝 orientation 候选负结果见 SOURCE_AST_AUDIT.public.json 和 INPUT_INVENTORY.public.json；不重新平滑、翻转或读原 MRI。

## 2. λ、SSD 与 fixed 阶段

官方保存 state.json 的 solve3 `regularization_lambda=9049.463427795125` 与本控制一致。installed `fnirt_costfunctions.h:297` 在 SSD-weighted 模式返回 `latest_ssd*lambda`，cpp:897–898 在 cost 评价中更新 latest_ssd；grad/hessian 使用保存状态。官方没有单独保存该 solve3 SSD。因此 **λ/150 不是精确 native SSD 记录**，不检查也不报告它为逐位复现。

当前 stage1 的原评价 λ 为9049.463406053183，随后为同点控制固定为9049.463427795125。替换 moving 后仍固定这个值，新的 SSD 不更新 λ。新 cost 使用相同、已经验收的 FNIT 弯曲能量 scalar。它不是重新运行官方 SSD-weighted objective，不代表 native cost/gradient 全部等价。

官方 initial_fixed 是 `Ref()`，4mm reference blur/stride4 后、global scale 前的 F32 `[24,28,24]`。checkpoint fixed 也是这个 raw 阶段。先要求 current_fixed.npy 与 checkpoint fixed、官方 Ref 与 checkpoint fixed 两项逐bit相等；不对不同阶段强行 rescale/normalize。只有原 state 公式 `Double scale*Double fixed → F32` 产生 scaled_fixed，两臂要求该结果不变。

## 3. 必要输入及恢复

只读取68项 NPZ 中32项必要 payload：coefficients/scale/fixed、M/coordinate_affine/affine_grid、field/warped/mask/residual/gradient_fsl、3个basis、18个预先保存的 bending Gram。逐项检查 shape/dtype/finite、逻辑值及原 byte-stride，并用独立 CPU storage 恢复。未保存的原对象身份/alias graph 不作恢复声明。

不读取 gradient_half、diagonal_half、bending_diagonal、spatial/cross/scale weights 或 H。当前 fullg、当前 FSL-order fullg、native fullg 均为已有 FP64 文件；参数只检查 packed point 的 SHA。当前 moving.npy 仅作文件身份绑定，基线使用自产的保存 warp/residual/derivative 状态；不再采样它。

影像 payload 仅三个：current_fixed.npy、official initial_fixed.nii.gz、official smoothed_moving.nii.gz。官方 moving 为原存储方向 F32 `[224,288,288]`，pixdim `.800000011920929,.7777777910232544,.7777777910232544`，100/SPM mean 强度、6mmFWHM；不重定向、重采样、归一化或平滑。source/input before-after 文件绑定12项；所有私有数组留服务器，输出仅 scalar/schema/hash。

## 4. 原源码数学与前置停止门

gradient_body.py 使用 stdlib AST 摘取原语句，SOURCE_AST_BINDINGS.public.json 保存完整 body identity：

1. evaluate 中 `count` 到 SSD 的原6个语句；没有调用 evaluate。弯曲能量由已验收 scalar 重建，cost 用原 assembly 固定 λ 语句。
2. linearize 中跳过首句 evaluate，保留 mask/count/sqrt/count、adjoint/bend_normal 和 gradient 赋值；在 spatial_weights 之前截断。没有 data_normal/matvec/diag/CPU normal cache。
3. 原 assembly 的 FSL-order控制：F32 derivative×residual×mask → Double adjoint/count；`+=lambda/count*bending.normal(coefficients)`；F32 fixed×residual×mask Double sum/count；最后2×pack。
4. 原 evaluate 的 F32 derivative projection；坐标仍调用当前 `_fsl_displacement_coordinates`，由保存的 F32 field、affine_grid 和 M 得到。

实际成熟 BendingOperator.normal 只在相同保存系数和18 Gram 上调用一次；四次 gradient 常量读取保持其值与布局不变，并核传入系数值/stride。它是固定正则化梯度项，不是新 Hessian callback/cache。

基线先核 residual 与 saved residual、count、SSD、固定 λ cost 各逐bit；full LM g 与当前 assembly `79e5d87f…`、FSL-order g 与 `a328e1b5…` 全1177项逐bit。任一失败停止，官方 moving 不读取、不采样。

## 5. 唯一替换与输出

基线通过后读取 official saved moving，原坐标构造一次，成熟 trilinear sampler 一次且 `derivatives=True`。要求 CPU/no-grad、坐标/warped/gradient_voxels F32，valid bool；shape 分别 `[3,24,28,24]`、`[24,28,24]`。valid mask 与保存 mask 逐bit、count14341 一致；首差停止，不再计算替换 g。保存坐标、mask、dtype/stride/hash并做 operands after-hash。

保持 field/coords生成规则、scale/fixed、basis、λ 和弯曲项；原 FNIT F32 projection 一次，两种 g-only prefix 各再一次。官方 totalg 只在四个自产梯度全部算完后读取，用作事后统计：全g、x/y/z三个392系数 block、global_scale 一个元素的不同bits、max、RMSE、relL2。系数/scale 是混合参数坐标，不写成 voxel 误差。

报告 baseline/替换的 SSD/cost、warped/residual/projected-gradient 差及相同 convention 的 g 改变。FSL-order仍用 FNIT projection/separable adjoint，并未恢复官方每一个累加次序；没有 native 同点中间缓存，所以总g改善或恶化都不证明唯一原因、完整等价或生产修复。

## 6. 数量、资源与失败收尾

唯一授权 worker 的上限/成功实际数量：sampler1、coordinates1、projection1、state-prefix2、LM-prefix2、FSL-order-prefix2、实际 bending.normal1、常量读取4；evaluate/linearize/gradient-method/dense-field-expansion/design-diagonal/PCG/SCG/H-callback/native 全0。无新 H/diag/normal-cache、原 MRI、全配准或 GPU。

nodecw7，亲和性32,36,40,44,48,52,56,60；Torch8/interop1、OMP/MKL/OpenBLAS/Numba8。记录实际 Python/Torch/NumPy/nibabel/Numba 版本及 canonical env/default 解析匹配；实际环境 prefix 只给 basename/路径hash，Python二进制给SHA/大小。CUDA_VISIBLE_DEVICES空，移除loader/BLAS override，RLIMIT_AS20,000,000,000B（address space不是物理峰值），私有 umask077、目录700、文件600。shared outerCPU锁后自身inner锁；等待19000s、controller19600s、science180s+TERM/kill5s，有限 detached 调度、不重试。

runner 科学失败后仍执行 after-preflight，并保留 science RC；worker 任意已开始科学的失败也保存已恢复 operands 的后置 hash、flags/CUDA/DSO snapshot及前后 source/input/freeze。异常不标 accepted；stdout/summary/三exit原记录保留。DSO只是前后文件身份快照，不宣称全程 syscall 审计。源/计时/JIT/文件读取等时钟仅观察，不作速度比。

## 7. 准备检查与后续授权

现在仅 stdlib source AST/compile、JSON/哈希和 bash-n；0 NumPy/Torch/nibabel 导入，0 scientific worker/callback/模型。freeze.public.json 绑定所有准备 payload，self-excluded。lifecycle_moving.py 只有明确 `--approved-lifecycle` 与 root 审过的 freeze SHA 才可六锁登记/one Popen。六锁内重读 INDEX JSON/MD，保留他项与权限，原子写入。

Root 审核本 freeze 后另行授权唯一执行；本准备版不自动上传、登记或派发。科学结果出来后再独立 report-only 提交，不更改 root 文档/安装或生产 runtime。
