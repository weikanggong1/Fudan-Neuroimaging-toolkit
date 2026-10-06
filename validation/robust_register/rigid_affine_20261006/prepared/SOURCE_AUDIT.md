# B 独立注册：源码和数学边界审查

固定 FreeSurfer 提交 `d932c45b7941662ea380a05efef580568b98d41a`，安装版二进制 SHA `82b20927…`；29 个原文件完整 SHA/URL 见 `SOURCE_BINDINGS.public.json`。原文件仅私密只读用于定义核对，没有复制进入发布内容。以下是来源对应，不是原软件数值验收。独立候选仍未运行真实 MRI 配准。

## 实际定义与本地实现

| 环节 | 固定原源码 | 实现边界 |
| --- | --- | --- |
| 两次命令参数 | `mri_robust_register.cpp:135–143,707,756–780,868–885` | robust、symmetry、sat50，Float 算法、每层 5 次、eps .01；rigid 后 MGH 重读再 affine；不附加 iscale、random subset 或其他优化器 |
| source 轴重排 | `Registration.cpp:3533–3684`，`utils/mriio.cpp:11578–11619,11629–11672` | 按列绝对最大与源冲突规则选择 signed permutation；复制/翻转体素，源几何和 Rsrc 按实际新 header 重建 |
| stored geometry | `utils/mri.cpp:102,587–675,747–789,1383–1410` | stored directions/delta/center 为 Float；方向乘 delta 经 Double 再写 Float；offset 用 `MatrixMultiplyD`；重排重新归一化 direction 但保留 permuted delta，再按 Float `MRIp0ToCRAS` 保存 center |
| common iso 网格 | `Registration.cpp:3740–3890,4076–4314`；`MyMRI.cpp:388–421` | 尺寸 ceil(extent/isosize−.0001)，双方 axiswise max，isosize=max(min source vox,min target vox)；保持原 stored center/directions；初始 reslice 固定 cubic B-spline |
| 数据类型 | `utils/mri.cpp:11806–11863` | uint8/int16/int32→Float 不进行 intensity rescale；Float 保持。候选拒绝未覆盖的 Double/其他 stored 类型，不在生产猜测原 histogram conversion |
| 初始化 | `Registration.cpp:3169–3285`；`CostFunctions.cpp:1308–1338` | 默认 `target centroid−source centroid`；强度加权坐标从 +1 开始。无质心初始化时用重采样后 Float header 矩阵转 Double 再求逆 |
| 金字塔级别 | `Registration.cpp:2893–2983,739–760,906–1025` | 每层≥16，strict `>16`；最高层的 highit 与预算耗尽保留；升分辨率只按源码倍增 translation |
| 金字塔数值 | `utils/mriBSpline.cpp:2451–2570`及`ReduceCentered/CenteredSpline`分支；`MyMRI.cpp`的5tap构建 | `[1,4,6,4,1]/16` Float 依次 x/y/z 平滑，21tap Centered Spline(little-l2) order3；奇轴尾端先截，再端点重复 mirror；Double 逐 tap 累加与 adjacent Haar average，x/y/z 每轴写 Float |
| cubic 系数和采样 | `utils/mriBSpline.cpp:42–163`及 cubic weights/sample/边界函数 | pole sqrt(3)−2；Double causal/anticausal，DBL_EPS horizon，每轴 MRI_FLOAT；mirror 周期2N−2。源码非负输入则裁负插值；禁止替换成保持 Double coefficients 的通用 FFT spline |
| trilinear 边界 | `utils/mri.cpp:10814`及`MRIsampleVolumeFrame`；`MyMRI.cpp:507–513` | `rint` ties-even 判断半体素外域、允许边缘 clamp；Float 4×4 coordinate，Double 八项有序插值，再 Float；不是 grid_sample zero padding |
| inverse/multiply | `utils/matrix.cpp:718–800`；`utils/numerics.cpp:1091` | Float 四项乘加；≤4方阵走 VNL Float cofactor。复用 FNIT 成熟 `_inverse_4x4_native`，不调用原二进制 |
| halfway | `MyMatrix.cpp:1287–1450`；`Registration.cpp:4586` | Double principal square root；半途 target=half×inverse(M)，都采样到 source geometry；4×4 Schur CPU，其余图像操作保持指定 device |
| derivatives/A/b | `MyMRI.cpp`的`getPartials`；`RegistrationStep.h:385–760`；`Transformation.h:785–803,1209–1221` | 原5tap/derivative Float 按序 x/y/z；avg/source−target；outside≈0 与 derivative 无效规则；row order z,x,y；rigid6/affine12 analytic columns |
| MAD/Tukey/QR | `Regression.cpp:206–328,363–435,632–659,689–709`；`Regression.h:109`；`RobustGaussian.cpp:344–368,394–411` | even median 双中位平均；Float MAD×Float1.4826；标准化 residual 为 Float。Tukey 的 ratio/product 是 Double，写回 Float；20 IRLS、2e−12、增误差恢复 p 与 w，等误差保留当前；Float QR，无 AᵀWA/ridge/SVD fallback |
| 参数更新/停止 | `RegRobust.h:172,243–295`；`Transformation.h:754–783,1185–1207`；`MyMatrix.cpp:2005–2049` | Double Quaternion rotvec 或 identity+rowmajor12；`inverse(target_half)×update×source_half`；两种模式均用 r100 的 affine distance，而非 torch LBFGS 收敛 |
| final voxel/RAS | `Registration.cpp:4392–4424`；`mri_robust_register.cpp:707,764,875`；`utils/mri.cpp:12992–13010` | Rtrg×Mfinal×inverse(Rsrc) Double；原命令转 MATRIX_REAL，再按 targetvox2ras×(voxel×sourceRAS2vox) 的 Float 括号转换。不能混用 MyMRI 的另一个 Double RAS helper |
| mapmovhdr | `utils/mri.cpp:13019–13080` | 默认保留原 delta，direction=newlinear/delta，center=translation+ordered Float linear×halfshape。仅原环境 opt-in 会改变 delta，benchmark 显式 unset。源体素完全不重采样 |

## 与成熟 FNIT helper 的关系

复用：`_nib.load_image`、`_transforms.AffineTransform`/LTA、`recon_all.ca_register_inverse._inverse_4x4_native`。原方法物理空间容器和接口保持。LTA 的成熟 writer 从 affine 分解几何；其 text 表示不承诺与官方字节相同，benchmark 按真实空间/shape/矩阵/重读几何比较并报告字段差。

不能直接复用的数值定义：通用 trilinear 外域规则、FFT spline 系数轴间舍入、常见 `interpolate(scale_factor=.5)` 金字塔，以及 autograd Dice 图像目标。候选以专属 adapter 实现，不修改其他 GPU 用户已有函数。

## 尚未完成的数值证明

1. 原 VNL QR/BLAS 和 Torch QR/并行 reduction 的浮点路径不同。算法控制已实现，**真实同输入拟合尚未运行**；不能由 local Float 线性系统合同推出真实最终 matrix 等价。失秩 fallback 未模仿 VNL 的所有行为，本候选直接报错。
2. centroid 的 Torch Double 并行求和与原 Double z/y/x 串行求和可能有尾差；Schur/Double inverse 的库实现也不同。每处 dtype/边界已声明，待真实 matrix/warp 门。
3. 公式中的 `FEQUAL` 宏在该 `utils/mri.cpp` 先包含 `float.h`，实际 FLT_EPS 为1.19209e−7；不能把宏文件的 fallback1e−5直接当 sampler 阈值。原 compiled source 对应关系仍通过安装build绑定，而非宣称所有编译开关逐 bit相同。
4. CUDA 默认 TF32 保留，但此 B 没有真实 GPU 运算/显存验收。CPU/GPU既有默认流水线未接入、没有改通用 GPU AST。

## 许可

原作者 Martin Reuter；Copyright ©2021 The General Hospital Corporation (Boston, MA), “MGH”；B-spline 来源 Thévenaz/Blu/Unser。改编保留模块内归属、[FreeSurfer v1 PartB](../../../licenses/FreeSurfer.txt)与修改标识。SciPy/NumPy/PyTorch/Numba/nibabel沿项目现有 Conda 安装；不转载上游源码集合、安装版二进制、私密 FreeSurfer license 或 MRI。
