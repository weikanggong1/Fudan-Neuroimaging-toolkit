# Final transform与Float inverse：源级定位

本审计只读固定源码、已有B统计和安装ELF元数据；新增数值仅六个保存4×4矩阵的独立合同，没有影像注册/评分。全部上游URL、大小、SHA在 `SOURCE_BINDINGS.json`；旧B六源码SHA仍见旧leaf的manifest。

## 1. 证实的计算顺序差异

成熟 `_inverse_4x4_native`（固定SHA `532e4601932964285ed98b55b8866528ebc80bbcb05a41da7cf62bd06100c389`）第93行通用minor展开，95–100行det以第一行和cofactor点积累计。新候选没有改这个文件；它在原固定inverse-GCAM用途上的既有验收不受影响。

FreeSurfer `matrix.cpp:259–265` 调用 `OpenLUMatrixInverse`；`numerics.cpp:1096,1120–1125`采用VNL Float4×4。ITK4.13.2 `vnl_inverse.h:112–153`规定16个adjugate元素的具体运算顺序，`vnl_det.hxx:27–56`按24项计算det，不等于Float下先算cofactor再点积。

例如adjugate[0,0]的两个正项，通用展开先计算 `(m12*m23)*m31` 再计算 `(m13*m21)*m32`；固定VNL先计算 `(m21*m13)*m32` 再计算 `(m31*m12)*m23`。三因子的中间舍入和正项累计顺序均不同。一般数学等价不能证明这两个Float程序bit相同。合同对四个仿射保存状态已实际证实非零差；对两个刚体只见signed-zero差，必须区分。

安装binary原SHA为 `82b20927fdf9f650da81699a339577fbd24a6a2d9281fa9452cd1c6569c3334a`，build绑定d932c45；编译字符串显示ITK4.13.2_centos7/VNL。固定FS CMakeLists:699也选择该版本。ELF无可用函数符号，不能声称已核其编译器融合/指令顺序完全相同。本次C++oracle编译原固定头文件，数值定义独立于新Python表。

## 2. 最终transform没有发现括号接线错误

| 源码边界 | 官方 | 旧B实验 |
|---|---|---|
| 原网格voxel矩阵 | `Registration.cpp:4422`，Rtrg×Mfinal×inverse(Rsrc)，Double | `registration.py:387`，同括号Double；但NumPy inverse与VNL Double算术尚无内部状态对照 |
| 转MATRIX_REAL | `mri_robust_register.cpp:704–707`，`MyMatrix.cpp:121–131`转Float | `native_matmul`接收final_voxel时显式Float32 |
| voxel→RAS | `mri.cpp:13001–13002`，target×(voxel×sourceRAS2vox) | `registration.py:389`同两级括号；source inverse委托上述成熟函数 |
| mapmovhdr | 命令:875–883，RAS×originalvox2ras，然后写存储几何 | `registration.py:219–243`同顺序、源体素不变 |

`MatrixMultiply`为Float逐项累计（matrix.cpp:717–789）；候选 `native_matmul:209–215`也明确每次Float乘/加，不用TF32。Double `MatrixMultiplyD`仅用于 `extract_i_to_r`中心offset（mri.cpp:649）；候选 `_compose_native:83–89`保留Double offset再写Float。

官方中心写法是offset+三项空间和；候选先三项空间和再加offset。按源码的左结合Float序列，仅最后加数交换，不是已经证实的关联顺序bug。没有将它作为修补点。编译器是否融合另属未验证指令边界。

## 3. MGH/LTA表示边界及其作用

官方 `MRIsetVoxelToRasXform:13052–13072`默认保留delta，允许非单位Mdc并保存Float center；`mriio.cpp:11317–11335`直接写这些Float字段。候选默认同样保留delta，不按仿射强制归一化Mdc。旧评分确认四份源体素/shape/dtype不变、各自LTA/header一致≤1e-5mm。

一个独立的确实存在的metadata差异：官方 `getVolGeom/writeVolGeom`保留原VOL_GEOM字段；成熟 `_transforms.py:119–135`从nibabel affine重新求列长度、归一化direction及center。不能全局改这个成熟writer；如后续要求保存字段逐bit，应由B实验自己的字段保留writer隔离处理。

官方LTA用 `%.15e`，成熟writer用 `.15g`；两者均有超过Float32往返所需的有效数字。旧共享warp scorer `score_byteorder_recovery.py:194–195`直接用保存MGH字段，经同一Float inverse/matmul/sampler计算，并不用LTA geometry作为warp输入。组合点指标先将LTA矩阵转Float。因此LTA text/metadata差异不能直接解释旧warp两项或support1体素失败；未做变量交换评分，不量化它的独立贡献。

## 4. 已保存指标能证明什么

旧B的刚体LTA本身已有微差，仿射各方读取自己的刚体MGH，故后续仿射不是同内部状态起点的solver对照。保存的Mdc/Pxyz_c不同，通过同一sampler得到不同warp，这是记录直接支持的关系。

没有保存那个支持集体素的位置/双方值/邻域，也没有官方same-A/b、pyramid、halfway/state trajectory。因此不能把误差全部归给inverse、Float header、QR或某个FEQUAL shortcut。六个合同矩阵是保存RAS元数据，尚非每个实际inverse调用点的输入；非零差证明源码定义不同，尚未证明真实注册最早分叉已定位。

## 5. 最小局部方案

在独立B命名空间仅替换 `_sampling.native_inverse`，转入这里的逐项Float候选；旧B6源码、成熟ca inverse和CPU/GPU生产源一字不动。下一次真实两阶段仍读相同原输入，保存MGH并重读，以原20门对旧官方输出评分。原门、目标mask和源图像不变，不用最终评分反馈优化；原B负结果永久保留。具体有限计划见NEXT_PLAN，尚未执行。
