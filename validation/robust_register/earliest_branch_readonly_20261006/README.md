# B 最早分叉观察点：只读源码审计

本次读取固定源码、索引和已有元数据；0 图像解码、0 采样、0 配准、0 求解、0 优化器更新、0 编译、0 GEMS/GPU。正式验收仍 **17/20，3 项失败**，relL2 门 `1e-5`、support 门0不变。support 角点分类是保存几何经过同一采样器得到的观察，尚未证明优化器原因。仿射各方读取自身不同刚体 MGH，不能用最终仿射差异作为同状态 solver 对照。

## 1. 唯一优先点

观察 **rigid 在重排和等距准备完成后、pyramid/halfway/QR 前的 Double 初始化矩阵 M0**。官方 `Registration.cpp:712` 调用初始化，`3192–3193` 求两侧质心，默认平移在 `3379–3381` 写入，`3393` 返回。候选 `registration.py:325–332` 对准备后的图像求质心并写 M0，pyramid 从338行开始。

这个位置既早又小：一对实际3元素质心和16元素矩阵即可观察；必须同时私密绑定两侧准备图像的 shape、dtype、outside值、几何 bits 和同一序列化顺序的体素 bits。准备状态不同，先查准备；准备相同而 M0 不同，再查质心求和/组装；M0 也相同，后续才考虑首个 A/b 入口。**这里选的是下一观察位置，尚未测得这个位置分叉。**

官方质心以 Double 按 z/y/x 串行累计 (`CostFunctions.cpp:1313–1330`)；B 转 Double 后先求轴边缘和再求加权和 (`registration.py:163–172`)。坐标均从+1开始，默认 outside0 下的正值图像公式相同；累计顺序不同。没有同数组/同状态结果，不能据此认定它导致当前失败。

## 2. 求解、库路径和停止

| 边界 | 固定官方 d932c45 | B/inverse_real 候选 | 源级结论 |
|---|---|---|---|
| 默认数值类型 | `Registration.h:68–71` doubleprec=false；`RegRobust.cpp:67–70`选 Float；小矩阵为Double | 图像、A/b、QR Float32；小矩阵Double，`registration.py:398–400` | 类型目标一致，运算顺序和 backend 未因此相同 |
| robust线性解 | `RegRobust.h:172`名为FloatSVD；实际 `Regression.cpp:403–431`使用 `vnl_qr<float>` | `_solver.py:36–49`显式 reduced QR、Q.T@rhs、triangular solve | 此活动路径是QR；`Regression.cpp:547`的SVD属于另一个非robust LSE路径，旧SVD替代实现被注释 |
| QR内核 | ITK4.13.2 `vnl_qr.hxx:23–30,51–60,185–213`→LINPACK sqrdc/sqrsl，无pivot | PyTorch2.5.1 upstream `BatchLinearAlgebra.cpp:2394,2416`→GEQRF/ORGQR，Float LAPACK绑定在867–884；再显式矩阵乘/三角解 | Source-defined库路径不同；没有 installed binary 逐指令/真实同A/b trace证明 |
| IRLS误差累计 | `Regression.cpp:284–295`以T(Float)逐行求 weighted residual error | `_solver.py:84–86`用Torch矩阵乘及 sum reductions | 是待观察边界；不能从最终矩阵微差指定原因 |
| IRLS停止 | max20、绝对EPS2e-12、误差不降停、严格升错回滚p/w，`Regression.cpp:206–208,297–320` | 同语义，`_solver.py:69,91–108` | 未发现本profile停止政策接线差；末位值可影响触发时机，实际同轨迹未捕获 |
| halfway | `MyMatrix.cpp:1321` complex Double Schur；1731–1761 调 zgees，内部三角递推 | `registration.py:190–216` SciPy complex Schur/递推，NumPy SVD只作退化门 | 不能把SVD gate称优化器solver；矩阵乘、inverse和库工作区仍待同输入观察 |
| 更新和外停止 | `RegRobust.h:230–232` `(inverse(mhi)×update)×mh`；264–275用 r100 affine distance | `registration.py:368–378`同括号，`_solver.py:177–180`系数2000 | max5/level、distance≤.01来源见既有 frozen PLAN；本次不改变 |

现场只读 package metadata 为 NumPy1.26.4、SciPy1.17.1、Torch2.5.1。SciPy实际 `_decomp_schur.py:169,172,201`选择 gees、先查最佳工作区，再调用；NumPy inverse/SVD调其扩展。**这是当前环境的源码元数据，不重建旧B运行时库版本/动态provider。** Source hashes见 SOURCE_BINDINGS。

## 3. Double matrix → Float → MGH/LTA

1. 官方 `Registration.cpp:4422`计算 Double `Rtrg×Mfinal×inverse(Rsrc)`；B `registration.py:387`计算对应Double复合。前者VNL inverse、后者NumPy inverse的同状态结果未捕获。
2. 官方 `mri_robust_register.cpp:704–707`经 `MyMatrix.cpp:121–131`转 MATRIX_REAL(Float)；B `registration.py:389`通过 native_matmul 输入边界转Float，再执行 `target×(voxel×inverse(source))`。官方同括号见 `utils/mri.cpp:12998–13002`。inverse_real只隔离替换 native_inverse，solver/registration仍原B字节；loader位置见绑定清单。
3. 官方先 `mri_robust_register.cpp:754–779`生成LTA，再875–884作mapmovhdr并写MGH。B先在API内建立transform/header，worker135–138先nib.save MGH，再transform.save LTA。顺序不同；尚无证据它改变本例内存矩阵或是原因。
4. 官方 `utils/mri.cpp:13052–13072`保留原delta，Float写directions/center；`mriio.cpp:11317–11335`写Float字段。B `_header_after_matrix:219–243`相同存储目标，禁止nibabel重导字段；体素不重采样。原合法非单位Mdc不得被当作需修正的归一化错误。
5. 官方 LTA matrix `%18.15le` (`transform.cpp:3255–3258`)，几何 `%.15e` (460–464)；成熟FNIT writer `.15g` (`_transforms.py:125,155`)，从 affine重建几何。均可提供Float32往返需要的有效数字，但metadata重建与存储字段保留是不同路径。旧共享warp评分读取保存MGH字段，不读LTA geometry；不能把文本格式差异直接作为support/warp失败原因。

## 4. 有限草案及实现门

[PLAN.draft.json](PLAN.draft.json) 给出一组两侧rigid初始化前缀：两worker、各1次准备和初始化，随后硬退出；0 full rigid/affine API、0 halfway、0 QR/IRLS、0更新、0最终MGH/LTA、0GEMS/GPU、0重试。建议原CPU8、共同锁、20GB、lock120/child120/outer420。原初始化前准备需要真实图像与条件性reslice，所以此组须另行授权，**本次没有运行**。

两侧tap、controller、prepared字节序和runtime/provider还未实现/冻结。现有verbose输出不足以证明IEEE64逐bit状态；`--debug`会继续写/采样、等待交互，不能把它当受控早退出。若无法审计既有安装binary的精确tap，应停；如改用独立冻结源码参考，需要另批构建并单列身份，不能宣称其等于安装binary轨迹。本草案不提供不存在的执行CLI，不自动进入下一A/b或全B。

私密模式700保存实际质心/M0/几何与必要 prepared arrays。公开只给source SHA、比较类别/计数、状态及资源收据；不包含影像、atlas pixels、坐标或矩阵值。19/23旧报告和生产源码保持原样。

## 5. 源与范围

37个已有固定源码绑定全部现场 bytes/SHA通过；另仅下载ITK4.13.2 QR/SVD和PyTorch2.5.1线性代数源用于只读路径核对。原软件源码保持私有，不复制进此leaf。完整SHA、原实现URL和精确行号见 [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)。根 README/INDEX已现场读取，正式repo HEAD为 e41dc9d，工作区干净；本leaf在既有B独立WT追加。
