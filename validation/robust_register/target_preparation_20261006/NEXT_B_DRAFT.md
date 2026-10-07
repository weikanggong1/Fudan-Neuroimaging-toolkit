# 下一B：独立 robust rigid→affine 计划草稿

**状态：只读源码与计划；0注册、0 GEMS whole、0默认替换。** A仅在这一个1mm实例上通过目标掩膜/头信息门，不能代替B。真实B计算在完整源码/参数/冻结/基准计划审阅之后另获批准。

## 来源与许可

原安装版build `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`；mri_robust_register的已读二进制SHA `82b20927fdf9f650da81699a339577fbd24a6a2d9281fa9452cd1c6569c3334a`。参考源提交[d932c45](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)，14个相关文件的只读SHA见 `NEXT_B_SOURCE_BINDINGS.public.json`，未复制发布原文件。FreeSurfer许可证v1.0的PartB允许改编，须保留MGH归属、许可与修改标识。生产用PyTorch/nibabel和自有函数，不调用原软件或Surfa；原软件仅在隔离benchmark使用既有私密许可，许可内容不读入报告。

## 固定两次调用与头信息边界

```bash
mri_robust_register --mov alignedAtlasImage.mgz --dst targetMask.mgz \
  --lta rigid.lta --mapmovhdr alignedAtlasImage.mgz --sat 50 -verbose 0
mri_robust_register --mov alignedAtlasImage.mgz --dst targetMask.mgz \
  --lta affine.lta --mapmovhdr alignedAtlasImage.mgz --affine --sat 50 -verbose 0
```

使用A实际保存并SHA绑定的target/reflectedAtlas。第一次刚体更新头信息后，第二次按真实MGH存储边界重读其输出，再执行affine。mapmovhdr仅更新 `new_vox2ras = ras2ras @ old_vox2ras`，不重采样atlas体素。禁止用旧官方精细labels/affine做初始化或反拟合。

源初始化：robust/sat50、symmetryTrue、iscaleFalse、nomultiFalse、每层maxit5、epsit0.01、highit−1、inittransTrue、initorientFalse、initscalingFalse、doubleprecFalse、subsample−1。这不是GEMS mesh1e−10/30步，也不使用torch默认LBFGS。原RegRobust默认float A/b与FloatSVD/QR，小坐标/4×4状态Double；GPU以FP32/默认TF32分别验证实际运算，不使用FP16。

## 必須先补的底层源码定义

1. `makeIsotropic/setSourceTarget/getGPLimits` 的尺寸、center/2相位、RAS↔voxel链与centroid初始化；不能只按shape猜common space。
2. pyramid5tap `[1,4,6,4,1]/16`、BSpline half-downsample；梯度滤波 `[.03504,.24878,.43234,.24878,.03504]`、导数 `[-.10689,-.28461,0,.28461,.10689]`。**utils层MRI padding、BSpline边界和linearTransform尚未验收**，不能直接替换为grid_sample/Gaussian。
3. 对称halfway matrix square-root/inverse和有效row选择。A的目标nearest源规则不等于注册的连续插值规则。
4. analytic A/b、even median/MAD、TukeyIRLS最多20次/eps2e−12、sigma0时权重1、weightedQR、误差上升时恢复；需进一步核rank-deficient原分支与pivot/FloatSVD定义，不能改成AᵀWA。
5. 实际停止距离 `sqrt(AffineTransDistSq(Mnew,Mold,r=100))<=0.01`，刚体同样按原代码判停；并保留各层maxit5预算耗尽记录。

## 独立接口与最小验收

拟提供rigid/affine的独立Python API及双阶段组合。原GEMS默认alignment不变，任何新profile显式opt-in；不把CPU mesh profile启用等同于配准替换。CPU模块/采样/IRLS/小矩阵合同先行，保留CPU8/common锁、≤32GB RAM/20GB VRAM预检。体素A/b/filter/weights按slab批量处理，避免逐voxel Python循环，保持原median/QR/接受/停止数学。

真实运行拟分开授权：

- 一次官方最低层same-state analytic step参考，记录source/input/pyramid/A/b/weights/update；各state/sample一致后同点cost/residual/update relL2≤1e−5，不事后改门。
- 两次完整独立rigid/affine组件比较，fixed grid/header/值/order门exact，比较RAS矩阵、LTA读写定义以及bbox八角点/atlas代表点，事先目标位移RMS≤0.001mm/max≤0.01mm。保留MGH重读产生的float32边界，不以矩阵四舍五入匹配代替物理点位移。
- 精度通过后才CPU原/new同节点8核ABBA与GPU配对；记录import/IO/pyramid/filter/sample/A-build/MAD/QR/update/保存的时钟、CPU RSS/GPU allocation和TF32。稳定速度结论不得来自A一次scope不同的API计时。
- 组件通过后只验证recipe准备输出（working image/header/transform/stage mask/Gaussian超参数）；不自动再跑GEMS。最终完整ROI仍按每区Dice≥0.95、硬体积差≤5%，整体拟合另获批准。

本草稿未准备/派发B worker，也未接默认流程；底层边界审计未完前不进入数值benchmark。
