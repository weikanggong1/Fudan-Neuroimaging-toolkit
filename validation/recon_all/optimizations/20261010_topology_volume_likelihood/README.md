# 完整首缺陷 unMRI 体积聚合与 curvbak 反馈

## 1．功能与验证范围

独立Python/Numba移植固定内部函数的完整体积聚合及顺序反馈。保留k→j→i、三体素边界、±0.5mm、计数初值1、原NPY、FP32累加、max距离1–2mm、源float索引乘法、重复映射的逐次反馈。不调用原版程序；距离场由调用者提供，本门不代表距离内核、动态 mutation fitness、GA或原始T1整例通过。

## 2．Python、全部输入输出

`defect_volume_likelihood()` 的15个具名参数、类型/空间/单位、源索引公式、完整返回结构、失败行为与中文注释示例见[功能页](../../../../docs/recon_all/TOPOLOGY_VOLUME_LIKELIHOOD.md)。输入为同候选uint8局部强度、float32有符号距离/logP、源均值/geometry/fx/fy/fz/curvbak与有序原状态映射。输出源float32八字段及完整110,534个顶点的curvbak副本；不回写参考或生产文件。

## 3．命令行与复现

使用 `tools/benchmark_recon_topology_volume_likelihood.py`，具名 `--capture` / `--native-repeat` 为两个FNITVOL1快照，`--observer-receipt`为编译来源，`--output`为新独立目录，`--threads 4`为可恢复线程预算。只在诊断读取参考，不供生产。API容差预先固定为逐位0差异。

## 4．固定原软件对照

原 `mrisComputeDefectMRILogUnlikelihood_wkr` 体积聚合/反馈（3021–3122行），提交d932c45…，属于mris_fix_topology内部步骤、无独立CLI。参考以独立源码编译观察器在真实首个unMRI前后采集；program SHA `f9ca8e5e…`，两次完整capture SHA `7991e020…`相同。原源码/程序/输入SHA见source_rules与reports，不发布这些程序或影像。

## 5．真实精度、时间和图

公开ds000114 sub-07 ses-test LH，真实上游由79a41cdd连续链生成，本轮作为冻结同输入阶段。67缺陷条目，21×19×16局部volume，110,534顶点。8字段及全curvbak与参考不同数0、max0、P990，38顶点实际变化；原生/自有两次重复均全0。完整值见scores.csv和report.json，原参考字段不进入候选公式。

同主机、总4线程（原参考8–11、候选100–103）：热完整API **0.001911s**，首次API **0.210241s**（合同缓存已可用）；含加载/校验/两次API/NPZ/hash **1.148384s**，plot **2.246772s**，工具入口墙钟 **7.008993s**。后者排除先前解释器/bootstrap与最后报告写出。原程序 **6.723703/5.090987s**包含距离/准备/捕获，不是本聚合组件计时，无可用原生比值、无整例速度结论。GPU不用、显存null，共享负载未排除；整体等效未评估。

![当前真实首缺陷的冻结距离、更新和误差](reports/aggregate.png)

左侧距离是源输入而非自有距离核输出；右侧是真实同索引零误差，不将局部map当整例QC。

## 6．版本、失败与来源

7合同覆盖有序取消、源索引乘法与非对称截断、哨兵/边界、重复/丢弃/越界反馈和输入不变。首次部署在元数据路径核验失败、未启动计算，记录deployment_v1_failure；v2完整运行源码701d7ee…，工具6c830d2a…，两个exit0。原统计初值与原参考完整来源在服务器保留。

公开导出只隐藏私有路径/主机，不改变数值/类型；每份原始/公开SHA与叶级保真见PUBLIC_EXPORT_MANIFEST。PUBLICATION_MANIFEST绑定全部独占当前载荷。完整距离计算、mutation后缺陷、第二例和完整GA尚未验证。

## 7．参考与许可

[固定unMRI源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp#L3021-L3122)、[固定体积索引](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/include/mrisurf_metricProperties.h)、[原缺陷状态](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.h)。文献和来源通知见功能页。公共载荷只含FNIT独立源码、复现脚本、数值与授权公开数据衍生PNG，不含原cpp、MRI、NPZ、权重、许可内容或服务器地址。
