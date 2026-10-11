# 完整九缺陷面缓存与只读末态对照

本目录仅保存服务器安全导出的类型指标、完成收据和派生PNG。完整MRI、NPZ、原日志、地址和私有执行记录留在服务器。实际功能与具名参数见[中文七节说明](../../../../docs/recon_all/TOPOLOGY_GENETIC_REPAIR_CACHE.md)。

## 实测版本与完成门

- 缓存九缺陷v3：源码MF `3060001b4eff1359afefde017512fa4f2cd581e284b515e8e25bbc0a5a24f104`；采用早期Python阻挡/诊断缓存包装，没有采用当前Numba/query-half/distance route。9/9、4879评分、9索引准备、4888索引消费，完整fresh CLI15196.729397秒。
- 标准orig：父真实完成后，三轮remesh、ORIGINAL清理与严格质量均完成，CLI183.192466秒；等待另列。Premesh108746V/217488F、orig114414V/228824F，source与独立F64相交均0；Euler2/单连通/其余结构0。
- 原域缺陷标签真实F32 morph写出并读回相同：110534顶点、fnum221116、0–9、1996非零。该文件不配压缩后的顶点索引。
- 旧Python基线实际PATCH清单 `9ddb21a2…`、9/9完成：11NPZ/404数组字段、有序面/F32XYZ/六项F64几何均0差。九完整控制按类型逐项JSON相等，不证明浮点符号零或JSON文本字节一致；历史字段ordered_nine_controls_bit_exact保留原件。两文件仅surface前导创建文字不同，footer同字节。69已记录模块内容闭合为56PATCH+13显式继承；原077归档容器未定位，单列未验。
- 旧Python完整fresh CLI31419.875133秒，内部tool wall31414.436612秒。不同四核与共享时段，不属于受控速度配对。旧RNG只18353710总数可比，没有旧bounds/value SHA。
- 固定独立FreeSurfer程序default/verbose同输入重复XYZ/ordered faces/numeric geometry相同，但该原生网格108710V/217416F，不能逐索引比缓存产物。原生程序只用于benchmark，不进入自有算法。
- 共同scannerRAS的完整双向点到三角面：自产→原生mean0.001129989/P99约2.16e−9/max1.089618mm；原生→自产mean0.001789094/P99约2.14e−9/max6.828374mm。0.1mm仅显示阈值。各自真实头的坐标约定变换，无拟合/配准/重排序。

## 目录

- `premesh/`：九缺陷完成阶段的typed metrics和producer身份收据。
- `orig/`：标准后续orig的typed metrics及原执行绑定。
- `python_baseline/`：完成的404字段后验对照；比较6.684100秒单列，不计入GA。
- `native_shape/`：完整距离typed指标/旧严格F64同头失败/两张真实T1与误差图；188.753740秒为独立诊断。
- `figures/`：自产修复轮廓和阶段计时；位移是修复运动，不是原生误差。
- `contracts.json`：31项实际focused合同，本地0.227秒，不是MRI benchmark。
- `source_bindings.json`：各完成阶段自己的实际MF/程序/产物SHA和继承边界。当前HEAD与本目录实测科学源码分开。
- `setup_history.json`：旧准备失败和导出schema修复，原回执不覆盖。
- `PUBLIC_MANIFEST.json`：本次发布的公开文件逐大小/SHA。

完整原始T1 recon-all、后续脑区指标及整体等效均 `not_assessed`。当前首缺陷距离serial/parallel完整配对独立运行，不包含在本目录科学结论中。
