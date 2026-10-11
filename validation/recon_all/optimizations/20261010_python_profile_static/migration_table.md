# v4 固定 Python 配方：静态迁移核对

| 阶段 | v4 实际实现 | 原版阶段程序 |
| --- | --- | --- |
| profile/程序发现 | 固定21选项，不消费原版程序/能力探测 | 未选中 |
| conform/SynthStrip/Talairach | 自有Python/Numba与Torch GPU | 未选中 |
| N4/nu | SimpleITK Python API通用ITK CPU库；自有nu转换 | 未选中 |
| 两轮normalize/brainmask | 自有Torch GPU与有序Numba CPU混合 | 未选中 |
| SynthSeg/辅助网络 | 自有Torch GPU | 未选中 |
| GCA registration | 完整自有Python/Numba EM与Torch GPU评分/求逆 | 未选中 |
| CA normalize/corpus callosum | 自有Python/Numba CPU | 未选中 |
| WM segmentation | 自有Python/Numba有序CPU与Torch GPU直方图/缓存平面 | 未选中 |
| WM aseg/pretess/Ento-ACJ | 完整自有Python/Numba CPU编辑与Torch GPU后编辑 | 未选中 |
| filled | 完整自有Python与有序Numba CPU切割/堆/标签 | 未选中 |
| tessellation/topology | 自有Torch tessellation与Python/Numba数字拓扑 | 未选中 |
| remesh/intersection | 自有Python/Numba有序CPU | 未选中 |
| white.preaparc/white/pial | 完整四轮Python/Numba CPU与Torch GPU候选/清理 | 未选中 |
| inflate/sphere | 自有Torch inflation；sphere CPU目标/步长与GPU平均/收尾 | 未选中 |
| sphere.reg | 自有Python/Numba CPU目标/步长与Torch/Triton GPU有序平均 | 未选中 |
| avg_curv/extra annotation | 自有Python/Numba CPU采样与标签映射 | 未选中 |
| GCSA annotation | 自有Torch GPU特征与有序Numba CPU Gibbs | 未选中 |
| defects volume | 自有Torch GPU投射与Python格式写出 | 未选中 |
| thickness/area/curv | 自有Torch GPU几何与SciPy/Numba CPU空间查询 | 未选中 |
| curv.stats/八派生图 | 自有Torch离散曲率与有序Numba统计/系统libm | 未选中 |
| jacobian/contrast/ribbon/volume maps | 自有Python/Numba与Torch CPU | 未选中 |
| ROI/aseg/wmparc/brainvol stats | 自有GPU统计基量与Python/Numba CPU归约/写出 | 未选中 |
| MNI affine/nonlinear/inverse/check | 自有SynthMorph Torch GPU与GPU采样/逆场 | 未选中 |
| 输出/网格校验 | 自有Python/NumPy/Numba CPU | 未选中 |

仅核对上述冻结源码调用条件，不替代实际整例执行/动态库/文件访问审计或干净环境验收。原版混合分支保留为独立对照，不被此配方消费。数字拓扑不是 FS GA；整体指标等效尚未判定。函数/文件 SHA 与已发现 WM 报告字段问题见 static_audit.json。
