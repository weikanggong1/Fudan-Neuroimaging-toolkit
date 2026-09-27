# recon-all 数值验收门槛

`fnit-recon-all` 目前结合 Python/PyTorch 与六个由固定 FreeSurfer 8.2 源码在 Conda 内编译的 C++ 程序，其中三个必需、三个可选。运行时不用安装集群的 FreeSurfer。[构建记录](native_cpp_conda_20260927/six_target_build_20260927/REPORT.md)列出程序及哈希。仍需单独提供 FreeSurfer 许可证、模型权重和模板。

最近一次从同一 T1 完整运行的 v3 共执行 39 步，对照未经修改的 FreeSurfer 8.2 被试，严格比较通过 **19/138** 项：19 个 MRI 文件通过，47 个预期文件缺失，72 个已有文件不同。上游 WM/filled 的 13 个 MRI 文件在体素、数据类型、仿射和 MGH 头前 284 字节上相同。v3 的首个表面差异发生在 `qsphere.nofix`。此后代码改用[左右半球均匹配的 Python quick sphere](native_cpp_conda_20260927/v3_e2e_20260927/quick_sphere_lh.json)，以 `-ga` 拓扑模式写出 `orig.premesh`，再由[Python remesh](REMESH_VALIDATION.md)生成 `orig`。这些修改尚未经过新的整例 138 项比较。[v3 耗时及 ROI/顶点结果](native_cpp_conda_20260927/v3_e2e_20260927/benchmark_summary.json)只记录当时版本，不代表当前代码的精度或速度。

## 发布前要核对的内容

1. **整例运行。** 从同一真实 T1 和空被试目录开始，记录源码版本、输入、权重、模板与程序哈希，以及构建过程、命令、主机、设备、退出状态和各步耗时。进程树只应包含声明的六个 Conda 编译程序、Python 及其库，不应依赖已安装的 FreeSurfer/FSL 运行环境。
2. **体积图。** 核对每个 MRI 输出的全部体素、数据类型、仿射和 MGH 头；有差异时给出数量与首个坐标。`aseg.mgz`、`aparc+aseg.mgz`、`wmparc.mgz` 等下游图尚未通过整例验收。
3. **表面。** 逐半球比较 `orig`、`white`、`pial`、`inflated`、`sphere`、`sphere.reg` 的有序顶点、有序面片、体积几何信息、拓扑和标签。若顶点数或拓扑不同，最近邻距离只能用于诊断，不能算逐点通过。
4. **顶点指标。** 在相同顶点顺序下比较厚度、white/pial/mid 面积、顶点体积、曲率及其他输出图。每张图、每个半球都记录最大误差、分布和超阈值顶点数；验收要求超阈值数为零。
5. **脑区和统计。** 比较注释图的有序 ID，并逐行核对 aparc、a2009s、DKTatlas、aseg 和 wmparc 的脑区结果及全局指标。缺失文件、数值差异和格式差异分开记录。
6. **耗时。** 数值门槛通过后，才在相同主机、输入和线程预算下逐步配对比较候选版与官方版。记录冷/热运行、I/O、设备、GPU 初始化和共享节点负载。未达到等价输出的整例耗时不能作为加速结论。

[严格比较器](compare_complete_subject.py)检查 138 个预期文件：39 个 MRI 输出、18 个表面几何、46 张顶点图、12 个注释和 23 个统计文件。官方存档被试与自身比较通过 138/138；人工改动 0.02 mm 厚度或 0.02 mm² 面积时，比较器分别只报对应图失败。这证明比较器能检出小误差，并不证明候选重建已通过。

## 当前实测边界

生产入口在保存的 T1 MRI 前缀之后仍采用近似表面流程。v3 的 **19/138** 是旧完整运行结果；当前代码尚无新的整例通过率，不能据此声称皮层指标一致或重建加速。后续[左半球候选 MRI 中间结果复跑](white_connected_prefix_20260927/README.md)中，`orig.premesh` 和 `orig` 的有序几何完全一致；`white.preaparc` 平均位移 0.000421 mm，50 个顶点超过 0.1 mm；三轮 `smoothwm` 平均位移 0.000311 mm，18 个顶点超过 0.1 mm。这表明先前将该输入差异归因于 Conda 拓扑修复的判断需要修正。 [首轮内存对照](../../../docs/recon_all/WHITE_PYTHON_FIRST_PASS.md)还显示当前 Conda 白质放置器从首轮便与已安装官方程序出现尾差；Python 首轮与官方第 17 步最大顶点差为 1.64×10⁻⁵ mm，但剩余三轮尚未实现。该测试既不是从 T1 开始的单进程重建，也未覆盖右半球。

[修复后的候选左半球完整 `sphere`](candidate_sphere_first_difference_20260927/full_stage/README.md)在相同候选输入上与官方的 106,622 个有序顶点和 213,240 个有序面全部一致；Python/官方墙钟为 884.56/328.61 秒。两者与归档官方球面的平均差都为 2.959 mm。[上游审计](candidate_sphere_first_difference_20260927/full_stage/UPSTREAM_FIRST_DIFFERENCE.md)发现第一个已保存的几何差异在 `white.preaparc`。这一阶段验收不能外推到 `sphere.reg`、注释、最终 white 或整例。

独立的 [final white](../../../docs/recon_all/FINAL_WHITE_CONDA.md) 和 [pial.T1](../../../docs/recon_all/PIAL_T1_CONDA.md) Conda 接口只在冻结的官方输入上测试。后者有 5,571 个顶点超过 0.1 mm；相同输入下的独立 Python pial.T1 几何则与官方一致。recon-all 的 SynthMorph Talairach 仿射调用现已[局部关闭 TF32](talairach_tf32_isolation_20260927/README.md)，在此 T1 上将 eTIV 误差从 822.547 降到 0.895 mm³；有限续跑中的 GCA、`norm` 和 `aseg.presurf` 仍匹配。此后尚未重新进行 138 项整例比较。最终 white/pial 几何、双侧 `sphere.reg`、注释、后处理体积图、全部顶点图及逐脑区统计仍需从真实 T1 连续验收。
