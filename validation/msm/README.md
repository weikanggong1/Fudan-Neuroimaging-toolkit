# MSMSulc 配准验证

用法与参数见 [MSMSulc 功能页](../../docs/msm/README.md)。本目录保存匿名汇总；原生球面、逐点成本表和真实 BOLD/CIFTI 留在验证服务器。

## 对照定义

- 两端使用同一例真实 UKB 的双侧初始球面、脑沟图和 HCP 参考资源。
- 配置为四级 `AFFINE,DISCRETE,DISCRETE,DISCRETE`，`simval=3,2,2,2`，最大迭代数 `50,10,15,15`；控制网格、标签网格、应变项和提前停止规则相同。
- 正式精度参照为 FSL Conda 的 `fsl-newmsm 1.0 h442c261_5`，单 CPU 线程。已核对其 45 个算法与优化器源码文件和 MIT 源码提交 `260718953547743c028a45f8c885d163441df87a` 逐字节一致。8 线程测量单独报告。
- 球面角差按相同原生顶点计算。490 帧时间相关先在每个有效灰质坐标内计算 Pearson r，再取算术均值；常数时序不计 r。
- 投影双方使用同一固定 clean volume、几何、ROI 和 Workbench。每套注册球面各自生成 32k midthickness 面积表面，再做 ADAP_BARY_AREA；皮层下路径相同。
- 配准耗时包含输入读取和球面/报告写盘，排除 Python 导入、CUDA 上下文初始化、离线精度计算及 BOLD 投影。冷、热调用与带 profiler 的调用分开。

## 已定位的差异

官方三角形面积在构网时缓存，归一化坐标后不会重算。FNIT 现保存同一缓存面积。相同真实几何下，修复后归一化特征误差降至约 `5×10⁻¹⁶`，float32 特征逐位一致。

官方浮点配置为 `Option<float>`，再提升为 double。直接使用 Python double 会改变步长和应变系数。按官方精度读取 shear/bulk 后，同一真实几何、特征和标签的 2,560 项候选成本最大误差从 `4.03×10⁻⁷` 降到 `7.99×10⁻¹⁴`。这两个检查点验证局部算子，完整球面及时间序列指标需另行测量。

刚性初始化须保留每次实际旋转后的 SOURCE 坐标；用累积旋转矩阵重新生成坐标会丢失舍入历史。在真实调用中，GPU 标量除法与指数的舍入组合使 77 次成本中有 7 次相差最多 1 ULP，并通过顶点/边上的邻域选择放大。采用独立 C++ double 除法、libm 指数和原顺序累加后，全部 77 次成本及每次 40,962 个逐顶点值逐位一致；组合运算的反事实测试中，完整 AFFINE 与首级 DATA/控制点坐标也逐位一致；正式实现还需复测这一检查点。这项修复位于 `fnit.msm`；surface pipeline 继续调用它，并保存最终球面质控。

HOCR 重复高阶项的合并顺序已修复。当前严格浮点编译配置的 [13,000 个单元用例](native_hocr.current.public.json)对官方 ELC→FastPD 标签均一致，包括重复项与接近并列的成本。该项是优化器回归测试，不作为真实脑配准精度结论。

## 复测

验证服务器准备私有 JSON，包含 `inputs`、`official_spheres`、`config_file` 以及可选 `projection`。`inputs` 为 L/R 两套 `MSMSulcInputs` 的文件路径；`projection` 使用同一固定 volume 和几何。正式测量前，独立运行官方 newMSM 生成参照球面。

```bash
python tools/benchmark_msmsulc.py \
  --case-json /absolute/path/case.private.json \
  --source-root /absolute/path/Fudan-Neuroimaging-toolkit \
  --output-dir /absolute/path/new-benchmark-directory \
  --device cuda:0 --warm-repeats 1 --profile-full
```

脚本调用 FNIT 配准与 Workbench 投影，不调用官方 newMSM。`report.safe.json` 保存来源哈希、实际配置、精度、阶段墙钟、CUDA kernel/同步事件及显存；输出目录中的球面和 CIFTI 为私有验证文件。

GPU 整卡利用率包含其他进程的负载；Torch 峰值已分配显存属于本进程。阶段计时为嵌套主机墙钟，CUDA kernel 时间单列，带 profiler 的时间不替代冷/热配准时间。
