# 官方细分割的重复性：源码审查

## 审查范围与版本

2026-10-02 在 gpucw1 读取实际安装的 FreeSurfer 8.2。build stamp 为 `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`，SAMSEG 为 `0.5a0+17.g2ce2b6b`。审查对象是当前对照所用的 Python `segment_subregions`，包括脑干、丘脑和左右海马–杏仁核；不是旧 MATLAB 可执行程序，也没有重跑 recon-all。

[安装源码审计](official_source_audit.json)记录 8 个 Python 源文件、原生 GEMS `.so`、`segment_subregions`、`fspython` 和 `mri_robust_register` 的路径、大小和 SHA-256。CLI 和核心 Python 文件与 SAMSEG 上游 `2ce2b6b` 的内容逐字节一致，见[上游内容校验](upstream_source_identity.json)。原生注册源码按实际 FS build 的 `d932c45` 核查，见[注册源码清单](registration_source_identity.json)。这里只保存审计信息，没有复制发布官方源码。

## 固定输入、参数和线程

官方 CLI `samseg/cli/segment_subregions.py` 第 29–37 行列出参数，第 51 行设置 GEMS 线程数；没有 `--seed` 参数。8 个被调用的 Python 源文件的 AST 审查未发现显式随机数调用。第 70–73 行为海马–杏仁核固定处理左、右侧。`core.py` 第 105 行的随机临时目录名字只影响文件位置，未用作拟合参数。

官方 GEMS 的[栅格器头文件第 8–20 行](https://github.com/freesurfer/samseg/blob/2ce2b6b/gems/kvlAtlasMeshRasterizor.h#L8)说明：同一线程数下计算有确定性；改变线程数会改变每个线程的浮点累加，细小舍入差异可能在迭代中放大。`CROSS_THREAD_REPRODUCIBLE` 编译选项改用 float128 线程累加器。当前安装二进制的该编译选项没有可核实的构建记录，记为未知。

[栅格器实现第 101–126 行](https://github.com/freesurfer/samseg/blob/2ce2b6b/gems/kvlAtlasMeshRasterizor.cxx#L101)按固定 round-robin 次序给每个线程分配四面体；[代价与梯度实现第 184–219 行](https://github.com/freesurfer/samseg/blob/2ce2b6b/gems/kvlAtlasMeshPositionCostAndGradientCalculator.cxx#L184)按固定线程顺序合并累加。由此，改变线程数属于数值路径敏感性对照；不能直接称为改变随机种子的重复试验。

## 初始线性配准的随机分支

安装 `samseg/subregions/core.py` 第 270–271 行调用两次 `mri_robust_register`，分别执行刚性和 affine 配准，均使用 `--sat 50`，没有 `--subsample` 或 seed 参数。

FS [mri_robust_register.cpp 第 70 行](https://github.com/freesurfer/freesurfer/blob/d932c45/mri_robust_register/mri_robust_register.cpp#L70)将 `SSAMPLE` 默认设为 −1；第 138 行将它写入默认参数。[RegistrationStep.h 第 443–447 行](https://github.com/freesurfer/freesurfer/blob/d932c45/mri_robust_register/RegistrationStep.h#L443)只有在 `subsamplesize > 0` 时才启用含随机 offset 的采样分支。因此当前官方细分割的默认初始配准不进入这个分支。`Registration.cpp` 第 3169–3220 行由影像的 RAS/header 或强度质心初始化矩阵，未采用随机姿态初始化。

不能据此宣称整套 FreeSurfer 在所有机器、版本和编译条件下逐位相同；当前同输入、同参数、同机器、同 4 线程的实际重复结果才是本轮的容差依据。

## 正确的逐脑区判定

需分别报告三个量：

1. 官方重复之间每个标签的 Dice、改变体素数、硬/软体积差：官方本次计算重复波动。
2. FNIT 重复之间的相同指标：我们自身的计算稳定性。
3. FNIT 与各次官方结果的指标：跨实现差异。

逐区验收分别记录运行稳定性和跨实现精度。FNIT 自身波动落在官方本次重复范围内，即达到这项稳定性要求；按照本任务允许部分体素差异的标准，不要求跨软件 Dice 为 1。官方重复完全一致时，较低的跨实现 Dice 属于固定算法或预处理差异，不能解释为官方随机性。优化前后另比较这些固定差异，保留具体脑区的改善和回退。

固定同阶段 `norm/aseg/wmparc` 的比较用于判断细分割算法。raw 的差异还包含 SynthSegPlus、偏置校正和重采样差异，另行报告。扫描–重扫、改变图像噪声、不同 recon-all 输入、网格平移和不同软件版本均不是固定输入下的算法随机波动；不能混入同一容差。双方空标签仍记 `both_empty`，不计为 Dice 1；一方空标签且另一方非空则 Dice 为 0。

## 官方重复命令

```bash
# 已处理的公开真实 T1；三项 mri 文件保持同一内容和 SHA-256。
benchmark_subjects_dir=/absolute/path/reconall_reference_gpucw1
benchmark_output_dir=/absolute/path/reproducibility/official_r1
export FREESURFER_HOME=/public/software/apps/Freesurfer/8.2.0-1
export PATH="$FREESURFER_HOME/bin:$PATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4

# 每个结构使用独立输出和临时目录；再用 r2/r3 启动新进程。
for structure in brainstem thalamus hippo-amygdala; do
    segment_subregions "$structure" \
        --cross fs_sub01 \
        --sd "$benchmark_subjects_dir" \
        --threads 4 \
        --out-dir "$benchmark_output_dir/$structure" \
        --temp-dir "$benchmark_output_dir/temp_$structure"
done
```

官方 CLI 不提供 seed 设置；不应添加它不识别的 `--seed`，也不应修改官方源代码强加随机初始化后作为默认重复性范围。若需测试线程敏感性，单独固定 `--threads 1` 并同步调整上述环境变量，保持软件、输入、atlas 和全部其他参数相同。
