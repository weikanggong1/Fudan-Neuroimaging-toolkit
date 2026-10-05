# MS-HBM CPU 官方对照与优化验证（2026-10-04）

本轮基线为 `cc9402734faeba93b3a13c29932fa1392eaccf62`。完整真实 490 帧的冻结/优化 FNIT CPU1/CPU8、13 项体积/参数/CLI 矩阵和 4 次完整 GPU ABBA 已完成。独立全帧几何与网络核查完成，既有单元门槛 7 项通过、1 项因 CPU 节点无 CUDA 跳过。固定原 vendor 的实际 reader 已读入完整 490 帧，独立比较的全部 29,111,880 个皮层值差异为 0；原版／冻结／优化 FNIT 的完整 CPU1、CPU8 六项配对已启动，尚无新的官方分区精度/时间结果。先读取服务器固定入口 `FNIT/README.md`、`FNIT/INDEX.md`，再使用本任务私有绑定；旧源码、旧输出与 Conda prefix 保留原位置。

最新七节功能与结果说明见[CPU benchmark](../../../docs/mshbm/cpu_benchmark_20261004.md)，机器可读完整旧新证据见 [cpu_hoist_control.public.json](cpu_hoist_control.public.json) 与 [cpu_hoist_receipts.public.json](cpu_hoist_receipts.public.json)。

其他已完成证据：[完整 CPU 功能矩阵](full_api_matrix.public.json)、[跨 API/CLI 的科学文件核查](cross_api_controls.public.json)、[独立全部帧核查](independent_oracles.public.json)、[GPU ABBA](gpu_abba_control.public.json)、[既有测试门槛](focused_tests.public.json)。[21 条完整新进程观测 CSV](fnit_complete_observations.csv)分开记录函数链、过程墙钟、user/system CPU time、非函数链开销、分步骤与源码 SHA；非函数链开销包含导入、校验和报告 I/O，不等于纯解释器启动。

## 当前功能与覆盖矩阵

| 功能 | 完整真实数据覆盖 | 官方参照／检查 | 精度与速度记录 |
|---|---|---|---|
| 单 run CIFTI | 完整 490 帧、两侧各 32,492 顶点，取 59,412 皮层顶点；前后各 245 帧 | 干净 CBIG 单被试 wrapper，全局 top 10% profile、原版完整迭代 | 完整 profile 值、64,984 标签、逐网络 Dice；CPU1/CPU8 启动到写出与 API 分步 |
| NPY 布局 | 同一完整时序的 `T×59412`、转置、`T×64984` 和转置 | 与 CIFTI 入口逐值比较；保留所有帧和顶点 | 读取及标签／输出一致性；不将格式转换时间混入推断速度 |
| 多文件 session | 真实多 run 资源待协调者选择；单 run 的两个 pseudo-session 可用于接口控制 | CBIG `num_sess` 与一文件一 session；完整 run，不复制 run 冒充独立采集 | session 数、每 run 全部帧、profile／标签；独立采集与接口控制分列 |
| censor | 无 censor 主样例完整 490 帧；可选控制预定义为同 run 的 DVARS P95，完整执行尚待完成 | 同一 490 行 0/1 向量，先保留帧后拆半 | 保留帧数、profile、网络 TSV 时间轴；该控制不作为运动质控建议，不降低主速度样例的工作量 |
| HCP_40 prior／mask | 官方固定先验、fsLR32k mask／seed／graph；`--assets` 使用相同 NPZ 控制 | `mu/sigma/epsil/theta` 与原始 MAT 逐值相同，medial mask 0 差 | 资产 SHA、shape、标签 0 与网络 1–17、缺失网络行为 |
| `w`／`c` | 默认 200／50；改变 prior／MRF 权重的真实输入控制 | 官方同值参数 | 各配置原版标签对照、收敛记录、耗时；不改变配置换速度 |
| 迭代上限 | Python `50/101/300/101` 原有预算 | 官方 outer 50、EM/lambda 101、M-step 按原收敛结束 | 原预算主 benchmark、未收敛报错；原版 M-step 无 300 次上限，应记录此接口差别 |
| 输出／CLI | NPY、左右标签、CIFTI dlabel、17 网络时序与 Pearson 矩阵、provenance | 原版标签；网络均值／FC 用完整原始时序独立核查 | 数值、shape、dtype、网络顺序与缺失网络 NaN；真实 CLI 和 API 逐值比较 |
| MNI 2 mm 体积 | `91×109×91×490`，完整双侧中层表面、同网格皮层 mask | 表面推断仍对照 CBIG；CBIG 没有原生体积入口，投影及回映射另作完整 SciPy 几何检查 | 读取／采样／profile／推断／输出／回映射分别计时；全部帧及体素误差 |
| `device/frame_chunk/max_distance_mm` | CPU 主样例；H100 原默认 TF32、chunk 8、距离 3 mm，并覆盖现有参数控制 | 相同冻结输入，旧／新完整 volume API 输出、时间与显存 | core 本身无 CUDA；修改共享 CPU core 时仍核查 GPU 输入的完整 API |
| 随机种子 | 当前公开 API 和 CLI 没有种子参数，推断无随机抽样 | 重复运行相同输入和预算 | 记录确定性；不新增不存在的参数 |

## 已现场核验的资源

完整官方 CIFTI 为 `490×91282 float32`，TR 0.735 s；左右 cortex 的原始 fsLR 顶点数均为 32,492。完整 MNI BOLD 为 `91×109×91×490 float32`，2 mm 网格。主样例 CIFTI SHA 为 `ebf2f4ac854e5e521819b5182233442a6b18271c81e62c4609f03cc0cbb4dcdd`。

FNIT HCP_40 NPZ 为 1,497,800 字节，SHA `aece34ff3651a10e44c8905d5eac32a1e322acd5d3b028d54c0fbe05ad3f7c17`；官方 `Params_Final.mat` 为 956,976 字节，SHA `3d6cd0b4aa77bf9549c2edc7b37baf667d211c0b7db952cd22aba89f27be6002`。四组 prior 参数逐值差为 0，59,412 顶点 mask 差为 0。两个 CBIG 群体 MNI 中层表面的大小与 SHA 已与 `assets_setup.py` 固定清单核对。HCP_40 算法／先验沿用 CBIG MIT；Caret 来源表面只复用原站获取的资源，仍不随 FNIT 分发。私有影像和路径不进入此仓库。

官方代码 commit 为 `b69b822a15e2a94f1e439606552fc44b6858cf3c`。现有磁盘 `CBIG_MSHBM_generate_individual_parcellation.m` 添加了 `.npy` 兼容条件，磁盘 SHA 为 `ddfdb0420a8e6e8c283300880b3d131368a25a930f213bcd1821fd225c03b4ed`；原 Git 对象 SHA 为 `ce5c5a8596994c19c88e1a0ddb65dd28b4e3c31ded5887a89dacd667f4d0b3bb`。简化 wrapper、`CBIG_MSHBM_read_fmri` 和 `CBIG_ComputeCorrelationProfile` 的磁盘 SHA 也与原对象不同。`export_reference.py` 从固定 Git 对象导出干净函数和所需 MATLAB/mesh 资源，写入服务器此任务的私有 reference，并记录原磁盘与干净文件 SHA；不会修改原 CBIG checkout。旧报告绑定原脚本，不作为新测量。

`nodecw8` 的 MATLAB 正式尝试因授权 hostid 不匹配退出，保留完整失败记录。`nodecw10` 的实际合法启动随后返回 0，版本 `9.5.0.944444 (R2018b)`，实际 `disp(version); exit(0)` 耗时 114.553 s，无许可证错误。官方 CBIG、冻结 FNIT 与优化 FNIT 均在 nodecw10 同一组物理核重跑；CPU1 `[2]`、CPU8 `[2,6,10,14,18,22,26,30]`，同 socket 且 core 均不同，实际 `flock` 串行。节点负载约 2,500，保留墙钟与 CPU time，不能作稳定速度结论或跨节点相除。

nodecw8 的冻结/优化 FNIT 控制使用 CPU1 `[2]`、CPU8 `[2,14,18,22,26,30,34,38]` 与 task03 锁。528 个干净参考文件及 fsLR900 downsample seed 已部署到任务私有目录，完整输入预检和实际冻结源码导入通过。

## 复现接口

完整资源绑定放在外层 `outputs/fnit-fmri-cpu-20261004/task03_resources.private.json`，不提交。正式运行先将下面的占位路径替换为本任务的冻结源码、私有绑定和新的空输出目录。CPU 亲和性、共用锁及 `/usr/bin/time -v` 由统一协调器施加。

```bash
# 安装时／准备时导出参考，输出必须为新目录；只在服务器私有目录保存原软件资源。
python export_reference.py --checkout /private/original-CBIG \
  --output /private/task03/reference/CBIG-clean

# 原 reader 实际读入探针通过后，独立核查完整皮层值。探针和影像均留在私有目录。
python verify_reference_probe.py --timeseries /private/complete_run.dtseries.nii \
  --assets /private/hcp40_fslr32k_17.npz --probe-dir /private/task03/reader-probe \
  --expected-input-sha256 ebf2f4ac854e5e521819b5182233442a6b18271c81e62c4609f03cc0cbb4dcdd \
  --output /private/task03/reader-probe-values.public.json

# 另行准备 censor 接口控制；阈值和向量只用于这个额外样例，主 benchmark 保留全部帧。
python prepare_censor_control.py --timeseries /private/complete_run.dtseries.nii \
  --expected-input-sha256 ebf2f4ac854e5e521819b5182233442a6b18271c81e62c4609f03cc0cbb4dcdd \
  --output-dir /private/task03/censor-control

# 完整真实 surface CPU API；--threads 只接受统一的 1 或 8。
python benchmark_fnit.py --binding /private/task03/surface.json \
  --kind surface --threads 8 --device cpu --output-dir /private/task03/fnit-surface-new

# 同一官方 profile 到最终完整标签。
python benchmark_fnit.py --binding /private/task03/profiles.json \
  --kind profiles --threads 8 --device cpu --output-dir /private/task03/fnit-profile-new

# 保持生产完整 API 调用；GPU 回归使用同一冻结输入和参数。
python benchmark_fnit.py --binding /private/task03/volume.json \
  --kind volume-api --threads 8 --device cuda:0 --output-dir /private/task03/fnit-volume-new
```

`benchmark_fnit.py` 的私有 JSON 字段为 `source_root`、`case`（匿名名）、`assets`（可省略）、`timeseries`（surface 路径列表）或 `profiles`（原版 MAT profile 列表）或 `volume/left_surface/right_surface/cortical_mask`；另有 `expected_frames`、可选的 `censor` 路径列表、`w/c`、`frame_chunk/max_distance_mm`。预期帧数为单 run 整数，多个完整 run 为逐 run 整数列表。输出是正常功能文件和不含私人路径的 `report.public.json`。adapter 自身时间含验证和报告开销；正式过程时钟由协调器记录，函数阶段时钟另列，不把嵌套阶段重复相加。

MATLAB 私有绑定字段为 `cbig_clean_dir`、`cbig_sd_dir`、可选的现有依赖 `matlab_startup`、`timeseries` 或 `profiles`、可选 `censor`、`w/c`。下面是原版单被试 wrapper 的完整调用；`profiles` 模式调用未改动的 `CBIG_MSHBM_generate_individual_parcellation`。不调整原版迭代阈值。

```matlab
addpath('/path/to/frozen/validation/fmri_cpu_20261004/task03_mshbm');
run_reference('/private/task03/matlab-binding.json', ...
              '/private/task03/cbig-new', 8, 'full');
```

`baseline_controller.py --config /private/controller.private.json --prepare` 只导出参考与预检；`--start` 才取得 CPU 锁并顺序执行完整 FNIT CPU1、CBIG CPU1、CBIG CPU8、FNIT CPU8。每项先保留一次初基线；精度或速度差异需用后续重复测量判断。计算节点没有 Git 时，生产源码 SHA 取实际已导入模块的 `__file__`，并核查等于冻结 `PYTHONPATH`；Git commit 字段与冻结基线身份分别记录。准备回执见 `preparation.public.json`。

`projection_oracle.py` 检查完整体积的每个 frame 与皮层顶点；它是独立几何核查，不是 CBIG 原生功能，也不作为官方速度参照。

`reference_probe.m` 必须实际从干净导出解析 `ft_read_cifti`、`ft_write_cifti`、`gifti`、`xmltree` 和 CBIG 主函数；同一 `cifti-matlab` 子树包含所需 FieldTrip private helpers 与 GIFTI/XML 类。`verify_reference_probe.py` 直接按原 CIFTI brain-model axis 重建左右半球顶点身份，与 MATLAB 写出的全顶点矩阵比较全部 29,111,880 个有效皮层值。该探针只验证实际依赖与输入 ABI；完整官方计时必须另外执行，不把探针计时当 benchmark。

正式队列全部成功后，按相同线程和物理核比较原版与冻结／优化 FNIT：

```bash
python compare_official.py \
  --official-results /private/task03/cbig_cpu8 \
  --fnit-results /private/task03/fnit_cpu8 /private/task03/fnit_candidate_cpu8 \
  --queue-status /private/task03/queue_status.public.json \
  --reader-values /private/task03/reader_probe_values.public.json \
  --output /private/task03/official_cpu8_precision.public.json
```

该聚合器要求官方 commit、实际成功退出、同一完整输入 SHA、全值 reader 门槛、固定资产 SHA、实际 FNIT 源码 SHA 和相同 CPU 线程／affinity 均匹配。它直接比较全部 64,984 顶点和固定身份的网络 1–17，分别报告左右半球的标签差、背景差和逐网络 Dice，不重排列网络标签。

固定原 [`ft_read_cifti`](https://github.com/ThomasYeoLab/CBIG/blob/b69b822a15e2a94f1e439606552fc44b6858cf3c/external_packages/matlab/default_packages/cifti-matlab/ft_read_cifti.m) 用文件名点分段选择输出数据字段。多段 UKB 文件名在第四次实际 probe 中未生成 CBIG 要求的 `dtseries`；参考适配器需使用私有 `complete_run.dtseries.nii` 兼容链接，并验证它与原文件的完整 SHA 相同。它只改变参考入口的文件名，不改影像内容、顶点身份、原软件函数或生产 FNIT 读入行为。

可选 censor 控制按完整皮层计算 `DVARS[t] = sqrt(mean((x[t]-x[t-1])**2))`，保持第 0 帧，剔除超过同 run DVARS 第 95 百分位数的帧。`prepare_censor_control.py` 先核对完整输入 SHA，再写出全部 490 行向量及私有逐帧 DVARS；公开报告只保存阈值、保留/剔除帧数和 SHA。原 CBIG 与 FNIT 必须使用相同向量，完整时间轴读取和原有迭代上限仍保留；该额外样例的耗时单独记录。

`reference_feature_controls.py` 提供独立的 CPU8 附加队列：原版／优化 FNIT 的相同 censor 控制，以及 `w=100/c=25` 的无 censor 控制。它要求六项主队列已经成功完成，重新核对完整输入、667 个原版文件和冻结 core，随后沿用相同八个物理核与 CPU 锁。每个配置写入新目录和各自私有绑定；公开回执保存参数、censor SHA、实际退出码与耗时，不保存逐帧向量。此附加队列目前只完成准备，没有可报告的官方结果。

```bash
python reference_feature_controls.py \
  --config /private/task03/controller.private.json \
  --output-root /private/task03/official-feature-controls
```

## 完整 CPU 热点与优化结果

完整冻结 CPU1 的诊断 line profile 表明：38 次 `profile.T @ posterior` 合计 56.265 s，19 次 `profile @ direction` 合计 15.676 s，7 次 data-logit 乘法合计 5.811 s；诊断运行排除在正式速度表之外。profile 为 float32、状态原本为 float64，混合 dtype 在重复 BLAS 乘法中转换整张矩阵。

候选在验证 profile 后一次提升为 float64。完整函数链 CPU1 为 85.646→44.162 s（1.94×），CPU8 为 63.490→12.642 s（5.02×）；推断阶段分别为 80.709→38.150 s、59.144→8.314 s。两个预算下完整标签均 0 差，两个 profile SHA 相同，六个科学文件逐字节相同，outer/EM 次数均 `(1,5)、(2,2)`。CPU8 κ 的最大绝对差为 `1.45974e-10`、相对差为 `1.71817e-13`，目标值差 0；不将浮点历史尾差标成全 bitwise 相同。

最大 RSS 的实际单位为 KiB，换算 CPU1 2.617→2.870 GiB、CPU8 2.621→2.873 GiB。全部预算、阈值、顶点和帧保持原值；新 core 的实际导入 SHA 为 `eabb4d62c810fc180d71f41c0e783bcf2dfd9c1bb6ae42fcc699e44859a9aa9e`。

旧 490 帧记录约 74–97 s，属于旧共享节点和 FNIT 对 FNIT 输入的下游运行，不是新的官方 MATLAB 速度。初步资源预算为每个完整双 session 数 GB RAM、每项数分钟；CPU1、官方启动与不同收敛过程需要实测后修订。

## 更新记录与参考

- 2026-10-04/05：完成资源与接口核对、完整 CPU1/CPU8、体积/参数/CLI 矩阵、GPU ABBA 和独立全帧核查；四种 NPY 布局读取与完整 CIFTI 逐值相同。用户直接确认共享 GPU 串行运行后由协调者启动；各失败 attempt 保留。固定原 vendor 的实际全值读取通过，完整原版 CPU1/CPU8 配对已启动，结果待完成。
- 2026-10-01：490 帧官方发布数据与 FNIT 上游输出的下游 MS-HBM 对照；两边推断均为 FNIT，保留原版本和报告。
- 2026-09-27：100 帧真实样例的旧 profile／标签对照；绑定当时脚本与源码，不能代替本轮完整数据对照。

Kong et al., *Spatial Topography of Individual-Specific Cortical Networks Predicts Human Cognition, Personality, and Emotion*, Cerebral Cortex (2019), [doi:10.1093/cercor/bhy123](https://doi.org/10.1093/cercor/bhy123)。原实现：[固定 CBIG MS-HBM 代码](https://github.com/ThomasYeoLab/CBIG/tree/b69b822a15e2a94f1e439606552fc44b6858cf3c/stable_projects/brain_parcellation/Kong2019_MSHBM)；[CBIG MIT 许可](https://github.com/ThomasYeoLab/CBIG/blob/b69b822a15e2a94f1e439606552fc44b6858cf3c/LICENSE.md)。
