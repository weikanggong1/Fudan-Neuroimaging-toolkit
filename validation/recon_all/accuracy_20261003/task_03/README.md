# GCA、归一化和 WM/filled 精度诊断

> 本页保留为旧冻结 `816e5610417a4c587caf321049438a9554139016` 输入的历史诊断参考。2026-10-04 准备的十例候选树以 `cc9402734faeba93b3a13c29932fa1392eaccf62` 为 main 基线；其中 CA 文件 SHA 与本页对应的已完成两例同输入回执一致，此句对应候选准备时点，后续3a真实连续链另见主报告。历史回执没有记录原 driver 退出码，sub-02 的性能稳定性仍未确认。

本轮基线为 `816e5610417a4c587caf321049438a9554139016`。`source_freeze.json` 固定体积模块源码。旧两例 `8d750e2` 自产检查点只用于定位；十例新公开 T1 的最终连续链由协调者统一执行，不能用旧两例替代。

## 功能和输入输出

`volume_probe.py` 比较冻结的真实体积、LTA、逐标签 Dice；固定官方 LTA 回放 CA normalize；完整 EM 执行四组 nu/mask 交叉诊断。生产接口不改变，交叉影像只写到独占 benchmark 目录。每组合单独取得 `/tmp/fnit-shared-benchmark.lock`，总线程 4。

配置 JSON 参数：`commit` 为运行源码身份；`output` 是新建的独占结果根；`assets` 为授权资产目录；`atlas` 是已校验单通道 GCA；`binary` 是 Conda 固定源码构建的完整缓存 EM 程序；`binary_sha256` 为该程序 SHA-256；`cases` 是被试列表，每项 `id` 标识、`fnit` 是冻结自产被试目录、`official` 是隔离官方被试目录。两目录必须含 `mri/`，nu 为 3D uint8 conformed T1，brainmask 与 nu 同网格；LTA 为附两侧几何的 VOX_TO_VOX 4×4 矩阵。ctrl_pts 是 256³×6 float32 体积。输出各模式的 `report.json`；每个运行下生成独立 `mri/norm.mgz`、`ctrl_pts.mgz` 和 LTA，不覆盖输入。

`--config` 必须给配置文件；`--mode audit` 比较已有冻结体积，`normalize` 使用同一官方 nu/mask/LTA 回放归一化，`cross` 完整重算四组合 EM，再送相同归一化。输出目录已存在时报错，输入缺失/程序哈希不符/网格不一致/程序失败明确失败。比较只有网格完全对应时才按同索引进行，记录 different/max/P99 与 changed-only P99；filled 和 aseg 逐标签报告 Dice。

`tail_probe.py` 同官方完整输入回放 GPU brain normalize、既有 Numba 去噪及 filled。它只读取官方影像作为算子诊断，分别生成自己的输出并比较，不组成生产链。device 固定 `cuda:0`，TF32 开启，使用逐阶段锁、显式 CUDA 同步、nvidia-smi 同期进程采样和 Torch 峰值。记录的去噪和 filled 是既有 CPU 实现，没有把 GPU allocated 的零值称为整流程显存。

## 调用

```python
# 成熟生产子函数接口保持不变。下例为冻结输入的诊断回放。
from fnit.recon_all.ca_normalize_python import run_ca_normalize
normalization_report = run_ca_normalize(
    nu_path=frozen_nu_file,            # 3D uint8 T1，conformed voxel grid
    mask_path=frozen_brainmask_file,   # 与 T1 同网格
    gca_path=verified_gca_file,        # 校验过的固定图谱
    lta_path=frozen_voxel_lta_file,    # source voxel → atlas voxel
    norm_path=diagnostic_norm_file,   # 同网格 uint8 输出
    ctrl_path=diagnostic_controls_file, # 六帧 float32 控制点输出
)
```

```bash
PYTHONPATH=src python validation/recon_all/accuracy_20261003/task_03/volume_probe.py --config frozen_config.json --mode audit
PYTHONPATH=src python validation/recon_all/accuracy_20261003/task_03/volume_probe.py --config frozen_config.json --mode normalize
PYTHONPATH=src python validation/recon_all/accuracy_20261003/task_03/volume_probe.py --config frozen_config.json --mode cross
PYTHONPATH=src python validation/recon_all/accuracy_20261003/task_03/tail_probe.py --config frozen_config.json
```

原软件完整 EM 对应 `mri_em_register -uns 3 -mask brainmask.mgz nu.mgz RB_all_2020-01-02.gca transforms/talairach.lta`；归一化对应 recon-all 内 `mri_ca_normalize`，后续 brain 为 `mri_normalize -aseg ... -mask ...`，去噪为 `AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz`；filled 为 `mri_fill` 的 aseg-guided 路径。评分缓存及控制点选择没有独立官方 CLI。实际原软件完整参数由冻结 recon-all 日志记录，本诊断不编造短命令代替完整调用。

## 当前实测与历史

`audit.json` 为 2026-10-03 gpucw1 实时重读冻结真实文件生成，并非把旧报告改日期。两例 orig 数值/affine 一致。sub01 nu/brainmask 各 2 体素差异，LTA max 0.0005735158920288086，norm 490692（max 8/P99 1），brain 705288（max 21/P99 2），filled 4844；sub02 nu 29、brainmask 17，LTA max 7.450580603862722e-9，norm 41（max 3/P99 0），brain 4502，filled 112。仅凭此不能把放大归因于 GCA 缓存或某个算子。

冻结同输入 normalize、tail 和四组合完整 EM 已排队，完成结果另存，不将排队视为实测完成。当前无生产修复或速度收益声明；整体等效 `not_assessed`。十例新数据所有受影响链和整例指标仍须完成。

此前 2026-10-02 评分缓存能力版本 2 保留上游 ROMP partial ranges/最终规约；缓存键逐项检查当前 variance/prior，均值、坐标、影像每次读。只替换搜索似然，原始稳定样本和全样本 EM 优化、梯度、接受拒绝及停止均保留。现有完整缓存二进制现场 SHA-256 为 `70911f6e3ed8f240f2fc745ad1060664c893da47ff3cb6d591d219760f59079f`。旧优化结果见 `validation/recon_all/optimizations/20261002_parallel/task_03/RESULTS.md`，不代替本轮结果。

## 来源和参考

- [固定 FreeSurfer 8.2 GCA/EM 源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/gca.cpp)
- [mri_em_register](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mri_em_register)
- [mri_ca_normalize](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mri_ca_normalize)
- Fischl et al. Whole brain segmentation: automated labeling of neuroanatomical structures in the human brain. Neuron 33, 341–355 (2002).

无新增生产依赖；诊断使用现有 Conda NumPy、nibabel、PyTorch、Numba、SciPy 和 Python 标准库 fcntl。

## 第一阶段交接

协调者要求本任务先提交审计后返回，以便后续任务启动。`checkpoint.json` 记录原执行 session、服务器 Python/shell/native PID、完整日志和 receipt 路径、逐组合/逐组件锁粒度。正在运行和排队的计算均保留，未取消已取得锁的子进程。当前首组完整 EM 已在运行；sub01 固定输入 CA normalize 已完成且 norm/ctrl_pts 零体素差异，sub02、cross 和 tail 尚未完成。原工具 SSH 会话继续保留，未尝试将正在运行进程迁入 tmux；请保持原会话和共享主连接打开，通过报告和日志只读接管。待结果完整及十例输入可用后，由协调者唤醒同一任务继续归因和回归。

## 第二阶段已返回结果

冻结官方输入的 CA normalize 回放已完成：sub01 norm 与 ctrl_pts 都零差；sub02 norm 有 11 个体素不同、ctrl_pts 有 6 个元素不同，集中于 `(112,90,95)` 三轮右侧 WM 控制点：FNIT 三帧标签为 41、三帧均值为 107.820549，参考为 0。`normalize_completed.json` 保存完整来源和输出哈希。该残差的根因仍待坐标算子诊断，不归为随机性或编译器差异。

sub01 完整 EM 四组合已完成，详见 `cross_progress.json`、`contribution_progress.json/.csv`：

| nu | mask | LTA max差 | norm不同体素 | ctrl不同元素 | EM含I/O秒 |
|---|---|---:|---:|---:|---:|
| 官方 | 官方 | 0 | 0 | 0 | 188.24 |
| FNIT | 官方 | 0.000573515892 | 490692 | 10617 | 190.77 |
| 官方 | FNIT | 0 | 0 | 0 | 221.69 |
| FNIT | FNIT | 0.000573515892 | 490692 | 10617 | 242.32 |

两例的 EM `mask>=5` 和 CA `mask!=0且!=1` 支持集均零差（`input_support.json`）。sub01 两处 nu 差异都位于有效 mask 内；四组合证明 mask 数值尾差没有额外贡献，nu 输入改变了注册和控制点。CA 图谱峰值 107、影像峰值 112 在四组合中相同；三轮实际控制点数量从 1925/2383/2337 变为 1940/2329/2386。这里是上游数值差异经离散坐标和控制点选择传播的结果，不能当作已证实的 CA 代码 bug。完整缓存 EM 的官方同输入 LTA 零差。sub02 其余交叉组合仍须完成。

尾段同完整官方输入回放已完成的步骤均零体素差异：sub01 brain 61.10秒、denoise 20.84秒、filled 50.87秒；sub02 brain 61.36秒、denoise 18.78秒，filled 尚等待锁。阶段秒数包括函数加载/计算/读写，不包括锁等待和之后的差异统计；不是 AB/BA 速度稳定验收。`tail_progress.json` 的 nvidia-smi 为原始同期快照。已采样父 PID 峰值 1,323,302,912 字节；没有 ancestry 快照，不宣称父子合计或整例显存，也不把 Torch allocated 的零值当实际零显存。

`diagnose_ca_coordinates.py` 对应两例同官方 nu/mask/LTA，进程内临时替换 atlas source 坐标：复用现有 VNL FP32 affine inverse、逐项 FP32 累计，并在 nint 前提升 float 坐标。它核对原生 `MatrixMultiply` 精度契约，生产源码未改；结果仍等待共享锁。该诊断会逐例释放锁，不再启动已有的 normalize/cross/tail 队列。原生固定源码 gca.cpp SHA 为 `52439c0c2218b4e685f03365ad6d4889be41fa5b2dff613897d6461be1b3a872`，matrix.cpp 为 `14d64078a37252c0e1b8d3354140bab46e8c41601975e414d1fe78dea94fd52c`。

下一条安全命令为只读收集，不重新计算（在服务器已部署脚本目录内）：

```bash
python collect_results.py \
  --receipts /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/task_03/frozen_v1 \
  --output /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/task_03/collected
```

`--receipts` 为四种诊断的结果根，`--output` 为独占报告目录。输出各模式完成标记、四组合 CSV、尾段 CSV 和父 PID 显存采样摘要；缺报告记 `receipt_absent`，没有 `complete:true` 记 `running_or_waiting`。缺少父子进程树快照记 `not_available`。真实十例全链和整例仍由协调者统一调度，本文不替代它们。

## 2026-10-03 04:08:37 UTC 最新只读检查

已在服务器执行现有 `collect_results.py`，报告见 `latest_collected.json` 和 `status_040837UTC.json`。四种诊断 receipt 的 SHA-256 与上一交接完全相同，本次没有新增完成结果，也未追加计算。

- 两例固定官方输入 CA normalize 已完成，原结果仍为 sub01 norm/ctrl 零差、sub02 norm 11 个体素/ctrl 6 个元素不同。
- 完整 EM 交叉已完成 5/8 组合：sub01 四组和 sub02 官方 nu/官方 mask；其余三组仍待锁。
- 同输入尾段完成 5/6 阶段：sub01 brain、denoise、filled，sub02 brain、denoise，全部零体素差；sub02 filled 待锁。
- FP32 坐标诊断完成 0/2 例，首例尚待锁。**坐标舍入假设尚未经过本次同输入实测，不能把它写成根因或精度改善。** 没有生产坐标修复。

三个原父进程 `94174`、`98758`、`114339` 均存活（状态 S），`/proc/locks` 确认等待同一共享锁。当前持锁者 `54960` 是协调者的新 T1 首例官方运行；旧基线锁驱动 `51564` 已为 Z，基线首例完成状态由协调者验收。本检查只核实任务3诊断状态，不将协调者首例结果扩写为十例完成。所有原 session、队列与共享 SSH 主连接继续保留。

坐标诊断脚本现场 SHA-256 为 `01bd6bf0ea79411dbe3e07cf468b4b373adad96c84cb361bbe24922365c9358d`，与本地提交脚本一致。下一步是现有队列获锁后读回新增 receipt，再决定是否需要一般坐标修复和同输入两例配对验证；本次不启动额外长作业，不改变任何阈值。

## 完成的隔离诊断与生产坐标修复

原队列均已完成，完整 receipt 分别为 `cross_completed.json`（8/8）、`tail_completed.json`（6/6）、`coordinates_fp32_completed.json`（2/2）。`collected_completed.json`、`completed_isolation.json` 和 `cross_completed.csv` 汇总最新实测，不用旧 `*_progress` 文件代表最新状态。

两例 FP32 坐标诊断的 norm 与六帧 ctrl_pts 都为零差，网格/dtype相同。sub02 原 11 个 norm 体素（max 1）和 6 个 ctrl 元素残差消除，sub01 保持零差。sub02 八组诊断中，官方 nu 的两组均 norm11/ctrl6，FNIT nu 的两组均 norm41/ctrl6，mask 不增加差异；两例完整缓存 EM 在官方同输入都复现官方 LTA。六项 tail 回放全部零差，sub02 filled 本次完成耗时 54.12秒。

成熟子函数修复已落在 `ca_normalize_python.atlas_samples`，内部新增 `prior_to_source_coordinates`，复用 VNL affine inverse并按原生逐项 FP32 运算；不缩放平移，nint 前提升已形成的 float 坐标。核对完整原生链发现：`GCAfindAllSamples` 初始 x/y/z 是 prior×spacing 的 atlas 坐标；`GCAcomputeSampleCoords` 仅在浮点 source 坐标位于闭区间 `[0,size-1]` 时覆盖它，越界应保留初始坐标。生产 helper 同时保留这条规则，没有体素特例或边缘 clip。三项 standalone 契约测试已在 gpucw1通过，涵盖负半整数、FP32 half-add、spacing/平移和255/255.1浮点边界；它们不代替真实 T1 benchmark。

完整生产接口、所有参数/异常、原软件命令和版本记录见 [CA_NORMALIZATION.md](../../../../docs/recon_all/CA_NORMALIZATION.md)。无新依赖，不改变现有 GPU/TF32 设置，其他控制点/偏置场算子保留。

生产 helper 两例 ABBA 短回归是新的独立验证，仍待共享锁：Python PID115793、shell115434、原工具session83207；报告为服务器 `task_03/production_fix_abba_v1/report.json`、日志 `task_03/production_fix_abba_v1.log`。每个完整 CA 调用后释放锁，共8次调用；不启动原始T1整例。该 PID 不在原始守护器启动时的固定watchlist，但协调父整例提交当前仍暂停；不宣称已动态加入。必须读到完成receipt后才确认生产 helper 的精度与配对耗时，不把诊断的28.01/25.55秒当配对结论。

安全恢复命令（只读）：

```bash
cat /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/task_03/production_fix_abba_v1/report.json
```

不要重新执行原ABBA脚本覆盖/重复已有队列。生产修复当前source SHA为 `167be9d72496b7279b98e52b8f0d6c0808d2c09d02df24af1fed0fd331100b9b`；基线仍为公开816的冻结模块，SHA `abb436826e600101c2c5087e2367e8a36a7c038c99eda10bc4f5be1e7486661a`。十例新T1连续链及最终指标仍由协调者完成，整体等效 `not_assessed`。

## 2026-10-03 08:42:18 UTC：生产 helper ABBA 已完成

`production_fix_abba_latest.json` 是服务器原始完整receipt，`production_fix_abba_completed.json/.csv` 提供机器可读汇总与分步骤。原报告 `complete:true` 在全部八次调用后写入，PID115793已退出，日志没有报错；原驱动没有独立退出码receipt，故不虚构exit_code=0。现场候选模块SHA仍为 `167be9d72496b7279b98e52b8f0d6c0808d2c09d02df24af1fed0fd331100b9b`，与本次执行报告一致，基线模块SHA为 `abb436826e600101c2c5087e2367e8a36a7c038c99eda10bc4f5be1e7486661a`。

| 冻结同输入 | ABBA基线秒（A1/A2） | 候选秒（B1/B2） | baseline norm/ctrl差 | candidate norm/ctrl差 |
|---|---|---|---|---|
| sub01 | 27.5265 / 25.0385 | 27.0813 / 24.9093 | 0/0，两次相同 | 0/0，两次相同 |
| sub02 | 24.7578 / 25.4525 | 27.2879 / 25.7171 | 11/6，两次相同 | 0/0，两次相同 |

四次候选和官方 norm/ctrl 数值一致，同网格、同dtype；因此候选相对基线的变化为sub01 0/0，sub02 11/6。后者由候选与官方完全相等和基线对官方的同网格计数严格推出，机器报告明确标注比较方法，不把它冒充另一次直接文件比对。

耗时包含每次加载、计算和读写，不包括共享锁等待及之后的差异统计。均值sub01基线26.2825秒、候选25.9953秒（观察-1.09%）；sub02基线25.1052秒、候选26.5025秒（观察+5.57%）。**sub02两次候选都慢于两次基线，当前不能宣布速度稳定通过。** 分步骤增量分布在setup、控制点selection、bias与写出，未隔离坐标算子自身成本；不直接归因于共享负载或随机性，也不把5%等未授权容差当门槛。本轮请求只读检查完成状态，未追加任何新长计算、未重启原队列。

当前可确认：两例冻结完整输入下的一般坐标修复精度通过；速度观察待进一步处理；新十例自产连续链、严格138项及最终脑区指标仍由协调者验收，整体等效 `not_assessed`。此前pending checkpoint保留为历史交接，不再代表最新执行状态。
