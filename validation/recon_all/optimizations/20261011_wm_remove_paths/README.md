# 完整 WM 路径阻断：同输入验证

## 1. 功能简介

本目录绑定新增 `wm_remove_paths.py` 与两个标准 WM 文件入口。生产计算为自有 NumPy/Numba及成熟Torch子阶段；固定原版源码程序仅隔离生成参考。补齐此前证明范围以外的有序路径搜索，同时保留原 int32/float32 临时存储。源规则、API、流程图和空间说明见 [功能专页](../../../../../docs/recon_all/WM_REMOVE_PATHS.md)。

## 2. Python、输入与输出

输入来自公開 ds000114 sub-06/sub-07 的 `79a41cdd` 自产连续链：同网格三维 `wm.seg`、`brain`、`aseg.presurf`、`entowm`。WM/brain为uint8，aseg原float32存储精确整数；完整文件API自动保留这个类型。输出为新uint8 WM，维持affine和surface-RAS矩阵。旧CPU与新CPU/Torch都不读取参考输出。

- `summary.json`：两例全部比较、每个整数值Dice、WM>=5前景Dice、API/CLI时间及采样内存。
- `reports/v1` / `reports/v2`：逐次实际过程、原输入/输出/源码/程序SHA与完整比较。原始生产者收据SHA仍绑定原测试，不改标当前整例。
- `source_manifest_v1.json` / `source_manifest_v2.json`：实际 `2474392b` 加最小overlay，17模块的完整SHA；算法模块相同，v2只有benchmark工具注册修复。
- `ten_case_roi_scan.json`：现有冻结3a十例的真实文件SHA、原dtype、两侧ROI范围。只读扫描不代表重新执行十例。
- `test_receipt.json`：本地20合同、9.23秒；不是活跃分支真实benchmark。
- `source_rules.json`：固定源码SHA/行号、左右读写差异、顺序、停止和限制。

`PUBLIC_EXPORT_MANIFEST.json` 对服务器脱敏产物分别记录 `original_sha256` 与 `published_sha256`、大小和数值类型保真；发布文档/复现脚本由 `PUBLICATION_MANIFEST.json` 绑定当前实际文件字节。两者作用不同，不用公开SHA替代原收据SHA。

## 3. CLI与复现

完整参数和具名调用见 [功能专页](../../../../../docs/recon_all/WM_REMOVE_PATHS.md)。

```bash
# 从两个冻结run导出已完成收据和公开衍生PNG；仅读取，输出必须是新目录
python collect_receipts.py \
  --fnit-root /path/to/FNIT \
  --source /path/to/FNIT/workspaces/wm_remove_paths_20261011_v2 \
  --v1-run /path/to/FNIT/runs/wm_remove_paths_20261010_v1 \
  --v2-run /path/to/FNIT/runs/wm_remove_paths_20261011_v2 \
  --output /path/to/new-public-metadata
```

上述路径分别为已授权任务根、v2实际源码、原始v1和补充v2运行，以及新导出目录；不导出MRI、权重、许可、原源码或程序。

```bash
# 仅读取显式case/relative_input清单，重验原存储与ROI，不运行任何编辑或参考程序
python scan_roi.py \
  --fnit-root /path/to/FNIT \
  --input-manifest ten_case_roi_scan.json \
  --output new-ten-case-scan.json
```

首版只读脚本的case显示名使用错误目录层级；本报告按已记录relative_input更正，图像没有重读，数值和输入SHA没有改动，原stdoutSHA与修复范围写在报告中。公共参数化复现脚本属于后续整理版本，其SHA单列，不冒称首版执行代码。

## 4. 原软件调用

```bash
mri_edit_wm_with_aseg -keep-in -fix-ento-wm entowm.mgz 3 255 255 \
  -fix-acj aseg.presurf.mgz 255 255 -fill-seg-wm -fix-scm-ha 1 \
  wm.seg.mgz brain.mgz aseg.presurf.mgz reference.mgz
```

仅在独立benchmark运行，固定提交d932c45…、运行程序SHA `239b181c45ed5cee91a112f7bcaa923226b2f0e1037974b18f5946642d1ed57d`。该程序是本次源码构建/重定位的独立参考，不是复制预装二进制；两个参考重复的decoded结果完全相同。

## 5. 最新实际精度、时间与脑图

| 文件API，秒 | sub-07 | sub-06 |
| --- | ---: | ---: |
| 既有CPU | 25.4038 | 21.6262 |
| 新CPU两次 | 24.9591 / 15.8292 | 19.9118 / 12.5974 |
| 原固定源码参考两次 | 30.3222 / 39.8935 | 31.7058 / 26.8098 |
| 新混合一次 | 44.1880 | 14.8702 |

两例五类完整对照包括新CPU/参考、新混合/参考、新混合/CPU、参考重复和CPU重复：全部decoded0差、最大/P99=0、shape/dtype/affine/tkr一致、所有逐值Dice和WM前景Dice=1。旧CPU/参考同样0差。`strict_decoded_stage_passed=true`；完整MGZ原字节不同，`strict_file_bytes_passed=false`。实际执行完成与解码一致不能替代逐字节诊断或整例等效。

原始图像读写、校验及GPU同步计入API墙钟，fresh CLI还计入启动、导入、哈希和采样；CUDA初始化在API计时前，首次Numba JIT包含在API。共享节点，子计算3线程加采样1线程，总预算4。原版时刻波动、首次JIT和顺序差异保留，未使用此组观察宣称稳定提速。

显式物理GPU0的UUID是 `GPU-anonymized-device-005`，CUDA_VISIBLE_DEVICES=0后逻辑cuda:0；两次调用者已在API前同步初始化该目标CUDA。TF32均开，autocast关。allocated553,648,128 / reserved641,728,512字节；采样目标卡上界1,164,967,936字节。父子进程所有权未解析，合计为null；请求0.5秒、最大实际间隔5.0806/1.7903秒，不能保证未采样瞬间或整例峰值。父进程已有CUDA并同时持有GPU张量的worker共存未新增实测，OS干净隔离也未验证。

20侧真实ROI均满足不可达证明，两个文件阶段的路径填充均0。活跃搜索仅有意义合同覆盖，暂无真实active MRI；源100000条件触发未测。整个recon-all、脑区指标与整体等效未在本子任务重跑，整体 `not_assessed`。

![sub07当前T1/WM及差值](plots/sub07_wm_paths.png)
![sub06当前T1/WM及差值](plots/sub06_wm_paths.png)

上行为公开自产brain和WM轮廓，下行为本次同输入CPU相对固定源差值图（max0），不是完整recon-all最终表面QC。

## 6. 更新与失败记录

- v1：两例hybrid及sub07旧/新CPU/native全部完成；sub06旧基线缓存载入因动态模块未注册而失败，exit1及失败日志SHA保留在 `v1_failure.json`。
- v2：仅注册明确模块名供Numba缓存导入，独立目录/缓存补sub06，全部退出0；没有重写成功GPU数据，也没有改算法。
- 两次参考/新CPU重复性在同输入、同设备、同预算下实际确认。旧H100及10月10日结果不改标签，不用历史时间代替本次。
- 无新增依赖，主页Conda已包含通用数值和影像库；本轮没有独立安装或整例隔离声明。

## 7. 参考与源码

- [固定WM编辑源](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_edit_wm_with_aseg/mri_edit_wm_with_aseg.cpp)
- [类型与MRIvox](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/include/mri.h)
- Dale AM, Fischl B, Sereno MI. *NeuroImage*. 1999;9:179–194.
