# 10 人公开 DWI 与 T1 数据：固定清单与采集核验

## 数据来源与许可

本次使用 OpenNeuro **ds003138，版本 1.0.1**，题名为 *Tidying Up White Matter: Neuroplastic Transformations in Sensorimotor Tracts following Slackline Skill Acquisition*。固定 Git 提交为 `0c57cc4966e2e181fb3536405327a64f65f3d022`。原作者 README 描述了 60 人、三个时间点的研究，每个时间点包含高分辨率 T1、多 shell DWI 和反向相位编码数据。[原作者 README](https://github.com/OpenNeuroDatasets/ds003138/blob/0c57cc4966e2e181fb3536405327a64f65f3d022/README)

版本元数据明确写明 **CC0**，版本 DOI 为 **10.18112/openneuro.ds003138.v1.0.1**。本次从 OpenNeuro 自有公开 S3 下载，不使用第三方镜像；报告只引用公开数据编号和汇总结果。[固定版本元数据与许可](https://github.com/OpenNeuroDatasets/ds003138/blob/0c57cc4966e2e181fb3536405327a64f65f3d022/dataset_description.json)，[OpenNeuro 版本页面](https://openneuro.org/datasets/ds003138/versions/1.0.1)

## 事前固定的 10 人

按该版本可用的被试编号升序选取前 10 人，只用 `ses-1`。编号为：

| 序号 | 公开被试编号 | Session |
|---|---|---|
| 1 | sub-82KK02101 | ses-1 |
| 2 | sub-82KK02102 | ses-1 |
| 3 | sub-82KK02103 | ses-1 |
| 4 | sub-82KK02104 | ses-1 |
| 5 | sub-82KK02105 | ses-1 |
| 6 | sub-82KK02106 | ses-1 |
| 7 | sub-82KK02107 | ses-1 |
| 8 | sub-82KK02108 | ses-1 |
| 9 | sub-82KK02109 | ses-1 |
| 10 | sub-82KK02112 | ses-1 |

清单在运行 pipeline 前固定。文件完整性和采集元数据是事前核查条件；不根据运行速度、相关系数、拟合图或其他输出替换、剔除被试。运行失败应保留在结果表中并说明原因。

运行目录以 `case01` 至 `case10` 对应上表顺序；每人分别执行 TBSS 与 MMORF 两个分支，共 **20 个配对比较**。两侧读取同一份规范化原始输入，各自生成 TOPUP、脑掩膜、EDDY、拟合和配准结果。公开仓库保存来源清单、脚本与匿名汇总；原始 MRI 和逐人完整输出保存在私密 benchmark 目录。

## 每个人的输入

| 输入 | 文件数 | 原始影像形状 | 内容 |
|---|---:|---|---|
| AP shell 1 | 4 | 120 × 120 × 68 × 21 | NIfTI、JSON、bval、bvec；b0 × 1，b1000 × 20 |
| AP shell 2 | 4 | 120 × 120 × 68 × 31 | NIfTI、JSON、bval、bvec；b0 × 1，b2000 × 30 |
| AP shell 3 | 4 | 120 × 120 × 68 × 65 | NIfTI、JSON、bval、bvec；b0 × 1，b3000 × 64 |
| PA shell 1/2/3 | 6 | 各 120 × 120 × 68，单 b0 | 每个 shell 一个 NIfTI 和一个 JSON |
| T1 | 2 | 224 × 288 × 288 | 一个完整 T1 NIfTI 和一个 JSON |
| 合计 | **20** |  |  |

DWI/EPI 为 **2 mm 各向同性**。所有 10 人的各 shell DWI 和 PA EPI 三维尺寸一致，三个维度均为偶数；各影像 sform 的元素最大绝对差不超过 **6.104 × 10⁻⁵**，bval 数与对应第四维一致。T1 体素大小约 **0.8000 × 0.7778 × 0.7778 mm**。

10 人的 **200 个文件共 786,381,691 bytes**，其中十个 T1 和十个 T1 JSON 共 **207,226,509 bytes**。这个总量包含所有必需的 T1、DWI、PA EPI 和 sidecar，不只包含扩散影像。

### 文件路径示例

以第一个被试为例，其原始文件位于：

```text
sub-82KK02101/ses-1/
├── anat/sub-82KK02101_ses-1_T1w.{nii.gz,json}
├── dwi/sub-82KK02101_ses-1_acq-shell1_dir-AP_run-1_dwi.{nii.gz,json,bval,bvec}
├── dwi/sub-82KK02101_ses-1_acq-shell2_dir-AP_run-1_dwi.{nii.gz,json,bval,bvec}
├── dwi/sub-82KK02101_ses-1_acq-shell3_dir-AP_run-1_dwi.{nii.gz,json,bval,bvec}
├── fmap/sub-82KK02101_ses-1_acq-shell1_dir-PA_run-1_epi.{nii.gz,json}
├── fmap/sub-82KK02101_ses-1_acq-shell2_dir-PA_run-1_epi.{nii.gz,json}
└── fmap/sub-82KK02101_ses-1_acq-shell3_dir-PA_run-1_epi.{nii.gz,json}
```

匿名原始文件 URL 的格式为 `https://s3.amazonaws.com/openneuro.org/ds003138/<上述相对路径>`。完整逐文件路径、大小、上游校验和和影像头记录在冻结的下载清单中。

## 相位编码与输入准备协议

三个 AP DWI 的 `PhaseEncodingDirection` 均为 **`j-`**，三个 PA EPI 均为 **`j`**；两类文件的 `TotalReadoutTime` 均为 **0.126134 秒**。本次将三个 AP shell 按 shell 1、2、3 顺序完整拼接为 **117 帧**，相应拼接 bval 和 bvec；不裁片、不重采样，不删除扩散方向。

反向输入采用 **shell 1 PA 单 b0**，它与第一 shell AP b0 的 TE 相同。其余 PA 文件保留在原始下载中。部分 shell 2/3 PA JSON 的 `IntendedFor` 仍指向 shell 1；本次显式绑定 shell 1 文件，并以实际影像头和采集参数核查配对，不依赖这些错误链接推断关联。

TOPUP 的 AP b0 从完整 AP 输入的三个候选 b0 中按固定流程选择，PA 候选只有上述 shell 1 b0；选中的原始帧号及候选相关得分写入各自报告。两套实现的 TBSS 与 MMORF 分支均将**各自 TOPUP 输出的两张校正 AP/PA b0 平均图**输入 SynthStrip，随后将自产脑掩膜用于 EDDY。FNIT 使用自身 PyTorch GPU SynthStrip，独立参考使用原 SynthStrip CPU。MMORF 另对原始 T1 独立提取脑部。

拼接输入的完整结构如下，两个实现共用这些输入文件，不共用处理输出：

```text
inputs/case01/
├── raw/AP.nii.gz       # 117 帧，shell 1→2→3，保留原始空间与解码后体素值
├── raw/AP.bval         # 对应 117 个 b 值
├── raw/AP.bvec         # 3 × 117 梯度，保留原始坐标约定
├── raw/AP.json         # 固定 AP 相位编码与 readout，另保留逐 shell TE 记录
├── raw/PA.nii.gz       # 原 shell 1 PA 单 b0
├── raw/PA.bval         # 对应单帧 b0
├── raw/PA.json         # 固定 PA 相位编码与 readout
├── T1w.nii.gz          # 原始 T1 输入
└── input_manifest.json # 原始文件哈希、拼接与逐值保留核验、采集限制
```

## 不同 TE 的模型解释范围

三个 shell 的 TE 分别为 **104、113、125 ms**，各自包含一个同 TE 的 b0。拼接保留了真实采集数据，但 TE 不同会引入跨 shell 的 T2 加权和信号尺度差异。如果模型将三个 b0 一起估计 S0，也会受到这个采集差异影响。此数据用于比较 **相同原始输入和固定处理协议下 FNIT 与参考软件的输出及耗时**；这个比较不能同时证明 NODDI 或张量参数的生物学正确性。输入准备不加入未经验证的跨 shell 强度缩放。

## 下载完整性与核验范围

固定 Git 版本中的 NIfTI annex 使用 **MD5E**，没有提供 SHA-256。事前核查已逐文件确认 annex 大小、annex MD5 与匿名 S3 的大小和单段 ETag 相等；小型 JSON、bval 和 bvec 另记录 SHA-256。完整下载后继续核对文件大小和原作者 MD5，并计算本地 SHA-256 绑定实际 benchmark 输入。

事前影像检查只读取每个 NIfTI 的前 **64 KiB 压缩 HTTP 206 range**，从中提取影像头；没有用 voxel 数据计算质量或筛选被试。实际数据可读性、有限值和完整 pipeline 的成功率应在后续运行中单列记录。

[dataset_manifest.json](dataset_manifest.json) 的 `full_image_downloaded=false` 和 header range 说明记录的是**事前固定清单时的检查范围**，不是后来 benchmark 的下载或执行状态。完整下载由 `download_cohort.py` 按清单校验；`prepare_inputs.py` 的逐人 `input_manifest.json` 再核对解码后的拼接体素、bval/bvec 和原始网格。十人清单均已准备，实际运行报告使用本地文件 SHA-256 绑定输入；清单不随输出效果重新筛选。

对应 FNIT 基线为 `5d84c7ddec099f1b94772d273c0f76a79934c9bb`；部署目录的 433 个运行时 Python 文件已逐路径核对 SHA-256，详见 [source_binding.public.json](source_binding.public.json)。原 MMORF 首次启动异常、同输入构造诊断和整链恢复运行分别保存，处理规则见 [PROTOCOL.md](PROTOCOL.md)；这些运行状态不改变固定数据、空间、分辨率或十人分母。

## 参考资料

- [OpenNeuro ds003138 v1.0.1](https://openneuro.org/datasets/ds003138/versions/1.0.1)
- [原始数据仓库与固定提交](https://github.com/OpenNeuroDatasets/ds003138/tree/0c57cc4966e2e181fb3536405327a64f65f3d022)
- [README：研究设计、T1、多 shell 与反向 PE](https://github.com/OpenNeuroDatasets/ds003138/blob/0c57cc4966e2e181fb3536405327a64f65f3d022/README)
- [dataset_description.json：CC0 与版本 DOI](https://github.com/OpenNeuroDatasets/ds003138/blob/0c57cc4966e2e181fb3536405327a64f65f3d022/dataset_description.json)
