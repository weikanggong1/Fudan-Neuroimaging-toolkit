# 丘脑、海马与杏仁核：同输入真实 T1 对照

## 输入和验收

病例 `fs_sub01` 的原始 T1 经已有 recon-all 处理后，固定使用相同的 `norm.mgz`、`aseg.mgz`、`wmparc.mgz` 作阶段输入。三文件 SHA-256 依次为 `d9b6b94c365d2b2b789edbed016953ae64e6cae65072c53ecb42f12410ab1615`、`29fbcff5dff5300599ee863e0c7c425867af725134631741685b41ca44639bc2`、`7cf2c8f0ab1405da795b162ea9d19cf40cb41eedab40d41abb56a6f3baaf9f79`。官方参考为同一 gpucw1 上 FreeSurfer 8.2.0-1 已保存的横断面结果；FNIT 扩展由 Conda GCC 11 编译，在 gpucw1 的 Conda ITK 5.4.7、Surfa 0.6.3、PyTorch 2.5.1 环境中运行，并以 NumPy 1.24.3、SciPy 1.10.1 的独立 Python 环境复核结果。扩展没有链接 FreeSurfer 运行库。`environment-gems-native.yml` 已通过离线 Conda 求解，但尚未在全新 Conda 环境中完成三组运行。

对每个非零硬标签，验收要求 Dice≥0.95 且相对硬体积差≤5%。体积差按同网格标签体素数计算；`.volumes.txt` 的后验软体积不是这项指标。两幅 FSvoxelSpace 图的 shape 和 affine 相同，无须对标签再次重采样。`compare_nuclei.py` 对候选与参考标签的并集逐一计数。

## 结果

| 结构 | 官方非零标签数 | 达标数 | 最小逐区 Dice | 前景 Dice | 不同体素数 | FNIT 墙钟（并发运行） | 官方命令墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 左海马/杏仁核 | 28 | 18 | 0.8000 | 0.9923 | 160 | 432.56 s | 双侧合计 1005.17 s |
| 右海马/杏仁核 | 27 | 14 | 0.8966 | 0.9930 | 191 | 409.29 s | 双侧合计 1005.17 s |
| 双侧丘脑核团 | 44 | 38 | 0.8000 | 0.9975 | 120 | 491.47 s | 596.84 s |

机器可读逐区记录：[左侧](nuclei_left_sub01.json)、[右侧](nuclei_right_sub01.json)、[丘脑](nuclei_thalamus_sub01.json)。左侧 `7006` 仅有 6 个官方参考体素，Dice 为 0.80、硬体积差 33.3%；右侧最小 Dice 来自 `7007`，为 0.8966；丘脑 `8216` 的官方参考只有 6 个体素，Dice 为 0.80。微小核团里一两个边界体素可同时改变 Dice 和相对体积。不能因整体前景 Dice 接近 1 而认定逐区已通过。

另将五份后验软体积表按同名结构比较。左海马 22/22、左杏仁核 10/10、右海马 22/22、右杏仁核 10/10 个条目相对差≤5%；丘脑为 51/52，唯一超出的是 `Left-L-Sg` 的 5.12%。各组最大差依次为 2.98%、2.40%、4.46%、3.14% 和 5.12%。逐项数值见[软体积报告](nuclei_soft_volumes_sub01.json)。软体积接近并不消除小核团硬标签差异。

![同一真实 T1 上的官方标签、FNIT 标签与边界差异；每行取差异体素最多的一层](nuclei_comparison_sub01.png)

## 阶段时间与差异定位

| FNIT 日志阶段 | 左侧 | 右侧 | 丘脑 |
|---|---:|---:|---:|
| 预处理 | 13 s | 14 s | 7 s |
| 初始图谱配准 | 1 s | 1 s | 1 s |
| 初始网格拟合 | 83 s | 78 s | 92 s |
| 强度网格拟合 | 305 s | 285 s | 367 s |
| 其余阶段、启动与写盘（外层墙钟减去以上日志整数秒） | 30.56 s | 31.29 s | 24.47 s |

官方日志对左、右、丘脑分别记录粗标签网格拟合 94、98、110 秒，强度网格拟合 351、331、385 秒。FNIT 与官方都主要耗时在强度网格拟合；FNIT 在该阶段用 Conda 内编译的 C++/ITK 核心，避免 Python 逐四面体循环。本轮 FNIT 三项任务同时运行，官方参考是在不同时间运行；这些墙钟只描述各自的实际运行，不能据此计算同等负载下的提速。逐区验收也尚未通过。

已逐阶段核对的输入：左、右和丘脑的预处理 `tempImage.mgz` 与 `targetMask.mgz` 在全部体素及仿射上相同。左右海马的最近邻粗掩膜也逐体素相同；三线性海马掩膜最大强度差为 `2.38×10⁻⁷`。配准后的 `alignedAtlasImage.mgz` 的强度体素逐个相同，但最终仿射最大差为左侧 `7.63×10⁻⁶ mm`、右侧 `3.81×10⁻⁶ mm`、丘脑 `2.64×10⁻⁴ mm`。受控试验使用已保存的官方配准头信息后，FNIT 的左侧、右侧、丘脑分别达到 28/28、27/27、44/44 逐区验收；它只能定位误差来源，不能作为独立流程的精度结果。当前要继续修正的是独立刚体及仿射配准的数值轨迹，以及微小核团对随后的网格优化与最近邻写回的敏感性。

同一官方程序用不同线程数生成的两份丘脑标签，按相同阈值也只有 27/44 区相互达标，前景 Dice 为 0.9940。这说明当前逐区阈值对 4–10 个体素的核团高度敏感；本报告仍按既定阈值判定 FNIT **未通过**。另外一次仅改变配准重采样头信息的试验，使 FNIT 丘脑从 38/44 降到 27/44；该变更已回退，未用较差结果替换本报告基线。

## 复核命令

```bash
# --norm、--aseg、--wmparc：三幅同网格输入；--atlas-root：经哈希校验的图谱目录
# --output-dir：分结构输出根目录；--threads：ITK 线程数
fnit-nuclei run --norm /absolute/path/subjects/fs_sub01/mri/norm.mgz \
  --aseg /absolute/path/subjects/fs_sub01/mri/aseg.mgz \
  --wmparc /absolute/path/subjects/fs_sub01/mri/wmparc.mgz \
  --atlas-root /absolute/path/nuclei_atlases \
  --output-dir /absolute/path/fnit_nuclei --threads 4

# --cross：已有官方 subject；--sd：其父目录；--threads：官方线程数
segment_subregions thalamus --cross fs_sub01 --sd /absolute/path/subjects --threads 4
segment_subregions hippo-amygdala --cross fs_sub01 --sd /absolute/path/subjects --threads 4

# --reference：已保存的官方原网格标签；--candidate：FNIT 对应标签
# --output：逐标签 Dice、硬体积差和通过状态的 JSON
python validation/subregions/compare_nuclei.py \
  --reference /absolute/path/official/ThalamicNuclei.FSvoxelSpace.mgz \
  --candidate /absolute/path/fnit_nuclei/thalamus/ThalamicNuclei.FSvoxelSpace.mgz \
  --output /absolute/path/thalamus_compare.json

# --reference：官方后验软体积表；--candidate：FNIT 同名表；--output：逐结构比较 JSON
python validation/subregions/compare_nuclei_volumes.py \
  --reference /absolute/path/official/ThalamicNuclei.volumes.txt \
  --candidate /absolute/path/fnit_nuclei/thalamus/ThalamicNuclei.volumes.txt \
  --output /absolute/path/thalamus_volumes_compare.json
```

## 参考文献与原实现代码

- [Iglesias 等，2018，丘脑核团概率图谱](https://pmc.ncbi.nlm.nih.gov/articles/PMC6215335/)。
- [Iglesias 等，2015，海马亚区图谱](https://doi.org/10.1016/j.neuroimage.2015.04.042)。
- [Saygin 等，2017，杏仁核图谱](https://doi.org/10.1016/j.neuroimage.2017.04.046)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer)；[原流程说明](https://surfer.nmr.mgh.harvard.edu/fswiki/SubregionSegmentation)。
