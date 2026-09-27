# MCA/dura 与静脉窦 PyTorch 分割：冻结同输入验收

`fnit.recon_all.aux_seg.mri_mcadura_seg` 与 `mri_vsinus_seg` 从 `nu.mgz`
生成 256³ MCA/dura 和静脉窦标签图。前者读取 MNI152 双侧先验，后者
读取静脉窦先验和 `synthseg.rca.mgz` 以清除皮层重叠；二者都使用
`synthmorph.1.0mm.1.0mm/reg.targ_to_invol.lta`、外置 H5 权重以及
PyTorch 推理。现有整例尚未生成该 LTA，也未将这些模型纳入
`recon-all` 默认权重下载清单，所以本次是**冻结官方输入的独立配对**。

```python
from fnit.recon_all.aux_seg import mri_mcadura_seg, mri_vsinus_seg

mri_mcadura_seg("SUBJECT/mri/nu.mgz", "SUBJECT/mri/mca-dura.mgz",
                "SUBJECT/mri/transforms/synthmorph.1.0mm.1.0mm",
                "/path/to/assets", device="cpu")
mri_vsinus_seg("SUBJECT/mri/nu.mgz", "SUBJECT/mri/vsinus.mgz",
               "SUBJECT/mri/transforms/synthmorph.1.0mm.1.0mm",
               "/path/to/assets", ctxseg_path="SUBJECT/mri/synthseg.rca.mgz",
               device="cpu")
```

外部 `FS_TORCH_MODEL_DIR` 指向三个小型 H5 模型所在的权重目录。
官方对应命令为：

```bash
mri_mcadura_seg --i SUBJECT/mri/nu.mgz --o SUBJECT/mri/mca-dura.mgz --threads 4 --synthmorphdir SUBJECT/mri/transforms/synthmorph.1.0mm.1.0mm
mri_vsinus_seg --s SUBJECT --rca-synthseg --threads 4 --synthmorphdir SUBJECT/mri/transforms/synthmorph.1.0mm.1.0mm
```

冻结官方被试 `a_official` 的 `nu.mgz`、先验 LTA、SynthSeg 标签图、
FreeSurfer 8.2 模型与先验作为**相同输入**。原实现虽有两张图各
0/16,777,216 体素差异，却写成 int32，并从 affine 重建了有差异的
MGH 头。现从 `nu.mgz` 复制头部、将数据写成 float32；两张图各
**0/16,777,216 体素差异**，数据类型、284 字节 MGH 头和 affine
与官方均一致。MGZ 压缩文件哈希不同，不能视为文件逐字节相同。
[输入/输出哈希和逐项结果](aux_seg_same_input_20260927.json)。

headcw CPU 4 线程独立 Python 调用耗时 MCA/dura 4.65 秒、静脉窦
7.01 秒；归档的官方 recon-all 日志分别记录 118 秒、55 秒。
两者不是同时间、同负载的配对速度试验，不据此宣称稳定加速。
修复后 `test_aux_seg.py` **4/4 通过**。接入整例前仍须生成并验收
MNI152 先验注册 LTA；不能用现有 MNI305 Talairach LTA 替代。
