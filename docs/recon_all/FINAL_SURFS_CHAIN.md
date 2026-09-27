# `brain.finalsurfs.mgz` 的 Python 掩膜和编辑链

`fnit.recon_all.finalsurfs_python.run_finalsurfs(subject_dir, device="cpu")` 从 `subject_dir/mri` 读取六张 conform 后的 MGZ：`brain.mgz`、`brainmask.mgz`、`mca-dura.mgz`、`vsinus.mgz`、`entowm.mgz`、`aseg.presurf.mgz`。它在同一目录写出 `brain.finalsurfs.mgz` 和人工编辑前的相同检查点 `brain.finalsurfs.manedit.mgz`，返回前者路径。输入必须处在同一体素网格。掩膜操作用指定 CPU/CUDA 设备上的 PyTorch，最后两步编辑用 CPU NumPy/SciPy；不调用 FreeSurfer 程序。

Python 调用：

```python
from fnit.recon_all.finalsurfs_python import run_finalsurfs

final_mgz = run_finalsurfs(
    subject_dir="/path/to/subjects/sub01",  # 六张 MGZ 输入所在的被试目录
    device="cpu",  # 掩膜步骤运行设备，可选 "cuda:0"
)
# final_mgz 是输出 mri/brain.finalsurfs.mgz 的路径。
```

命令行：

```bash
python -m fnit.recon_all.finalsurfs_python /path/to/subjects/sub01 --device cpu
```

CLI 位置参数是被试目录；`--device` 选择掩膜计算设备。函数同时写出 `brain.finalsurfs.mgz` 和相同的 `brain.finalsurfs.manedit.mgz`，两者均位于该目录的 `mri/` 下。

从 `subject_dir/mri` 执行的 FreeSurfer 8.2 等价命令：

```bash
mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz
mri_mask -oval 1 -invert brain.finalsurfs.mgz mca-dura.mgz brain.finalsurfs.mgz
mri_mask -oval 1 -invert brain.finalsurfs.mgz vsinus.mgz brain.finalsurfs.mgz
mri_edit_wm_with_aseg -sa-fix-ento-wm entowm.mgz 2 255 255 brain.finalsurfs.mgz brain.finalsurfs.mgz
mri_edit_wm_with_aseg -sa-fix-acj aseg.presurf.mgz 255 255 brain.finalsurfs.mgz brain.finalsurfs.mgz
cp brain.finalsurfs.mgz brain.finalsurfs.manedit.mgz
```

## 真实 T1 对照

[保存的对照报告](../../validation/recon_all/python_gpu_port/finalsurfs_chain_20260927/report.json)使用去标识 `sub-01_T1w.nii.gz`（SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`）及其在 gpucw1 完成的 FreeSurfer 8.2 重建。两组完整的五步复跑读取**相同的六张官方保存输入图**，各输入哈希均记录在报告中。Python 生成的最终图与检查点均为 **0/16,777,216 个差异体素**，包含函数 I/O 用时 8.83 秒；五条官方命令包含进程启动和 I/O 用时 8.21 秒。这是在同一主机的一次非配对运行，不能由此判断稳定速度差。

FNIT v5 当时的 T1 前缀在此影像上已使 `brain`、`brainmask`、`entowm`、`aseg.presurf`、`nu` 的体素与官方相同，但尚未生成 `mca-dura`、`vsinus`。第二次复跑取其中**四张 FNIT 生成的输入**，只借用**两张官方辅助标签图**，9.13 秒得到 0/16,777,216 个最终图差异体素。这是有条件的上游测试，不是 FNIT 整例结果。

[字节分析](../../validation/recon_all/python_gpu_port/finalsurfs_chain_20260927/byte_analysis.json)显示，解压后的 MGH 头和体素负载相同。Python 文件只在尾部来源标记少一个字节：原生命令复跑及保存的官方输出在 `UNKNOWN` 后还有一个空字节；此尾部不参与体素或表面计算。

可选 `--native-white-preaparc` 调度现在会先生成被试 MNI152 LTA 和两张辅助分割，再调用本函数；默认路径不调用。前缀连通性与后续皮层指标还需从原始 T1 验收，冻结同输入结果不能证明整例形态指标一致。
