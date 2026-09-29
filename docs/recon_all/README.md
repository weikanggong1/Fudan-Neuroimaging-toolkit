# 单幅 T1w 的 recon-all 重建

[返回首页](../../README.md) · [安装与原生程序](CONDA_CPP_BUILD.md) · [阶段与官方命令](CONDA_CPP_STAGES.md) · [验收范围](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)

`fnit-recon-all` 从一幅 T1w 生成体积分割、双侧皮层表面、顶点指标、脑区标注和统计。标准路径依次执行拓扑修复、`white.preaparc`、球面生成与配准、最终 white、[Conda 源码构建的四轮 pial 放置](NATIVE_PIAL_PLACEMENT.md)和后处理。必要程序或资产缺失时，入口在运行前报错；阶段失败时抛出异常并保存报告。当前支持单幅 T1w；多 T1、T2/FLAIR 和纵向重建不在此接口的范围内。

本页描述当前源码的调用方式。阶段的同输入结果不代表从原始 T1 连续重建已通过验收。现版整例的输出完整性、数值比较与资源记录应分别查看运行 JSON 和[验收说明](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)；旧版本整例数据不作为现版精度或速度结论。

## 安装

在仓库根目录创建[主页 Conda 环境](../../environment.yml)，然后运行[原生程序安装脚本](../../tools/setup_recon_all_native_conda.sh)。脚本从固定 FreeSurfer 源码提交编译所需程序并安装至当前 Conda 环境；不会调用系统安装的 FreeSurfer。模型、模板及个人许可证单独提供。

```bash
conda env create -f environment.yml
conda activate fnit
bash tools/setup_recon_all_native_conda.sh
fnit-setup-weights --model recon-all --dest /data/fnit-weights
fnit-setup-recon-all-assets --dest /data/fnit-assets
fnit-setup-weights --model recon-all --dest /data/fnit-weights --verify-only
fnit-setup-recon-all-assets --dest /data/fnit-assets --verify-only
```

默认资产组包含标准单 T1 流程所需的 98 个文件，入口逐项检查大小及 SHA-256。安装脚本也支持开发者用 `FNIT_RECON_ALL_SOURCE` 指向已准备好的固定源码。构建的命令、补丁、哈希和 Conda 记录见[安装说明](CONDA_CPP_BUILD.md)。2026-09-29 已从主页环境文件创建新 Conda 环境，并从已校验的固定源码归档编译安装所需程序；全新联网检出、资产重新下载及该安装产物的连续整例仍需单独验收。

## 运行

```bash
export FS_LICENSE=/private/license.txt
fnit-recon-all /data/sub01_T1w.nii.gz /data/subjects/sub01 \
  --weights-dir /data/fnit-weights \
  --assets-dir /data/fnit-assets \
  --device cuda:0 --threads 4
```

`FS_LICENSE` 指向用户自己的许可证文件。默认从激活的 Conda 环境 `bin/` 寻找原生程序；`--native-bin-dir` 可指定已核验的开发者构建目录。没有 GPU 时可使用 `--device cpu`。`subject_dir` 须不存在或为空。

```python
from fnit.recon_all.native_free import run_recon_all_python

report = run_recon_all_python(
    t1="/data/sub01_T1w.nii.gz",  # 一幅原始 T1w NIfTI 的路径
    subject_dir="/data/subjects/sub01",  # 空的被试输出目录
    weights_dir="/data/fnit-weights",  # 已校验的模型权重目录
    assets_dir="/data/fnit-assets",  # 已校验的模板和图谱目录
    device="cuda:0",  # PyTorch 阶段的设备；无 GPU 时为 "cpu"
    threads=4,  # 原生程序与 CPU 算子的线程数
    native_bin_dir=None,  # None 表示使用当前 Conda 环境的 bin/
)
# report 是运行报告字典；仅在全部阶段与文件完整性检查通过后返回。
```

`run_recon_all_python(...) -> dict` 的输入参数均在上例中列出。入口把原始 T1 重采样到 1 mm、256³ 的 conform 网格；`mri/orig.mgz`、分割图与最终体积图均在该网格上。`surf/lh.*`、`surf/rh.*` 使用该被试的 surface RAS，网格文件保存有序顶点和三角面；`surf/H.thickness` 等顶点图及 `label/H.*.annot` 与对应半球的顶点顺序对齐。原始 NIfTI 仿射和 conform 网格不能互换使用。

主要输出位于被试目录下：

| 路径 | 内容与结构 |
| --- | --- |
| `mri/*.mgz`、`mri/transforms/*` | 体积分割、强度图和配准变换；MGH 体积为 conform 网格，部分 MNI 辅助 NIfTI 为图谱网格。 |
| `surf/H.white`、`H.pial`、`H.sphere.reg` | 双侧有序三角网格；`H` 为 `lh` 或 `rh`。 |
| `surf/H.thickness`、`H.area`、`H.volume`、`H.curv` 等 | 每顶点标量，顺序与同侧表面一致；厚度与几何位置单位为 mm，面积为 mm²，体积为 mm³。 |
| `label/H.*.annot` | 每顶点脑区编码及颜色表。 |
| `stats/*.stats` | 体积和皮层分区统计。 |
| `fnit-native-free-run.json` | 阶段耗时、原生程序哈希、输出清单、文件完整性、网格质量和数值验收状态。 |

固定单 T1 profile 的全部 138 个相对路径由[清单](../../src/fnit/recon_all/expected_outputs.py)定义。`report["outputs"]` 是实际存在的 `{相对路径: 绝对路径}` 映射；`report["output_validation"]` 给出 138 项存在性检查；`report["mesh_validation"]` 逐侧检查闭合球面拓扑、顶点顺序、有限坐标及 white/pial 自相交；`report["numeric_validation"]` 单独记录参考结果的数值验收，默认是 `not_run`。`report["stages"]` 为按执行顺序排列的阶段名、秒数和可得的 PyTorch GPU 峰值字节数。默认关闭 CUDA 分配缓存时，父进程的 PyTorch 峰值接口不可用，以 `gpu_memory_mode` 说明，整例显存仍需进程级外部采样。`status="complete"` 只表示全部阶段执行、输出存在性和网格质量检查通过，不表示已与官方结果达到数值门槛。运行失败会抛出异常，部分失败信息写在 JSON 中；入口前置校验失败时可能尚未创建 JSON。

批量 Python API `run_recon_all_python_batch(jobs=..., weights_dir=..., assets_dir=..., devices=..., threads=..., native_bin_dir=None)` 中，`jobs` 是按顺序排列的 `{"t1": 路径, "subject_dir": 空目录}` 列表；`devices` 是可用设备列表；其余参数与单被试一致。返回值为同序的报告列表；任一被试失败时抛出 `RuntimeError`。每个设备一次运行一例。

## 验证与边界

```bash
python validation/recon_all/python_gpu_port/compare_complete_subject.py \
  /data/reference-sub01 /data/subjects/sub01 \
  --report /data/sub01-comparison.json
```

比较需要单独生成的真实 T1 官方参考目录。[固定 138 项比较器](../../validation/recon_all/python_gpu_port/compare_complete_subject.py)检查体积、表面、顶点图和统计；其中同输入阶段验证、FNIT 自产上游连续链、从原始 T1 开始的整例验证应分别记录。若双侧网格顶点数不同，空间最近点距离只能用于定位差异，不能证明同源顶点一致。资源报告给出 PyTorch allocated/reserved 峰值；进程总 GPU 占用仍需外部采样，GB 与 GiB 均须注明。标准流程默认允许 TF32，SynthStrip、SynthSeg 和 Talairach affine 有经过阶段验证的 FP32 例外；不自动启用 FP16/BF16。

各阶段的输入、输出、官方命令和真实数据记录见[阶段索引](CONDA_CPP_STAGES.md)。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 官方 recon-all 说明](https://www.freesurfer.net/fswiki/recon-all)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
