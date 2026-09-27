# 完整 Conda YAML 安装与真实 T1 单阶段复核

2026-09-27 在 headcw 新建独立环境；gpucw1 用同一共享环境检查 glibc 2.17 和 H100 CUDA。此次测试使用仓库 `environment-recon-all-cpp.yml`，SHA-256 为 `4239f99c4c0223e362623dbc244504cf0f296cadf337b6c3df6f2840938bbd66`；`pyproject.toml` SHA-256 为 `2de15b1be3736935f9b97487f050576f41e072b6d8c2e94faf79f8832bd74996`。测试仓库是已发布 main 的文件快照，和上述两个本地文件的哈希相同。

## 环境与程序

共享存储上的独立目录为 `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_yaml_install_20260927`。实际创建命令：

```bash
CONDA_OVERRIDE_GLIBC=2.17 conda env create \
  -p /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_yaml_install_20260927/env \
  -f environment-recon-all-cpp.yml
```

`-p` 是新环境目录；`-f` 是仓库内完整的依赖 YAML。安装进程退出 0，环境占约 11 GiB。导入检查得到 `fnit 0.13.0`、`torch 2.5.1`、`nibabel 5.4.2`、`numba 0.67.0`、`SimpleITK 2.5.6` 和 `ants 0.6.3`；`run_recon_all_python`、批量 API、`fnit-recon-all --help`、权重与模板安装入口均可用。gpucw1 上 `torch.cuda.is_available()` 为真，H100 PCIe 上的 float32 矩阵乘法完成。新建的 PyTorch 会话尚未进入 FNIT 推理入口时 `torch.backends.cuda.matmul.allow_tf32=False`；FNIT 的 GPU 入口会显式启用 TF32，少数已验证的精度敏感阶段会局部关闭并恢复。

在该新环境中执行仓库脚本：

```bash
bash tools/build_recon_all_fs_cpp_conda.sh \
  /path/to/freesurfer-source-d932c45 \
  /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_yaml_install_20260927/native-build
```

第一个参数是固定 FreeSurfer 8.2 源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 的洁净源码目录；本次使用预先校验的源码归档，树哈希 `df2ace4b904dc722090782895c251ceac65b8c52f192c2abcf7ce3daabc83585`。第二个参数是独立构建输出目录。构建退出 0，产出 `mri_em_register`、`mri_segment`、`mri_edit_wm_with_aseg`、`mris_fix_topology`、`mris_fix_topology_fnit`、`mris_inflate`、`mris_place_surface` 七个程序。`ldd` 无缺失共享库、无系统 FreeSurfer/外部 ITK 路径；最高 glibc 符号需求为 2.14。gpucw1 的 glibc 为 2.17，在不设置已安装 FreeSurfer 路径或库路径时，七个程序均通过动态加载和启动检查。`mris_fix_topology` 仅用于诊断，当前整例调用拓扑变体。程序仍需外置模板/图谱和私有 `FS_LICENSE`。

## `mri_segment` 同输入测试

输入来自 OpenNeuro ds000114 sub-01 真实 T1 被试保存的 `mri/antsdn.brain.mgz`，SHA-256 为 `9bf196431ac355130060b9f1eb130747d359df1363e88678a1d09190371a597b`。在 headcw 用新环境编译出的程序运行：

```bash
mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz
```

`-wsizemm 13` 指定局部窗口宽度 13 mm；`-mprage` 启用此 T1 类型的分割参数；`antsdn.brain.mgz` 是去噪脑图输入；`wm.seg.mgz` 是与输入同空间的 uint8 MGZ 初始白质分割输出。官方等价命令相同，但调用 FreeSurfer 8.2 自带的 `mri_segment`。本次另用 `module load freesurfer` 提供的官方 8.2.0-1 程序对完全相同的输入补跑一次，只用于准确度与时间参考；候选输出只由 Conda 编译程序生成。

| 检查 | 新环境源码编译程序 | 官方保存结果 |
| --- | ---: | ---: |
| 输出大小 | 256 × 256 × 256，uint8 | 256 × 256 × 256，uint8 |
| 逐体素差 | 0 / 16,777,216 | 参考 |
| 仿射与 MGH 头前 284 字节 | 完全相同 | 参考 |
| MGZ 压缩文件 SHA-256 | `fcb9b16c5670da7690892f3d5dc15343d5e470e92a6f92888096ed4698517f99` | `94895a0563f7d3c527538b238362ca28240863e4c13305e93b076d9dbdf8f522` |
| 本次运行墙钟、最大 RSS | 41.04 秒；150,304 KiB | 63.93 秒；144,260 KiB |

两份 MGZ 压缩字节不同，解码后的体素、仿射和头字段相同；与原先保存的官方结果比较也为 0 个不同体素。新编译 `mri_segment` 的 SHA-256 为 `b81b219de0b5c777e9d24a6106d7a492b8d3bf49ab8df2f6372df5e2efad373a`。两条命令在同一 headcw 节点、相同输入下顺序各运行一次；这次观测为 Conda 41.04 秒、官方 63.93 秒。未控制其他用户负载，也没有重复计时，不能把单次差值当作稳定加速比。[旧配对报告](../native_cpp_conda_20260927/MRI_SEGMENT_CONDA_PAIR.md)的 51.85 秒官方计时使用另一份输入哈希，不能与本次时间配对。

这次验证说明完整 Conda YAML 能安装 Python 环境、从固定源码编译并在目标 GPU 节点加载所需 C++ 程序，且一个真实 T1 白质分割阶段与官方逐体素一致。**没有**用新环境从原始 T1 完成整例重建；最终 white/pial、注册球面、逐顶点和脑区指标仍按[发布门槛](../RELEASE_GATES.md)分别验收。
