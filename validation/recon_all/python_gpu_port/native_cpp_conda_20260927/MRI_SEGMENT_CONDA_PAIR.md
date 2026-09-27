# `mri_segment`：同输入 Conda C++ 与 Python 配对

这是一份早期**独立子函数验收**记录，固定输入为当时保存的 `antsdn.brain.mgz`（256³ uint8；SHA-256 `378548a3f58f74878450525a734c15ff19ce935dff913133f52aa149bb91d2ce`）。后续 v3 整例已接入去噪与 Conda C++ `mri_segment`；本报告的时间和哈希不能替代[当前完整 YAML 环境的另一份同输入测试](../conda_yaml_install_20260927/README.md)，也不能代表整例提速。

## 功能与用法

`mri_segment` 从去噪脑图生成初始白质分割 `wm.seg.mgz`。官方 FreeSurfer 8.2、同一提交源码经 Conda 编译的程序和现有 Python 函数使用相同输入，分别运行：

```bash
$FS_BIN/mri_segment -wsizemm 13 -mprage antsdn.brain.mgz official.wm.seg.mgz
FREESURFER_HOME=/path/to/external/assets FS_LICENSE=/path/to/private/license.txt \
  /path/to/conda-build/bin/mri_segment -wsizemm 13 -mprage antsdn.brain.mgz conda.wm.seg.mgz
```

```python
from fnit.recon_all.mri_segment import segment_white_matter_mgz
segment_white_matter_mgz(
    source_path="antsdn.brain.mgz",  # 去噪脑图，256³ uint8 MGZ
    output_path="python.wm.seg.mgz",  # 初始白质分割输出，原空间 uint8 MGZ
)
```

编译命令为 `cmake --build /path/to/conda-build/build --parallel 4 --target mri_segment`；主页的 `tools/build_recon_all_fs_cpp_conda.sh` 已将其列为第七个目标。单目标增量编译在 headcw 用时 4.05 秒。编译器、ITK 和其余链接依赖来自项目 Conda 环境；二进制需要 `FREESURFER_HOME` 指向外置数据目录以及私有 `FS_LICENSE` 环境变量，不链接已安装的 FreeSurfer 库，最大 glibc 符号版本为 2.14。编译二进制 SHA-256 为 `0842a77762043c4f78f8be46799b32f02f24da1bb05b9e9d8231055e951b9309`。

## 同输入结果

| 实现 | 与官方不同体素 / 16,777,216 | MGH 头部、仿射、类型 | 单次墙钟 | 最大 RSS |
|---|---:|---|---:|---:|
| 官方 FreeSurfer 8.2 | 0 | uint8 参考 | 51.85 s | 153,884 KiB |
| Conda 编译 C++ | **0** | 与官方一致，uint8 | **40.45 s** | 160,124 KiB |
| 现有 Python | **0** | 与官方一致，uint8 | 105.82 s | 1,933,132 KiB |

MGZ 压缩文件 SHA-256 三者不同；验收以解码后的全部体素、MGH 头部和仿射为准。三条命令在同一 headcw 节点依次运行一次，未控制其他用户负载，因此时间是观测值，不能当作稳定加速比。原始哈希与机器可读结果见 [`mri_segment_conda_same_input.json`](mri_segment_conda_same_input.json)。

随后实际运行了加入第七目标的完整构建脚本；固定源码归档树哈希验证通过，七个程序均在 Conda 环境构建、安装并通过动态链接检查，脚本退出 0、用时 104.29 秒。重编后 `mri_segment` 二进制 SHA-256 改为 `cf0ba5982db19aefbf2bd25aa0fc9f314b6275850beab71269b1d4b362dcd04a`；在同一输入复测仍为 **0/16,777,216 个体素差**、MGH 头部和仿射全同，单次耗时 49.90 秒。[七目标哈希](bin_seven.sha256)、[来源验证](build_provenance_seven.txt)、[构建计时](build_seven.time)和[复测计时](mri_segment_sevenbuild.time)保留此次构建证据。构建时间与两次 C++ 运行时间均为不同负载下的单次观测。

随后在目标 gpucw1（CentOS7 CMake 分支）再次完整运行七目标脚本，退出 0、用时 149.45 秒。重编后 `mri_segment` SHA-256 为 `f3c46df3f178e932d42f52c61118696f5893e8f7def7a53eb2cbb5b7d222e5f3`；它在 headcw 的相同冻结输入上仍为 **0 个体素差**、MGH 头部及仿射全同，单次耗时 41.39 秒。[gpucw1 七目标哈希](bin_seven_gpucw1.sha256)、[构建时间](build_seven_gpucw1.time)和[本次运行时间](mri_segment_gpucw1build.time)分别保存。重编使二进制字节哈希改变，数值配对仍须以具体输出为准。

本报告生成时，整例尚未连通 `brain.mgz → AntsDenoiseImageFs → antsdn.brain.mgz → mri_segment → mri_edit_wm_with_aseg → mri_pretess → mri_fill`。后续 v3 已连通并完成[整例比较](v3_e2e_20260927/BENCHMARK.md)，但最终 white/pial 和脑区指标尚未通过；本报告保留当时的独立阶段数据。
