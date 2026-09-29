# 最终 smoothwm：同输入逐点验证

真实 `sub-01_T1w.nii.gz` 的 FreeSurfer 8.2 `recon-all.log` 记录了两个不同的平滑阶段：

| 输出 | 官方命令 | 输入 |
| --- | --- | --- |
| `surf/H.smoothwm.nofix` | `mris_smooth -nw -seed 1234 H.orig.nofix H.smoothwm.nofix` | 未修复拓扑的 `orig.nofix`；默认 10 轮 |
| `surf/H.smoothwm` | `mris_smooth -n 3 -nw -seed 1234 H.white.preaparc H.smoothwm` | 放置后的 `white.preaparc`；3 轮 |

`H` 为 `lh` 或 `rh`。最终平滑命令见该被试官方 `scripts/recon-all.log` 的第 4675、4681 行。标准 runner 先运行 Conda white.preaparc 放置，再用 Python 平滑 3 轮得到 `smoothwm`。该路径另有[真实 T1 保存阶段 LH 连通检验](../../validation/recon_all/python_gpu_port/white_connected_prefix_20260927/README.md)；最终 white/pial 放置及完整重建尚未验收。`smoothwm.nofix` 使用独立的 `orig.nofix` 输入和 10 轮平滑。

## 函数与调用

`fnit.recon_all.smooth_surface_python.smooth_surface(input_path, output_path, iterations=10, device="cpu") -> None` 读取 FreeSurfer 三角表面，输出有序面及体积几何不变的三角表面。生成最终 `smoothwm` 时，`input_path` 为 `surf/H.white.preaparc`，`output_path` 为 `surf/H.smoothwm`，`iterations=3`。`device` 接受 `cpu` 或 CUDA 设备字符串；下述实测使用 CPU float32。函数按有序一环邻居平均顶点，不额外写曲率文件，对应官方命令的 `-nw`。创建者注释导致文件字节不同；数值验收比较有序坐标、面和体积几何。

Python API：

```python
from fnit.recon_all.smooth_surface_python import smooth_surface

smooth_surface(
    input_path="surf/lh.white.preaparc",  # 已放置的输入表面
    output_path="surf/lh.smoothwm",  # 平滑后的输出表面
    iterations=3,  # 最终 smoothwm 平滑三轮
    device="cpu",  # 计算设备；也可传 CUDA 设备字符串
)
```

函数返回 `None`；输出写入 `output_path`，保留输入的有序面和体积几何。

独立 Python 命令：

```bash
python -m fnit.recon_all.smooth_surface_python \
  surf/lh.white.preaparc surf/lh.smoothwm --iterations 3 --device cpu
```

命令行两个位置参数依次对应 `input_path`、`output_path`；`--iterations` 指定平滑轮数（默认 10），`--device` 指定 `cpu` 或 CUDA 设备（默认 `cpu`）。

对应的官方命令：

```bash
mris_smooth -n 3 -nw -seed 1234 \
  surf/lh.white.preaparc surf/lh.smoothwm
```

右侧调用将 `lh` 改为 `rh`。独立函数本身不生成 `white.preaparc`；标准 recon-all 路径由 Conda 放置函数提供这一输入。

## 冻结官方输入的真实数据对照

在 `gpucw1`，两种实现均读取这例去标识真实 T1 已保存的**同一组官方双侧 `white.preaparc`**。T1 SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`，官方程序版本为 FreeSurfer 8.2.0。交替运行 3 组配对测试，计入文件 I/O 和原生程序启动，不计 Python 导入。比较范围为有序 float32 坐标、有序面和 FreeSurfer 体积几何。

| 半球 | 顶点 / 面 | 完全相同的坐标分量 | 最大差异 | Python 墙钟中位数 | 官方墙钟中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LH | 106,622 / 213,240 | 319,866 / 319,866 | 0 mm | 7.889 s | 3.153 s |
| RH | 105,541 / 211,078 | 316,623 / 316,623 | 0 mm | 6.634 s | 3.609 s |

有序面和体积几何字段也完全一致。3 次耗时、输入哈希和原生命令重放结果见 [`report.json`](../../validation/recon_all/python_gpu_port/smoothwm_final_same_input_20260927/report.json)。这证明平滑函数在**冻结的同输入**条件下数值一致，不等于候选上游及整例输出一致。历史 LH 全候选保存阶段试验中，最终 `smoothwm` 仍有 18 个顶点位移 >0.1 mm；下游指标尚未验收。

在包含 FreeSurfer 可执行程序和本 Python 包的环境中，按以下命令重跑配对 benchmark：

```bash
python validation/recon_all/python_gpu_port/benchmark_smooth_surface.py \
  --left-surface SUBJECT/surf/lh.white.preaparc \
  --right-surface SUBJECT/surf/rh.white.preaparc \
  --native-binary /path/to/mris_smooth \
  --output-dir /path/to/output --iterations 3 --device cpu --repeats 3
```

benchmark 脚本的 `--left-surface`、`--right-surface` 是两侧同输入表面，`--native-binary` 是官方 `mris_smooth`，`--output-dir` 保存配对结果，`--iterations` 和 `--device` 与函数参数相同，`--repeats` 指定配对次数。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
