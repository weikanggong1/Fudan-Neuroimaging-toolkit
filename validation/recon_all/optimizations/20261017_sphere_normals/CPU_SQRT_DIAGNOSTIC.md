# 保存后球面CPU张量故障诊断

## 功能和当前证据

[`diagnose_surface_cpu_sqrt.py`](diagnose_surface_cpu_sqrt.py) 只读取已经保存
的同一个真实球面，定位 `face_area_normals` 的CPU张量操作。它不重跑sphere，
不修改生产、质量判据或网格，也不产生精度验收或速度结论。

首轮benchmark在保存与旧生产逐字节相同的sub-06左球面后退出SIGSEGV。
后续仅CPU的保存后QA在独立进程、新Numba缓存及boundscheck下也崩溃，
栈位于 `squared.double().sqrt().float()` 所在行。该行包含三个操作，
仅凭Python行号不能确定是哪一个失败，也不能直接归因于MKL、环境或线程。
球面头已核对为129,674顶点、259,344面和4,668,910字节；这不是内存stride
证明，内存布局由诊断工具实测记录。

## 输入、输出及具名参数

| 参数 | 格式及含义 |
| --- | --- |
| `--surface` | 已保存的普通三角球面；原坐标surface RAS/mm，面为有序顶点索引，只读 |
| `--surface-sha256` | 必需64位SHA，计算前后须匹配同一球面完整字节 |
| `--source-root` | 冻结FNIT仓库根，包含 `src/fnit/recon_all/mris_register_nonlinear.py` |
| `--face-kernel-sha256` | 必需64位SHA，验证实际加载的原函数源码，没有替换函数体 |
| `--output` | 必须不存在的新私有目录，保存 `state.private.json` |
| `--mode` | 必需：`original`调用原函数；`split`逐项展开到double、sqrt和float转换，仅用于定位 |
| `--import-order` | 默认 `numpy-first`，NumPy/nibabel在Torch之前导入；`torch-first`作为另一项显式条件，必须另开进程和新目录 |

固定Torch intra-op 4线程、interop 1，Numba/BLAS环境4线程，CUDA隐藏；
不自动改变线程数、浮点精度或选择其他数学实现。`split`拆开表达式会改变
临时量的生命周期和地址，不能因它成功就宣布原问题已经修复。

每个操作前后及NumPy统计前后持久化phase；结果包括shape、dtype、字节/
元素stride、contiguous、storage offset、有限值、范围、完整数组字节SHA
和地址对齐，以及本进程映射的Torch/MKL/OpenMP/libm等库路径和stat。
没有复制库或影像。原始内存地址与运行库路径仅保留私有，不发布。

SHA错误在计算前停止；Python异常保留失败报告并非零退出；原生SIGSEGV
由faulthandler写日志，JSON保留最后一个已完成阶段。日志需由外层重定向，
EXIT由外层记录。所有步骤均为诊断，不用其操作墙钟比较性能。

## 命令示例

下面路径是通用示例；实际源和球面SHA由协调者现场核对。先在原租约内
串行运行一个进程，不与整例或其他数值任务重叠：

```bash
python -X faulthandler -u diagnose_surface_cpu_sqrt.py \
  --surface /data/fnit/runs/previous-sphere/lh.sphere \
  --surface-sha256 5d943f0eb3207d574dd82b60f2b1680d6363dfe16be7b019af5abcde66c15c3e \
  --source-root /data/fnit/source_5ca07f99 \
  --face-kernel-sha256 47107b4862ae695e06398615534158b8dfcfecd8654dabe83301160c5eb17688 \
  --mode split \
  --import-order numpy-first \
  --output /data/fnit/runs/cpu-sqrt-split-new
```

需要对照原函数时改为 `--mode original`，并使用另一新目录；不自动循环
测试或重启失败进程。现有对照输入和输出始终保留。法向和面积为内部步骤，
没有独立原软件CLI；完整原命令仍为 `mris_sphere INPUT OUTPUT`，本诊断
不调用该程序。

## 已执行的同文件CPU诊断

得到追加授权后，在原服务器按顺序运行以下保存文件诊断，没有重跑sphere
或整例，没有占用GPU，也未改变其他活动任务。

| 范围 | 结果 |
| --- | --- |
| `split`，NumPy先导入 | EXIT0；double、sqrt、float三步均完成 |
| 原 `face_area_normals`，NumPy先／Torch先导入 | 各EXIT0 |
| 原 `inspect_surface`，普通独立进程 | 3/3 EXIT0 |
| 原 `inspect_surface`，gdb下运行 | EXIT0，未截获native故障帧 |
| 原v1工具、原导入顺序、source-only PYTHONPATH、CPU0–3及新cache/boundscheck | EXIT0 |

split实测平方范数为连续float32 `(259344,)`，元素stride为1，范围约
`1.5646e-5–65.12033`；转换后的double、平方根和最终float均连续且有限。
面索引为0–129673，未发现形状、索引、非有限或负平方范数异常。这些结果
不证明旧崩溃不存在，也不能确定MKL、线程、亲和性或导入顺序就是原因。
旧SIGSEGV日志、失败benchmark和全部新诊断目录均保留。

当前没有科学算法修复或质量判据替换。下一步由协调者使用已加phase回执
及faulthandler的工具进行新A/B；若再失败，先根据真实phase对同一保存文件
定位，不因本表成功而宣布已修复。所有诊断操作耗时不计入性能benchmark，
也不代替完整网格质量或两例端到端回归。
