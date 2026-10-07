# white.preaparc 四臂同输入阶段诊断

本工具只比较同一份已保存 orig、autodet stats、WM、brain 和 aseg 输入。按顺序运行官方 A、官方 B、Conda 标准版、Conda fast，各用四线程，共用一个有有限等待期限的排他锁。禁止把它称为原始 T1 连续整例验收。完整 Python white placement 尚未存在，本工具不修改 placement 算法。

## Python 调用、输入与输出

```python
from prewhite_same_input import run
report = run(plan_path='/absolute/private/plan.json',  # 现场绑定的私有 JSON
             output='/absolute/private/new-result')  # 新目录，其父目录须已存在
```

所有路径必须是运算机器上的绝对路径。私有计划与回执包含真实输入、程序及资源路径，不能上传公共仓库。计划字段：

| 字段 | 意义与格式 |
|---|---|
| `threads` | 必须为整数 4；四臂串行，总线程预算 4 |
| `hemi` | `lh` 或 `rh`，决定 surface 和 stats 文件名 |
| `timeout_seconds` | 每臂程序超时，正数且不超过 86400 秒；超时杀死进程组 |
| `lock_timeout_seconds` | 共用锁最大等待，正数且不超过 3600 秒 |
| `shared_lock` | 与其他运行任务实际使用同一文件系统锁的绝对路径 |
| `source_root` | 冻结生产源码根目录；从其 `src` 导入 FNIT 相交检测器 |
| `srcSHA` | 下述整个 `src` 树的 SHA-256，不是 Git commit |
| `inputs` | 列表；每项含 `path`、`size`（字节）、`sha256` 和 `relative_path` |
| `arms` | 依次为 `official_A`、`official_B`、`conda_standard`、`conda_fast` |

每臂字段为 `name`、`binary`（含 `path,size,sha256`）、`assets_dir`（运行时 FREESURFER_HOME）、`assets`（每个显式资源含 `path,size,sha256`）和 `license_path`。许可证只检查文件存在，不读取、不复制、不计算哈希。资产清单应覆盖实际使用的隐式 LUT 等资源；脚本只能核查计划声明的资源，不能证明未声明资源从未被读取。官方 A/B 的程序和资源必须完全相同。

输入 `relative_path` 位于 `surf` 或 `mri`，不允许 `..`；必须覆盖 `surf/{hemi}.orig`（有序顶点与三角面）、`surf/autodet.gw.stats.{hemi}.dat`（冻结灰白边界统计）、`mri/wm.mgz`（白质）、`mri/brain.finalsurfs.mgz`（放置影像）和 `mri/aseg.presurf.mgz`（表面前分割）。可额外绑定并复制 `orig.premesh` 等来源文件，工具不会重建 stats。MGZ 空间沿用原输入，表面坐标为 surface RAS，位移单位 mm。

每臂独立复制为 `<output>/<arm>/subject/{mri,surf}`，不建立链接。结果包括各臂 `command.log`、white.preaparc、mrisps.wpa.mgz，以及根目录 `receipt.json`。回执带 schema、runner SHA、实际源码 SHA、Python SHA/版本、主机、CPU 型号、CPU affinity、四线程和 CUDA 禁用状态；官方重复比较额外记录坐标字节确切一致及不同顶点/坐标数。回执记录输入与程序执行前后 SHA、日志和输出 SHA、时间、退出码、坐标 `<f4` 字节 SHA、有序面 `<i4` 字节 SHA、坐标有限性、Euler、边闭合、有序面保持和参与自相交的独立面数。相交使用当前 FNIT `mark_intersections` 原谓词；零相交门槛保持。执行完成与几何过关分为 `status` 和 `zero_intersection_gate`。

只有顶点数组形状和有序三角面都与官方 A 一致，才计算官方 B/标准/fast 相对 A 的逐顶点最大、P99、平均欧氏位移；否则记录对应关系未证明，不以最近点代替。`wall_seconds_including_lock_wait` 包含计划读取、预检、复制、执行、读出、检测及哈希；锁等待另列，写最终回执的磁盘耗时不包含在该字段。

源码 SHA 的定义：遍历 `source_root/src` 全部普通文件，排除 `__pycache__` 和 `.pyc`，拒绝符号链接；按路径排序，对每项拼接 UTF-8 `相对路径 + NUL + 十进制字节数 + NUL + 文件SHA256 + LF`，计算整体 SHA256。可在计划制作时调用同模块 `source_binding(source_root)`。读源码、输入、程序、资产期间发生变化或执行后绑定变化会失败；已有输出目录直接拒绝。在创建任何输出之前读取冻结计划，拒绝嵌套于 source_root、输入 subject 共同根、每个输入父目录、资产目录或程序父目录的输出；这类拒绝不创建目录或回执。其他预检或执行异常在新目录保留失败回执。无需许可证正文或网络连接。

## 命令行

```bash
python prewhite_same_input.py \
  --plan /absolute/private/plan.json \
  --output /absolute/private/new-result
```

输出路径必须显式提供。运行环境隐藏 CUDA、删除继承的 LD_PRELOAD/LD_AUDIT，并固定 OpenMP、BLAS、ITK 线程变量为 4；同时设置 NUMBA_NUM_THREADS=4，PATH 仅含当前 Python 父目录、当前臂程序父目录、/usr/bin 和 /bin。继承 LD_LIBRARY_PATH 的原值只记录在私有回执，动态库没有全面绑定，不能称动态依赖已核验。原程序本身是否另外生成线程仍需实际监测。程序 cwd 为独立 arm 的 mri 目录。

## 对应原软件调用

工具原样构造当前 white.preaparc 的 `mris_place_surface` argv：`--adgws-in --wm --threads 4 --invol --lh/--rh --i --o --white --seg --restore-255 --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 --outvol`。各路径指向独立副本。它只执行计划里绝对路径且哈希核验通过的阶段程序。

## 验证与更新记录

2026-10-07：新增隔离四臂 runner，语法编译与标准库合同测试属于程序验证；真实四臂诊断已完成：官方A/B重复逐位一致；Conda标准/fast彼此逐位一致，但与官方A的P99位移0.026694 mm、最大位移0.666025 mm。四臂同一检测器均保留16个相交面，zero-intersection gate失败；这不能归因于fast差异。耗时和绑定见[脱敏机器报告](runtime/server_refresh_20261007/prewhite_same_input_public.json)，没有生成新版脑图。真实运行后保留完整私有回执，并由协调者另制作脱敏比较。历史结果保持其原版本和输入绑定，不改标当前。

## 来源

原算法与阶段命令：[FreeSurfer mris_place_surface](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_make_surfaces/mris_place_surface.cpp)。表面重建背景：Dale, Fischl & Sereno (1999), NeuroImage 9:179–194；Fischl, Sereno & Dale (1999), NeuroImage 9:195–207。相交计数复用 FNIT `mris_remove_intersection_python.py`。
