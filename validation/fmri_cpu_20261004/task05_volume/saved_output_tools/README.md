# 完整 volume 的已保存输出核对工具

## 1. 功能

这些工具只读取 benchmark 已完成运行的绑定回执、完整输出和原版节点，不执行新的影像处理。FNIT 运行时不导入这些验证工具。

## 2. 输入与输出

- CPU binding：私密 JSON，绑定两份冻结源码、输入/资产 SHA、四个后端与线程预算队列、实际 guard、CPU affinity 和已授权锁；由完整运行控制器生成。
- 原版 manifest / inputs / wrapper：实际 fMRIPrep 25.2.4 运行时使用的文件及 SHA。许可证内容不读取、不复制。
- 每个队列：两份源码各保存 first/warm 完整结果，四张四维 BOLD 完整保留 180 帧，另有 mask 和全部运动/BBR 变换。
- 输出目录必须是新目录。`comparison.public.json` 只含聚合值与 SHA；`inventory.private.json` 含实际路径和回执，留在私密服务器。
- 官方角色 manifest 依据原版保存的节点输出字段确认 native/T1w/MNI6-res2 preproc、reference、mask、全部 HMC 和 BBR；需要原始移动/固定网格及变换方向。不能仅凭相似文件名配对。

## 3. 命令行

```bash
# 变量分别是实际冻结回执路径、其 SHA 和一个新的私密输出目录。
FNIT_CPU_BINDING=/private/benchmark/binding.private.json
FNIT_CPU_BINDING_SHA256="actual_binding_sha256"
FNIT_COMPARISON_OUTPUT=/private/benchmark/new_saved_comparison
python validation/fmri_cpu_20261004/task05_volume/saved_output_tools/collect_cpu16.py \
  --binding "$FNIT_CPU_BINDING" \
  --binding-sha256 "$FNIT_CPU_BINDING_SHA256" \
  --collector-queue fnirt_cpu1 \
  --output "$FNIT_COMPARISON_OUTPUT"
```

`--collector-queue` 选择已完成并授权的一线程核组；工具从 binding 读取同一实际 affinity 和锁。全部四队列完成 16 次完整调用后才能收集；不自行申请新资源。各工具的完整参数可用 `--help` 查看。

## 4. 原版保存节点

`export_reference_saved.py` 必须在原来的固定镜像内运行，用同版本 Nipype 读取已有 `result_*.pklz`；不会重新启动 fMRIPrep。容器须复用原运行的 TemplateFlow 路径绑定和 runtime home，否则 traits 的 existing-file 校验会失败。绑定缓存为只读，保留原版 launcher/payload 双退出码、180 帧、STC OFF、1/8 线程与完整源码/输入门槛。先导出私密 inventory，再据实际输出字段构造 `compare_official_saved.py` 的角色 manifest。

## 5. 精度与读取边界

`volume_numeric.py` 对全部值分八帧读取，保持原归约顺序、dtype、finite 与加载/存储头门槛。实际环境无 indexed_gzip，CPU16 使用的 v3 仅将三个 `nib.load` 改为 `keep_file_open=True`，减少 gzip 前缀重复解压；计算方法未变。当前 CPU16 每组十科学输出 395,140,404 个值，新旧与缓存输出均精确相同，见 [完整报告](../final_merged_cpu_v1.public.json)。

官方比较只允许同一物理网格的无损轴交换/翻转；不插值、不拟合强度。变换方向、单位与空间单独核查。FNIT 没有独立原生 pre-scale/pre-highpass preproc 输出，不能把 clean 当作 preproc。原版 preproc 与 FNIT PICA/AROMA/clean 的范围分别说明。

本次官方保存输出比较采用 v2 comparator：working-native 没有声明时间单位时保留完整帧数、有限值和 SHA，明确记录 `physical_time_axis_checked=false`。v4 numeric 另给 `physical_pair` 两处加载启用持久 gzip 流，去掉这两个关键字后 AST 与 v3 完全相同；归约、数值门槛及空间/时间检查不改。[官方比较报告](../official_saved_comparison_v2.public.json)只完成四组同物理网格 MNI，四组 T1w 网格不同，未声明整链等价。

## 6. 记录

- 本轮完整测量绑定 `6f624040 → 98019133`；CPU16 与 GPU4 保持性通过。
- CPU16 的冻结 v3 工具见提交 `23f75b32`；当前官方读取使用 comparator v2（`d4425507…`）及 numeric v4（`b26b2c45…`）。
- v2 保留原算法；v3 仅维持 gzip 打开流，CPU16 使用的收集器 SHA `756f72638d3ae566b8decdd256d12ebc8d45808504d80158aede124bd667d0ad`，CPU16 v3 数值工具 SHA `f4150e1d4c2b9f3dbcb7535d04772e7e69c1925cbc721033f8d2ad69b7013b77`。
- [合并源码核查](../final_source_bridge_20261006.public.json)区分实际冻结闭与 Git tracked 文件，不将旧计时改标为合并后的新实测。

## 7. 参照

[fMRIPrep](https://github.com/nipreps/fmriprep)、[NiWorkflows](https://github.com/nipreps/niworkflows)、[Nibabel](https://github.com/nipy/nibabel)。功能接口和原软件调用见 [volume 文档](../../../../docs/fmri/README.md)。
