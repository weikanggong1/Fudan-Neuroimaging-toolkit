# 三种 recon-all 来源：真实输入预检

2026-10-03，功能源码起点 `231dfaa1`；来源调度实现 `7d141bcc`，阶段生命周期/PATH与完整文档 `0de2ed3d`。本目录保存 CPU 只读预检与专用测试，**没有重新运行 recon-all，也没有评测新的 T1→SC 整链精度/速度**。整链与模板重用评测由本轮协调报告单列。

## 数据与实际结果

真实数据为 OpenNeuro [ds001226](https://github.com/OpenNeuroDatasets/ds001226/tree/fb4d0fda44f2ab7a732fb4ab6cd62add09dc1cd7)，固定快照 `fb4d0fda44f2ab7a732fb4ab6cd62add09dc1cd7`，数据许可 CC0。核对以前已经完成的官方结果与此前另一 FNIT 重建队列的当前输出。完整服务器私有记录保留原始 T1、报告及 anatomy 文件的 SHA-256。仓库公开摘要只保存完成检查、脑体积网格、white/pial 顶点/面数和检查时间。

| 来源 | 真正通过 | 未完整/缺失 | 范围 |
|---|---:|---:|---|
| 官方已有 subject | 10/10 | 0 | 七个核心文件均可读，体积几何相同，white/pial 有一致顶点与有序面。 |
| FNIT已有 subject | 6/10 | 4 | CON01/03/04/05/06/07 的全部138项存在、公开报告 complete、网格报告 passed；核心几何通过。 |
| FNIT其他4例 | 0/4 | 4 | 预检时 CON08/09 目录已存在但输出未齐；CON10/11 未出现目录。目录存在不能视为完成。 |

六例 FNIT 报告中的实际 `source_T1w` 与对应原始 BIDS T1 的体素和 affine 全部一致；原始 T1 的当前 SHA-256 与固定下载 manifest 一致。`numeric_validation` 仍为 `not_run`，本次检查不能证明官方数值等价。

官方10例的这次 CPU 格式/几何/内容检查为4.34–6.20秒/例；完整FNIT六例为1.31–4.51秒/例。这些时间包含本次读文件、解压及内容hash，与原始T1重建的时间口径不同。

## FNIT 原生程序来源

现场使用以前的 FNIT 独立 Conda 构建产物，而非系统 FreeSurfer 二进制：14个必要程序存在，当前文件 SHA-256 全部与该环境 `installed-bin.sha256` 一致。`build-provenance.txt` 绑定源提交 `d932c45b7941662ea380a05efef580568b98d41a` 与源码树 SHA-256 `313afb62ea5b5c7d5aa9d78659403b126c63e7cdd91465391ce6a2138c93693c`。可选 `mris_place_surface_white_fast` 不在该环境；成熟入口会保留完整原始放置程序。

统一 DWI 环境的默认 `bin/` 没有这些产物，因此新执行 FNIT recon 时要显式提供已核对的 `native_bin_dir`，或完成主页 FNIT recon-all安装。本报告保存manifest/patch/Conda记录与程序哈希，不复制二进制、第三方源码、许可证或影像。

## 专用测试

`focused_tests.txt`：34 passed，6.34秒，CPU运行、GPU不可见。覆盖BIDS处理、内容哈希、auto无官方回退、用户目录只读、无效体积/表面、FNIT完整输出/报告、失败不发布缓存、修改资源/输出后新尝试、运行中源改变拒绝完成、先重建后DWI CUDA以及官方PATH优先。

预检源码为 `7d141bcc` 的backend；随后的 `0de2ed3d` 只改变准备阶段顺序和官方子进程环境，几何预检函数不变。专用测试单独针对这次后续修改运行，两个来源不能改标为同一个新的 MRI 整例。

- [功能与完整参数](../../../docs/connectome/recon_backends.md)
- [逐例公开预检摘要](real_anatomy_preflight.public.json)
- [已有 recon-all 精度与速度](../../recon_all/optimizations/20261002_parallel/FINAL_RESULTS.md)
