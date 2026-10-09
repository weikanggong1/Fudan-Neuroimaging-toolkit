# 两例原始 T1→nu 与完整 N4 缓存隔离收据

## 范围

公开ds000114 sub-06/sub-07，A100-SXM4-80GB、Xeon Platinum 8369B、CPU0–3、4线程预算、TF32默认。native N4按既有接口拟合/重建各1线程；SynthStrip/Talairach已有FP32例外保留。原始链来自`803aec50`加本模块冻结v1；worker冻结v2只校正本模块docstring，执行AST哈希相同。核心完整N4与B-spline源文件SHA在原始报告/worker报告中分别绑定。

本目录不含MRI、权重或许可证。26份白名单原件通过目录/主机文本替换后发布；3800个数值、布尔和null叶节点未改，PNG逐字节保留。私有原始包SHA：`dba43f23da59bb7df97377a3dc72ea4df7085a70e51873ccb16963dc0bc0f28e`，原件仍私有保留。`FNIT_ROOT`是多个私有目录前缀的统一占位，`BENCHMARK_HOST`是主机占位；不能从占位推断实际挂载结构。

## 收据

| 文件 | 覆盖范围 |
|---|---|
| [raw_pair/summary.json](raw_pair/summary.json) | 四个新空目录从原始T1连续到nu；输入/资源/源码SHA、完整墙钟、强度/几何误差与采样历史 |
| [cached_worker/summary.json](cached_worker/summary.json) | 两例自产orig冻结输入；冷CLI与已初始化父CUDA API四组exec回归、LTA矩阵补充、doc-only AST桥接 |
| `cached_worker/<case>/<mode>/child.json` | 完整200轮输出/实际精度、allocator、线程、源码、allocated/reserved |
| [source/raw_v1_patch_manifest.json](source/raw_v1_patch_manifest.json) | 原始连续链实际运行v1源码SHA |
| [source/worker_v2_patch_manifest.json](source/worker_v2_patch_manifest.json) | 隔离worker实际运行v2源码SHA |
| [logs/unit_v2.xml](logs/unit_v2.xml) | 十项契约2.91s；只验证路由/隔离等软件契约，不替代真实benchmark |
| [figures/raw_chain_nu_error.png](figures/raw_chain_nu_error.png) / [figure_receipt.json](figures/figure_receipt.json) | 实际原始链RAS轴向误差图、选片规则、输入/绘图源码与图片SHA |
| [runtime_receipt.json](runtime_receipt.json) | 完成后同runtime核验CPU/GPU、native程序SHA及14个动态库SHA；未验证物理隔离 |
| [public_export_manifest.json](public_export_manifest.json) / [publication_receipt.json](publication_receipt.json) | 原始/公开文件SHA、数值保留核验和原包SHA |

## 结论

原始链到nu合计墙钟461.280→302.519s，观察缩短34.4%，不是完整recon-all。两例orig、SynthStrip和Talairach数据/矩阵相同；N4 nu0首次出现4016/3259个体素全部+1，后处理nu最大差3，不能称为随机误差或已无实质影响。严格复现未过，整体脑区/表面指标等效未判定。

隔离缓存四组完整N4含exec/退出约10–11s，对缓存关闭完整算法新增体素差0；既有系统偏移仍保留。父allocator/精度不变，无半精度、轮次减少或原生参考复制。各组仅一次；未重新运行cached-worker原始链，也未测其完整recon-all提速。父子PID归属未解决，tree峰值null，实际采样间隔可超过0.5s；报告卡/全部进程观测上界，未宣称精确连续峰值。

## 复现和接口

[原始输入链中文说明](../../../../../../docs/recon_all/INPUT_N4_CHAIN.md)、[缓存隔离中文说明](../../../../../../docs/recon_all/N4_CACHED_WORKER.md)。复现脚本位于本优化目录上两级：`benchmark_input_chain.py`、`benchmark_input_n4_worker.py`、`plot_input_chain.py`。报告收集器 `collect_input_chain_reports.py --fnit-root <private-FNIT-root> --output <new-report-directory>` 只复制固定白名单；公开导出器 `publish_reports.py`另行替换目录/主机文本，原始数值与哈希不改。
