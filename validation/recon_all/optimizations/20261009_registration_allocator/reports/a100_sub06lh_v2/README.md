# 球面配准子进程缓存配对（A100，2026-10-09）

同一公开 ds000114 sub-06 左半球、同一 FNIT 自产 sphere/smoothwm/sulc 与图谱，完整 fresh exec ABBA：disabled / enabled / enabled / disabled。每次独立冷 JIT，CPU 4 线程；节点动态共享 GPU 负载。

| 项目 | 实测结果 |
|---|---|
| 四次完整 exec 墙钟 | 466.583 / 417.828 / 471.134 / 413.757 s |
| 关闭 / 开启缓存中位数 | 440.170 / 444.481 s；开启慢 0.979%，无整步收益 |
| 严格同输入复现 | 四次文件 SHA、有序面、坐标及逐步轨迹相同；与既有自产结果也相同 |
| 网格 | 130,679 顶点 / 261,354 面；最终径向负面及零面均 0 |
| 优化引入退化 | 本次未观察到；整体指标等效与整例提速未评价 |
| CPU 生命周期合同 | 32/32，63.12 s，CUDA 隐藏 |
| 已初始化父 CUDA 合同 | 7/7，12.839 s，活张量和父禁用缓存环境保持；两侧子 worker 实际启用缓存 |

生产注册缓存保持默认 inherit。通用子进程选择只改变 fresh exec 环境，不修改父进程实际缓存、TF32、CPU 总预算或活张量。

## 报告与身份

- [原始完整执行报告](registration/summary_v1_original.json)：四次算法 worker 全完成，但末尾 metadata 中 allocator 同名覆盖导致中位数聚合失败；保留原失败状态。
- [只读派生统计](registration/derived_metrics_v2.json)：从上述已完成 worker 推导中位数和完整比较，不重跑算法、不改原结果。当前脚本修复字段并绑定分析 SHA。
- [源码补丁身份](source/source_patch_manifest.json)、[冻结源码](source/frozen_source.json)、[硬件及源码收据](hardware_source_receipt.json)。注册本体使用冻结 803aec50；调度合同使用 39cbf95d 加补丁。
- [CPU 合同](contracts/pytest.xml)、[真实 CUDA 生命周期合同](contracts/live_cuda.json)。它们不读取影像，不替代真实 benchmark。
- [导出去敏感清单](public_export_manifest.json)、[公开收据复核](publication_receipt.json)：只替换私有路径和主机；数字、布尔和 null 保持。

开启缓存的 allocator allocated/reserved 为 14,113,792 / 23,068,672 字节；关闭缓存不可用，记录 null。目标卡采样峰值 43,429,920,768 字节与全部计算进程合计上界 43,413,143,552 字节含其他任务，不能当作 FNIT 峰值。进程树峰值未确认；名义采样 0.5s、最大实际间隔 8.295s、5 次查询失败。本报告不宣称整步 20GB 资源验收通过。

完整输入、接口、命令、计时范围和参考见[中文说明](../../../../../docs/recon_all/SPHERE_REGISTRATION_ALLOCATOR.md)。报告不包含影像、网格、权重、许可证或认证内容。
