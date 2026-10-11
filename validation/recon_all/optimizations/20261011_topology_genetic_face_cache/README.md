# 完整默认首缺陷GA：显式保留面索引缓存

本目录是已实际退出0的首缺陷完整阶段白名单指标和真实MRI派生图。新结果完整写出后才打开旧赢家；参数10/10/−1、seed1234、CPU8–11共4线程、CUDA隐藏，未混入状态缓存。

495搜索+1最终评分、29代记录（索引0–28）、1,362,381随机调用、完整80字段/3NPZ和控制均与旧自有完整GA一致。CLI2263.422952→1329.628818秒，搜索/最终写回API2182.039244→1252.548780秒；时段共享负载不同，只是墙钟观察，不称受控吞吐、原版最终赢家逐位复现或整例提速。

- [七节功能及完整参数](../../../../docs/recon_all/TOPOLOGY_GENETIC_FACE_CACHE.md)
- [完整指标](metrics.json)、[显式缓存调用](face_cache.json)、[496个评分](candidate_timings.json)
- [实际共享负载](shared_load.json)、[完成和产物SHA](METRICS_COMPLETE.json)
- [完整墙钟图](full_wall_and_load.png)、[自产最终缺陷真实MRI图](own_defect_on_MRI.png)
- [两项身份门合同](focused_test_receipt.json)、[公开逐文件SHA](PUBLIC_MANIFEST.json)

图中位移为自产首候选到自身最终状态的运动；此处未执行全9缺陷/全半球SOAP压缩/标准remesh/最终orig或原始T1整例。CPU峰值RSS1,648,036KiB，GPU/allocated/reserved为null。原完整记录服务器保留，MRI/网格NPZ/私有日志和路径不发布，数值与类型保真证据由两份EXPORT_RECEIPT记录。
