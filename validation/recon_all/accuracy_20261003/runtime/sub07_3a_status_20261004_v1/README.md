# sub-07候选3a实际状态

实际CLI整例完成 2026-10-04T14:52:32.654906+00:00，138/138齐全、双侧meshpassed，8个startup均首次complete、未见CUDA OOM。CLI入口 2656.9304494736716s；内部API总墙钟 2653.813531429041s；pipeline 2647.7085681217723s。

自身父子同期采样峰 12027166720B，1264次；请求2s，最大间隔4.73319700313732s，查询失败0；不是连续峰。

guard完成并queue_drivers_resumed于2026-10-04T14:56:29.187476+00:00，父114563已非T。旧磁盘queue可能显示admission_or_execution false，实际source在post-evaluation完成后的finally才save，不能据此判算法失败。评估独立观察状态见summary；整体等效仍未评估。

原始影像不读取/不归档；输入SHA核对config/actual launch/原准入资源清单一致。只读3个实际核心生产脚本和monitor/driver/queue脚本核对库存SHA。
