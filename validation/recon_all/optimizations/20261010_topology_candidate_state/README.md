# DVS 首次真实完整恢复

本组是公开 ds000114-sub06 LH 历史 3a 的冻结阶段，非当前 CFFF79 连续候选或完整 GA。固定源独立参考在真实 record、首次候选评分及 RP 更新之后、真实 full restore 之后各捕一次。两次参考完全相同。

- [候选完整比较](reports/candidate_v1.json)：24 字段组及有效存储 0bitdiff；自身重复 0bitdiff。
- [参考重复收据](reports/reference_v2.json)：原输入/程序 SHA、两次退出和捕获 SHA。
- [状态相位](reports/phase_metadata.json)：N126778、F246985→247092→246985；167 记录、127 恢复、40 跳过。
- [失败 wrapper v1](reports/reference_failure_v1.json) 与 [失败 observer v1](observer_failure_v1.json) 保留，不改标为成功。
- [最新契约](contracts-v5.json)：23 本地契约；该记录写入时真实授权仍待确认，随后真实验证由新的独立收据描述。
- face-only 未在这次真实源调用中执行；无完整 GA、最终网格或整例等效结论。

复现先用 `tools/benchmark_recon_topology_candidate_state.py` 的 `load_state_capture()`/`dump_state_checkpoint()` 将自己的带哈希冻结捕获转换为无 pickle NPZ，再按功能页 CLI 调用。`driver.py` 保留本次私有 benchmark 控制逻辑；`reports/source_manifest_v1.json` 记录实际执行源码，私有 namespace 仅用于避免导入生产调度，不修改 DVS 算法。

计算按源选择性恢复字段，不读取 reference-after；参考 after 只在计算完成后用于比较。冷 CLI 81.03663 秒包含全表面逐行比较和读写，保存/恢复本身分别 0.00556/0.01125 秒。参考与候选主机和计时边界不同，无速度比。

这里的源观察器只发布 FNIT 自有记录函数和注入位置，原 C++ 主体、库、程序、原始 MRI、全网格字段与许可证不入 Git。公开指标按固定字段白名单导出。
