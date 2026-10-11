# 完整候选几何剖析及撤回状态打包

- `profile_v1.json`：ce96基线完整两序列，每次geometry/state分别cProfile，分离首次JIT；完整参考门和重复门通过。
- `readonly_geometry_profile_v1.py`：实际只读透明driver；不修改数值返回值。
- `abba_v1.json` / `summary_v1.json`：四个非剖析fresh child全部162字段/控制字节相同；完整CLI中位115.987567→123.799664秒，慢6.7353%。状态打包已撤回。
- `source_freeze_v1.json`：失败性能版本源码/driver/打包哈希，不回标后续版本。
- `contracts.json` / `contracts.xml`：被撤回版本本地合同18+14，保留真实结果；不是MRI benchmark。

公开仅FNIT代码/标量/逐字段gate/哈希，无参考数组、原版源码程序、许可、账户或服务器路径。参考after仅在候选保存后比较。全过程、最终网格质量及整体等效未在此阶段判定。

复现入口见 `tools/benchmark_recon_topology_candidate_geometry_hotspots.py` 与[七节说明](../../../../../docs/recon_all/TOPOLOGY_CANDIDATE_GEOMETRY_HOTSPOTS.md)。失败候选源码仍在私有冻结包，发布当前代码不含该state修改。
