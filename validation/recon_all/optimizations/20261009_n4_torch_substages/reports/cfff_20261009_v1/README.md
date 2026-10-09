# 两例 A100 完整 N4 阶段收据

本目录是公开 ds000114 sub-02/sub-07 的冻结同输入 N4 验证，不是原始 T1 recon-all 整例。两例 native 各两次重复、只增加导出的诊断、完整 Torch CPU、GPU v1 和 device-divisor 变体均已执行。全部候选完成 200 次自产反馈；生产默认不变，严格完整 N4 复现未通过，整体指标等效及最终分割/网格/脑区影响未评价。

- `complete_pair/summary.json`：实际作业、线程、亲和性、程序哈希及两例参考重复结果。
- `complete_pair/<case>/`：CPU/GPU/变体的完整数值、同步计时、原生子段及固定 lattice/log/exp 诊断。
- `file_interfaces/summary.json`：已初始化 CUDA API 与冷 CLI 的 MGZ shape、uint8、affine、zooms、图像类型、量化值及文件哈希。4 项合同通过，NIfTI 尚未单独实测。
- `summary_metrics.csv`：从十份候选原始 JSON 提取的同节点指标；不包含整例提速估计。
- `receipt_cfff_20261009_v1.json`：CPU/GPU、独立 Conda GCC/ITK 构建、实际动态库、源码哈希和测试工具。一个预期头文件名不可用，明确保留；动态库检查不是干净环境整例隔离验证。
- `unit_v1.log` / `unit_v1.xml`：冻结 v1 的七项结构测试通过。独立测试层为 pytest 8.3.5、pluggy 1.6.0、packaging 26.3、iniconfig 2.3.1，没有修改运行环境的计算库。
- `memory_visibility_probe.json`：容器/驱动 PID 不同；进程树归属不确定时峰值为 null。整卡采样 27.47 GB 包含其他或未决进程，不能当作 N4 自身峰值。

冻结完整源码基线为 `a756fffb`，实际运行以每份候选 `source_sha256` 为准。v1 和变体各自独立冻结；不同收据中的 native 程序 SHA 由本节点的 RPATH 重定位产生，没有拿旧主机结果当作本次实测。所有强度、体素误差方向和失败门保留。

## 原始文件与公开副本

私有原始包为 1,907,898 字节，SHA-256：
`29947cd327eae9cdba5e7e8213b9738b4a79b95b19441a99a75312dbd8bba34d`。
其中 88 个导出文件在发布前逐项核验；原始包另行私有保留。

公开副本将实际存储目录前缀和容器主机名替换为 `FNIT_ROOT` 与 `BENCHMARK_HOST`；派生 CSV 使用 LF 换行，保证 Git clone 后的文件 SHA 仍能校验。所有算法、输入、程序和产物 SHA 保留。`public_export_manifest.json` 给出 91 个初始发布文件各自的原始/公开 SHA，包含原始导出清单及后续镜像/CSV收据；`public_export_validation.json` 核对数值和哈希未改变，JSON/XML 可解析，PNG 字节未改变。原始文件 SHA 不能直接校验脱敏后的 JSON、日志或换行转换后的 CSV。README 与最终验证收据属于额外的公开说明，不是假装服务器生成的原始产物。

公开内容不包含 MRI 数组、浮点场、许可证、凭据或服务器地址。复现需要单独获取声明的公开输入并按 manifest 核验；路径占位符不是可直接访问的数据下载地址。

完整调用、参数、坐标空间、精度与速度范围见 [N4 中文说明](../../../../../../docs/recon_all/N4_COMPLETE_TORCH_20261009.md)。
