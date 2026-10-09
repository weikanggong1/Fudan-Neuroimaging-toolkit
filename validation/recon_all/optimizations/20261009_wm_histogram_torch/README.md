# WM histogram 冻结真实输入验证，2026-10-09

## 范围和版本

本组使用两例公开 ds000114 T1 的既有 FNIT `antsdn.brain.mgz` 冻结输入，
验证 `MRIhistoSegment` 两遍完整候选；不是原始 T1 整例。原生参考通过
固定源码 Conda 构建的 `mri_segment` 生成。候选算法只读取本阶段强度
与三值输入，不访问参考输出、不执行原生程序、不按被试添加规则。

基线 `57cc115f68133069930565d885195ab6f2b18f2b` 加执行文件 SHA-256
覆盖层；JSON 记录实际旧 CPU 模块、新 Torch 模块、脚本、辅助采样器、
三份上游源码、输入、标签与参考程序哈希。之后仅在现有完整 WM API
增加显式后端，默认 CPU 不变；子阶段原始报告不回写源码版本。

## 实测结果

| 数据 / 遍次 | 旧 Python 秒 | Torch CUDA 秒 | 提速倍数 | 两者对原生差异体素 |
| --- | ---: | ---: | ---: | ---: |
| sub-07 / 第一遍 | 37.3980 | 0.36340 | 102.91 | 0 |
| sub-07 / 第二遍 | 16.1604 | 0.13218 | 122.26 | 0 |
| sub-06 / 第一遍 | 44.3566 | 0.38950 | 113.88 | 0 |
| sub-06 / 第二遍 | 25.0434 | 0.20080 | 124.72 | 0 |

旧 CPU 和新 GPU 各运行两次，ABBA，每次全部候选的 uint8 输出均相同，
标签 1/128/255 Dice=1，最大/P99误差为0；两次强度分类输入亦与原生
相同。计时显式同步目标 CUDA，包含张量克隆与临时数据，只测预加载
数组子阶段，排除文件读取、传输与写出，不用本表宣称整例提速。

GPU0 H100、Xeon Gold 6430，线程4。节点另有 connectome 工作；本组
不代表独占稳定吞吐。两例进程采样峰值约0.801/0.797 GB（十进制），
PyTorch allocated约0.218 GB、reserved约0.250/0.245 GB。请求间隔
0.25秒，最大实际间隔0.728/0.411秒，不保证捕获连续峰值。

原生 `-diag-write -diag-verbose` 完整命令时间只作参考生成，其大量
写出与同期负载不能作为 histogram 或生产速度比较。完整 WM 三方
文件 API 与原始 T1 整例单列，整体指标等效未判定。无干净环境执行、
动态库和文件访问证据，隔离部署仍未验证。

## 文件

- `sub-07/paired_histogram_v1/report.json`、`sub-06/paired_histogram_v1/report.json`：完整指标和版本。
- 对应 `process_memory.json`：全部原始采样，没有裁掉未取到峰值的间隔。
- `cpu_tests_v1.log`：标准库 unittest，6 CPU通过、GPU项未请求。
- `cpu_tests_v2.log`：增加完整接口无效选项早期拒绝测试；7 CPU通过、GPU项未请求。
- `cpu_tests_v2_receipt.json`：实际代码/测试SHA与测试范围。
- `summary.csv`：上述子阶段结果，便于机器汇总。
- `reproduce.sh`：变量使用者显式填入本地资产和参考路径，没有凭据或许可证内容。

真实影像与输出 MGZ 保留在服务器本任务目录，不入 Git。没有新增包，
使用主页环境已有 Torch、NumPy、SciPy、Numba、nibabel。

## 完整接口后续验证

已有 `segment_white_matter` / `segment_white_matter_mgz` 增加显式
`histogram_backend="torch"` 和 `histogram_batch_size=2048`。默认 CPU
保留。`benchmark/recon_wm_segment_histogram_backend.py` 已准备完整
文件 API 三方配对、冷缓存单列、全部阶段剖析、原生无诊断写出及重复
运行检查。节点新 SSH 当时拒绝连接，因此完整 API 计时尚未执行；
不能用上述子阶段倍数宣称完整 WM 或整例加速。仍等待已授权 GPU 窗口。
