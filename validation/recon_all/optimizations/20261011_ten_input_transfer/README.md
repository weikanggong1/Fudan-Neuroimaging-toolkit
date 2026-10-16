# 十例公开原始T1实体核查与accelerator-b传输（2026-10-11）

## 1. 功能与流程

十例已全部在accelerator-b可用，合计107,511,490字节；只传入原始公开影像与许可元数据，没有官方结果。此为输入转移验收，未启动新的整例，不批准完整EM、GA或pial算法/质量门。

```mermaid
flowchart LR
 A[原服务器固定索引及十例冻结清单] --> B[源端10/10大小/SHA及CC0元数据]
 B --> C[仅打包缺少8例；WSL复验包内逐文件SHA]
 C --> D[accelerator-b接收8例并复用已有2例]
 D --> E[接收端10/10重新哈希及影像头核对]
 E --> F[退出0、收据、私有绑定与INDEX登记]
 F --> G[等待算法质量及最终冻结集成版本]
```

## 2. 输入、输出和Python读取

原固定十例不更换：ds000030 1.0.0中的10159、10171、10189、10193、10206；ds000114 1.0.2中的04至08，均ses-test。两个CC0许可元数据固定SHA与源端、接收端匹配。源文件均为原始三维float32 T1，保持原始网格和头；不以参考分割或曲面替代输入。

```python
import json
from pathlib import Path

receiver_receipt_path = Path("receiver_receipt.public.json")  # 只有哈希、标量、公开数据身份与头
receiver_receipt = json.loads(receiver_receipt_path.read_text())
verified_case_count = len(receiver_receipt["files"])
verified_input_bytes = receiver_receipt["verified_bytes"]
assert receiver_receipt["receiver_exitcode"] == 0
assert verified_case_count == 10 and verified_input_bytes == 107_511_490
assert receiver_receipt["whole_started"] is False  # 输入可用不代表重建已开始
```

`transfer_manifest.public.json`：冻结清单身份、十例预期大小/SHA、许可元数据、8例传输与2例复用方式、包SHA及保留的失败。`receiver_receipt.public.json`：实际接收大小/SHA、shape/dtype/affine/qform/sform、退出0和完成时间。`readiness.public.json`：独立对冻结cohort逐项比较影像头和SHA，以及尚未通过的启动门。`registration.public.json`：现场核对INDEX计数、字节、manifest/收据SHA及旧缺失记录存在。

原始文件、包、日志及十例路径绑定留私有服务器目录。公开报告不含服务器地址、账号或私有绝对路径。source和receiver仅用通用Python/文件工具，不调用原版阶段程序或GPU。

## 3. 传输调用与参数边界

复用已有SSH ControlMaster和持久tmux，不请求或记录密码；本次没有重新从上游下载。传输包只允许八个冻结相对文件名、普通文件，拒绝额外成员、绝对路径、父目录跳转或链接。接收端先验证整包，再复用原两例，并重新校验十例；大小或SHA不符即保留失败收据、退出1，不登记为完成。

影像头全部和固定cohort相同；仅复验头和字节，不执行预处理或MRI算法。完成门为接收端十例逐文件通过、107,511,490字节、退出0、收据与INDEX一致。tmux、socket或发送端退出均不能单独证明完成。

## 4. 原软件调用

数据转移没有原软件对应命令，也未调用FreeSurfer/FSL等程序。未来参考软件只在独立基准中运行；生产不读取官方subject或阶段结果。

## 5. 精度、耗时与验证范围

本次验收为数据完整性：10/10大小/SHA一致，存储dtype、shape、affine及qform/sform代码与冻结清单一致。8例传入88,501,448原始字节，2例accelerator-b复用19,010,042字节；总计107,511,490字节。传输时间不是重建benchmark，没有脑图或科学精度/性能新结果。

数据准备已从2/10补齐10/10。完整EM/CA、GA/orig、最终pial、冻结集成commit/独立安装、资源/许可保留、硬件/空目录复查和可信GPU父子PID归属仍按前版协议等待；不得重标旧v2/v3的不完整EM全流程。

## 6. 更新与失败记录

前版首批准备报告只在accelerator-b已登记输入找到2例，另外8例缺失状态保留原样；本次通过原服务器只读固定索引找到八例，核查源端十例后转移补齐。没有覆盖旧准备报告或任何冻结环境/运行目录。

首次本地tmux只读查询遇沙箱socket权限限制，经授权只读权限重试后复用既有连接。首次本地包校验使用旧Python没有的 `hashlib.file_digest`，在逐文件比较前抛AttributeError；改用流式SHA256校验同一包，未重传或修改影像。全部记录在manifest失败历史中。源端、WSL和accelerator-b三段均有独立证据，收据仍区分数据完成与重建未开始。

## 7. 数据来源

固定OpenNeuro ds000030 1.0.0（提交4070b6eea231517bfeff42527a46d9d166ac4e13）与ds000114 1.0.2（提交6299834614e9ae7df1e2fc5922545331b5c7f022）。许可字段CC0及元数据SHA见transfer_manifest；本次复用已有冻结文件，没有新增下载或改变队列选择。
