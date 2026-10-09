# 最终white两例双侧同输入回归

## 范围与版本

本组从公开ds000114 sub06/sub07的FNIT自产七个前置文件开始，执行完整四轮
最终white、源规则相交清理及表面/MRI写出。它是冻结同输入阶段验证，未将
这个实验分支接入生产默认，也不是原始T1整例。接口说明见
[最终white中文页](../../../../docs/recon_all/PYTHON_FINAL_WHITE.md)。

源码基线为`3644f50c`，新增final分支的实际模块SHA保存在每次报告和
`source_binding.json`。正式交付的final接口、共享白质优化器、rip子函数和
限幅模块均与四次实际候选源码SHA相同；不把历史报告改标为最新main。

同机参考为独立Conda源码构建的标准`mris_place_surface`，源码提交
`d932c45b7941662ea380a05efef580568b98d41a`、程序SHA
`78b64b7395aa0db0592ab6912fc026128b221c56d9f802db225fa59c18ceda44`。
每侧新运行两次，解码坐标、有序面、MRI体素/几何/dtype均稳定。其描述性
文件头可不同，不能把字节不同当作坐标不稳定。

当前生产auto选择的white专用程序是`mris_place_surface_white_fast`，SHA
`810c8086eb1b3088a1952be6f205500d99de2f3feff8c29476c99a5e67b8b427`。
本组没有该程序的同输入性能运行；表中标准参考不能替代生产性能控制。
跨环境官方安装结果和当前生产整例由协调者另行评估。

## 完整阶段结果

同A100主机，每次四线程；GPU2和GPU4分别运行独立案例，固定亲和性保存在
私有原始报告。候选使用CPU梯度/边界、Numba有序接受、GPU保守候选和
source_torch清理，TF32开启，不使用FP16/BF16。

| 被试/侧 | 四轮总步数 | 候选API/s | 标准参考两次完整调用/s | 最终坐标/有序面/MRI |
|---|---:|---:|---|---|
| sub07 LH | 31 | 173.576610 | 158.639500 / 155.284668 | 0差异 |
| sub07 RH | 29 | 180.302315 | 199.459015 / 203.159993 | 0差异 |
| sub06 LH | 35 | 220.791466 | 271.848697 / 266.599552 | 0差异 |
| sub06 RH | 35 | 248.107786 | 296.885479 / 303.769064 | 0差异 |

四侧均1个连通分量、Euler=2，无边界边/非流形边/索引退化面/重复面，最终
source标记相交数为0。输入SHA在候选和两次参考运行前后相同。所有误差均
在证明有序面与顶点对应之后计算；mean/P99/max表面距离为0mm。

候选API包含验证、MRI加载、传输、当前进程JIT、四轮优化、清理、读写和
逐步报告写出；CUDA上下文1.80–3.69秒另计。外部monitor完整进程墙钟另列。
共享GPU负载和采样延迟变化，本组不是隔离ABBA速度门，也不据此宣布整例
提速。严格阶段输出复现通过；生产后端性能控制、原始T1整例和整体指标
等效未由本组判定。

## 具体误差定位

v17首次完整实现错误保留了中线冻结。标准命令中的`--rip-label`实际设置
`RipMidline=0`，并关闭轮间RipMidline；独立rip-surface只执行具名BG和247。
透明原生probe先与原生最终输出0差异，随后首步比较发现117个多余ripflags
和553个目标强度差异。v17最终最大偏移2.803446mm，4,205顶点超过0.1mm。
v19修正解析语义，复用完整优化器并保留原边界/接受/清理规则，四侧全部
匹配标准参考；没有放宽阈值或读取参考来修补候选。

`complete_report.json`单列失败v17和透明v18诊断。诊断包含额外逐步影像I/O，
不用于性能比较。修复前后图取同一参考顶点70241、同一MRI y=110平面，
图形元数据保留输入和脚本SHA；全部0误差直接标注为0mm。

## 显存与统计范围

候选PyTorch allocated峰值690,836,480–708,015,616字节，reserved峰值
1,277,165,568–3,732,930,560字节。外部采样请求0.25秒，实际最大间隔
2.17–2.97秒，容器/主机PID归属未解析，进程树峰值为null。同期整卡上界
13,404,995,584/27,653,046,272字节含其他项目，不能当作FNIT占用；本组没有
完成独立进程树20,000,000,000字节验收。原始完整采样留在私有运行目录，
公开报告保留完整采样文件SHA、数量、间隔和归属状态。

## 复现

完整候选和原生参考先按专页的具名命令生成独立目录。随后运行只读比较器：

```bash
python validation/recon_all/python_gpu_port/publish_final_white_receipts.py \
  --run-root /data/diagnostic/final-white \
  --output /data/reports/final-white-complete.json \
  --case sub07:lh:v19:v17 \
  --case sub07:rh:v19:v19 \
  --case sub06:lh:v19:v19 \
  --case sub06:rh:v19:v19 \
  --include-failed-case sub07:lh:v17:v17 \
  --diagnostic-directory /data/diagnostic/final-white-first-difference-v18 \
  --production-reference-binary /opt/fnit/native/bin/mris_place_surface_white_fast
```

`run-root`含`final_white_{subject}_{hemi}_candidate_{version}`和对应
`native_repeat_{version}`目录；`case`逐项指定真实case、侧、候选/参考冻结
版本。`include-failed-case`保留已完成但不匹配的失败，`diagnostic-directory`
记录透明probe，`production-reference-binary`只记录名称/SHA，不执行程序。
输出不得已存在；原始报告未完成、输入变化、输入SHA不同或没有双重复时
明确报错。比较器复用现有`compare_subject._topology`。

## 文件

- `complete_report.json`：四侧全部原始去敏报告、实际输入/源码/程序SHA、
  每轮/试步/清理、计时/显存、参考重复性、坐标/MRI和网格质量。
- `sub07_figure_comparison.json`：生成修复前后脑图时实际使用的比较收据。
- `final_white_v17_reference_plane.*`、`final_white_v19_reference_plane.*`：
  真实MRI叠加、局部异常区边界、全顶点误差图和机器元数据。
- `contract_receipt.json`：本地CPU环境21项接口/默认preaparc契约通过，
  不能代替本组真实T1 GPU benchmark。
- `source_binding.json`、`SHA256.json`：交付源码与实测代码、文件校验。

新接口复用已有nibabel、NumPy、Numba、SciPy和PyTorch，无新增依赖。
