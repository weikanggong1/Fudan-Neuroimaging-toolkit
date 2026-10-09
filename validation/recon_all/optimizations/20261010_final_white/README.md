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
v19的标准参考不能替代生产性能控制；后续v20补测当前生产程序，单独记录。
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
提速。严格阶段输出复现通过；原始T1整例和整体指标等效未由本组判定。

## 当前生产white_fast补测（v20）

四次fresh native-only运行均使用上述实际生产程序SHA，输入与v19逐项相同，
完整命令的线程为4、CPU亲和性与各侧候选相同；生产程序是CPU计算。
候选及标准双参考不重跑，原有v19报告不改标为v20。

| 被试/侧 | 生产fast完整调用/s | 候选v19 API/s | fast对标准参考 / 候选对fast |
|---|---:|---:|---|
| sub07 LH | 200.891559 | 173.576610 | 坐标、有序面、MRI均0差异 |
| sub07 RH | 183.399960 | 180.302315 | 坐标、有序面、MRI均0差异 |
| sub06 LH | 235.018735 | 220.791466 | 坐标、有序面、MRI均0差异 |
| sub06 RH | 225.571066 | 248.107786 | 坐标、有序面、MRI均0差异 |

候选实测模块SHA与运行后SHA相同，七个输入SHA在所有运行前后相同。
每侧fast只运行一次，fast独立重复性未测；标准参考已有双重复稳定证据。
共享主机跨时段负载、候选API与原生子程序计时边界不同，故本表不构成
ABBA速度门，不用一次比值决定生产替换，也不宣称整例提速。原生默认保持。

`production_fast_report_v20.json`保存完整去敏原始fast报告、每次输入/程序/
候选源码SHA、四组表面与MRI比较和外部monitor收据SHA。采样包含其他
项目的整卡占用，不能把CPU参考monitor的整卡值当作本接口显存。

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

生产程序按同一benchmark加`--native-only --reference-repeat 1`写入独立
`production_fast_v20`目录后，只读汇总如下：

```bash
python validation/recon_all/python_gpu_port/compare_final_white_production.py \
  --run-root /data/diagnostic/final-white \
  --output /data/reports/final-white-production-v20.json \
  --expected-fast-sha256 810c8086eb1b3088a1952be6f205500d99de2f3feff8c29476c99a5e67b8b427 \
  --case sub07:lh:v19:v17:v20 \
  --case sub07:rh:v19:v19:v20 \
  --case sub06:lh:v19:v19:v20 \
  --case sub06:rh:v19:v19:v20
```

`run-root`为已有三组冻结产物的共同目录；`output`为新JSON路径，不覆盖；
`expected-fast-sha256`必须是实际生产选择的程序哈希；每个`case`依次指定
被试、半球、候选版本、标准参考版本、fast版本。工具不执行放置算法，
输入未完成/改变/不匹配、fast程序SHA错误或子程序失败会报错；面顺序不
匹配时不报告同索引距离。输出逐项比较坐标、面、MRI体素/affine/dtype，
保留源码和运行哈希、计时边界及原始报告SHA；一次运行不会转写为双重复。

## 文件

- `complete_report.json`：四侧全部原始去敏报告、实际输入/源码/程序SHA、
  每轮/试步/清理、计时/显存、参考重复性、坐标/MRI和网格质量。
- `production_fast_report_v20.json`：当前生产程序的四侧完整同输入补测；
  v19收据与严格契约保持独立，不能由本表外推原始T1整例。
- `sub07_figure_comparison.json`：生成修复前后脑图时实际使用的比较收据。
- `final_white_v17_reference_plane.*`、`final_white_v19_reference_plane.*`：
  真实MRI叠加、局部异常区边界、全顶点误差图和机器元数据。
- `contract_receipt.json`：本地CPU环境21项接口/默认preaparc契约通过，
  不能代替本组真实T1 GPU benchmark。
- `source_binding.json`、`SHA256.json`：交付源码与实测代码、文件校验。

新接口复用已有nibabel、NumPy、Numba、SciPy和PyTorch，无新增依赖。
