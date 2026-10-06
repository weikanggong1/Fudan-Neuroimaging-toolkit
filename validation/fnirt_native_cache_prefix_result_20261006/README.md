# FNIRT 冷初始化前缀：运行库修正后仍存在操作数位差

## 1. 目的、结果和冻结版本

本报告记录固定真实输入的 `cf-prefix → grad-prefix` 操作数诊断。v2 原生子进程正常退出，但实际动态库身份门失败；v3 修正链接动态 tags 后，执行和身份校验通过。两版保存操作数 SHA 相同：与既有 Python state 比较，obj 有 2425/16128 个 FP32 元素不同位，mask 差异为 0。v3 将 obj 的最大绝对差量化为 `9.918212890625e-05`、相对 L2 为 `2.1212013838169472e-07`。本轮尚未正式采纳数值容差，也未验证完整注册链，不能据此宣称科学等价。

| 身份 | v2 | v3 |
| --- | --- | --- |
| 冻结源码 commit | `fde3f59c9d35af3b41d34c75ce3b2c7b0fd64e53` | `d38b327b7b1a934f2978f91a2fccbf3f1a94d3b4` |
| observer 程序字节数 | 2263672 | 2263672 |
| observer 程序 SHA-256 | `5125821a36012d96f9bf47a4a86bf5bba9f4b71f9626e151027ba6b6f8bcc74a` | `0cadc734182a1f0a47b4c7ca694c8efa6ab71204dbf85f967e2524de0257e193` |
| PLAN SHA-256 | `b8ddb683fe4b4fa41e80f4b600c59699892a0252bf222d2a12a9f4f35ab7d793` | `0a1a1f503ffe0c1d3b8595e159adc6ad2998b965de5b6f9d7f70d5792d85e87c` |

数学实现基线为 `7ff215ee86c49414b2fa6156fdf6769aa54d00f8`，capture 时正式仓库 HEAD 为 `b98db76fe60f565cf616df04638428de26ad33a0`。两版各 19 个 Git 文本 blob、17 项冻结 payload 和原 64 项资源绑定独立核对通过。observer 数学源体逐字节相同，SHA 为 `7cdebfe36c8bd7536418f143a35493b8280285e00a79769e3867467f1452785c`。本目录只发布四份结果文本；源码准备叶、SDK、二进制、影像、参数和数组不随报告发布。

## 2. 输入、既有 Python 状态和比较范围

原生初始化使用冻结的真实影像、仿射、配置和 FP64 参数文件。原数组、参数和几何保留在私密运行目录，公共记录仅含身份摘要和统计。

既有 Python checkpoint 来自一次 current `linearize` 及其内部 `evaluate(derivatives=True)`，绑定 solve3 第二个 accepted 参数点。旧 solve2 仅提供保存的预处理输入来源。该 checkpoint 保存了重建的 state 与 unpacked weights，未保存历史 Python cache 对象身份。

| 原生操作数 | Python 保存成员 | 比较合同 |
| --- | --- | --- |
| Deriv 后 `obj` | `state_warped` | FP32 同索引字节；显式映射原生 X 连续存储与保存的 C-order |
| Deriv 后 `mask` | `state_mask` | char 与 bool 编码均须为 0/1，逐索引比较 |
| `sref` | 无直接保存成员 | 只保留原生字节 SHA，当前无直接比较对象 |

源码出处为 `stage1_linearize.py:217,249–250` 和 `registration.py:897–916,994`。registration.py 字节与数学基线相同，SHA 为 `caab8ffccfd47225018d15803e0acee70c05b0212a22a4b720b034454c75e4da`；其 scaled fixed 已采用 `(scale * fixed.to(scale.dtype)).to(F32)`。本轮比较只恢复两个 NPZ 成员和三份 native 数组，不重新求 residual、scale、采样、gradient 或 H。

## 3. 原始执行和保存字节结果

每版各执行一次编译，并由根任务另行授权一次 binary-bound capture；无 native 重试。

| 执行门 | v2 首次 capture | v3 唯一 capture |
| --- | --- | --- |
| native child | RC 0，两个 typed exit 到达 | RC 0，两个 typed exit 到达 |
| controller | RC 1：`Actual loaded DSO outside frozen closure` | RC 0，源码、资源、binary、HEAD 和实际 DSO 门通过 |
| supervisor / outer | 两者 valid 均为 false；outer RC 1 | 两者 valid 均为 true；outer RC 0 |
| 冻结控制器中的保存比较 | 未执行，止于 DSO 门 | 已执行 |
| 首次 FAIL | 保留 | v3 独立 FAIL 不存在，v2 FAIL 保留 |

v2 随后单独授权一次保存字节后验比较，RC 0、额外 native 0，source 与保存输入身份不变。该比较得到 obj 位差 2425、mask 差异 0，未覆盖原失败状态。v2 helper 未记录幅度指标。

v3 在原冻结比较 helper 中对保存的 FP32 值以 Python Double 统计，relative L2 的分母为对应既有 Python 操作数的 L2。仅纳入有限配对：

| obj 统计域 | 元素数／有限配对 | 最大绝对差 | 相对 L2 | 不同位的 signed-zero 配对 |
| --- | ---: | ---: | ---: | ---: |
| 全网格 | 16128／16128 | `9.918212890625e-05` | `2.1212013838169472e-07` | 0 |
| 两侧相同 valid mask | 14341／14341 | `9.918212890625e-05` | `2.1212013838169472e-07` | 0 |

两域差值 L2 均为 `0.0008373486472112933`，Python 对照 L2 均为 `3947.5207474386307`。所有纳入值有限；obj 位差不是 signed-zero 配对造成。mask 差异仍为 0。指标没有转化为正式精度门，RC 0 也没有判定位差为零。

v3 controller、supervisor、outer 到回执分别为 40.944997、41.069332、41.163640 秒；v2 outer 为 34.656333 秒，v3 编译 outer 为 20.640585 秒。这些时间包含校验、初始化和收尾，仅描述诊断运行，不能用于注册或内部算子的速度比较。

## 4. 观察器边界与自然调用

固定原 `cf` 在 `ScaledRef` 返回后、SSD 前 typed exit；同一个新建 cost 对象随后以相同参数进入 `grad`，在 `Deriv` 返回后、Jte 前 typed exit。grad 会复用 cf 已建立的 cache。

| 自然调用 | cf-prefix | grad-prefix |
| --- | ---: | ---: |
| SetDefFieldParams / SetScaleParams / IMap | 各 1 | 各 1 |
| Ref | 8 | 8 |
| Robj / Mask / ScaledRef | 各 1 | 各 1 |
| Deriv / RobjDeriv | 0 | 各 1 |
| 新增 cache getter | 0 | 0 |

两版调用计数以及 pre/post obj、mask、sref、borrowed ref SHA 完全相同。Deriv 前后 obj 字节变化、mask 字节相同。obj 变化说明此次 Deriv 路径改写了它的值；尚不能定位相对 Python 差异的原因。mask 相同排除了此次调用改变 mask 值的解释。

完整 cf、完整 grad、objective SSD、新 scale reduction、Jte、bending、intensity Gradient、H、diag、PCG 和 SCG 均为 0；latest_ssd 未写入。原 grad 自然的 mask sum 在 Deriv 前保留。本轮没有保存 derivative 数组、pre obj/mask 数组或完整 Ref 数组。

## 5. ELF 链接修正与实际运行库

v2 实际加载 22 个 DSO：19 个符合冻结 native 身份，2 个仅符合原 64 绑定中的 compiler-only 身份，1 个未绑定。三个差异为 Conda libgcc_s、libstdc++ 和 libz。Conda libz 与冻结 FSL libz 均为 112800 字节，但 SHA 分别为 `9a1046cd4a615a1295b4b7612eefc201c31d7d13669a1859c688b5ae642693ef` 和 `c25b897365422036ada99c415a1df32c88c0b9c24a43bc76d860a72ec71cb42e`。

根只读 binary 的确切 tags：v2 DT_RPATH 为 `Conda/lib:FSL/lib`，DT_RUNPATH 为空。这解释了实际 loader 与旧 explicit-first 静态模型的分叉。v2 `metadata_elf.py:41–42,75–80` 合并 RPATH/RUNPATH 并优先显式目录；该模型并未实现完整 loader 语义。

v3 唯一编译 argv 变化是在尾部加入 `-Wl,--enable-new-dtags`；原 native allowlist 不变。实际编译 tags 的 DT_RPATH 为空，DT_RUNPATH 为 `Conda/lib:FSL/lib` 并含固定 FSL 完整目录分量；运行环境的固定 FSL 搜索路径可优先。静态闭合为新 binary 加 22 个原允许 DSO，共 23 项，无 missing。v3 capture 的 22 个实际 DSO 按 resolved path、大小和 SHA 全部符合原 native 允许集合。

根读取的 v2/v3 `.text` 均为 806033 字节、SHA `c14c1803d41ea6defb3967973046481000592b2c0c19ca5dc3dcf34805f4e7ee`；本独立审查仅读其原始元数据。链接修正消除了此次实际库身份问题，却没有改变这些已观察操作数。ABI1 旧对象与新编译器的链接成功及有限前缀运行，仍不能建立完整 solver ABI 或历史运行等价。

## 6. 原始收据、控制树和离线核验

独立审核绑定的 v2 原始文本 18 份、v3 编译文本 9 份、v3 capture 原始文本 15 份。原始 SHA 全部吻合；v2 后验比较 JSON 与根嵌入结果一致，v3 幅度 JSON 与根统计记录一致。各版 987 条实际编译依赖路径与 `.d` 集合一致；capture 前逐项校验，本报告不宣称这 987 项在 capture 前后均逐项复核。

v2 capture、v3 编译、v3 capture 分别记录 5、9、5 个唯一 owner PID/start ticks 已退出。supervisor 观察和收养的子树已退出，`resource_release_safe=true`，关闭后实际 LOCK_NB 探测成功、`/proc/locks` 为空。v3 capture 的 supervisor 未发信号。上述是原始收据时点的观察，不推定未观察历史进程的完整性。

公共清单只保存文本收据身份、库 basename 与 SHA、源 blob 身份和统计，不含私密绝对路径、PID、影像几何或原数组。`SOURCE_BINDINGS.public.json` 自包含两版冻结 metadata 原文和 38 个 Git 文本 blob 身份；不需要 main 持有准备源码即可执行默认离线闭合校验：

```bash
python verify_report.py
```

默认校验四个公共文件、冻结 metadata 字节和身份记录闭合；它不重算未附带的源体或私密原始收据。有冻结 Git 历史时，可通过 `--repo /path/to/source_repository` 重算 38 个 blob；有本地原始文本时，可同时提供 `--raw-v2`、`--bindings-v2`、`--raw-v3-compile`、`--raw-v3-capture`、`--bindings-v3` 做完整文本审核。核验脚本只读 Git/text metadata，不调用观察器、不读取数组。本次独立审核新增 compile/native、MRI/checkpoint 数组读取和上传均为 0。

## 7. 结论、后续最小诊断与源出处

v3 执行和身份门已通过；与既有 state 的 obj 位差及幅度已量化，mask 字节一致。现有两个保存成员适合作为 Deriv 后操作数对照，不能代表 sref、完整 SSD、gradient、H 或 solver 一致。正式精度与性能 gates 保持原状态。

相同参数不能恢复旧 solve3 的历史调用顺序、trial/accepted 状态、derivative cache 或 latest_ssd。本轮没有定位优化器或历史 cache 原因，也未正式接受 `2.1212013838169472e-07` 为数值容差。

最小后续源码提议是保留 Deriv 前已生成的 obj/mask 字节向量，分别比较 pre、post 与既有 Python 成员，区分差异是否已在 Deriv 前出现。此提议仍需新冻结、独立审核和单独授权，本报告不执行它。

固定 FSL 6.0.7.4 `fnirt_costfunctions.cpp` SHA 为 `300bcf80839cb1591870845534f87778aa06d04785c22d76bf5100cd2474006a`，cf/grad 边界行分别为 883/971，Jte 自 973 起；自有 observer 入口见 `native_cache_observer.cpp:129–168`、保存入口 172–174。准备源码 commit 和逐 blob SHA 自包含于本目录清单。本次发布限结果文本，不复制 SDK、原软件二进制、影像、权重或冻结准备源码叶。
