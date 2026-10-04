# 十例精度候选串行计划的 CPU 准备

本脚本只准备验证配置，不启动 queue、GPU 或算法。使用既有九例计划 v2 与已完整运行的 sub-06 评估模板，九例保留原始 T1、5 次 initialized_cuda_api / 4 次 CLI；加初始 sub-06 CLI 后十例为 5 API / 5 CLI、GPU UUID、精度与总线程 4。初始 sub-06 先做独立官方评估，随后按 v2 原优先顺序运行九例并逐例评估。

## 输入、输出与参数

`build_precision_ten_case_queue_v2.py --output NEW_ROOT` 的唯一参数 output 是独占的新队列根目录，默认 `FNIT/runs/recon_accuracy_20261003/precision_candidate_3a_ten_case_queue_v1`。脚本内固定服务器 canonical root、3a 完整 commit、03cc 源码归档 SHA、原 v2 plan SHA、sub-06 模板 SHA、queue/guard/admission/driver/evaluator/monitor SHA；更换版本须另审查和新 builder，不得覆盖冻结运行源码。

先检查初始实际 completion 为 complete、整数退出码 0、138/138、mesh passed 和实际输入/输出路径，核对原 v2 的 2089 文件绑定；原始影像、权重和许可只在服务器检查，内容不复制进 metadata。九例已有 nominal/subject/attempt/admission 任何目的存在即拒绝，不覆盖历史执行。

输出：九份 `evaluation_configs/<真实case>.json`、可执行 `plan.json`、`preparation_manifest.json`。评估配置仅在初始模板上修改 case、official_config、evaluated_config（未来实际 retry_config）、evaluated_resources（未来实际 admission）、output；源码归档与 evaluator 来源不改。队列配置引用原 v2 whole config、retry root 和 admission 路径；原 v2 的已有 plan/config 文件只读，按它们指定的未存在 nominal diagnostics 新建准备回执。

冻结 `execute_whole_case.py` 没有 `--prepare-only`，此命令**未执行**。builder 只导入冻结 `after_startup_stage_whole.py` 并调用成熟 `prepare_launch(config_path, guard)`；不调用任何模块 main，不重跑 GPU stage。该函数核对输入、源码归档的全文件及 Python 集合、关键 scheduler/native_free SHA、资源 inventory，独占写 nominal launch。launch 明确 `status=prepared_not_executed`、`algorithm_entered=False`，不能当作历史执行。manifest 记录每次原函数/模块 SHA、config/guard SHA、exit、墙钟、launch SHA，以及总准备墙钟；异常保存失败 manifest，不发布完整 plan。

## 可复现 CPU 命令

```bash
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$FNIT_PYTHON" "$FNIT_ROOT/workspaces/recon_accuracy_20261003/precision_candidate_3a_ten_case_queue_v1/build_precision_ten_case_queue_v2.py" \
  --output "$FNIT_ROOT/runs/recon_accuracy_20261003/precision_candidate_3a_ten_case_queue_v1"
```

成功后的 manifest 中 `queue_command` 是供主线程审阅后启动的准确命令，本准备任务不执行。queue 的全部输入、初始失败继续、比较/算法分开计数、取消语义见 `PRECISION_CANDIDATE_QUEUE.md`。候选原始 T1 整例与官方数值比较各保留实际回执；准备通过不等于剩余九例完成、指标等效或 main 推送。

此脚本没有原软件独立命令或新增数值算法；完整官方参照仍为 [FreeSurfer recon-all](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)。CPU 准备墙钟单独记录，不计入算法/官方比较 benchmark；本次没有新增依赖。

## 本次准备恢复记录

九例 prepare_launch 只实际执行一次。原 v1 builder 末尾把剩余 CLI 数量误写为 3，实际应为 4；此错误发生在九例 CPU 准备完成后，queue/算法未启动，失败 builder 和 manifest 字节保留。修正 v2 的 `--resume-prepared` 只重新核对已完成 config/guard/launch/模块 SHA，再复用这些回执生成 metadata，绝不重新调用 prepare_launch。独立恢复 audit 还重核各 launch 的完整 resource inventory、prepared 标记与实际 output/attempt/admission 仍不存在。恢复墙钟与原 prepare 墙钟分列，不能伪称原始整例重试。
