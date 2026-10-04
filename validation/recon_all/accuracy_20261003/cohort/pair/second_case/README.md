# 第二例统一目录单对评估

复用第一例的`evaluate_pair.py`，SHA256为`68f33d7fad3cc702b5bc3b8628bdae6f174ed497248235fbcb9bf7a5a426ebed`；没有改生产算法或旧冻结源码。全部函数、输入输出、参数、对应原软件与文献说明沿用上层pair/README.md。

本例`ds000030_sub-10171`原始输入SHA256为`6f7587d216cc979940f9ecde0dbe9dd4546cb2f2ca8158cf7b80946c3fc67af0`。baseline为816e5610417a4c587caf321049438a9554139016，完整输出138/138，入口2593.3357972232625秒；official为FS8.2.0 d932c45，入口5507.655824495945秒，done SHA为`1a450cd8ad400d06b4899f4cc666188dca96d7bbeafb2ec54a44a4d08be57b6f`，log SHA为`3a04b021b76859a906e96b59d8a8ee68a19cf6e9d0e65c86108112412cd2e480`。比较仅使用这些真实已完成输出，candidate未冻结。第三例失败记录保持原状。

统一目录F=`/cwStorage/home/gongwk/Notebook_code/FNIT`：

- 源码及配置：`F/workspaces/recon_accuracy_20261003/pair_tools_v1`。
- 产物及检查点：`F/runs/recon_accuracy_20261003/evaluation/ds000030_sub-10171/checkpoint.json`。
- 日志及PID：`F/logs/recon_accuracy_20261003/evaluation_ds000030_sub-10171.{log,pid}`。
- 本次PID100248已启动，初始检查点为verify_binding等待共享锁；该状态不等于哈希复验或数值评估完成。后续每段计算时间和等待时间分别记录。

启动前已核对FNIT正式repo为干净main7af34e6d，但评估仍显式导入冻结baseline816来源，实际fnit.__file__为`F/legacy/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/baseline_runtime_816e5610/src/fnit/__init__.py`。原Conda prefix保持不变。统一INDEX.json/INDEX.md新增本例定位，保留其他任务字段和旧入口。

```bash
# 仅作复现说明；当前PID运行时不要重复启动。
FNIT_SERVER_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_RECON_BASELINE="$FNIT_SERVER_ROOT/legacy/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/baseline_runtime_816e5610"
FNIT_RECON_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
cd "$FNIT_SERVER_ROOT/workspaces/recon_accuracy_20261003/pair_tools_v1"
PYTHONPATH="$FNIT_RECON_BASELINE/src" "$FNIT_RECON_PYTHON" evaluate_pair.py --config ds000030_sub-10171.json
```

本轮复验第一例检查点已经complete、138严格诊断通过2项；这里仅登记既有首例状态，没有重启其进程。第二例数值结果尚未产出，整体等效not_assessed。
