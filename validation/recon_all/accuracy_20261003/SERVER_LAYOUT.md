# FNIT服务器位置规则（2026-10-03）

用户最新目录要求优先；继续本轮十例精度任务时，先读取服务器`/cwStorage/home/gongwk/Notebook_code/FNIT/README.md`及`INDEX.md`、`INDEX.json`，由同一索引定位已有源码、冻结运行和环境。

## 已核对的位置与版本

- 正式仓库：`FNIT/repo`。2026-10-03 gpucw1现场核对为干净`main`，HEAD与本地origin/main均`7af34e6d072e843fb2558c931bb2781f1d4b0be9`；后续使用重新核对，未以本次检查确认远端未缓存的新状态。
- 既有精度任务：索引`legacy/freesurfer_synth`链接至保留原实体的`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth`；其`work/recon_standard_20260929/accuracy_20261003`保留现有运行。
- 既有任务环境：`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env`，原prefix保留。
- 通用环境：`FNIT/envs/default`解析到`/cwStorage/home/gongwk/Notebook_code/fnit_conda_env_956b1a9`，它与本轮既有环境不同，不能由位置通知自动切换。
- 本轮基线仍为`816e5610417a4c587caf321049438a9554139016`；工作分支、固定源码、程序和资源哈希按各自回执保留，不把历史结果改标为7af34e6d。

## 新任务与执行

新任务源码放`FNIT/workspaces/<任务名>`，新运行产物放`FNIT/runs/<任务名>`；传输包放`FNIT/archive/transfers`，日志放`FNIT/logs`。新增或变更位置后同步统一索引，保留兼容入口。现有冻结源码、环境和活跃输出继续按原实体身份执行。

解释器使用已核对的实际prefix；`PYTHONPATH`显式指向本次工作树的`src`。用`fnit.__file__`和源码SHA核实实际导入，不把默认环境中/tmp旧树当作main。新调用例：

```bash
# 新任务：先确认索引中的工作树和环境，不将这里示例用于更换已在跑的冻结任务。
FNIT_SERVER_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_WORKSPACE="$FNIT_SERVER_ROOT/workspaces/example-task"  # 本任务的独立源码树
FNIT_ENV_PREFIX="$(readlink -f "$FNIT_SERVER_ROOT/envs/default")"  # 原环境的真实prefix
PYTHONPATH="$FNIT_WORKSPACE/src" "$FNIT_ENV_PREFIX/bin/python" \
  -c 'import sys, fnit; print(sys.executable); print(sys.prefix); print(fnit.__file__)'
```

本次仅同步工作位置，没有启动新的benchmark、改动冻结源码或迁移环境。后续继续已授权的五项精度任务和十例评估，仍按原算法、精度、线程和GPU预算验证。位置规则本身不是算法或安装验收结果。
