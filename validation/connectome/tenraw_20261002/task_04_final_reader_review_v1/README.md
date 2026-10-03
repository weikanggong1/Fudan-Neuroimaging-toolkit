# Task4 最终 reader 的独立只读审查

本目录只保存协调者子任务的元数据审查证据。未运行 MRI、GPU、追踪或矩阵科学统计；未修改 Task4 源码和产物。

## 当前结论

- Task4 最终提交为 `177db9d0db52d0663f50d37fa5509cca57a3cbbf`。提交中的 reader SHA 与实际 v7 冻结源码相同：`9be317df1bc38fdaf993ccb16b29e52f902c681ecbf80a3d85c08521ef4ee235`。原数值比较源码保持 `120a3e1a431ad23177bc9a4a7c17291e52e374085e0fe7314115b656de8db676`，没有改数学或判定规则。
- 两个旧缺口已修复：初始目录检查包含 canonical raw 的真实父目录、实际替代 run 和官方 producer；ready 后再次检查 qualified job/source 和 symlink 解析后的官方目录。`validate_gpu_run` 的实际 touched 输出 SHA ledger 被保留，并在最终报告生成前执行完整 snapshot 的结束 SHA 检查。
- 实际 terminal snapshot 含 9099 个文件；1480 个 required 输出的 path/SHA 与 snapshot 逐项相等，20 条 chain 每条 74 项。101 个唯一 canonical raw 文件均有相同 SHA 绑定。两个科学源树的 2388 个源文件由本次独立审查重新读取并校验字节 SHA；其中 60 个 JSON 保持源码字节校验，不把打包清单中的相对资源路径误当运行输入。
- 七个历史 JSON 必须逐项匹配冻结 mixed 配置的精确 path/SHA/role，仍校验完整文件字节，仅停止递归展开其旧状态观察。实际通用 walker 的六条例外恰为三个官方 controller 的历史/当前观测，没有跳过真实 raw、MRI 输出或 required 文件。
- 三个 controller 的 fixed workload、选中病例 completed 行和未选中病例不被派发规则已独立复验。旧 groupB 的整体 status 会因轮询改变，CON03/05/07/09 的 completed 行保持逐值相同，CON11 仍 waiting；正式 CON11 来自新 producer 的 completed 行。当前 status 的全字节 SHA 不必等于结束时 SHA，这个例外只用于 controller，真实输入输出 SHA 不放宽。
- 实际 20 份比较全部完成且整体 gate 均为 `failed`；2782/4800 个判定接受。`FNIT_self` 和 `population` 均为 `not_assessed`，未证明全链匹配。此处读取已保存结论，没有重新计算矩阵统计。

## 证据和复核

[actual_metadata_review.json](actual_metadata_review.json) 记录实际 headcw 只读检查、实际配置/ready/snapshot/final-report 的路径和 SHA、七份历史绑定与 controller 前后状态；[delivery_exact_bytes_review.json](delivery_exact_bytes_review.json) 记录最终 commit、归档与 122 个 payload 的逐字节 SHA 校验。归档另含一份自身索引，索引字节也核对相同。没有复制大 snapshot 或 20 份矩阵报告。

独立检查核对了 raw/output **ledger 的完整覆盖**，未再次读取全部大型 MRI 输出内容；实际 reader 已在结束前执行两处完整 snapshot SHA 校验。这个证据层级在 JSON 中明确区分。本地针对两个缺口、历史记录及 mutable controller 的九个 stdlib 元数据测试通过；fixture 不进行影像或数值 benchmark。

```bash
# 已认证的 headcw 连接；只读元数据和科学源码字节，不启动科学求解。
ssh -S /tmp/fnit-bwas-headcw.sock -o ControlMaster=no -o BatchMode=yes \
    -p 39516 gongwk@10.190.248.228 'python3 -B -' \
    < review_final_raw_reader.py > actual_metadata_review.json

# 本地核对 Task4 最终提交和已经交付的压缩证据成员。
python3 -B review_final_raw_reader.py \
    --delivery-repository /tmp/fnit-connectome-tenraw-20261002/task_04 \
    > delivery_exact_bytes_review.json

# 只执行 Task4 的 stdlib 元数据 guard 测试。
cd /tmp/fnit-connectome-tenraw-20261002/task_04
python3 -B -m unittest discover -s tests/connectome \
    -p test_final_raw_envelope_guards.py -v
```

脚本仅使用 Python 标准库。`--actual-root` 可指定实际完成产物目录，默认指向本次 v7；`--delivery-repository` 选择本地精确提交/归档模式，两者均只读输入。冻结源码和提交的预期 SHA 写明在脚本中；来源变更会报错，不能将旧证据改标为新版本。
