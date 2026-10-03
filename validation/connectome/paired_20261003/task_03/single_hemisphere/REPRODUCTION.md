# 单侧 surface ROI 的 CPU 回归

实际测试源码为 `3e43f05fc4020741f6607e7f74fb1bf1e5b65e6c`，部署到已登记的 `FNIT/workspaces/connectome_paired_20261003_v1/task_03/single_hemi_cpu_v1`，没有修改既有冻结运行目录。

在包含该版本 `src/`、`tests/` 与 `pyproject.toml` 的独立目录中执行以下命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 PYTHONPATH=src \
taskset -c 4-7 python -m pytest -q \
  tests/connectome/test_single_hemisphere_templates.py \
  tests/connectome/test_template_pairs.py \
  tests/connectome/test_paired_e2e.py
```

附带 `driver.py` 是本次实际执行的元数据驱动脚本：复制到上述独立目录，与 `src/`、`tests/` 同级，并使用相同环境/affinity 执行。它记录实际导入位置、指定源码与测试文件 SHA、全部源码内容聚合 SHA 前后、CUDA 初始化状态和 `pytest.main` 时间。不要直接在本证据子目录执行，该目录只保存结果而不含冻结源码。

实际 **52 passed in 5.34s**；`pytest.main` 为5.499901213683188秒，完整驱动含导入/两次源码校验为8.809553013648838秒。退出码0，CUDA 初始化前后False。小型模拟 MRI/表面仅用于单元测试，不构成真实 MRI、GPU 或官方等效 benchmark。没有导出真实输入影像或其哈希。

| 文件 | SHA-256 |
|---|---|
| `focused_cpu_report.json` | `82aacfe6f06e85a925d108cbf1dab379fb6fc8f2b8b34bf368580da31f202041` |
| `focused_cpu.log` | `cc17fab7686ddcdf224eeaba6500a5b7a71f95b8fc18ecc554587cae0c9b4e53` |
| `driver.py` | `eb26f1cdd40153167fe7ef0bc9624802e2a603cca3405a228101ebb882b71f65` |

已有双半球非空节点的搜索、映射与投影运算没有改变。新增空半球分支只对明确背景输出 Int32零；两个半球仍须有合法标签文件和几何，全模板至少含一项真实/声明节点。原双半球矩阵与端点聚合逻辑由旧测试继续验证。
