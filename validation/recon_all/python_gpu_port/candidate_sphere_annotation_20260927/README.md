# 真实 T1 左半球：自产平滑白质表面到球面

本次接续[已保存的自产 MRI、拓扑和白质预放置阶段](../white_connected_prefix_20260927/README.md)，输入为去标识的 `sub-01_T1w.nii.gz`（SHA-256：`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`）。候选被试位于 gpucw1 的 `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_finalsurfs_stage_20260927/white_candidate_no_official_topology_fslicense`。官方 FreeSurfer 8.2 被试 `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official` 只用于输出比较。这是保存阶段的接续试验，尚非从原始 T1 在单一进程完成的整例重建。

旧 v5 被试的左侧表面有 106,695 个顶点、213,386 个面，`sulc` 和 `aparc.annot` 也各有 106,695 项；当前候选与官方被试均为 106,622 个顶点、213,240 个面，因此不能复用旧注释。[网格数量审计](saved_v5_audit.json)保留了逐文件记录。

## 输入、输出与命令

Conda 编译的 `mris_inflate` 读取自产 `surf/lh.smoothwm`，写入 `surf/lh.inflated` 和每顶点 `surf/lh.sulc`；对应官方命令为 `mris_inflate LH.SMOOTHWM LH.INFLATED`。随后 Python `run_standard_sphere(inflated, smoothwm, output, finish_device="cpu") -> dict` 读取两个同面序网格，写入 `surf/lh.sphere`，返回投影、度量、逐步优化和总耗时。独立命令与官方对应命令如下：

```bash
python -m fnit.recon_all.sphere_standard_run \
  CANDIDATE/surf/lh.inflated CANDIDATE/surf/lh.smoothwm CANDIDATE/surf/lh.sphere \
  --finish-device cpu --report sphere_report.json
mris_sphere -threads 4 -seed 1234 CANDIDATE/surf/lh.inflated CANDIDATE/surf/lh.sphere
```

Python 使用该 scratch 目录中的源码快照，`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMBA_NUM_THREADS` 均为 4。实际运行的 `sphere_standard_run.py` 与 `sphere_standard_line_search.py` SHA-256 分别为 `7544ccb77153a7ac242b88d1c9752bdd08ce5765ba483cd644a4075e94e90405` 和 `8777cb96ca93ad0584d34e3f2d2f13827b1302dbaf600fc9454fdd39cd323296`，与此仓库文件一致。[audit.py](audit.py) 接受 `候选被试 官方被试 输出JSON` 三个参数，只读比较文件哈希、同索引坐标与面、`sulc`，若注释存在还比较逐顶点标签。

## 逐顶点结果

下表的表面误差是在相同顶点索引计算的三维欧氏位移；三个网格的面数量与排列均与官方相同。

| 自产输出 | 相对保存的官方输出：均值 / P99 / 最大值 | 超过 0.1 mm 的顶点 |
| --- | ---: | ---: |
| `lh.smoothwm` 输入 | 0.000311208 / 0.006172390 / 0.188170 mm | 18 |
| `lh.inflated` | 0.001061059 / 0.002213085 / 0.004201193 mm | 0 |
| Python `lh.sphere` | **3.480691977 / 5.641300784 / 6.873138365 mm** | **106,606 / 106,622** |

`lh.sulc` 的绝对标量误差均值 / P99 / 最大值为 0.000505999 / 0.008345132 / 0.580831528，191 个值完全相同。[逐文件审计](white_candidate_new_sphere_audit.json)含 realpath 和 SHA-256；候选 `lh.sphere` 的 SHA-256 为 `7748a5108c4d722f90f22c584f82659db46b6d57bd2eb825e76ee1429174de9a`。

Conda 膨胀耗时 [12.72 秒](white_candidate_lh_inflate_time.txt)、峰值 RSS 166,052 KiB；Python 球面函数耗时 1144.18 秒，进程墙钟 [1149.97 秒](white_candidate_lh_sphere_time.txt)、峰值 RSS 749,556 KiB。[完整 331 步轨迹](white_candidate_lh_sphere_report.json)记录了每次步长和耗时。此前冻结官方输入的 Python 运行耗时 372.82 秒、243 步，不能直接当作本次候选输入的速度对照。

## 同候选输入的官方程序对照

为判断微小上游误差是否也影响官方优化，将**自产** `lh.inflated` 和 `lh.smoothwm` 逐字节复制到隔离目录。运行前后 `cmp` 均相同；两份[输入哈希](candidate_sphere_official_control/input_sha256.txt)分别为 `f3a2128de469dc3c8d06e2649f85f420b26c84f95aea3c7912f677b4e57afd30` 和 `6d1a5d30639f3678c2e825d69b816f9bc35d5411546145f6c0bcab29a5fd0b91`。官方程序 SHA-256 为 `c34ca308a7fa03acdb3f689bf6125cf3d0198c37c3a62992a29f68e631c73612`。设置私有 `FS_LICENSE`、`FREESURFER_HOME`、`SUBJECTS_DIR` 和 `OMP_NUM_THREADS=4` 后，用 600 秒超时保护执行：

```bash
CONTROL=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_finalsurfs_stage_20260927/candidate_sphere_official_control
/public/software/apps/Freesurfer/8.2.0-1/bin/mris_sphere -threads 4 -seed 1234 \
  "$CONTROL/subjects/sub01/surf/lh.inflated" "$CONTROL/subjects/sub01/surf/lh.sphere"
```

| 同索引球面比较 | 均值 / P99 / 最大位移 | 超过 0.1 mm 的顶点 |
| --- | ---: | ---: |
| 官方程序使用自产输入 vs 原官方整例球面 | 2.959344 / 5.437959 / 7.066581 mm | 106,579 / 106,622 |
| Python vs 官方程序，**输入完全相同** | 1.652636 / 2.952841 / 3.557925 mm | 106,495 / 106,622 |

两组比较的有序面均相同；[三方审计](candidate_sphere_official_control_audit.json)含文件路径和哈希。官方程序耗时 [328.61 秒](candidate_sphere_official_control/time.txt)、峰值 RSS 217,164 KiB；[原始日志](candidate_sphere_official_control/mris_sphere.log)保留了运行信息。Python 与官方同输入的单次墙钟为 1149.97 / 328.61 秒，但两次顺序运行时共享节点负载未控制，不据此给稳定速度比。

初始负面面积比例在候选/冻结输入下分别为 0.022116% / 0.022112%。第 0、1 次线搜索步长分别是 1027.937 / 1027.952、18450.785 / 18396.798；第 2 次变为 **399.365 / 132.050**，第 8 次首次出现优化调度分歧。NumPy 二次拟合曾发出[浮点溢出警告](white_candidate_lh_sphere_warnings.log)，目前不能单凭警告确定误差来源。

这项控制说明：自产上游网格的小误差足以让官方球面也偏离原整例结果；相同自产输入下 Python 与官方程序仍有 1.653 mm 平均差异，移植本身也未通过。依逐点验收规则，本次停在 `sphere`，没有据此生成 `sphere.reg`、`aparc.annot` 或最终 white。下一步需先收敛上游网格误差和同输入球面误差。
