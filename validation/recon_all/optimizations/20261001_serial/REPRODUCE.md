# 本轮验证与结果复现

本目录的报告绑定实际运行版本，不读取官方结果修补生产输出。输入影像、权重、图谱和许可证留在已授权服务器；这里只保存日志、哈希、标量、CSV和脑图。所有新输出目录须为空，JSON/归档文件须不存在。

## 已执行的生产版本

- 起点：0c8ab327c1a4d10eaf6c8066a16d050e09e30c90。
- 整合计算：ff372d73f106e850b999fbc95ae0b324cd315cbf，src tree 319ff4ccd7632c637721b55522640714ce0d37f2。
- 源码归档SHA-256：071e48c97a1f8d88bd920e4753ed676033e47bed5a3fa0bc22e06c24338fd126。
- sub01为预初始化CUDA API，sub02为CLI；均是原始T1和空目录，H100同一UUID、4线程。
- 首例基线复用已有c248520 GPU API，生产src与0c8相同；第二例GPU CLI基线本轮另测。不同日期的共享负载和缓存无法视为相同。
- 后续main的MCFLIRT更新与本轮调用链无交集，实际测试提交不改标。

## 原始T1整例与只读比较

完整服务器参数在whole/integrated_main/whole_sub01_integrated_ff372d7.json、whole_sub02_integrated_ff372d7.json。配置包含python、weights、assets、device、threads、gpu_uuid、code_root、code_commit、source_archive_sha256、input、output、diagnostic_root、invocation及native_bin_dir。路径可按实际授权环境修改；源码与资源SHA必须重新核验，不能只修改commit标签。

~~~bash
# 原始T1、空输出目录和程序束由配置声明；许可文件只提供路径，不复制内容。
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/whole_sub01.json  # 实际源码归档、GPU UUID和预初始化API参数
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/whole_sub02.json  # 相同线程预算的CLI，第二个空目录
python validation/recon_all/python_gpu_port/collect_hotspot_whole_comparison.py \
  --config /bench/whole_comparison.json  # 两例自产目录、优化前和独立官方参考；只读
~~~

whole_comparison的additional_quality=[baseline, official]要求为三方分别生成质量报告。质量候选不足或超时显示incomplete，不作为通过。原网格不对应时使用双向点到三角面距离，不能逐索引比较。

## 连续链、全域逆场与结果页

以下均为独立benchmark内部步骤，没有官方独立等价CLI；对应重建参考为recon-all -all -parallel -openmp 4 -itkthreads 1，官方只在独立参考目录执行。

~~~bash
comparison_config_path=/bench/whole_comparison.json  # 两例成功自产目录、固定比较器及实际提交
volume_drift_output_path=/bench/volume_mesh_drift.json  # 新JSON；conform体积及surface RAS/mm
inverse_error_output_path=/bench/whole_inverse_error.json  # 新JSON；全场及基线brainmask内距离/mm
inverse_candidate_mode=whole  # whole读取完成的整例；stage读取冻结阶段
python validation/recon_all/optimizations/20261001_serial/collect_serial_drift.py \
  --config "$comparison_config_path" \
  --output "$volume_drift_output_path"
python validation/recon_all/optimizations/20261001_serial/diagnose_mni_inverse_residual.py \
  --config "$comparison_config_path" \
  --candidate-mode "$inverse_candidate_mode" \
  --output "$inverse_error_output_path"
~~~

0.1mm仅为误差定位分箱，不是事后验收阈值。保留全域最大误差，不能只展示脑内残差。冻结stage模式还需stage-prefix，路径规则沿用当次mni_affine_*_sub01_retry1/sub02_retry1。

~~~bash
serial_reports_dir=/repo/validation/recon_all/optimizations/20261001_serial  # 报告及相对路径根
production_commit=ff372d73f106e850b999fbc95ae0b324cd315cbf  # 实际计算源码
whole_case_report_root=whole/integrated_main/whole  # reports下两例原始报告目录
resource_fingerprint_path=whole/integrated_main/reports/runtime_fingerprints_ff372d7.json  # 实测资源指纹
whole_summary_path="$serial_reports_dir/final_whole_summary.json"  # 新完整汇总JSON
volume_drift_path="$serial_reports_dir/whole/integrated_main/volume_mesh_drift_ff372d7.json"  # 自产连续链
archived_official_timing_path="$serial_reports_dir/whole/integrated_main/reports/official_runtime_archived_v2.json"  # 原日志e墙钟
inverse_error_path="$serial_reports_dir/whole/integrated_main/whole_mni_inverse_residual_ff372d7.json"  # 全域逆场
backend_control_report_dir="$serial_reports_dir/whole/integrated_main/backend_controls"  # 匹配生产的cuDNN控制
final_article_path="$serial_reports_dir/FINAL_RESULTS.md"  # 新中文结果页
final_metric_path="$serial_reports_dir/final_metric_summary.json"  # 新精简指标和源SHA
python "$serial_reports_dir/summarize_whole.py" \
  --reports "$serial_reports_dir" \
  --candidate-commit "$production_commit" \
  --case-root "$whole_case_report_root" \
  --resource-report "$resource_fingerprint_path" \
  --output "$whole_summary_path"
python "$serial_reports_dir/render_final_results.py" \
  --reports "$serial_reports_dir" \
  --summary "$whole_summary_path" \
  --drift "$volume_drift_path" \
  --official "$archived_official_timing_path" \
  --inverse "$inverse_error_path" \
  --backend-controls "$backend_control_report_dir" \
  --integration-proof "$serial_reports_dir/whole/integrated_main/source_after_main_merge.json" \
  --output "$final_article_path" \
  --metrics "$final_metric_path"
~~~

示例用完整变量名声明每个参数，并逐项中文注释；全部续行均可直接复制。

## GPU缓存与cuDNN控制

benchmark_synth_main_integration.py的source是带哈希自产orig及旧分割，output为空目录；weights/assets为已声明资源，device默认cuda:0、threads默认4、commit必填。返回run.json与影像/体积CSV，包含设备、float32、实际TF32/autocast、cuDNN启用/确定性策略、含加载/传输/额外mask和SDT写出的秒数。比较器只在计算结束后读旧结果，缺文件/几何或shape不符即失败。

旧独立控制仅使用cudnn.flags(allow_tf32=False)，PyTorch实际默认关闭cuDNN，不能代表生产策略。其原始字节保留，scope_correction.json标明限制。修正控制显式enabled=True、benchmark=False、deterministic=True、allow_tf32=False，并读取实际设置确认。所有这些控制在两例整例计时之后执行，不扣除任何等待时间。

run_synth_backend_controls.sh展示实际固定GPU UUID、OMP/BLAS/Numba线程4、缓存开启/关闭的四组命令。allocated/reserved unavailable与0不同；外部NVML统计为目标GPU上的同一时刻父子进程合计。未跑缓存开启的完整CLI/API，不据局部算子或阶段数据修改默认低显存策略。

## 传输、安装与资源

- transfer_serial_reports.py接收mapping（source/target列表）、output（新的tar.gz）和label（版本/范围）；只导出报告后缀，拒绝MRI/权重/许可证/重复或越界路径，归档含每文件大小与SHA。
- capture_serial_resources.py重新核验已声明的两例、11权重、102资产、14程序与当前依赖/硬件。缺失或不符失败，不读取许可证内容，也不证明隔离部署。
- validate_install_integrated.py固定实际ff372d7快照与已有主页Conda环境，构建wheel、编译扩展、独立target安装、API导入及CLI帮助；不重建Conda，不执行物理隔离整例。
- collect_official_runtime.py只提取既有官方日志的e墙钟和历史完整时间戳，保存源SHA及CPU/墙钟字段定义；并行及嵌套时间不能相加。
- render_final_results.py不做影像计算，也不改变标准；输入版本不符、输出已存在、数据缺失会报错。输出Markdown、精简JSON保留严格诊断、局部异常和整体等效not_assessed。
- verify_report_bundle.py接收root、manifest及可选新output；逐文件核对相对路径、原始字节大小和SHA，丢失或不符非零退出，不加载pstats/pickle、模型或影像。

~~~bash
public_report_root=/repo/validation/recon_all/optimizations/20261001_serial  # 当前公开报告根
public_manifest_path="$public_report_root/report_bundle_manifest.json"  # 带版本、相对路径、大小及SHA
bundle_verification_output=/tmp/fnit_bundle_verification.json  # 必须不存在的新检查JSON
python "$public_report_root/verify_report_bundle.py" \
  --root "$public_report_root" \
  --manifest "$public_manifest_path" \
  --output "$bundle_verification_output"
~~~

原始输入空间、MGZ conform网格、surface RAS及指标单位参见[调用页](../../../../docs/recon_all/README.md)。模型来源与许可证以资源清单和FNIT固定Release为准。无物理隔离环境整例时始终标记未验证。


## main 合并后回归与日志字段

run_postmerge_ece5e23_retry2.sh 使用相同目标GPU和4线程，原始T1整例目录仅作为只读冻结输入；新目录独立生成注册结果，再与自产结果比较。第一轮LTA封装失败和第二轮后端不匹配控制分别保留有效性说明。两例正确后端注册输出全部逐位相同；后续db479a0只增加日志字段，97项接口测试通过，未另跑该日志补丁整例。复现脚本及报告见whole/integrated_main/postmerge/。

结果生成器新增 --integration-proof，传入 whole/integrated_main/source_after_main_merge.json，以绑定这三类版本。

## 下一条性能验证

下条命令运行缓存开启的完整CLI，输出目录均须不存在；在已授权gpucw1执行。它不是本轮已完成结果。固定源码、原始T1、GPU UUID、4线程和资源，GPU同步剖析与NVML监控均开启；随后需同配置测试预初始化CUDA的API。当前保持低显存默认，不能根据孤立Synth阶段直接切换整例策略。

```bash
# 原始T1与空目录，显式指定权重/资产/原生程序、逻辑设备和线程；env移除禁用缓存变量。
env -u PYTORCH_NO_CUDA_MEMORY_CACHING PYTHONPATH='/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/candidate_final_d4cba29/src' CUDA_VISIBLE_DEVICES=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/../fnit_main_env/bin/python' '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/candidate_integrated_ff372d7/validation/recon_all/python_gpu_port/run_monitored.py' --gpu-uuid GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba --output '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/next_cache_enabled_sub02_monitor' --interval 1 -- '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/../fnit_main_env/bin/python' -m fnit.recon_all.native_free /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-02_T1w.nii.gz '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/next_cache_enabled_sub02' --weights-dir '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/../weights' --assets-dir '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/../assets' --native-bin-dir '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/native_bundle/bin' --device cuda:0 --threads 4 --profile-stages --cuda-allocator-cache enabled
```

MNI逆场局部误差另沿冻结输入诊断，不使用降低精度或改变分区阈值换取速度。当前最慢部分是双侧表面生成、最终white/pial，全部内部时间与来源已列在FINAL_RESULTS.md。

## 自产LTA读取回归

使用新源码的PYTHONPATH，运行以下命令；config只取cases的candidate，官方参考路径不被读取。输出必须不存在。

```bash
comparison_config=/authorized/run/whole_integrated_ff372d7_comparison_config.json  # 两例自产整例路径
lta_report_output=/tmp/fnit_real_lta_reader.json  # 新输出JSON
python validate_lta_real_outputs.py \
  --config "$comparison_config" \
  --output "$lta_report_output" \
  --commit d4cba2907856ed545f10e36f9c2b9e914e84585b
```

最新格式回归六文件矩阵差0，100项相关测试通过；两例原始T1整例仍绑定ff372d7。
