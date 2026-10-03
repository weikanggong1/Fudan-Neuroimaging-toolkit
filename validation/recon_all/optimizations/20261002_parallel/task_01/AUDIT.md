# 半球调度审计（测量前）

基线 f07cf59c7f51ae1393e2578f141b4d130ed7801a；默认串行不变，双侧 exec 为显式 opt-in。无新增依赖，使用既有 Python/PyTorch/Numba/nibabel 与项目 Conda 原生组件。

| 阶段 | 共享读取 | 私有写入 | 父进程发布/屏障 |
| --- | --- | --- | --- |
| surface | filled、norm、brain、wm、aseg.presurf、entowm、brain.finalsurfs、固定资产 | H.orig*、H.defect_labels、H.*label、filled-pretess255/127、H.white.preaparc、H.smoothwm、H.sphere、H.*曲率、autodet.gw.stats.H.dat、mrisps.wpa.mgz、原生 cwd 文件及日志 | 双侧完成后发布 H 文件，mrisps.wpa 最后取右侧；再串行 lh→rh surface.defects |
| register | H.sphere、H.smoothwm、H.sulc、左右 TIFF atlas | H.sphere.reg、H.avg_curv；临时 sulc seed 位于私有 surf | 双侧完成后才能 annotation |
| annotation | H.orig/sphere.reg、aseg、GCSA、ic4/ic7 | 三套 H.*.annot | 双侧完成后才能 final white/pial |
| finish_surface | H.white.preaparc、H.aparc、cortex、灰白统计、共有 MRI | H.white、H.pial.T1/pial、mrisps.white.mgz、H.pial.log（GPU厚度/面积/曲率/volume在父进程屏障后串行） | 双侧完成后发布；mrisps.white 最后取右侧；然后统一 exvivo/ribbon/volumes/stats |

`surface.defects.mgz` 左侧模板是 orig，右侧模板是左侧输出，标签编码分别1000/2000，清空/累计分别0/1。这一串行依赖不能用双侧各自产 volume 后相加代替。

私有目录完整复制 mri/surf/label/stats，禁止 hardlink 和符号链接共享可写文件；scripts 从空目录开始。两侧完成后先审计全部新建/修改/删除的路径，再逐文件临时写入+replace发布。未声明共享写入和删除输入即失败；失败终止进程组、回收 worker、删除私有影像，保留独立日志及失败 JSON。发布不是跨文件原子事务，发布失败明确留下 failed 和已发布路径，不能恢复成成功。

原生 topology 已固定 seed=1234、单线程，保持该例外。sphere 的随机状态是每次调用的局部复现状态，不共享 RNG。生产 parallel worker 不主动改变随机种子。Torch/Numba/OMP/BLAS/ITK 新进程预算各2，父进程等待期间不执行影像计算；Torch interop=1。原生单线程规则保留。总预算4，不能将各库配置数相加当作实测活跃线程。线程实效与环境分别记录。

父进程现有 TF32 默认和局部 FP32 例外保留；worker 将调度瞬间的 matmul/cudnn TF32 值传递到全新进程，各自作用域互不影响。已有 Python API 全局 TF32 设置沿用旧入口；不采用线程池做影像计算。caller autocast 开启时 parallel 在创建输出前拒绝，避免暗中引入未授权低精度。已有 CUDA Python API 不 fork CUDA 上下文，使用 exec Python worker。

组墙钟包含私有复制、exec/导入、计算、同步、发布、清理；worker sum 只为诊断，不能替代组墙钟。进程树显存按同一 nvidia-smi 快照汇总目标 UUID 的父子进程；采样间隔0.5秒，错误/最大间隔明确报告，未采样为 unavailable。外部 GPU 负载单列。

真实回归预声明：严格数值差异0，项目 `tests/recon_all/tolerances_numeric.json` 保持原门槛。整体指标等效始终 not_assessed。模拟数据仅检查调度隔离与取消，真实检查点才用于性能。冻结检查点不写入；资源复用现有已授权目录，不新下载、不复制进公开仓库，不发布真实影像或许可证。

各任务2—5继续使用原 callable 接口，本任务通过 `_hemisphere_operation` 统一调度。生产并行覆盖 surface/register/annotation/finish_surface；下游双侧体积分割和全脑统计保持串行屏障。新后端的参数由本任务接线，不能由其他任务同时修改此模块。

实测后的更新：GPU顶点指标AB/BA的双侧独立exec并行变慢，最终指标回到父进程串行；white/pial放置仍双侧独立。首轮CUDA同步失败保留，显式FP32首分配与UUID设备选择后AB/BA完成；两项初始化条件共同修改，不单独归因于显存容量。
