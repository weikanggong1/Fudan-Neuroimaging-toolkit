"""只读冻结源码审查；只解析 AST/哈希，不执行 FNIT 或影像计算。"""
import argparse, ast, hashlib, json
from pathlib import Path

MANIFEST_SHA = '21bfe987f5f2cfde992578208d8235b1ed90ad2c4608c95df1b0f4c6751b9d9a'
def sha(raw): return hashlib.sha256(raw).hexdigest()

def main(source, output):
    raw=(source/'SOURCE_MANIFEST.json').read_bytes()
    assert sha(raw)==MANIFEST_SHA, 'unexpected manifest'
    manifest=json.loads(raw)
    verified=[]
    for relative, expected in manifest['files'].items():
        p=source/relative
        if p.is_file():
            r=p.read_bytes()
            assert len(r)==expected['bytes'] and sha(r)==expected['sha256'], relative
            verified.append(relative)
    def ref(file, function=None):
        relative='src/fnit/'+file+'.py'
        r=(source/relative).read_bytes()
        item={'path':relative,'file_sha256':sha(r),'manifest_sha256':manifest['files'][relative]['sha256']}
        if function:
            nodes=ast.parse(r).body
            for part in function.split('.'):
                node=next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==part)
                nodes=node.body
            first=min([node.lineno]+[n.lineno for n in getattr(node,'decorator_list',[])])
            fragment=b''.join(r.splitlines(keepends=True)[first-1:node.end_lineno])
            item.update(function=function,line=first,end_line=node.end_lineno,function_sha256=sha(fragment),
                        function_sha_scope='exact UTF-8 lines including decorators and original line endings')
        return item
    def refs(*items): return [ref(*x) for x in items]
    node=next(n for n in ast.parse((source/'src/fnit/recon_all/python_profile.py').read_text()).body if isinstance(n,ast.FunctionDef))
    profile=ast.literal_eval(next(n.value for n in ast.walk(node) if isinstance(n,ast.Return)))
    assert len(profile)==21
    def s(name, implementation, guard, *evidence):
        return dict(name=name,implementation=implementation,selected_guard=guard,
                    original_FS_stage_selected=False,evidence=refs(*evidence))
    stages=[
      s('profile/程序发现','固定21选项，不消费原版程序/能力探测','backend=python；mode=torch；include_white=False',('recon_all/python_profile','python_profile_options'),('recon_all/native_free','_run_recon_all_python'),('recon_all/native_runtime_selection','select_native_optimizations')),
      s('conform/SynthStrip/Talairach','自有Python/Numba与Torch GPU','显式CUDA；Talairach子进程为sys.executable -m',('recon_all/input_talairach_chain','run_input_talairach_chain'),('recon_all/talairach_synthmorph','register_talairach')),
      s('N4/nu','SimpleITK Python API通用ITK CPU库；自有nu转换','simpleitk首分支返回，不解析原版N4程序',('recon_all/input_n4_chain','run_n4_stage'),('recon_all/n4_itk','correct_volume'),('recon_all/n4_simpleitk','correct_volume'),('recon_all/n4_wrapper','make_nu')),
      s('两轮normalize/brainmask','自有Torch GPU与有序Numba CPU混合','两轮device=CUDA；controls=torch；第二轮initial_bias=torch；动态ridge CPU',('recon_all/normalization/pipeline','normalize_t1'),('recon_all/normalization/aseg_pipeline','normalize_t1_aseg')),
      s('SynthSeg/辅助网络','自有Torch GPU','SynthSeg CUDA不进入CPU columns；EntoWM/MCA-dura/vsinus显式CUDA及既有精度例外',('recon_all/native_free','_run_white_mri_chain'),('synthseg_parc/cpu_conv','CPUInferenceConv3d.forward')),
      s('GCA registration','完整自有Python/Numba EM与Torch GPU评分/求逆','binary=None；isolated Python worker；inverse=torch；chunk1024',('recon_all/native_free','_run_native_em_register'),('recon_all/gca_torch_worker','run_isolated_registration'),('recon_all/gca_torch_worker','run_worker'),('recon_all/mri_em_register_python','register_t1')),
      s('CA normalize/corpus callosum','自有Python/Numba CPU','run_ca_normalize未传device，保持CPU默认；run_mri_cc自有',('recon_all/ca_normalize_python','run_ca_normalize'),('recon_all/mri_cc_python','run_mri_cc')),
      s('WM segmentation','自有Python/Numba有序CPU与Torch GPU直方图/缓存平面','wm_backend=torch-optimized；isolated Python worker',('recon_all/native_free','_run_torch_wm_segment'),('recon_all/wm_torch_worker','run_isolated_segmentation'),('recon_all/mri_segment','segment_white_matter_mgz')),
      s('WM aseg/pretess/Ento-ACJ','完整自有Python/Numba CPU编辑与Torch GPU后编辑','wm_edit_backend=python；byte-path几何guard保留；生产没有被试SHA白名单',('recon_all/native_free','_run_python_wm_edit'),('recon_all/edit_wm_aseg_core_python','write_wm_asegedit_profile'),('recon_all/edit_wm_aseg_core_python','remove_paths_to_cortex')),
      s('filled','完整自有Python与有序Numba CPU切割/堆/标签','fill_backend=numba；初始边界CPU，不是纯GPU',('recon_all/fill_cutting_plane_python','fill_mgz')),
      s('tessellation/topology','自有Torch tessellation与Python/Numba数字拓扑','topology_backend=python；数字拓扑不同于FS GA',('recon_all/native_free','_surface_pair'),('recon_all/native_free','_prepare_native_topology'),('recon_all/topology_volume_python','run_volume_topology')),
      s('remesh/intersection','自有Python/Numba有序CPU','remesh_scalar_storage=python；intersection binary=None；source_numba完整修复',('recon_all/mris_remesh_python','remesh_surface'),('recon_all/mris_remove_intersection_python','remove_intersection_surface'),('recon_all/mris_remove_intersection_python','run_remove_intersection')),
      s('white.preaparc/white/pial','完整四轮Python/Numba CPU与Torch GPU候选/清理','surface_placement_backend=python；compiled retained MHT是@njit/objmode，不是CPP库',('recon_all/surface_placement_python','placement_options'),('recon_all/surface_placement_python','run_preaparc_python'),('recon_all/surface_placement_python','run_final_white_python'),('recon_all/surface_placement_python','run_pial_python'),('recon_all/place_surface_snapshot','_ordered_retained_snapshot_step')),
      s('inflate/sphere','自有Torch inflation；sphere CPU目标/步长与GPU平均/收尾','inflate=torch；sphere_finish=torch；averaging_device=CUDA；normals默认CPU',('recon_all/native_free','_run_accurate_sphere_pair'),('recon_all/inflate_torch','TorchInflationContext'),('recon_all/sphere_standard_run','run_standard_sphere')),
      s('sphere.reg','自有Python/Numba CPU目标/步长与Torch/Triton GPU有序平均','两pass传CUDA averaging_device；overlap_device=cpu；不走CPU CPP平均器',('recon_all/native_free','_hemisphere_operation'),('recon_all/mris_register_run','run_register_sphere'),('recon_all/mris_register_sulc_run','run_register_sulc'),('recon_all/mris_register_smoothwm_run','run_register_smoothwm'),('recon_all/mris_register_average_numba','RegistrationGradientAverager.__call__')),
      s('avg_curv/extra annotation','自有Python/Numba CPU采样与标签映射','python_only=True；avg_curv平均复用average_marked_values，不走CPP平均器',('recon_all/native_free','_run_avg_curv'),('recon_all/mrisp_paint_python','paint_atlas_values'),('recon_all/native_free','_project_exvivo_annotations')),
      s('GCSA annotation','自有Torch GPU特征与有序Numba CPU Gibbs','annotation_gibbs_backend=numba；按表面版本缓存',('recon_all/gcsa_label_python','GCSAFeatureCache'),('recon_all/gcsa_label_python','label_surface')),
      s('defects volume','自有Torch GPU投射与Python格式写出','defects_backend=torch；提前return；LH→RH共享输出顺序保留',('recon_all/native_free','_run_defects_volume'),('recon_all/defects_label_volume_torch','defects_to_volume')),
      s('thickness/area/curv','自有Torch GPU几何与SciPy/Numba CPU空间查询','Python/CUDA分支，没有原版指标程序',('recon_all/native_free','_run_surface_metrics'),('recon_all/surface_thickness_gpu','thickness_map'),('recon_all/surface_area_gpu','area_map'),('recon_all/surface_curvature_gpu','curvature_map')),
      s('curv.stats/八派生图','自有Torch离散曲率与有序Numba统计/系统libm','python_only=True；source_libm=Numba np.arccos，不是原FS库',('recon_all/native_free','_run_curvature_stats'),('recon_all/curvature_stats_python','write_curvature_stats'),('recon_all/discrete_curvature_torch','_source_acosf')),
      s('jacobian/contrast/ribbon/volume maps','自有Python/Numba与Torch CPU','jacobian与contrast明确CPU；标签/体积映射为自有函数',('recon_all/native_free','_project_parcels'),('recon_all/native_free','_project_wmparc'),('recon_all/native_free','_run_recon_all_python')),
      s('ROI/aseg/wmparc/brainvol stats','自有GPU统计基量与Python/Numba CPU归约/写出','SurfaceStatsCache CUDA；no-th3脑区定义保留',('recon_all/native_free','_write_hemisphere_stats'),('recon_all/surface_stats_cache','SurfaceStatsCache'),('recon_all/anatomical_stats_file','write_anatomical_stats')),
      s('MNI affine/nonlinear/inverse/check','自有SynthMorph Torch GPU与GPU采样/逆场','parallel-late；postprocess=gpu；三个原版program None；Python fresh exec及CPU mesh2+2',('recon_all/mni_aux_chain','run_mni_aux_chain'),('recon_all/mni_mesh_parallel','run_mni_stage'),('recon_all/mni_mesh_parallel','run_mni_and_validate'),('recon_all/mni_nonlinear_chain','run_mni_nonlinear_chain')),
      s('输出/网格校验','自有Python/NumPy/Numba CPU','join后完整检查，失败抛异常；整体等效未判定',('recon_all/native_free','_complete_output_validation'),('recon_all/native_free','_validate_meshes')),
    ]
    excluded=[
      dict(name='CPU sphere/register CPP averager',selected=False,guard='Averager CUDA分支不调用average_gradients_exact_cpu',evidence=refs(('recon_all/mris_register_average_numba','average_gradients_exact_cpu'),('recon_all/mris_register_average_numba','RegistrationGradientAverager.__call__'),('recon_all/_average_cpu_cpp','_bind'))),
      dict(name='SynthSeg CPU columns CPP/ctypes',selected=False,guard='CPUInferenceConv3d需要image.device=cpu；CUDA不惰性导入或构建',evidence=refs(('synthseg_parc/cpu_conv','CPUInferenceConv3d.forward'))),
      dict(name='SynthMorph CPU joint Eigen CPP/ctypes',selected=False,guard='requires moving and fixed CPU；GPU不导入_cpu_eigen',evidence=refs(('synthmorph/models','_cpu_joint_inference_enabled'),('synthmorph/models','AffineNetwork.forward'),('synthmorph/models','_cpu_joint_matrix_sqrt'))),
    ]
    report=dict(schema_version=1,scope='manual static selected call-path audit; not execution or OS isolation',
      frozen_source=dict(git_commit=manifest['git_commit'],has_overlay=True,label=manifest['label'],source_manifest_sha256=sha(raw),declared_files=775,
        tar_sha256='5fdc490a8f3a3cb9c6abc924de76da4ae7d01422f01702703597458d62e593c7',tar_bytes=7024640,
        server_manifest_verification='775/775 size and SHA verified read-only before subset export',
        downloaded_subset_verified=len(verified),downloaded_python_modules=sum(x.endswith('.py') for x in verified)),
      profile=profile,profile_source=ref('recon_all/python_profile','python_profile_options'),stages=stages,unselected_local_CPP_routes=excluded,
      conclusion='No original FS stage executable, original FS stage-linked library or CPP build fallback selected in reviewed frozen fixed-profile paths',
      generic_compiled_libraries_allowed=['SimpleITK/ITK','PyTorch/CUDA/Triton','NumPy/SciPy','Numba and system libm'],
      Python_workers='sys.executable fresh exec for Talairach/GCA/WM/hemi/MNI; not original FS program execution',
      IO_and_monitoring='nibabel.freesurfer.io is format IO; nvidia-smi is monitoring; asset path FREESURFER_HOME does not prove program use',
      findings=[dict(id='WM-METADATA-001',severity='metadata-only',line=1456,
        description='顶级WM implementation错误写Conda C++ edit；实际wm_edit_backend=python为完整自有编辑，edit_implementation正确；已通知root，新版修复，不改变冻结v4',
        evidence=refs(('recon_all/native_free','_run_recon_all_python'),('recon_all/native_free','_run_python_wm_edit')))],
      boundaries=dict(static='manual guard/call-path review with exact function SHA; not formal reachability proof',runtime_execution_audit='not performed by this task',
        clean_OS_isolation='not_verified',new_MRI_GPU_jobs=0,precision='TF32 default and existing per-stage FP32 exceptions; no new FP16/BF16',
        time_and_memory='not measured by this static audit',strict_reproduction='not established by static review',overall_metric_equivalence='not_assessed',
        topology='independent digital topology differs from FS GA'))
    output.mkdir(parents=True,exist_ok=True)
    (output/'static_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    table='| 阶段 | v4 实际实现 | 原版阶段程序 |\n| --- | --- | --- |\n'
    for row in stages: table+='| '+row['name']+' | '+row['implementation']+' | 未选中 |\n'
    (output/'migration_table.md').write_text('# v4 固定 Python 配方：静态迁移核对\n\n'+table+'\n仅核对上述冻结源码调用条件，不替代实际整例执行/动态库/文件访问审计或干净环境验收。原版混合分支保留为独立对照，不被此配方消费。数字拓扑不是 FS GA；整体指标等效尚未判定。函数/文件 SHA 与已发现 WM 报告字段问题见 static_audit.json。\n')
    print(json.dumps(dict(stages=len(stages),function_references=sum(len(x['evidence']) for x in stages),subset_verified=len(verified),report_sha256=sha((output/'static_audit.json').read_bytes())),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();main(a.source,a.output)
