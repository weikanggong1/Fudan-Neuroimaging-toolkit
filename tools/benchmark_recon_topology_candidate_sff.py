"""完整同输入候选 SFF→自产 K1/K2→已验证五项评分；不运行 GA。"""
from __future__ import annotations
import time
_ENTRY = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import nibabel as nib
import numba
import numpy as np
import torch

from benchmark_recon_topology_neighbor_fields import load_neighbor_capture
from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from benchmark_recon_topology_distance_fields import load_distance_capture
from benchmark_recon_topology_unmri_fields import load_volume_capture
from benchmark_recon_topology_patch_fitness import (
    difference, fitness_reference, prepared_arguments, source_modules, json_safe,
)
from fnit.recon_all import topology_candidate_sff as implementation
from fnit.recon_all.topology_patch_fitness import evaluate_prepared_patch_fitness
from fnit.recon_all.thread_budget import thread_budget


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, neighborhood_capture, neighborhood_repeat, statistics_capture, statistics_repeat,
        volume_capture, distance_capture, brain, fnit_joint_pdf, reference_fitness,
        reference_repeat, source_manifest, output, threads=4, post_neighborhood=None):
    """同一 first candidate 完整拟合及两个自产评分，不读取参考 K 作计算输入。

    neighborhood_* 为 SFF 前原 vt.vtotal/法向/状态；statistics_* 为同候选
    SFF 后只读诊断。brain/joint-PDF、volume/distance 为已声明准备阶段
    输入；距离从几何重算，反馈独立副本。所得 K 替换完整曲率输入后
    执行五项现有评分。original reference 只在候选写出之后比较。
    post_neighborhood 如存在，还比较完整主切向量后态；未捕获时该项
    not_assessed。output 新空目录；参数不修改生产，无官方程序调用。
    """
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError('fresh candidate SFF output required')
    before = load_neighbor_capture(capture_path=neighborhood_capture)
    before_repeat = load_neighbor_capture(capture_path=neighborhood_repeat)
    after = load_statistics_capture(capture_path=statistics_capture)
    after_repeat = load_statistics_capture(capture_path=statistics_repeat)
    vertex = before['vertex_fields']
    if (not np.array_equal(vertex[:, :3], after['vertex_fields'][:, :3])
            or not np.array_equal(vertex[:, 3:6], after['vertex_fields'][:, 6:9])
            or not np.array_equal(before['selected_vertices'], after['selected_vertices'])):
        raise ValueError('same ordered candidate origXYZ/normals/patch required')
    args = dict(original_vertices=vertex[:, :3], vertex_normals=vertex[:, 3:6],
        neighbor_offsets=before['neighbor_offsets'], neighbors=before['neighbors'],
        patch_vertices=before['selected_vertices'], initial_k1=vertex[:, 12],
        initial_k2=vertex[:, 13], initial_tangent_e1=vertex[:, 6:9], initial_tangent_e2=vertex[:, 9:12])
    input_hashes = {key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in args.items()}
    loaded = time.perf_counter()
    with thread_budget(threads=threads):
        candidate = implementation.candidate_second_fundamental_form(**args)
        repeated = implementation.candidate_second_fundamental_form(**args)
        volume = load_volume_capture(capture_path=volume_capture)
        distance = load_distance_capture(capture_path=distance_capture)
        image = nib.load(str(brain))
        with np.load(fnit_joint_pdf, allow_pickle=False) as archive:
            joint_pdf = archive['gray_white_pdf'].copy()
        scoring = prepared_arguments(statistics=after, volume_fields=volume,
            distance_fields=distance, brain_image=image, joint_pdf=joint_pdf)
        # The after-capture K fields are never consumed by the scoring call.
        scoring['curvature_inputs']['principal_curvature_k1_mm_inv'] = candidate['k1']
        scoring['curvature_inputs']['principal_curvature_k2_mm_inv'] = candidate['k2']
        own = evaluate_prepared_patch_fitness(**scoring, valid_faces=True,
            mri_weight=1., volume_unmri_weight=10., normal_dot_weight=1., quadratic_curvature_weight=1.)
        repeat_score = evaluate_prepared_patch_fitness(**scoring, valid_faces=True,
            mri_weight=1., volume_unmri_weight=10., normal_dot_weight=1., quadratic_curvature_weight=1.)
    output.mkdir(parents=True)
    tick = time.perf_counter()
    np.savez_compressed(output/'candidate.private.npz', **{key:value for key,value in candidate.items() if isinstance(value,np.ndarray)},
        fitness=np.array(own['fitness'], np.float64), distance_mm=own['distance']['distance_mm'],
        vertex_curvbak=own['vertex_curvbak'])
    candidate_io = time.perf_counter()-tick
    reference, stable = fitness_reference(reference_fitness)
    _, stable_repeat = fitness_reference(reference_repeat)
    comparisons = {name:difference(candidate[name], after['vertex_fields'][:,col]) for name,col in [('k1',10),('k2',11)]}
    comparisons.update({key:difference(np.array(own['composition']['tp_fields'][key],np.float32),
        np.array(reference['tp_fields'][key],np.float32)) for key in ['face_ll','vertex_ll','curv_ll','qcurv_ll','unmri_ll']})
    comparisons['fitness']=difference(np.array(own['fitness'],np.float64),np.array(reference['fitness'],np.float64))
    comparisons['distance']=difference(own['distance']['distance_mm'],distance['distance_after'])
    comparisons['feedback']=difference(own['vertex_curvbak'],volume['reference_curvbak'])
    for field, value in dict(fll=own['components']['mri']['area_weighted_face_bits'],
            vll=own['components']['mri']['area_weighted_vertex_bits'],
            cll=own['components']['normal']['cll_float32_bits'],
            qcll=own['components']['curvature']['qcll_float32_bits']).items():
        comparisons[field+'_uint32_bits']=difference(np.array(value,np.uint32),
            np.array(reference['tp_field_uint32_bits'][field],np.uint32))
    tangent = {'status':'not_assessed','reason':'post-SFF principal tangent fields not captured'}
    if post_neighborhood is not None:
        post = load_neighbor_capture(capture_path=post_neighborhood)
        if not np.array_equal(post['selected_vertices'],before['selected_vertices']):
            raise ValueError('post tangent candidate order mismatch')
        for name, cols in [('tangent_e1',slice(6,9)),('tangent_e2',slice(9,12))]:
            comparisons[name]=difference(candidate[name],post['vertex_fields'][:,cols])
        tangent={'status':'captured_and_compared','capture_sha256':sha(post_neighborhood)}
    own_repeat={name:difference(candidate[name],repeated[name]) for name in ['k1','k2','tangent_e1','tangent_e2','gram','rhs','hessian','fit_kind']}
    own_repeat.update(fitness=difference(np.array(own['fitness'],np.float64),np.array(repeat_score['fitness'],np.float64)),
        distance=difference(own['distance']['distance_mm'],repeat_score['distance']['distance_mm']),
        feedback=difference(own['vertex_curvbak'],repeat_score['vertex_curvbak']))
    native_repeat={key:difference(value,before_repeat[key]) for key,value in before.items()}
    native_repeat.update(post_k1=difference(after['vertex_fields'][:,10],after_repeat['vertex_fields'][:,10]),
        post_k2=difference(after['vertex_fields'][:,11],after_repeat['vertex_fields'][:,11]))
    reference_exact=stable==stable_repeat and all(row['byte_equal'] for row in native_repeat.values())
    input_unchanged=all(hashlib.sha256(value.tobytes()).hexdigest()==input_hashes[key] for key,value in args.items())
    matrix=difference(scoring['mri_inputs']['ras_to_voxel'],after['matrix'])
    exact=all(row['byte_equal'] for row in comparisons.values())
    repeated_exact=all(row['byte_equal'] for row in own_repeat.values())
    source=source_modules();source['topology_candidate_sff']=sha(implementation.__file__)
    from fnit.recon_all import topology_vnl_svd
    source['topology_vnl_svd']=sha(topology_vnl_svd.__file__)
    paths={name:path for name,path in locals().copy().items() if isinstance(path,Path) and name!='output'}
    report=dict(status='complete_frozen_candidate_sff_and_prepared_fitness',execution_complete=True,
        strict_sff_and_scoring_outputs=exact,candidate_repeat_exact=repeated_exact,reference_repeat_exact=reference_exact,
        stage_execution_gate_passed=bool(exact and repeated_exact and reference_exact and input_unchanged),
        overall_internal_byte_reproduction=bool(exact and repeated_exact and reference_exact and matrix['byte_equal']),
        preparation_matrix=matrix,principal_tangent_validation=tangent,inputs_unchanged=input_unchanged,
        comparisons=comparisons,candidate_repeat=own_repeat,reference_repeat=native_repeat,
        source_module_sha256=source,benchmark_sha256=sha(__file__),input_files_sha256={key:sha(path) for key,path in paths.items()},
        source_manifest=json.loads(source_manifest.read_text()),reference=reference,
        candidate_sha256=sha(output/'candidate.private.npz'),vertices=len(vertex),patch_vertices=len(before['selected_vertices']),
        fit_branches={str(kind):int(np.count_nonzero(candidate['fit_kind']==kind)) for kind in range(4)},
        seconds=dict(initial_input_load=loaded-started,first_sff_api=candidate['seconds'],repeat_sff_api=repeated['seconds'],
            first_prepared_five_api=own['seconds'],repeat_prepared_five_api=repeat_score['seconds'],candidate_npz_write=candidate_io,
            complete_script_before_report_write=time.perf_counter()-_ENTRY),
        process_rss_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        thread_budget=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),
        python=platform.python_version(),numpy=np.__version__,numba=numba.__version__,torch=torch.__version__,
        cuda_initialized=torch.cuda.is_initialized(),gpu_used=False,process_gpu_memory_bytes=None,
        scope='same-input candidate SFF and prepared scoring only; raw MRI/PDF preparation, retessellation and complete GA not assessed',
        overall_equivalence='not_assessed',source_component_speedup=None)
    (output/'report.private.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['neighborhood-capture','neighborhood-repeat','statistics-capture','statistics-repeat',
                 'volume-capture','distance-capture','brain','fnit-joint-pdf','reference-fitness',
                 'reference-repeat','source-manifest','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--post-neighborhood',type=Path)
    parser.add_argument('--threads',type=int,default=4)
    report=run(**vars(parser.parse_args()))
    if not report['stage_execution_gate_passed']:
        raise SystemExit(1)


if __name__=='__main__':main()
