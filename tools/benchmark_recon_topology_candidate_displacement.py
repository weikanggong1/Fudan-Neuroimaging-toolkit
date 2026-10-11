"""完整真实缺陷域位移→自产unMRI的冻结同输入回归；不执行GA。"""
from __future__ import annotations
import time
_START=time.perf_counter()
import argparse,hashlib,json,os,platform
from pathlib import Path
import numpy as np
from benchmark_recon_topology_unmri_fields import load_volume_capture
from benchmark_recon_topology_volume_likelihood import arguments
from fnit.recon_all import topology_candidate_displacement as implementation
from fnit.recon_all.topology_defect_volume_likelihood import defect_volume_likelihood
from fnit.recon_all.thread_budget import thread_budget
_IMPORT=time.perf_counter()-_START


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return dict(shape_equal=False,bitwise_equal=False)
    error=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(shape_equal=True,different_elements=int(np.count_nonzero(a!=b)),bitwise_equal=bool(a.dtype==b.dtype and a.tobytes()==b.tobytes()),max_abs=float(error.max(initial=0)),p99_abs=float(np.percentile(error,99)) if error.size else 0.)


def run(*,capture,native_repeat,observer_receipt,output,threads):
    """FNITVOL1原orig/fx、完整映射/status→自产位移和完整体积反馈。

    capture/native_repeat是独立冻结参考，observer_receipt绑定程序/输入；
    output须不存在，threads正整数。全部未丢弃缺陷均求位移，参考
    active curvbak不入求解。以自产active值替换冻结的untouched旧状态
    后调用已有完整unMRI；其distance/MRI/PDF仍是冻结阶段输入，不能
    宣称原始MRI→全候选自产。输出私有NPZ、标准JSON与真实误差PNG，
    不写生产目录或启动原程序。异常输入/重复不稳定/格式错报错。
    """
    if output.exists():raise FileExistsError('new isolated diagnostic output required')
    if not isinstance(threads,int) or threads<1:raise ValueError('positive thread budget required')
    started=time.perf_counter();f=load_volume_capture(capture_path=capture);repeat=load_volume_capture(capture_path=native_repeat)
    vertex,defect=f['vertex_fields'],f['defect_vertices']
    args=dict(original_vertices=vertex[:,:3],prematch_positions_mm=vertex[:,6:9],translated_defect_vertices=defect[:,1],discarded_defect_vertices=defect[:,2]==1)
    with thread_budget(threads=threads):
        tick=time.perf_counter();a=implementation.candidate_displacement(**args);first=time.perf_counter()-tick
        tick=time.perf_counter();b=implementation.candidate_displacement(**args);warm=time.perf_counter()-tick
        # Only untouched state is frozen; active reference displacement is overwritten.
        initial=vertex[:,9].copy();initial[a['vertex_indices']]=a['displacement_mm']
        volume_args=arguments(f);volume_args['initial_vertex_curvbak']=initial
        tick=time.perf_counter();volume=defect_volume_likelihood(**volume_args);downstream=time.perf_counter()-tick
    active=a['vertex_indices'];names=['white_ll','gray_ll','white_mean_intensity','gray_mean_intensity','max_distance_mm','unmri_ll','nwhite','ngray']
    score=np.asarray([volume[name] for name in names],np.float32)
    comparisons=dict(all_active_displacement=difference(a['displacement_mm'],vertex[active,9]),downstream_eight_fields=difference(score,f['reference_results']),downstream_full_curvbak=difference(volume['vertex_curvbak'],f['reference_curvbak']))
    self_repeat={key:difference(value,b[key]) for key,value in a.items()}
    native={key:difference(f[key],repeat[key]) for key in ['vertex_fields','defect_vertices','reference_results','reference_curvbak']}
    output.mkdir(parents=True);np.savez_compressed(output/'candidate.private.npz',**a,downstream_scores=score,downstream_full_curvbak=volume['vertex_curvbak'])
    report=dict(status='complete_frozen_candidate_displacement_regression',strict_active_displacement_passed=comparisons['all_active_displacement']['bitwise_equal'],strict_downstream_passed=all(comparisons[name]['bitwise_equal'] for name in ['downstream_eight_fields','downstream_full_curvbak']),candidate_repeat_exact=all(row['bitwise_equal'] for row in self_repeat.values()),native_selected_fields_repeat_exact=all(row['bitwise_equal'] for row in native.values()),comparisons=comparisons,candidate_repeat=self_repeat,native_repeat=native,
        complete_defect_entries=len(defect),active_entries=len(active),discarded_entries=int(np.count_nonzero(defect[:,2]==1)),patch_vertices=len(f['patch_vertices']),full_vertices=len(vertex),
        seconds=dict(top_level_imports=_IMPORT,first_complete_displacement_api=first,repeat_complete_displacement_api=warm,downstream_complete_unmri_api=downstream),
        module_sha256=sha(implementation.__file__),benchmark_sha256=sha(__file__),source_inputs_sha256=dict(capture=sha(capture),native_repeat=sha(native_repeat),observer_receipt=sha(observer_receipt)),observer_receipt=json.loads(observer_receipt.read_text()),candidate_npz_sha256=sha(output/'candidate.private.npz'),cpu_threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),python=platform.python_version(),numpy=np.__version__,gpu_used=False,process_gpu_memory_bytes=None,native_component_seconds=None,original_component_speedup=None,
        scope='complete non-discarded mapped defect displacement plus frozen-input unMRI propagation; no upstream smooth/match/distance/MRI generation or complete GA',overall_equivalence='not_assessed')
    _plot(vertex,active,a['displacement_mm'],output/'displacement_error.png');report['figure_sha256']=sha(output/'displacement_error.png')
    report['seconds']['api_with_load_npz_and_png_io_before_report_write']=time.perf_counter()-started;report['seconds']['script_wall_before_report_write']=time.perf_counter()-_START
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report


def _plot(vertex,active,displacement,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    error=displacement.astype(np.float64)-vertex[active,9].astype(np.float64)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,value,title in [(axes[0],displacement,'Matched displacement (mm)'),(axes[1],error,'FNIT minus source (mm)')]:
        p=ax.scatter(vertex[active,1],vertex[active,2],c=value,s=12,cmap='viridis');ax.set(xlabel='surface RAS y (mm)',ylabel='surface RAS z (mm)',title=title);ax.set_aspect('equal');fig.colorbar(p,ax=ax)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','native-repeat','observer-receipt','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4);report=run(**vars(parser.parse_args()))
    if not all(report[key] for key in ['strict_active_displacement_passed','strict_downstream_passed','candidate_repeat_exact','native_selected_fields_repeat_exact']):raise SystemExit(1)


if __name__=='__main__':main()
