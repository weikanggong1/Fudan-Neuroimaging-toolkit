"""观察首次完整 patch fitness，固定源码仅供隔离 benchmark，完成后有意停止。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from build_volume_observer import BEFORE, AFTER, sha
from build_statistics_observer import OBSERVE


FITNESS = r'''
  static bool fnit_full_fitness_captured=false;
  const char* fnit_fitness_path=getenv("FNIT_TOPOLOGY_FITNESS_CAPTURE");
  if(fnit_fitness_path && !fnit_full_fitness_captured) {
    fnit_full_fitness_captured=true;
    FILE* report=fopen(fnit_fitness_path,"w");
    if(!report) abort();
    float fields[12]={dp->tp.face_ll,dp->tp.vertex_ll,dp->tp.curv_ll,dp->tp.qcurv_ll,
      dp->tp.mri_ll,dp->tp.unmri_ll,dp->tp.fll,dp->tp.vll,dp->tp.cll,dp->tp.qcll,
      dp->tp.mrill,dp->tp.unmrill};
    double weights[4]={l_mri,fnit_effective_l_unmri,l_qcurv,l_curv};
    double parameters[4]={parms->l_mri,parms->l_unmri,parms->l_qcurv,parms->l_curv};
    fprintf(report,"{\"tp_fields\":[");
    for(int i=0;i<12;i++) {
      fprintf(report,"%s",i?",":"");
      if((double)fields[i]<=1e300 && (double)fields[i]>=-1e300) fprintf(report,"%.17g",(double)fields[i]);
      else fprintf(report,"null"); // unused source diagnostic fields can be NaN
    }
    fprintf(report,"],\"tp_field_uint32_bits\":[");
    for(int i=0;i<12;i++) {
      unsigned int bits;memmove(&bits,&fields[i],sizeof(float));
      fprintf(report,"%s%u",i?",":"",bits);
    }
    fprintf(report,"],\"effective_weights\":[");
    for(int i=0;i<4;i++) fprintf(report,"%s%.17g",i?",":"",weights[i]);
    fprintf(report,"],\"parameter_weights\":[");
    for(int i=0;i<4;i++) fprintf(report,"%s%.17g",i?",":"",parameters[i]);
    fprintf(report,"],\"tp_field_sizes\":[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d],\"parameter_weight_sizes\":[%d,%d,%d,%d],\"FLT_EPSILON\":%.17g,",
      (int)sizeof(dp->tp.face_ll),(int)sizeof(dp->tp.vertex_ll),(int)sizeof(dp->tp.curv_ll),(int)sizeof(dp->tp.qcurv_ll),
      (int)sizeof(dp->tp.mri_ll),(int)sizeof(dp->tp.unmri_ll),(int)sizeof(dp->tp.fll),(int)sizeof(dp->tp.vll),
      (int)sizeof(dp->tp.cll),(int)sizeof(dp->tp.qcll),(int)sizeof(dp->tp.mrill),(int)sizeof(dp->tp.unmrill),
      (int)sizeof(parms->l_mri),(int)sizeof(parms->l_unmri),(int)sizeof(parms->l_qcurv),(int)sizeof(parms->l_curv),(double)FLT_EPSILON);
    fprintf(report,"\"face_check_result\":%d,\"fitness\":%.17g,\"unmri_capture_calls\":%d,\"patch_vertices\":%d,\"patch_faces\":%d}\n",
      fnit_face_check_result,ll,fnit_unmri_capture_calls,dp->tp.nvertices,dp->tp.nfaces);
    if(fclose(report)!=0) abort();
    fprintf(stderr,"FNIT_TOPOLOGY_FULL_FITNESS_CAPTURE_STOP_BEFORE_CONTROLLER\n");
    exit(0); // benchmark stop only, no production topology result
  }
'''


def build(*,native_build,base_observer,output):
    """初始stats观察器→同候选MRI/体积/曲率/法向与最终合成的只读检查点。"""
    native_build,base_observer,output=map(Path,(native_build,base_observer,output))
    if output.exists():raise FileExistsError('fresh benchmark build required')
    receipt=json.loads((base_observer/'build.private.json').read_text())
    base,old_archive=base_observer/'mrisurf_defect.cpp',base_observer/'libutils.a'
    if sha(base)!=receipt['patched_defect_sha256'] or sha(old_archive)!=receipt['archive_sha256']:
        raise ValueError('frozen initial statistics observer source changed')
    text=base.read_text()
    before='  float white_ll = dp_defect->white_mean_ll;\n'
    after='  dp_nonconst->tp.unmri_ll = (white_ll + gray_ll);\t// MODIFIER\n'
    fit='  l_unmri = parms->l_unmri;\n  if (mrisCheckDefectFaces(mris, dp) < 0) ll -= 10000000;\n'
    if any(text.count(anchor)!=1 for anchor in (before,after,fit)):
        raise ValueError('fixed source full fitness anchors changed')
    # 记录每次unMRI，首个完整fitness结束前的最后一份距离/反馈属于同一候选。
    volume_before=BEFORE.replace('if(fnit_volume_path && !fnit_volume_captured)',
        'if(fnit_volume_path)').replace('fnit_volume_captured=true;',
        'fnit_volume_captured=true; fnit_unmri_capture_calls++;')
    volume_after=AFTER.replace('    exit(0); // intentional benchmark checkpoint, no production topology output\n','')
    patch_stats=OBSERVE.replace('FNIT_TOPOLOGY_STATISTICS_CAPTURE','FNIT_TOPOLOGY_PATCH_STATISTICS_CAPTURE').replace('tp.','dp->tp.')
    patch_stats=patch_stats.replace('    #if MATRIX_ALLOCATION',
        '    MATRIX* VoxelFromSRASmatrix=GetSurfaceRASToVoxelMatrix(mri);\n    #if MATRIX_ALLOCATION')
    patch_stats=patch_stats.replace('    double scores[8]',
        '    MatrixFree(&VoxelFromSRASmatrix);\n    double scores[8]')
    fit_replacement='  double fnit_effective_l_unmri=l_unmri;\n  l_unmri = parms->l_unmri;\n  int fnit_face_check_result=mrisCheckDefectFaces(mris, dp);\n  if (fnit_face_check_result < 0) ll -= 10000000;\n'+patch_stats+FITNESS
    text='static int fnit_unmri_capture_calls=0;\n'+text.replace(before,volume_before+before,1).replace(after,after+volume_after,1).replace(fit,fit_replacement,1)
    output.mkdir(parents=True);patched=output/'mrisurf_defect.cpp';patched.write_text(text)
    command=list(receipt['compile_argv']);obj=output/'mrisurf_defect.cpp.o'
    for flag,value in [('-o',str(obj)),('-c',str(patched)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:
        command[command.index(flag)+1]=value
    archive=output/'libutils.a';shutil.copyfile(old_archive,archive)
    link=[str(archive) if x==str(old_archive) else x for x in receipt['link_argv']]
    binary=output/'mris_fix_topology_fitness_observer';link[link.index('-o')+1]=str(binary)
    started=time.perf_counter();compiler=Path(command[0])
    with (output/'build.log').open('w') as log:
        subprocess.run(command,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')),'r',str(archive),str(obj)],check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')),str(archive)],check=True)
        subprocess.run(link,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(status='complete_readonly_full_fitness_observer_build',scope='first complete patch fitness, not controller/GA completion',
        base_receipt_sha256=sha(base_observer/'build.private.json'),observer_builder_sha256=sha(__file__),
        auxiliary_builder_sha256={p.name:sha(p) for p in [Path(__file__).with_name('build_volume_observer.py'),Path(__file__).with_name('build_statistics_observer.py')]},
        patched_defect_sha256=sha(patched),archive_sha256=sha(archive),observer_program_sha256=sha(binary),
        compile_argv=command,link_argv=link,seconds=time.perf_counter()-started,
        tp_field_names=['face_ll','vertex_ll','curv_ll','qcurv_ll','mri_ll','unmri_ll','fll','vll','cll','qcll','mrill','unmrill'],
        weight_names=['l_mri','l_unmri','l_qcurv','l_curv'],
        volume_capture='last call before first complete fitness; same candidate, not initial statistics',
        patch_statistics_format='existing FNITFIT1, complete same-candidate geometry/normals/PDF/curv/CSR')
    (output/'build.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:p.add_argument('--'+name,type=Path,required=True)
    build(**vars(p.parse_args()))


if __name__=='__main__':main()
