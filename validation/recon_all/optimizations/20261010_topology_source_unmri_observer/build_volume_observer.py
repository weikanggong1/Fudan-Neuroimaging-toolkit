"""隔离构建首个真实缺陷 unMRI 聚合的固定源码观察器，仅供 benchmark。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


BEFORE = r'''
  static bool fnit_volume_captured=false;
  FILE* fnit_volume_capture=NULL;
  const char* fnit_volume_path=getenv("FNIT_TOPOLOGY_VOLUME_CAPTURE");
  if(fnit_volume_path && !fnit_volume_captured) {
    fnit_volume_captured=true;
    fnit_volume_capture=fopen(fnit_volume_path,"wb");
    if(!fnit_volume_capture) abort();
    auto write=[&](const void* data,size_t size,size_t count) {
      if(fwrite(data,size,count,fnit_volume_capture)!=count) abort();
    };
    const char magic[8]={'F','N','I','T','V','O','L','1'};
    write(magic,1,8);
    int header[5]={mris->nvertices,mris->nfaces,dp_defect->nvertices,dp->tp.nvertices,dp->tp.nfaces};
    write(header,sizeof(int),5);
    float means[5]={dp_defect->white_mean_ll,dp_defect->gray_mean_ll,
      dp_defect->white_mean,dp_defect->gray_mean,(float)NPY};
    write(means,sizeof(float),5);
    for(int vno=0;vno<mris->nvertices;vno++) {
      const VERTEX* v=&mris->vertices[vno];
      float fields[13]={v->origx,v->origy,v->origz,v->nx,v->ny,v->nz,
        v->fx,v->fy,v->fz,v->curvbak,v->area,(float)v->marked,(float)v->ripflag};
      write(fields,sizeof(float),13);
    }
    for(int fno=0;fno<mris->nfaces;fno++) write(mris->faces[fno].v,sizeof(int),3);
    for(int fno=0;fno<mris->nfaces;fno++) {
      const FaceNormCacheEntry* normal=getFaceNorm(mris_nonconst,fno);
      float fields[4]={normal->nx,normal->ny,normal->nz,mris->faces[fno].area};
      write(fields,sizeof(float),4);
    }
    for(int p=0;p<dp_defect->nvertices;p++) {
      int fields[3]={dp_defect->vertices[p],dp_defect->vertex_trans[dp_defect->vertices[p]],dp_defect->status[p]};
      write(fields,sizeof(int),3);
    }
    write(dp->tp.vertices,sizeof(int),dp->tp.nvertices);
    write(dp->tp.faces,sizeof(int),dp->tp.nfaces);
    MRI const* volumes[4]={mri_defect,dp->mri_defect_white,dp->mri_defect_gray,mri_distance};
    for(int n=0;n<4;n++) {
      MRI const* volume=volumes[n];
      int dims[4]={volume->width,volume->height,volume->depth,volume->type};
      float geometry[9]={volume->xstart,volume->ystart,volume->zstart,
        volume->xsize,volume->ysize,volume->zsize,volume->xend,volume->yend,volume->zend};
      write(dims,sizeof(int),4);write(geometry,sizeof(float),9);
      for(int x=0;x<volume->width;x++) for(int y=0;y<volume->height;y++) for(int z=0;z<volume->depth;z++) {
        float value=(n==0)?MRIvox(volume,x,y,z):MRIFvox(volume,x,y,z);
        write(&value,sizeof(float),1);
      }
    }
    if(fflush(fnit_volume_capture)!=0) abort();
  }
'''

AFTER = r'''
  if(fnit_volume_capture) {
    float results[8]={white_ll,gray_ll,int_w,int_g,max_distance,
      dp_nonconst->tp.unmri_ll,(float)nwhite,(float)ngray};
    if(fwrite(results,sizeof(float),8,fnit_volume_capture)!=8) abort();
    for(int vno=0;vno<mris->nvertices;vno++) {
      float curvbak=mris->vertices[vno].curvbak;
      if(fwrite(&curvbak,sizeof(float),1,fnit_volume_capture)!=1) abort();
    }
    if(fclose(fnit_volume_capture)!=0) abort();
    fprintf(stderr,"FNIT_TOPOLOGY_VOLUME_CAPTURE_COMPLETE_STOP_BEFORE_GA\n");
    exit(0); // intentional benchmark checkpoint, no production topology output
  }
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, native_build, base_observer, output):
    """只读 observer 再构建；新目录、完整命令和源码/程序哈希绑定。"""
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError('new isolated benchmark build required')
    receipt=json.loads((base_observer/'build.private.json').read_text())
    base,old_archive=base_observer/'mrisurf_defect.cpp',base_observer/'libutils.a'
    if sha(base)!=receipt['patched_defect_sha256'] or sha(old_archive)!=receipt['archive_sha256']:
        raise ValueError('frozen complete-statistics observer source/archive changed')
    text=base.read_text()
    before='  float white_ll = dp_defect->white_mean_ll;\n'
    after='  dp_nonconst->tp.unmri_ll = (white_ll + gray_ll);\t// MODIFIER\n'
    if text.count(before)!=1 or text.count(after)!=1:
        raise ValueError('fixed unMRI aggregation capture anchors changed')
    output.mkdir(parents=True)
    patched=output/'mrisurf_defect.cpp'
    patched.write_text(text.replace(before,BEFORE+before,1).replace(after,after+AFTER,1))
    command=list(receipt['compile_argv']);obj=output/'mrisurf_defect.cpp.o'
    for flag,value in [('-o',str(obj)),('-c',str(patched)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:
        command[command.index(flag)+1]=value
    archive=output/'libutils.a';shutil.copyfile(old_archive,archive)
    link=[str(archive) if token==str(old_archive) else token for token in receipt['link_argv']]
    binary=output/'mris_fix_topology_volume_observer';link[link.index('-o')+1]=str(binary)
    started=time.perf_counter();compiler=Path(command[0])
    with (output/'build.log').open('w') as log:
        subprocess.run(command,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')),'r',str(archive),str(obj)],check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')),str(archive)],check=True)
        subprocess.run(link,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(status='complete_isolated_volume_observer_build',scope='first actual source unMRI aggregation; intentional stop, not complete GA',
        base_receipt_sha256=sha(base_observer/'build.private.json'),observer_builder_sha256=sha(__file__),
        patched_defect_sha256=sha(patched),archive_sha256=sha(archive),observer_program_sha256=sha(binary),
        compile_argv=command,link_argv=link,seconds=time.perf_counter()-started,
        binary_format='FNITVOL1; int32[5]; float32[5] means; vertices(N,13); faces(F,3); face fields(F,4); source/trans/status(K,3); patch V/F; four dimensions[4]/geometry[9]/float32 volumes; result[8]; curvbak(N)',
        vertex_fields=['origx','origy','origz','nx','ny','nz','fx','fy','fz','curvbak','area','marked','rip'],
        volume_order=['uint8 defect intensity represented F32','white log-P','gray log-P','calculated signed distance'],
        result_fields=['white_ll','gray_ll','int_w','int_g','max_distance','unmri_ll','nwhite','ngray'])
    (output/'build.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    build(**vars(parser.parse_args()))


if __name__=='__main__':
    main()
