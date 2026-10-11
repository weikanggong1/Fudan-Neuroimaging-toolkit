"""固定源码真实首次有向距离检查点；只在独立 benchmark 编译和运行。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


BEFORE_DISTANCE = r'''
  static bool fnit_distance_seen=false;
  FILE* fnit_distance_capture=NULL;
  const char* fnit_distance_path=getenv("FNIT_TOPOLOGY_DISTANCE_CAPTURE");
  if(fnit_distance_path && !fnit_distance_seen) {
    fnit_distance_seen=true;
    fnit_distance_capture=fopen(fnit_distance_path,"wb");
    if(!fnit_distance_capture) abort();
    auto write=[&](const void* data,size_t size,size_t count) {
      if(fwrite(data,size,count,fnit_distance_capture)!=count) abort();
    };
    const char magic[8]={'F','N','I','T','D','S','T','2'};
    write(magic,1,8);
    int incidents=0;
    for(int vno=0;vno<mris->nvertices;vno++) incidents+=mris->vertices_topology[vno].num;
    if(mris->nvertices<1) abort();
    unsigned char* mentioned=(unsigned char*)calloc((size_t)(unsigned int)mris->nvertices,1);
    if(!mentioned) abort();
    for(int p=0;p<bufferSize;p++) for(int c=0;c<3;c++) mentioned[buffer[p].face->v[c]]=1;
    int queried=0;
    for(int vno=0;vno<mris->nvertices;vno++) queried+=mentioned[vno]!=0;
    int header[10]={mris->nvertices,mris->nfaces,incidents,bufferSize,queried,
      mri_distance->width,mri_distance->height,mri_distance->depth,mri_distance->type,maxThreads};
    write(header,sizeof(int),10);
    float geometry[9]={mri_distance->xstart,mri_distance->ystart,mri_distance->zstart,
      mri_distance->xsize,mri_distance->ysize,mri_distance->zsize,
      mri_distance->xend,mri_distance->yend,mri_distance->zend};
    write(geometry,sizeof(float),9);
    int algorithm_dims[4]={mri_defect->width,mri_defect->height,mri_defect->depth,mri_defect->type};
    float algorithm_geometry[9]={mri_defect->xstart,mri_defect->ystart,mri_defect->zstart,
      mri_defect->xsize,mri_defect->ysize,mri_defect->zsize,
      mri_defect->xend,mri_defect->yend,mri_defect->zend};
    write(algorithm_dims,sizeof(int),4);write(algorithm_geometry,sizeof(float),9);
    int settings[5]={(int)use_sharedVertexPseudoNormalCache,(int)do_new_MRIDistance,
      (int)do_new_loop3,delta,(int)sizeof(dp->fitness)};
    write(settings,sizeof(int),5);
    for(int vno=0;vno<mris->nvertices;vno++) {
      VERTEX const* v=&mris->vertices[vno];
      float fields[6]={v->origx,v->origy,v->origz,v->x,v->y,v->z};
      write(fields,sizeof(float),6);
    }
    for(int vno=0;vno<mris->nvertices;vno++) {
      int flag=mris->vertices[vno].ripflag;write(&flag,sizeof(int),1);
    }
    for(int fno=0;fno<mris->nfaces;fno++) write(mris->faces[fno].v,sizeof(int),3);
    for(int fno=0;fno<mris->nfaces;fno++) {
      FaceNormCacheEntry const* normal=getFaceNorm(mris_nonconst,fno);
      float fields[3]={normal->nx,normal->ny,normal->nz};write(fields,sizeof(float),3);
    }
    for(int fno=0;fno<mris->nfaces;fno++) {
      int flag=mris->faces[fno].ripflag;write(&flag,sizeof(int),1);
    }
    int offset=0;write(&offset,sizeof(int),1);
    for(int vno=0;vno<mris->nvertices;vno++) {
      offset+=mris->vertices_topology[vno].num;write(&offset,sizeof(int),1);
    }
    for(int vno=0;vno<mris->nvertices;vno++) {
      VERTEX_TOPOLOGY const* vt=&mris->vertices_topology[vno];write(vt->f,sizeof(int),vt->num);
    }
    for(int vno=0;vno<mris->nvertices;vno++) {
      VERTEX_TOPOLOGY const* vt=&mris->vertices_topology[vno];
      for(int n=0;n<vt->num;n++) {int corner=vt->n[n];write(&corner,sizeof(int),1);}
    }
    for(int p=0;p<bufferSize;p++) {
      Entry const* e=&buffer[p];
      int fields[7]={e->fno,e->imin_nobnd,e->imax_nobnd,e->jmin_nobnd,e->jmax_nobnd,e->kmin_nobnd,e->kmax_nobnd};
      write(fields,sizeof(int),7);
    }
    for(int p=0;p<bufferSize;p++) {
      Entry const* e=&buffer[p];
      float fields[9]={e->x0,e->x1,e->x2,e->y0,e->y1,e->y2,e->z0,e->z1,e->z2};
      write(fields,sizeof(float),9);
    }
    for(int p=0;p<bufferSize;p++) {
      Entry const* e=&buffer[p];FACE const* face=e->face;
      int fields[3]={findOtherEdgeFace(mris,e->fno,face->v[0],face->v[1]),
        findOtherEdgeFace(mris,e->fno,face->v[1],face->v[2]),findOtherEdgeFace(mris,e->fno,face->v[2],face->v[0])};
      write(fields,sizeof(int),3);
    }
    for(int vno=0;vno<mris->nvertices;vno++) if(mentioned[vno]) write(&vno,sizeof(int),1);
    for(int vno=0;vno<mris->nvertices;vno++) if(mentioned[vno]) {
      float norm[3];
      if(use_sharedVertexPseudoNormalCache) memmove(norm,sharedVertexPseudoNormalCache[vno].elts,sizeof(norm));
      else computeVertexPseudoNormal(mris,vno,norm,dp->verbose_mode);
      write(norm,sizeof(float),3);
    }
    for(int i=0;i<mri_distance->width;i++) for(int j=0;j<mri_distance->height;j++) for(int k=0;k<mri_distance->depth;k++) {
      float distance=MRIFvox(mri_distance,i,j,k);write(&distance,sizeof(float),1);
    }
    free(mentioned);
    if(fflush(fnit_distance_capture)) abort();
  }
'''

AFTER_DISTANCE = r'''
  if(fnit_distance_capture) {
    for(int i=0;i<mri_distance->width;i++) for(int j=0;j<mri_distance->height;j++) for(int k=0;k<mri_distance->depth;k++) {
      float distance=MRIFvox(mri_distance,i,j,k);
      if(fwrite(&distance,sizeof(float),1,fnit_distance_capture)!=1) abort();
    }
    if(fclose(fnit_distance_capture)) abort();
    fprintf(stderr,"FNIT_TOPOLOGY_DISTANCE_CAPTURE_COMPLETE_STOP_BEFORE_VOLUME_FITNESS\n");
    exit(0); // isolated benchmark stop; no production repair/output
  }
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, native_build, base_observer, output):
    """复用已验证完整 fitness observer；新增真实距离前后捕获、独立编译。

    输入 native_build 为固定 reference 源码的构建树，base_observer 为
    complete-firstfit observer 源码/archive/命令收据，output 必须不存在。
    输出 binary/build.private.json/build.log 与编译后的独立 archive。
    观察器只用于隔离真实数据算子回归，不能安装成 FNIT 生产程序。
    只增加检查点读写与在首次距离完成后退出，不改变原运算、参数、
    bbox/thread reduction/查询顺序。原 OMP buffer 顺序原样保存。
    """
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError('new isolated distance benchmark directory required')
    receipt = json.loads((base_observer / 'build.private.json').read_text())
    base, old_archive = base_observer / 'mrisurf_defect.cpp', base_observer / 'libutils.a'
    if sha(base) != receipt['patched_defect_sha256'] or sha(old_archive) != receipt['archive_sha256']:
        raise ValueError('complete first-fitness observer source/archive changed')
    text = base.read_text()
    before = '  ROMP_PF_begin\n\n  int bufferIndex;\n'
    after = '  freeAndNULL(sharedVertexPseudoNormalCache);\n'
    if text.count(before) != 1 or text.count(after) != 1:
        raise ValueError('fixed active distance-loop anchors changed')
    output.mkdir(parents=True)
    patched = output / 'mrisurf_defect.cpp'
    patched.write_text(text.replace(before, BEFORE_DISTANCE + before, 1).replace(after, AFTER_DISTANCE + after, 1))
    command = list(receipt['compile_argv'])
    obj = output / 'mrisurf_defect.cpp.o'
    for flag, value in [('-o', str(obj)), ('-c', str(patched)), ('-MF', str(obj) + '.d'), ('-MT', str(obj))]:
        command[command.index(flag) + 1] = value
    archive = output / 'libutils.a'
    shutil.copyfile(old_archive, archive)
    link = [str(archive) if x == str(old_archive) else x for x in receipt['link_argv']]
    binary = output / 'mris_fix_topology_distance_observer'
    link[link.index('-o') + 1] = str(binary)
    started = time.perf_counter()
    compiler = Path(command[0])
    with (output / 'build.log').open('w') as log:
        subprocess.run(command, cwd=native_build / 'build', stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')), 'r', str(archive), str(obj)], check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')), str(archive)], check=True)
        subprocess.run(link, cwd=native_build / 'build', stdout=log, stderr=subprocess.STDOUT, check=True)
    source_root = native_build / 'source'
    result = dict(status='complete_readonly_distance_observer_build',
        scope='first actual candidate distance; stop before volume aggregation/controller/GA completion',
        base_receipt_sha256=sha(base_observer / 'build.private.json'), observer_builder_sha256=sha(__file__),
        patched_defect_sha256=sha(patched), archive_sha256=sha(archive), observer_program_sha256=sha(binary),
        compile_argv=command, link_argv=link, seconds=time.perf_counter() - started,
        fixed_dependency_sha256={str(p.relative_to(source_root)): sha(p) for p in [
            source_root / 'utils/mrisurf_metricProperties.cpp', source_root / 'include/mrisurf_base.h',
            source_root / 'include/topo_parms.h', source_root / 'utils/mrisurf_defect.h']},
        binary_format='FNITDST2; int32[10]; float32[9] storage geometry; int32[4] actual mri_defect dimensions; float32[9] actual mri_defect geometry; int32[5] settings; orig/currentXYZ(N,6); vertex rip(N); faces(F,3); normals(F,3); face rip(F); incident CSR/corners; buffer bounds(B,7)/entryXYZ(B,9)/otherFace(B,3); queryIDs(Q)/actual pseudo(Q,3); before+after XYZ distance volumes')
    (output / 'build.private.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-build', 'base-observer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
