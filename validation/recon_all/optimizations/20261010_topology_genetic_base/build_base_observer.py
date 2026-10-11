"""固定源码的初始 GA 基底只读 before/after 观察器；仅独立 benchmark。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CAPTURE = r'''
static void fnit_capture_genetic_base(MRIS* s, MRIS* source, DEFECT_LIST* dl,
                                     int* vtrans, int* ftrans, int phase) {
  const char* prefix = getenv("FNIT_TOPOLOGY_GENETIC_BASE_CAPTURE");
  static bool done[2] = {false, false};
  if (!prefix || done[phase]) return;
  done[phase] = true;
  char name[STRLEN];
  snprintf(name, sizeof(name), "%s.%s.bin", prefix, phase ? "after" : "before");
  FILE* out = fopen(name, "wb"); if (!out) abort();
  int en = 0, ei = 0, selected = 0;
  for (int v = 0; v < s->nvertices; v++) {
    en += s->vertices_topology[v].vnum;
    ei += s->vertices_topology[v].num;
  }
  for (int d = 0; d < dl->ndefects; d++) selected += dl->defects[d].nvertices;
  int h[18] = {s->nvertices,s->nfaces,en,ei,phase,source->nvertices,source->nfaces,
               s->max_vertices,s->max_faces,s->nsize,s->max_nsize,s->useRealRAS,
               s->hemisphere,s->type,s->status,s->origxyz_status,dl->ndefects,selected};
  fwrite("FNITBAS1",1,8,out); fwrite(h,sizeof(int),18,out);
  int vg[5] = {s->vg.valid,s->vg.width,s->vg.height,s->vg.depth,s->vg.ras_good_flag};
  double g[18] = {s->vg.xsize,s->vg.ysize,s->vg.zsize,
                 s->vg.x_r,s->vg.x_a,s->vg.x_s,s->vg.y_r,s->vg.y_a,s->vg.y_s,
                 s->vg.z_r,s->vg.z_a,s->vg.z_s,s->vg.c_r,s->vg.c_a,s->vg.c_s,
                 s->vg.s_r,s->vg.s_a,s->vg.s_s};
  fwrite(vg,sizeof(int),5,out); fwrite(g,sizeof(double),18,out);
  int namelen=(int)strlen(s->vg.fname); fwrite(&namelen,sizeof(int),1,out);
  fwrite(s->vg.fname,1,namelen,out);
  int caches[3]={s->vg.i_to_r__!=NULL,s->vg.r_to_i__!=NULL,s->vg.register_mat!=NULL};
  fwrite(caches,sizeof(int),3,out);
  if (s->vg.i_to_r__) fwrite(s->vg.i_to_r__->mat,sizeof(float),16,out);
  const MATRIX* matrices[2]={s->vg.r_to_i__,s->vg.register_mat};
  for (int k=0; k<2; k++) if (matrices[k]) {
    int dims[3]={matrices[k]->rows,matrices[k]->cols,matrices[k]->type};
    fwrite(dims,sizeof(int),3,out);
    for (int r=1;r<=dims[0];r++) for (int c=1;c<=dims[1];c++) {
      double value=*MATRIX_RELT(matrices[k],r,c); fwrite(&value,sizeof(double),1,out);
    }
  }
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX* p = &s->vertices[v];
    float a[42] = {p->x,p->y,p->z,p->origx,p->origy,p->origz,
        p->tx,p->ty,p->tz,p->cx,p->cy,p->cz,p->nx,p->ny,p->nz,
        p->val,p->val2,p->valbak,p->val2bak,p->imag_val,p->curv,p->curvbak,
        p->stat,p->mean,p->mean_imag,p->std_error,p->H,p->K,p->k1,p->k2,
        p->fx,p->fy,p->fz,p->e1x,p->e1y,p->e1z,p->e2x,p->e2y,p->e2z,
        p->area,(float)p->undefval,(float)p->old_undefval};
    fwrite(a,sizeof(float),42,out);
  }
  for (int v = 0; v < s->nvertices; v++) {
    int a[3] = {s->vertices[v].marked,s->vertices[v].ripflag,s->vertices[v].border};
    fwrite(a,sizeof(int),3,out);
  }
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX_TOPOLOGY* t = &s->vertices_topology[v];
    int a[7] = {t->nsizeCur,t->nsizeMax,t->vtotal,t->vnum,t->v2num,t->v3num,t->num};
    fwrite(a,sizeof(int),7,out);
  }
  for (int f = 0; f < s->nfaces; f++) {
    fwrite(s->faces[f].v,sizeof(int),3,out);
    int rip = s->faces[f].ripflag; fwrite(&rip,sizeof(int),1,out);
    fwrite(&s->faces[f].area,sizeof(float),1,out);
  }
  int offset = 0; fwrite(&offset,sizeof(int),1,out);
  for (int v = 0; v < s->nvertices; v++) {
    offset += s->vertices_topology[v].vnum; fwrite(&offset,sizeof(int),1,out);
  }
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX_TOPOLOGY* t = &s->vertices_topology[v];
    fwrite(t->v,sizeof(int),t->vnum,out);
  }
  offset = 0; fwrite(&offset,sizeof(int),1,out);
  for (int v = 0; v < s->nvertices; v++) {
    offset += s->vertices_topology[v].num; fwrite(&offset,sizeof(int),1,out);
  }
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX_TOPOLOGY* t = &s->vertices_topology[v];
    fwrite(t->f,sizeof(int),t->num,out);
    for (int k = 0; k < t->num; k++) { int c=t->n[k]; fwrite(&c,sizeof(int),1,out); }
  }
  int* labels=(int*)calloc(source->nvertices,sizeof(int));
  int* status=(int*)calloc(source->nvertices,sizeof(int));
  for (int d=0; d<dl->ndefects; d++) for (int k=0; k<dl->defects[d].nvertices; k++) {
    int v=dl->defects[d].vertices[k]; labels[v]=d+1; status[v]=dl->defects[d].status[k];
  }
  fwrite(labels,sizeof(int),source->nvertices,out);
  fwrite(status,sizeof(int),source->nvertices,out); free(labels); free(status);
  for (int d=0; d<dl->ndefects; d++) { int k=dl->defects[d].nvertices; fwrite(&k,sizeof(int),1,out); }
  for (int d=0; d<dl->ndefects; d++) fwrite(dl->defects[d].vertices,sizeof(int),dl->defects[d].nvertices,out);
  fwrite(vtrans,sizeof(int),source->nvertices,out); fwrite(ftrans,sizeof(int),source->nfaces,out);
  if (fclose(out)!=0) abort();
}
'''


def patch_source(text):
    """只在 fixed 三个唯一锚点插入读取，不重算指标或改写源状态。"""
    declaration = 'MRI_SURFACE *MRIScorrectTopology(\n'
    before = '  int newNVertices = 0;\n'
    after = '  mrisCheckVertexFaceTopology(mris);\n  \n  MRISclearMarks(mris);\n  MRISrestoreVertexPositions(mris, ORIGINAL_VERTICES);'
    if any(text.count(x) != 1 for x in [declaration,before,after]):
        raise ValueError('fixed GA base construction anchors changed')
    text=text.replace(declaration,CAPTURE+'\n'+declaration,1)
    text=text.replace(before,'  fnit_capture_genetic_base(mris,mris,dl,vertex_trans,face_trans,0);\n'+before,1)
    return text.replace(after,'  fnit_capture_genetic_base(mris_corrected,mris,dl,vertex_trans,face_trans,1);\n'+after,1)


def build(*, native_build, base_observer, output):
    """显式固定源码/完整 fitness observer→新 benchmark 程序和 private 收据。

    三路径具名传入；output 必须不存在。校验原 build receipt 对应源码/
    archive SHA 后复用原 Conda 编译 argv，仅替换自己的目标文件。捕获 raw
    TMP restore+metric 后的真正 source input，完整 copied attrs 与有序
    source 拓扑；after 只读真实新 base，禁止用 after 填补生产输入。
    不读取许可证内容、不发布源码 archive、不调用生产流程。异常抛出，
    build.log保留；构建完成不等于实际 reference/repeat 已完成。
    """
    native_build,base_observer,output=map(Path,(native_build,base_observer,output))
    if output.exists():raise FileExistsError('fresh GA base observer build required')
    receipt=json.loads((base_observer/'build.private.json').read_text())
    base,old_archive=base_observer/'mrisurf_defect.cpp',base_observer/'libutils.a'
    if sha(base)!=receipt['patched_defect_sha256'] or sha(old_archive)!=receipt['archive_sha256']:
        raise ValueError('frozen complete fitness observer source/archive changed')
    text=patch_source(base.read_text())
    output.mkdir(parents=True);patched=output/'mrisurf_defect.cpp';patched.write_text(text)
    command=list(receipt['compile_argv']);obj=output/'mrisurf_defect.cpp.o'
    for flag,value in [('-o',str(obj)),('-c',str(patched)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:
        command[command.index(flag)+1]=value
    archive=output/'libutils.a';shutil.copyfile(old_archive,archive)
    link=[str(archive) if x==str(old_archive) else x for x in receipt['link_argv']]
    binary=output/'mris_fix_topology_base_observer';link[link.index('-o')+1]=str(binary)
    started=time.perf_counter();compiler=Path(command[0])
    with (output/'build.log').open('w') as log:
        subprocess.run(command,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')),'r',str(archive),str(obj)],check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')),str(archive)],check=True)
        subprocess.run(link,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(status='complete_isolated_genetic_base_observer_build',
        scope='actual before/after ordered GA initial base; not candidate search or full GA',
        base_receipt_sha256=sha(base_observer/'build.private.json'),builder_sha256=sha(__file__),
        original_patched_defect_sha256=sha(base),patched_defect_sha256=sha(patched),
        archive_sha256=sha(archive),observer_program_sha256=sha(binary),
        compile_argv=command,link_argv=link,seconds=time.perf_counter()-started,
        format='FNITBAS1 header18; VG I5/D18/fname bytes/cache flags plus cache arrays; verticesF42; flagsI3; topologyI7; faces I4/F1; one-ring CSR; incident CSR; labels/status/source ordered defects/maps',
        after_inputs_consumed=False,face_normals_not_evaluated_by_observer=True)
    (output/'build.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    build(**vars(parser.parse_args()))


if __name__=='__main__':main()
