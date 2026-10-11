"""复用已有固定构建框架，只读记录真实 DVS 保存和恢复；仅benchmark。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CAPTURE = r'''
static DVS* fnit_dvs_first = NULL;
static void fnit_capture_dvs_state(MRIS const* s, DVS const* dvs, int phase) {
  const char* base = getenv("FNIT_TOPOLOGY_CANDIDATE_STATE_CAPTURE");
  static bool done[5] = {false,false,false,false,false};
  if (!base || dvs != fnit_dvs_first || done[phase]) return;
  done[phase] = true;
  char name[STRLEN];
  const char* suffix[5] = {"record","before_restore","restored","before_faces","faces"};
  snprintf(name,sizeof(name),"%s.%s.bin",base,suffix[phase]);
  FILE* f = fopen(name,"wb"); if (!f) abort();
  DEFECT const* d = dvs->defect;
  int entries=0,incidents=0,map_count=0;
  for(int v=0;v<s->nvertices;v++) {
    entries += mrisVertexVSize(s,v); incidents += s->vertices_topology[v].num;
  }
  for(int i=0;i<d->nvertices;i++) if(d->vertices[i]>=map_count) map_count=d->vertices[i]+1;
  for(int i=0;i<d->nchull;i++) if(d->chull[i]>=map_count) map_count=d->chull[i]+1;
  for(int i=0;i<d->nborder;i++) if(d->border[i]>=map_count) map_count=d->border[i]+1;
  int header[9]={s->nvertices,s->nfaces,entries,incidents,d->nvertices,d->nchull,d->nborder,map_count,phase};
  fwrite("FNITDVS1",1,8,f);fwrite(header,sizeof(int),9,f);
  for(int v=0;v<s->nvertices;v++) {
    VERTEX const* p=&s->vertices[v];
    float values[27]={p->origx,p->origy,p->origz,p->x,p->y,p->z,
      p->fx,p->fy,p->fz,p->nx,p->ny,p->nz,p->e1x,p->e1y,p->e1z,
      p->e2x,p->e2y,p->e2z,p->k1,p->k2,p->area,p->val,p->val2,
      p->val2bak,p->curvbak,(float)p->undefval,(float)p->old_undefval};
    fwrite(values,sizeof(float),27,f);
  }
  for(int v=0;v<s->nvertices;v++) {
    int flags[2]={s->vertices[v].marked,s->vertices[v].ripflag};fwrite(flags,sizeof(int),2,f);
  }
  for(int v=0;v<s->nvertices;v++) {
    VERTEX_TOPOLOGY const* t=&s->vertices_topology[v];
    int values[7]={t->nsizeCur,t->nsizeMax,t->vtotal,t->vnum,t->v2num,t->v3num,t->num};
    fwrite(values,sizeof(int),7,f);
  }
  for(int i=0;i<s->nfaces;i++) {
    fwrite(s->faces[i].v,sizeof(int),3,f);
    int rip=s->faces[i].ripflag;fwrite(&rip,sizeof(int),1,f);
  }
  int offset=0;fwrite(&offset,sizeof(int),1,f);
  for(int v=0;v<s->nvertices;v++) {offset+=mrisVertexVSize(s,v);fwrite(&offset,sizeof(int),1,f);}
  for(int v=0;v<s->nvertices;v++) {
    int count=mrisVertexVSize(s,v);
    if(count && !s->vertices_topology[v].v) abort();
    fwrite(s->vertices_topology[v].v,sizeof(int),count,f);
  }
  offset=0;fwrite(&offset,sizeof(int),1,f);
  for(int v=0;v<s->nvertices;v++) {offset+=s->vertices_topology[v].num;fwrite(&offset,sizeof(int),1,f);}
  for(int v=0;v<s->nvertices;v++) {
    VERTEX_TOPOLOGY const* t=&s->vertices_topology[v];
    if(t->num && (!t->f || !t->n)) abort();
    fwrite(t->f,sizeof(int),t->num,f);
    for(int i=0;i<t->num;i++) {int corner=t->n[i];fwrite(&corner,sizeof(int),1,f);}
  }
  fwrite(d->vertices,sizeof(int),d->nvertices,f);
  fwrite(d->chull,sizeof(int),d->nchull,f);
  fwrite(d->border,sizeof(int),d->nborder,f);
  fwrite(dvs->vertex_trans,sizeof(int),map_count,f);
  if(fclose(f)!=0) abort();
  if(phase==2) {
    fprintf(stderr,"FNIT_DVS_CAPTURE_STOP_AFTER_REAL_RESTORE\n");
    exit(0); // isolated observation only, never a production topology output
  }
}
'''


def build(*, native_build, base_observer, observer_tools, output):
    """新独立程序捕获record/first完整restore，不额外调用原算法。

    路径均必填；base为已绑定fullfitness观察器。复用现有编译器/归档/链接
    实现，不修改旧源码或库。首fullfitness旧观察停止移到真实restore之后；
    SOME路径若源实际调用可捕获face-only，否则无文件并标未测。异常SHA、
    锚点或目录拒绝。输出私有program/完整SHA/argv，不发布原C++主体。
    """
    native_build,base_observer,observer_tools,output=map(Path,(native_build,base_observer,observer_tools,output))
    if output.exists():raise FileExistsError('fresh DVS observer workspace required')
    builder=observer_tools/'build_candidate_neighborhood_observer.py'
    sys.path.insert(0,str(observer_tools))
    spec=importlib.util.spec_from_file_location('fnit_dvs_reused_builder',builder)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt=json.loads((base_observer/'build.private.json').read_text())
    source,archive=base_observer/'mrisurf_defect.cpp',base_observer/'libutils.a'
    if sha(source)!=receipt['patched_defect_sha256'] or sha(archive)!=receipt['archive_sha256']:
        raise ValueError('bound first-fitness source/archive required')
    text=source.read_text()
    definition='static DEFECT_VERTEX_STATE* mrisRecordVertexState(MRIS const * const mris, DEFECT * const defect, int * const vertex_trans)'
    record='  dvs->vertex_trans = vertex_trans;'
    restore='  updateVertexStatistics(mris, mris_corrected, dvs, rp, dp, vertex_trans, dp->fitness);\n\n  /* restore the vertex state */\n  mrisRestoreVertexState(mris_corrected, dvs);'
    face='    mrisRestoreFaceVertexState(mris_corrected, dvs);'
    old_stop='    exit(0); // benchmark stop only, no production topology result\n'
    if any(text.count(anchor)!=1 for anchor in (definition,record,restore,face,old_stop)):
        raise ValueError('fixed DVS/first-fitness observation anchors changed')
    text=text.replace(definition,CAPTURE+'\n'+definition,1)
    text=text.replace(record,record+'\n  if (!fnit_dvs_first && getenv("FNIT_TOPOLOGY_CANDIDATE_STATE_CAPTURE")) fnit_dvs_first = dvs;\n  fnit_capture_dvs_state(mris,dvs,0);',1)
    text=text.replace(restore,restore.replace('  /* restore the vertex state */','  fnit_capture_dvs_state(mris_corrected,dvs,1);\n  /* restore the vertex state */')+'\n  fnit_capture_dvs_state(mris_corrected,dvs,2);',1)
    text=text.replace(face,'    fnit_capture_dvs_state(mris_corrected,dvs,3);\n'+face+'\n    fnit_capture_dvs_state(mris_corrected,dvs,4);',1)
    text=text.replace(old_stop,'    // Isolated DVS observer continues through the original restore.\n',1)
    output.mkdir(parents=True);staging=output/'bound_staging';staging.mkdir()
    (staging/'mrisurf_defect.cpp').write_text(text);os.link(archive,staging/'libutils.a')
    prepared=dict(receipt,patched_defect_sha256=sha(staging/'mrisurf_defect.cpp'),
        link_argv=[str(staging/'libutils.a') if a==str(archive) else a for a in receipt['link_argv']])
    (staging/'build.private.json').write_text(json.dumps(prepared,indent=2)+'\n')
    module.CAPTURE=''
    result=module.build(native_build=native_build,base_observer=staging,output=output/'build')
    result.update(scope='actual first DVS record→first complete candidate→source full restore; not complete GA',
        wrapper_sha256=sha(__file__),reused_builder_sha256=sha(builder),
        original_fullfit_receipt_sha256=sha(base_observer/'build.private.json'),
        format='FNITDVS1 header9 N,F,vsizeEntries,numEntries,K,H,B,mapN,phase; Nx27F; Nx2flags; Nx7topology; Fx4; vsizeCSR; incident interleaved per-row int32face/corner; original selection3lists+translation')
    (output/'build'/'build.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-build','base-observer','observer-tools','output'):p.add_argument('--'+name,required=True,type=Path)
    build(**vars(p.parse_args()))
