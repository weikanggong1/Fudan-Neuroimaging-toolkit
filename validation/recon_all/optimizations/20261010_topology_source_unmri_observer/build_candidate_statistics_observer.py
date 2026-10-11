"""仅benchmark：前三次真实updateVertexStatistics调用的前后RP与顶点状态。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from build_volume_observer import sha


CAPTURE=r'''
static int fnit_rp_completed=0;
static bool fnit_rp_active=false;
static int* fnit_rp_marks_before=NULL;
static int* fnit_rp_counts_before=NULL;
static float* fnit_rp_fitness_before=NULL;
static float fnit_rp_incoming_fitness=0;
static void fnit_rp_begin(MRIS* s, DP* dp, RP* rp, float fitness) {
  fnit_rp_active=getenv("FNIT_TOPOLOGY_CANDIDATE_STATISTICS_CAPTURE") && fnit_rp_completed<3;
  if(!fnit_rp_active) { return; }
  fnit_rp_incoming_fitness=fitness;
  int n=s->nvertices,k=dp->defect->nvertices;
  fnit_rp_marks_before=(int*)malloc(sizeof(int)*n);
  fnit_rp_counts_before=(int*)malloc(sizeof(int)*k);
  fnit_rp_fitness_before=(float*)malloc(sizeof(float)*k);
  if(!fnit_rp_marks_before||!fnit_rp_counts_before||!fnit_rp_fitness_before) abort();
  for(int i=0;i<n;i++) fnit_rp_marks_before[i]=s->vertices[i].marked;
  for(int i=0;i<k;i++) {
    fnit_rp_counts_before[i]=rp->nused[i];
    fnit_rp_fitness_before[i]=rp->vertex_fitness[i];
  }
}
static void fnit_rp_end(MRIS* s, DP* dp, RP* rp, const int* vertex_trans) {
  if(!fnit_rp_active) { return; }
  const char* base=getenv("FNIT_TOPOLOGY_CANDIDATE_STATISTICS_CAPTURE");
  char name[STRLEN];snprintf(name,sizeof(name),"%s%d.bin",base,fnit_rp_completed);
  FILE* f=fopen(name,"wb");if(!f)abort();
  DEFECT* d=dp->defect;
  unsigned int h[4]={(unsigned int)s->nvertices,(unsigned int)d->nvertices,
                     (unsigned int)d->nborder,(unsigned int)dp->nedges};
  if(sizeof(int)!=4||sizeof(float)!=4||sizeof(unsigned int)!=4) abort();
  fwrite("FNITRST1",1,8,f);fwrite(h,sizeof(unsigned int),4,f);
  fwrite(&fnit_rp_incoming_fitness,sizeof(float),1,f);
  fwrite(fnit_rp_marks_before,sizeof(int),s->nvertices,f);
  for(int i=0;i<s->nvertices;i++) {
    int mark=s->vertices[i].marked;fwrite(&mark,sizeof(int),1,f);
  }
  for(int i=0;i<s->nvertices;i++) {
    float curv=s->vertices[i].curvbak;fwrite(&curv,sizeof(float),1,f);
  }
  for(int i=0;i<d->nvertices;i++) {
    int fields[2]={vertex_trans[d->vertices[i]],d->status[i]};fwrite(fields,sizeof(int),2,f);
  }
  for(int i=0;i<d->nborder;i++) {
    int id=vertex_trans[d->border[i]];fwrite(&id,sizeof(int),1,f);
  }
  for(int i=0;i<dp->nedges;i++) {
    const EDGE* e=&dp->etable->edges[i];int fields[3]={e->vno1,e->vno2,e->used};
    fwrite(fields,sizeof(int),3,f);
  }
  fwrite(fnit_rp_counts_before,sizeof(int),d->nvertices,f);
  for(int i=0;i<d->nvertices;i++) {
    int value=rp->nused[i];fwrite(&value,sizeof(int),1,f);
  }
  fwrite(fnit_rp_fitness_before,sizeof(float),d->nvertices,f);
  for(int i=0;i<d->nvertices;i++) {
    float value=rp->vertex_fitness[i];fwrite(&value,sizeof(float),1,f);
  }
  if(fclose(f)!=0)abort();
  free(fnit_rp_marks_before);free(fnit_rp_counts_before);free(fnit_rp_fitness_before);
  fnit_rp_marks_before=NULL;fnit_rp_counts_before=NULL;fnit_rp_fitness_before=NULL;
  fnit_rp_active=false;fnit_rp_completed++;
  if(fnit_rp_completed==3) {
    fprintf(stderr,"FNIT_STATISTICS_CAPTURE_STOP_AFTER_THREE_REAL_CALLS\n");
    exit(0); // isolated benchmark stop; never a completed topology output
  }
}
'''


def build(*,native_build,base_observer,output):
    """真实调用前后capture，不重复调用updateStatistics；第三次后有意停止。

    路径参数无默认：native_build固定Conda源码，base_observer完整fitness
    捕获构建且源码/archive与收据相同，output必须不存在。新程序在原调度
    下继续前三候选更新，可覆盖nused>0；只移除benchmark旧首fitness停止，
    不改任何源算法分支、seed/输入/参数。运行显式capture前缀，输出
    statistics0/1/2.bin；未生成GA最终标准文件，不构成生产替代。非法目录/
    SHA/anchor拒绝。输出私有编译收据和独立program，旧参考保持原样。
    """
    native_build,base_observer,output=map(Path,(native_build,base_observer,output))
    if output.exists():raise FileExistsError('fresh RP statistics observer required')
    receipt=json.loads((base_observer/'build.private.json').read_text())
    base,old=base_observer/'mrisurf_defect.cpp',base_observer/'libutils.a'
    if sha(base)!=receipt['patched_defect_sha256'] or sha(old)!=receipt['archive_sha256']:
        raise ValueError('frozen full fitness observer changed')
    text=base.read_text()
    anchor='static void updateVertexStatistics(\n'
    start=text.index(anchor,text.index(anchor)+1)
    stop=text.index('\nstatic int deleteWorstVertices(',start)
    section=text[start:stop]
    if section.count('\n{\n')!=1:raise ValueError('fixed update statistics function changed')
    section=section.replace('\n{\n','\n{\n  fnit_rp_begin(mris_corrected,dp,rp,fitness);\n',1)
    tail=section.rfind('\n}')
    if tail<0:raise ValueError('fixed statistics tail changed')
    section=section[:tail]+'\n  fnit_rp_end(mris_corrected,dp,rp,vertex_trans);'+section[tail:]
    text=text[:start]+CAPTURE+'\n'+section+text[stop:]
    old_stop='    exit(0); // benchmark stop only, no production topology result\n'
    if text.count(old_stop)!=1:raise ValueError('fixed first-fitness benchmark stop changed')
    text=text.replace(old_stop,'    // New isolated benchmark continues to three real RP updates.\n',1)
    text=text.replace('FNIT_TOPOLOGY_FULL_FITNESS_CAPTURE_STOP_BEFORE_CONTROLLER',
                      'FNIT_TOPOLOGY_FULL_FITNESS_CAPTURE_CONTINUE_FOR_RP_DIAGNOSTICS')
    output.mkdir(parents=True);patched=output/'mrisurf_defect.cpp';patched.write_text(text)
    command=list(receipt['compile_argv']);obj=output/'mrisurf_defect.cpp.o'
    for flag,value in [('-o',str(obj)),('-c',str(patched)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:
        command[command.index(flag)+1]=value
    archive=output/'libutils.a';shutil.copyfile(old,archive)
    link=[str(archive) if x==str(old) else x for x in receipt['link_argv']]
    binary=output/'mris_fix_topology_statistics_updates_observer';link[link.index('-o')+1]=str(binary)
    compiler=Path(command[0]);started=time.perf_counter()
    with (output/'build.log').open('w') as log:
        subprocess.run(command,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')),'r',str(archive),str(obj)],check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')),str(archive)],check=True)
        subprocess.run(link,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(scope='three actual RP statistics calls; intentional benchmark stop; not complete GA',
        status='complete_readonly_statistics_updates_build',base_receipt_sha256=sha(base_observer/'build.private.json'),
        observer_builder_sha256=sha(__file__),patched_defect_sha256=sha(patched),archive_sha256=sha(archive),
        observer_program_sha256=sha(binary),compile_argv=command,link_argv=link,seconds=time.perf_counter()-started,
        format='FNITRST1 magic8+u32 N,K,B,E+F32 incoming+marks before/after+curvbak+mapped/status+border+edges+RP counts before/after+RP fitness before/after; no padding')
    (output/'build.private.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:parser.add_argument('--'+name,type=Path,required=True)
    build(**vars(parser.parse_args()))


if __name__=='__main__':main()
