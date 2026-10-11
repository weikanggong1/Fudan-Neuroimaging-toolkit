"""捕获一次真实动态 retessellation 的完整基底/排序/overlap/输出，仅作 benchmark。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import hashlib


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

CAPTURE=r'''
static void fnit_capture_retessellation_state(MRIS* s, DVS* dvs, DP* dp,int phase) {
  const char* base=getenv("FNIT_TOPOLOGY_RETESS_STATE_CAPTURE");static bool done[2]={false,false};
  if(!base||done[phase]) { return; }
  done[phase]=true;
  DEFECT* defect=dvs->defect;EDGE_TABLE* table=dp->etable;
  char name[STRLEN];snprintf(name,sizeof(name),"%s.%s",base,phase?"after.bin":"before.bin");
  FILE* f=fopen(name,"wb");if(!f)abort();
  int entries=0,overlaps=0;
  for(int v=0;v<s->nvertices;v++)entries+=s->vertices_topology[v].vnum;
  if(table&&table->use_overlap)for(int e=0;e<dp->nedges;e++)overlaps+=table->noverlap[e];
  int h[11]={s->nvertices,s->nfaces,entries,dp->nedges,defect->nvertices,defect->nborder,defect->nchull,dp->retessellation_mode,defect->optimal_mapping,overlaps,phase};fwrite("FNITRTS1",1,8,f);fwrite(h,sizeof(int),11,f);
  for(int v=0;v<s->nvertices;v++){VERTEX* p=&s->vertices[v];float x[9]={p->origx,p->origy,p->origz,p->x,p->y,p->z,p->cx,p->cy,p->cz};fwrite(x,sizeof(float),9,f);}
  for(int v=0;v<s->nvertices;v++){VERTEX* p=&s->vertices[v];int flags[3]={p->marked,p->ripflag,p->fixedval};fwrite(flags,sizeof(int),3,f);}
  for(int i=0;i<s->nfaces;i++)fwrite(s->faces[i].v,sizeof(int),3,f);
  int offset=0;fwrite(&offset,sizeof(int),1,f);for(int v=0;v<s->nvertices;v++){offset+=s->vertices_topology[v].vnum;fwrite(&offset,sizeof(int),1,f);}
  for(int v=0;v<s->nvertices;v++){VERTEX_TOPOLOGY* t=&s->vertices_topology[v];fwrite(t->v,sizeof(int),t->vnum,f);}
  for(int e=0;e<dp->nedges;e++){EDGE* p=&table->edges[e];int row[4]={p->vno1,p->vno2,p->used,table->use_overlap && !(table->flags[e]&ET_OVERLAP_LIST_INCOMPLETE)};fwrite(row,sizeof(int),4,f);}
  for(int e=0;e<dp->nedges;e++){int order=dp->ordering?dp->ordering[e]:e;fwrite(&order,sizeof(int),1,f);}
  offset=0;fwrite(&offset,sizeof(int),1,f);for(int e=0;e<dp->nedges;e++){if(table->use_overlap)offset+=table->noverlap[e];fwrite(&offset,sizeof(int),1,f);}
  if(table->use_overlap)for(int e=0;e<dp->nedges;e++)fwrite(table->overlapping_edges[e],sizeof(int),table->noverlap[e],f);
  for(int i=0;i<defect->nvertices;i++){int row[2]={dvs->vertex_trans[defect->vertices[i]],defect->status[i]};fwrite(row,sizeof(int),2,f);}
  for(int i=0;i<defect->nborder;i++){int v=dvs->vertex_trans[defect->border[i]];fwrite(&v,sizeof(int),1,f);}
  for(int i=0;i<defect->nchull;i++){int v=dvs->vertex_trans[defect->chull[i]];fwrite(&v,sizeof(int),1,f);}
  int selected[4]={0,0,0,0};
  if(phase){selected[0]=dp->tp.nvertices;selected[1]=dp->tp.nedges;selected[2]=dp->tp.ninside;selected[3]=dp->tp.ndiscarded;}
  fwrite(selected,sizeof(int),4,f);
  if(phase){fwrite(dp->tp.vertices,sizeof(int),dp->tp.nvertices,f);fwrite(dp->tp.edges,sizeof(int),dp->tp.nedges,f);}
  if(fclose(f)!=0)abort();
}
'''


def build(*,native_build,base_observer,output):
    """新独立固定源码观察器；基底前/后原状态不作为生产输入。"""
    native_build,base_observer,output=map(Path,(native_build,base_observer,output))
    if output.exists():raise FileExistsError('fresh retessellation observer build required')
    receipt=json.loads((base_observer/'build.private.json').read_text());base=base_observer/'mrisurf_defect.cpp';old=base_observer/'libutils.a'
    if sha(base)!=receipt['patched_defect_sha256'] or sha(old)!=receipt['archive_sha256']:raise ValueError('frozen fullfit observer changed')
    text=base.read_text();definition='static int retessellateDefect_wkr(MRIS *mris, MRIS *mris_corrected, DVS *dvs, DP *dp)\n{'
    before='  /* count the number of potentially added edges */\n';after='  /* free the allocated memory for the intersection_table */\n  free(it);\n'
    if any(text.count(a)!=1 for a in [definition,before,after]):raise ValueError('fixed retessellation anchors changed')
    text=text.replace(definition,CAPTURE+'\n'+definition).replace(before,'  fnit_capture_retessellation_state(mris_corrected,dvs,dp,0);\n'+before).replace(after,after+'  fnit_capture_retessellation_state(mris_corrected,dvs,dp,1);\n')
    output.mkdir(parents=True);patched=output/'mrisurf_defect.cpp';patched.write_text(text)
    command=list(receipt['compile_argv']);obj=output/'mrisurf_defect.cpp.o'
    for flag,value in [('-o',str(obj)),('-c',str(patched)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:command[command.index(flag)+1]=value
    archive=output/'libutils.a';shutil.copyfile(old,archive)
    link=[str(archive) if x==str(old) else x for x in receipt['link_argv']];binary=output/'mris_fix_topology_retessellation_observer';link[link.index('-o')+1]=str(binary)
    compiler=Path(command[0]);started=time.perf_counter()
    with (output/'build.log').open('w') as log:
        subprocess.run(command,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')),'r',str(archive),str(obj)],check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')),str(archive)],check=True)
        subprocess.run(link,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(scope='first retessellation before/after, not complete GA',status='complete_readonly_retessellation_observer',base_receipt_sha256=sha(base_observer/'build.private.json'),observer_builder_sha256=sha(__file__),patched_defect_sha256=sha(patched),archive_sha256=sha(archive),observer_program_sha256=sha(binary),compile_argv=command,link_argv=link,seconds=time.perf_counter()-started,format='FNITRTS1 ordered complete base/edges/overlap/translated-domain/current states')
    (output/'build.private.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:parser.add_argument('--'+name,type=Path,required=True)
    build(**vars(parser.parse_args()))


if __name__=='__main__':main()
