"""仅benchmark：透明捕获原交叉/变异的完整输入、输出及真实随机调用。"""
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


CAPTURE=r'''
static int fnit_ga_completed[2]={0,0};
static bool fnit_ga_active=false;
static int fnit_ga_kind=0,fnit_ga_n=0,fnit_ga_alias=0,fnit_ga_ndraws=0;
static int* fnit_ga_parent1=NULL;
static int* fnit_ga_parent2=NULL;
static double* fnit_ga_draws=NULL;
static double fnit_ga_parameter=0;
static void fnit_ga_begin(int kind,DEFECT_PATCH* p1,DEFECT_PATCH* p2,DEFECT_PATCH* dst,double parameter) {
  fnit_ga_active=getenv("FNIT_TOPOLOGY_GA_ORDER_CAPTURE") && fnit_ga_completed[kind]<3;
  if(!fnit_ga_active)return;
  fnit_ga_kind=kind;fnit_ga_n=p1->nedges;fnit_ga_ndraws=0;fnit_ga_parameter=parameter;
  fnit_ga_alias=kind ? ((dst==p1 ? 1:0)|(dst==p2 ? 2:0)):0;
  if(fnit_ga_n<0 || (kind && p2->nedges!=fnit_ga_n))abort();
  size_t n=(size_t)fnit_ga_n;
  fnit_ga_parent1=(int*)malloc(sizeof(int)*(n+1));
  fnit_ga_parent2=kind ? (int*)malloc(sizeof(int)*(n+1)):NULL;
  fnit_ga_draws=(double*)malloc(sizeof(double)*3*(2*n+2));
  if(!fnit_ga_parent1 || !fnit_ga_draws || (kind&&!fnit_ga_parent2))abort();
  for(int i=0;i<fnit_ga_n;i++) {
    fnit_ga_parent1[i]=p1->ordering[i];
    if(kind)fnit_ga_parent2[i]=p2->ordering[i];
  }
}
static double fnit_ga_random(double low,double high) {
  double value=randomNumber(low,high); // exactly one original draw
  if(fnit_ga_active) {
    if(fnit_ga_ndraws>=2*fnit_ga_n+2)abort();
    double* row=fnit_ga_draws+3*fnit_ga_ndraws++;
    row[0]=low;row[1]=high;row[2]=value;
  }
  return value;
}
static void fnit_ga_end(DEFECT_PATCH* dst,EDGE_TABLE* table,const int* position_map) {
  if(!fnit_ga_active)return;
  int kind=fnit_ga_kind,n=fnit_ga_n,d=fnit_ga_ndraws;
  int use_overlap=!kind&&table->use_overlap ? 1:0;
  unsigned int m=0;
  if(use_overlap) for(int e=0;e<n;e++) {
    if(table->noverlap[e]<0)abort();m+=(unsigned int)table->noverlap[e];
  }
  char filename[STRLEN];snprintf(filename,sizeof(filename),"%s%s%d.bin",
    getenv("FNIT_TOPOLOGY_GA_ORDER_CAPTURE"),kind ? "crossover":"mutation",fnit_ga_completed[kind]);
  FILE* file=fopen(filename,"wb");if(!file)abort();
  if(sizeof(unsigned int)!=4||sizeof(int)!=4||sizeof(double)!=8)abort();
  unsigned int header[6]={(unsigned int)kind,(unsigned int)n,m,(unsigned int)d,
                          (unsigned int)use_overlap,(unsigned int)fnit_ga_alias};
  double parameter=kind ? fnit_ga_draws[2]:fnit_ga_parameter;
  if(kind && d!=1)abort();
  fwrite("FNITGAO1",1,8,file);fwrite(header,sizeof(unsigned int),6,file);
  fwrite(&parameter,sizeof(double),1,file);fwrite(fnit_ga_parent1,sizeof(int),n,file);
  if(kind)fwrite(fnit_ga_parent2,sizeof(int),n,file);
  fwrite(dst->ordering,sizeof(int),n,file);
  if(!kind) {
    fwrite(position_map,sizeof(int),n,file);
    int offset=0;fwrite(&offset,sizeof(int),1,file);
    for(int e=0;e<n;e++) {
      if(use_overlap)offset+=table->noverlap[e];fwrite(&offset,sizeof(int),1,file);
    }
    if(use_overlap)for(int e=0;e<n;e++)
      fwrite(table->overlapping_edges[e],sizeof(int),table->noverlap[e],file);
    for(int e=0;e<n;e++) {
      int incomplete=use_overlap && (table->flags[e]&ET_OVERLAP_LIST_INCOMPLETE) ? 1:0;
      fwrite(&incomplete,sizeof(int),1,file);
    }
  }
  fwrite(fnit_ga_draws,sizeof(double),3*d,file);
  if(fclose(file)!=0)abort();
  free(fnit_ga_parent1);free(fnit_ga_parent2);free(fnit_ga_draws);
  fnit_ga_parent1=NULL;fnit_ga_parent2=NULL;fnit_ga_draws=NULL;
  fnit_ga_active=false;fnit_ga_completed[kind]++;
  if(fnit_ga_completed[0]>=3 && fnit_ga_completed[1]>=3) {
    fprintf(stderr,"FNIT_GA_ORDER_CAPTURE_STOP_AFTER_SIX_REAL_OPERATORS\n");
    exit(0); // intentional benchmark stop; not a final topology output
  }
}
'''


def instrument_ordering_source(*, source_text):
    """固定源文本 → 透明观测文本；核对全部原调用位置，不执行/写文件。

    source_text为带两个完整固定函数的str，返回str；prototype/body、
    分配/释放或原随机调用位置不唯一/变化抛ValueError。原算法主体及
    randomNumber调用参数保留，只围绕真实调用收集before/after。
    """
    text=source_text
    crossover='static int mrisCrossoverDefectPatches(DEFECT_PATCH *dp1, DEFECT_PATCH *dp2, DEFECT_PATCH *dp_dst, EDGE_TABLE *etable)\n{'
    mutation='static int mrisMutateDefectPatch(DEFECT_PATCH *dp, EDGE_TABLE *etable, double pmutation)\n{'
    if text.count(crossover)!=1 or text.count(mutation)!=1:
        raise ValueError('unique fixed source ordering functions required')
    start=text.index(crossover);middle=text.index(mutation,start)
    end=text.index('\nstatic int mrisCopyDefectPatch(',middle)
    cross=text[start:middle];mut=text[middle:end]
    if cross.count('randomNumber(')!=1 or mut.count('randomNumber(')!=7:
        raise ValueError('fixed per-operator random call sites changed')
    if cross.count('  added = (int *)calloc')!=1 or cross.count('  free(added);')!=1:
        raise ValueError('fixed crossover state anchors changed')
    if mut.count('  dp_indices = (int *)calloc')!=1 or mut.count('  free(dp_indices);')!=1:
        raise ValueError('fixed mutation state anchors changed')
    cross=cross.replace('randomNumber(','fnit_ga_random(')
    cross=cross.replace('  added = (int *)calloc',
        '  fnit_ga_begin(1,dp1,dp2,dp_dst,0);\n  added = (int *)calloc',1)
    cross=cross.replace('  free(added);','  fnit_ga_end(dp_dst,etable,NULL);\n  free(added);',1)
    mut=mut.replace('randomNumber(','fnit_ga_random(')
    mut=mut.replace('  dp_indices = (int *)calloc',
        '  fnit_ga_begin(0,dp,NULL,dp,pmutation);\n  dp_indices = (int *)calloc',1)
    mut=mut.replace('  free(dp_indices);','  fnit_ga_end(dp,etable,dp_indices);\n  free(dp_indices);',1)
    return text[:start]+CAPTURE+'\n'+cross+mut+text[end:]


def build(*, native_build, base_observer, output):
    """固定Conda源码参考 → 新隔离观察程序，不部署生产、不迁出源码库。

    native_build为固定原构建；base_observer为带SHA/编译链接收据的原
    完整fitness observer，output必须不存在。仅观测原真实算子调用；
    每个randomNumber保持一次原调用及相同参数/返回值，未加入随机数。
    捕获每类前三调用后有意停止，exit0不等于GA完成。未设置capture
    环境变量时不捕获/停止。仅在获授权benchmark机器编译/运行。
    锚点/哈希/格式改变拒绝，编译失败保留日志并抛CalledProcessError。
    输出program、私有构建收据、patched源码，仅原位保留；不提交它们。
    """
    native_build,base_observer,output=map(Path,(native_build,base_observer,output))
    if output.exists():
        raise FileExistsError('fresh ordering observer workspace required')
    receipt=json.loads((base_observer/'build.private.json').read_text())
    old_source=base_observer/'mrisurf_defect.cpp';old_archive=base_observer/'libutils.a'
    if sha(old_source)!=receipt['patched_defect_sha256'] or sha(old_archive)!=receipt['archive_sha256']:
        raise ValueError('fixed benchmark source/archive differs from receipt')
    text=instrument_ordering_source(source_text=old_source.read_text())
    output.mkdir(parents=True);source=output/'mrisurf_defect.cpp';source.write_text(text)
    obj=output/'mrisurf_defect.cpp.o';command=list(receipt['compile_argv'])
    for flag,value in [('-o',str(obj)),('-c',str(source)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:
        command[command.index(flag)+1]=value
    archive=output/'libutils.a';shutil.copyfile(old_archive,archive)
    link=[str(archive) if arg==str(old_archive) else arg for arg in receipt['link_argv']]
    binary=output/'mris_fix_topology_ga_order_observer';link[link.index('-o')+1]=str(binary)
    compiler=Path(command[0]);started=time.perf_counter()
    with (output/'build.log').open('w') as log:
        subprocess.run(command,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')),'r',str(archive),str(obj)],check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')),str(archive)],check=True)
        subprocess.run(link,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(status='complete_readonly_ordering_observer_build',
        scope='three actual crossover and three actual mutation calls; intentional stop, not full GA',
        base_receipt_sha256=sha(base_observer/'build.private.json'),observer_builder_sha256=sha(__file__),
        patched_defect_sha256=sha(source),archive_sha256=sha(archive),observer_program_sha256=sha(binary),
        compiler_sha256=sha(compiler),compile_argv=command,link_argv=link,seconds=time.perf_counter()-started,
        format='FNITGAO1 magic8/u32 kind,N,M,D,use_overlap,alias/F64 parameter; conditional complete i32 fields/F64 low-high-value draws')
    (output/'build.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    build(**vars(parser.parse_args()))
