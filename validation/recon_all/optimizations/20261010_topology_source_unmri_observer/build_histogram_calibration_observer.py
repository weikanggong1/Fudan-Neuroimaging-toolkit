"""独立观测首个缺陷的灰白直方图输入、原PDF及平滑PDF，生产不执行。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from build_volume_observer import sha


CAPTURE = r'''
static bool fnit_hist_active = false, fnit_hist_complete = false;
static int* fnit_hist_selected = NULL;
static int fnit_hist_selected_count = 0;
static void fnit_capture_hist_calibration(MRIS* s, DEFECT* d,
        HISTOGRAM* hw, HISTOGRAM* hg, HISTOGRAM* hb, HISTOGRAM* hd, int phase) {
  const char* base = getenv("FNIT_TOPOLOGY_HIST_CALIBRATION_CAPTURE");
  if (!base || !fnit_hist_active) { return; }
  char name[STRLEN]; snprintf(name,sizeof(name),"%s.%s.bin",base,
      phase == 0 ? "before" : (phase == 1 ? "raw" : "smooth"));
  FILE* f = fopen(name,"wb"); if (!f) abort();
  int entries = 0;
  for (int v = 0; v < s->nvertices; v++) entries += s->vertices_topology[v].vtotal;
  int h[12] = {s->nvertices,s->nfaces,entries,d->nchull,d->nvertices,d->nborder,
               phase,fnit_hist_selected_count,hw->nbins,hg->nbins,hb->nbins,hd->nbins};
  fwrite("FNITHST1",1,8,f); fwrite(h,sizeof(int),12,f);
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX* p = &s->vertices[v];
    float fields[9] = {p->origx,p->origy,p->origz,p->nx,p->ny,p->nz,p->val,p->val2,p->val2bak};
    fwrite(fields,sizeof(float),9,f);
  }
  for (int v = 0; v < s->nvertices; v++) {
    int fields[2] = {s->vertices[v].marked,s->vertices[v].ripflag};
    fwrite(fields,sizeof(int),2,f);
  }
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX_TOPOLOGY* t = &s->vertices_topology[v];
    int fields[3] = {t->vnum,t->vtotal,t->nsizeCur}; fwrite(fields,sizeof(int),3,f);
  }
  for (int i = 0; i < s->nfaces; i++) fwrite(s->faces[i].v,sizeof(int),3,f);
  int offset=0; fwrite(&offset,sizeof(int),1,f);
  for (int v=0;v<s->nvertices;v++) {
    offset+=s->vertices_topology[v].vtotal; fwrite(&offset,sizeof(int),1,f);
  }
  for (int v=0;v<s->nvertices;v++) {
    const VERTEX_TOPOLOGY* t=&s->vertices_topology[v];
    fwrite(t->v,sizeof(int),t->vtotal,f);
  }
  fwrite(d->chull,sizeof(int),d->nchull,f);
  for (int i=0;i<d->nvertices;i++) {
    int fields[2]={d->vertices[i],d->status[i]}; fwrite(fields,sizeof(int),2,f);
  }
  fwrite(d->border,sizeof(int),d->nborder,f);
  if (fnit_hist_selected_count) fwrite(fnit_hist_selected,sizeof(int),fnit_hist_selected_count,f);
  HISTOGRAM* histograms[4]={hw,hg,hb,hd};
  for (int i=0;i<4;i++) {
    HISTOGRAM* p=histograms[i]; float bin_size=p->bin_size;
    fwrite(&bin_size,sizeof(float),1,f); fwrite(p->bins,sizeof(float),p->nbins,f);
    fwrite(p->counts,sizeof(float),p->nbins,f);
  }
  if (fclose(f)!=0) abort();
}
'''


def build(*, native_build, base_observer, output):
    """固定源码新观察器；真实入口/实际选点顺序/平滑前后不改变源计算。

    参数都是显式路径；native_build是已固定Conda源码构建，base_observer
    是已有完整fitness观察器且源码/archive必须匹配构建收据，output不能
    存在。输出独立benchmark程序及私有argv/源/program SHA；运行用capture
    前缀生成FNITHST1三phase。非法anchor或哈希报错；不修改旧观测或生产。
    """
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError('fresh calibration observer build required')
    receipt = json.loads((base_observer/'build.private.json').read_text())
    base, old_archive = base_observer/'mrisurf_defect.cpp', base_observer/'libutils.a'
    if sha(base) != receipt['patched_defect_sha256'] or sha(old_archive) != receipt['archive_sha256']:
        raise ValueError('frozen full fitness observer changed')
    text = base.read_text()
    start = text.index('static int mrisComputeGrayWhiteBorderDistributions(MRI_SURFACE *mris,',
                       text.index('static int mrisComputeGrayWhiteBorderDistributions(MRI_SURFACE *mris,')+1)
    stop = text.index('\nstatic void mrisComputeSurfaceStatistics(', start)
    section = text[start:stop]
    opening = '\n{\n'
    selected = '        nvertices++;\n'
    before = '  HISTOsmooth(h_grad_raw, h_grad, 2);\n'
    after = '  HISTOsmooth(h_border_raw, h_border, 2);\n'
    end = '  mrisMarkDefect(mris, defect, 0);\n'
    if any(section.count(a) != 1 for a in [opening, selected, before, after, end]):
        raise ValueError('fixed histogram calibration anchors changed')
    section = section.replace(opening, '\n{\n'
        '  fnit_hist_active = getenv("FNIT_TOPOLOGY_HIST_CALIBRATION_CAPTURE") && !fnit_hist_complete;\n'
        '  if (fnit_hist_active) {\n'
        '    fnit_hist_selected_count=0; fnit_hist_selected=(int*)malloc(sizeof(int)*mris->nvertices);\n'
        '    if (!fnit_hist_selected) abort();\n'
        '    fnit_capture_hist_calibration(mris,defect,h_white,h_gray,h_border,h_grad,0);\n'
        '  }\n', 1)
    section = section.replace(selected, selected+
        '        if (fnit_hist_active) fnit_hist_selected[fnit_hist_selected_count++]=vnt->v[n2];\n', 1)
    section = section.replace(before,
        '  fnit_capture_hist_calibration(mris,defect,h_white,h_gray,h_border,h_grad,1);\n'+before, 1)
    section = section.replace(after, after+
        '  fnit_capture_hist_calibration(mris,defect,h_white,h_gray,h_border,h_grad,2);\n', 1)
    section = section.replace(end, end+
        '  if (fnit_hist_active) { free(fnit_hist_selected); fnit_hist_selected=NULL; fnit_hist_complete=true; }\n'
        '  fnit_hist_active=false;\n', 1)
    text = text[:start]+CAPTURE+'\n'+section+text[stop:]
    output.mkdir(parents=True)
    patched=output/'mrisurf_defect.cpp'; patched.write_text(text)
    command=list(receipt['compile_argv']); obj=output/'mrisurf_defect.cpp.o'
    for flag,value in [('-o',str(obj)),('-c',str(patched)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:
        command[command.index(flag)+1]=value
    archive=output/'libutils.a'; shutil.copyfile(old_archive,archive)
    link=[str(archive) if x==str(old_archive) else x for x in receipt['link_argv']]
    binary=output/'mris_fix_topology_calibration_observer'; link[link.index('-o')+1]=str(binary)
    compiler=Path(command[0]); started=time.perf_counter()
    with (output/'build.log').open('w') as log:
        subprocess.run(command,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')),'r',str(archive),str(obj)],check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')),str(archive)],check=True)
        subprocess.run(link,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(scope='first calibrated tissue histograms before/raw/smooth; no production input',
        status='complete_readonly_histogram_calibration_build',base_receipt_sha256=sha(base_observer/'build.private.json'),
        observer_builder_sha256=sha(__file__),patched_defect_sha256=sha(patched),archive_sha256=sha(archive),
        observer_program_sha256=sha(binary),compile_argv=command,link_argv=link,seconds=time.perf_counter()-started,
        format='FNITHST1 header12; fields Nx9; marked/rip Nx2; topology Nx3; faces; vtotalCSR; chull; inside/status; border; actualselected; histogram bin_size,bins,counts x4')
    (output/'build.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    build(**vars(parser.parse_args()))


if __name__=='__main__':
    main()
