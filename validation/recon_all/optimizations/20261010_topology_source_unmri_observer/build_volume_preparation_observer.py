"""观察 defect 原 MRI→局部 intensity/PDF 准备，仅供隔离 benchmark。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from build_volume_observer import sha

CAPTURE = r'''
static void fnit_capture_volume_preparation(MRIS* surf, MRI* brain, EDGE_TABLE* et,
  TOPOLOGY_PARMS* parms, DEFECT* defect, HISTOGRAM* hw, HISTOGRAM* hg,
  MRI* intensity, MRI* white, MRI* gray, int phase) {
  const char* base=getenv("FNIT_TOPOLOGY_VOLUME_PREPARATION_CAPTURE");
  static bool seen[2]={false,false};
  if(!base || seen[phase]) { return; }
  seen[phase]=true;
  char name[STRLEN];snprintf(name,sizeof(name),"%s.%s",base,phase?"likelihood.bin":"creation.bin");
  FILE* f=fopen(name,"wb");if(!f)abort();
  int h[8]={surf->nvertices,et->nedges,intensity->width,intensity->height,intensity->depth,hw->nbins,hg->nbins,phase};
  fwrite("FNITPDF1",1,8,f);fwrite(h,sizeof(int),8,f);
  float g[9]={intensity->xstart,intensity->ystart,intensity->zstart,intensity->xsize,intensity->ysize,intensity->zsize,intensity->xend,intensity->yend,intensity->zend};fwrite(g,sizeof(float),9,f);
  float params[7]={(float)parms->volume_resolution,hw->bin_size,hg->bin_size,defect->white_mean,defect->gray_mean,defect->white_mean_ll,defect->gray_mean_ll};fwrite(params,sizeof(float),7,f);
  MATRIX* map=GetSurfaceRASToVoxelMatrix(brain);
  for(int r=1;r<=4;r++) { for(int c=1;c<=4;c++){double v=*MATRIX_RELT(map,r,c);fwrite(&v,sizeof(double),1,f);} }
  MatrixFree(&map);
  for(int i=0;i<surf->nvertices;i++){VERTEX* v=&surf->vertices[i];float x[3]={v->origx,v->origy,v->origz};fwrite(x,sizeof(float),3,f);}
  for(int i=0;i<et->nedges;i++){int e[2]={et->edges[i].vno1,et->edges[i].vno2};fwrite(e,sizeof(int),2,f);}
  fwrite(hw->bins,sizeof(float),hw->nbins,f);fwrite(hw->counts,sizeof(float),hw->nbins,f);fwrite(hg->bins,sizeof(float),hg->nbins,f);fwrite(hg->counts,sizeof(float),hg->nbins,f);
  for(int x=0;x<intensity->width;x++)for(int y=0;y<intensity->height;y++)for(int z=0;z<intensity->depth;z++){unsigned char v=MRIvox(intensity,x,y,z);fwrite(&v,1,1,f);}
  if(phase)for(int m=0;m<2;m++){MRI* image=m?gray:white;for(int x=0;x<image->width;x++)for(int y=0;y<image->height;y++)for(int z=0;z<image->depth;z++){float v=MRIFvox(image,x,y,z);fwrite(&v,sizeof(float),1,f);}}
  if(fclose(f)!=0)abort();
}
'''


def build(*,native_build,base_observer,output):
    """全新 benchmark 构建：创建后 mask 和 likelihood 后图，生产不读取。"""
    native_build,base_observer,output=map(Path,(native_build,base_observer,output))
    if output.exists():raise FileExistsError('fresh volume preparation observer build required')
    receipt=json.loads((base_observer/'build.private.json').read_text());base=base_observer/'mrisurf_defect.cpp';old=base_observer/'libutils.a'
    if sha(base)!=receipt['patched_defect_sha256'] or sha(old)!=receipt['archive_sha256']:raise ValueError('frozen fullfit observer changed')
    text=base.read_text()
    definition='static MRI *mriDefectVolume(MRIS *mris, EDGE_TABLE *etable, TOPOLOGY_PARMS *parms)\n{'
    creation='    mri_defect = mriDefectVolume(mris_corrected, &etable, parms);\n'
    after='                           defect->gray_mean,\n                           0,\n                           0);\n'
    # 固定源码的 random 与 genetic 配方均创建局部图；首次实际调用单独捕获。
    if tuple(text.count(a) for a in [definition,creation,after]) != (1,2,2):
        raise ValueError('fixed genetic volume preparation anchors changed')
    call='fnit_capture_volume_preparation(mris_corrected,mri,&etable,parms,defect,h_white,h_gray,mri_defect,mri_defect_white,mri_defect_gray,'
    text=text.replace(definition,CAPTURE+'\n'+definition).replace(creation,creation+'    '+call+'0);\n').replace(after,after+'    '+call+'1);\n')
    output.mkdir(parents=True);patched=output/'mrisurf_defect.cpp';patched.write_text(text)
    command=list(receipt['compile_argv']);obj=output/'mrisurf_defect.cpp.o'
    for flag,value in [('-o',str(obj)),('-c',str(patched)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:command[command.index(flag)+1]=value
    archive=output/'libutils.a';shutil.copyfile(old,archive)
    link=[str(archive) if x==str(old) else x for x in receipt['link_argv']];binary=output/'mris_fix_topology_volume_preparation_observer';link[link.index('-o')+1]=str(binary)
    compiler=Path(command[0]);started=time.perf_counter()
    with (output/'build.log').open('w') as log:
        subprocess.run(command,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')),'r',str(archive),str(obj)],check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')),str(archive)],check=True)
        subprocess.run(link,cwd=native_build/'build',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(scope='original MRI to defect intensity/PDF prepare, not GA controller',status='complete_readonly_volume_preparation_observer',base_receipt_sha256=sha(base_observer/'build.private.json'),observer_builder_sha256=sha(__file__),patched_defect_sha256=sha(patched),archive_sha256=sha(archive),observer_program_sha256=sha(binary),compile_argv=command,link_argv=link,seconds=time.perf_counter()-started,format='FNITPDF1: header8; geometry9F32; params7F32; MRI-only matrix16F64; origXYZ Nx3F32; potentialedges Kx2I32; white/gray bins/counts; U8 local mask/intensity XYZ; phase1 white/gray logPDF F32 XYZ')
    (output/'build.private.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:parser.add_argument('--'+name,type=Path,required=True)
    build(**vars(parser.parse_args()))


if __name__=='__main__':main()
