"""捕获完整 fitness 候选曲率重算前的真实 vtotal 邻域，只用于 benchmark。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from build_volume_observer import sha


CAPTURE = r'''
    const char* fnit_neighbors_path=getenv("FNIT_TOPOLOGY_CANDIDATE_NEIGHBORHOOD_CAPTURE");
    static bool fnit_neighbors_captured=false;
    if(fnit_neighbors_path && !fnit_neighbors_captured) {
      fnit_neighbors_captured=true;
      FILE* f=fopen(fnit_neighbors_path,"wb");if(!f) abort();
      int n=mris->nvertices,entries=0;
      for(int i=0;i<n;i++) entries+=mris->vertices_topology[i].vtotal;
      int header[3]={n,entries,dp->tp.nvertices};
      fwrite("FNITNBR1",1,8,f);fwrite(header,sizeof(int),3,f);
      int offset=0;fwrite(&offset,sizeof(int),1,f);
      for(int i=0;i<n;i++) {offset+=mris->vertices_topology[i].vtotal;fwrite(&offset,sizeof(int),1,f);}
      for(int i=0;i<n;i++) {
        const VERTEX_TOPOLOGY* vt=&mris->vertices_topology[i];
        int fields[3]={vt->vnum,vt->vtotal,vt->nsizeCur};fwrite(fields,sizeof(int),3,f);
      }
      for(int i=0;i<n;i++) {
        const VERTEX_TOPOLOGY* vt=&mris->vertices_topology[i];
        if(fwrite(vt->v,sizeof(int),vt->vtotal,f)!=(size_t)vt->vtotal) abort();
      }
      for(int i=0;i<n;i++) {
        const VERTEX* v=&mris->vertices[i];
        float fields[15]={v->origx,v->origy,v->origz,v->nx,v->ny,v->nz,
          v->e1x,v->e1y,v->e1z,v->e2x,v->e2y,v->e2z,v->k1,v->k2,v->area};
        fwrite(fields,sizeof(float),15,f);
      }
      fwrite(dp->tp.vertices,sizeof(int),dp->tp.nvertices,f);
      if(fclose(f)!=0) abort();
    }
'''


def build(*, native_build, base_observer, output):
    """使用已绑定完整 fitness 观察器的新独立构建，原运算保持不变。

    native_build 为固定 Conda 编译源码目录；base_observer 必须是已完成
    full-fitness 构建（源码和 archive 哈希匹配其收据）；output 必须不存在。
    输出独立程序、编译 argv/SHA 与 FNITNBR1 格式说明。运行时环境变量
    指向新文件，记录 first complete fitness 的曲率拟合前时点。
    不修改旧源码/程序/结果；异常目录、anchor 或 SHA 不匹配均报错。
    """
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError('fresh candidate neighborhood observer build required')
    receipt = json.loads((base_observer/'build.private.json').read_text())
    base, old_archive = base_observer/'mrisurf_defect.cpp', base_observer/'libutils.a'
    if sha(base) != receipt['patched_defect_sha256'] or sha(old_archive) != receipt['archive_sha256']:
        raise ValueError('frozen complete fitness observer source changed')
    text = base.read_text()
    anchor = '    /*compute the second fundamental form */\n    computeDefectSecondFundamentalForm(mris, &dp->tp);'
    if text.count(anchor) != 1:
        raise ValueError('fixed complete fitness curvature call anchor changed')
    text = text.replace(anchor, CAPTURE + anchor, 1)
    output.mkdir(parents=True)
    patched = output/'mrisurf_defect.cpp'; patched.write_text(text)
    command = list(receipt['compile_argv']); obj = output/'mrisurf_defect.cpp.o'
    for flag, value in [('-o', str(obj)), ('-c', str(patched)), ('-MF', str(obj)+'.d'), ('-MT', str(obj))]:
        command[command.index(flag)+1] = value
    archive = output/'libutils.a'; shutil.copyfile(old_archive, archive)
    link = [str(archive) if x == str(old_archive) else x for x in receipt['link_argv']]
    binary = output/'mris_fix_topology_neighborhood_observer'; link[link.index('-o')+1] = str(binary)
    started = time.perf_counter(); compiler = Path(command[0])
    with (output/'build.log').open('w') as log:
        subprocess.run(command, cwd=native_build/'build', stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')), 'r', str(archive), str(obj)], check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')), str(archive)], check=True)
        subprocess.run(link, cwd=native_build/'build', stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(scope='first complete fitness candidate before curvature fit; not controller/GA completion',
        status='complete_readonly_candidate_neighborhood_build', base_receipt_sha256=sha(base_observer/'build.private.json'),
        observer_builder_sha256=sha(__file__), patched_defect_sha256=sha(patched), archive_sha256=sha(archive),
        observer_program_sha256=sha(binary), compile_argv=command, link_argv=link,
        seconds=time.perf_counter()-started, format='FNITNBR1: header3; offsets N+1; vertex topology Nx3 vnum/vtotal/nsizeCur; ordered neighbors; fields Nx15 origXYZ/normal/e1/e2/k1/k2/area; selected vertex order')
    (output/'build.private.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['native-build','base-observer','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__=='__main__':
    main()
