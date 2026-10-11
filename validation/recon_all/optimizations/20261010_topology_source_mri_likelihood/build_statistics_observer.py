"""隔离构建固定源码的完整初始surface-statistics观察器；只供benchmark。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


OBSERVE = r'''
  static bool fnit_statistics_captured=false;
  const char* fnit_statistics_path=getenv("FNIT_TOPOLOGY_STATISTICS_CAPTURE");
  if(fnit_statistics_path && !fnit_statistics_captured) {
    fnit_statistics_captured=true;
    FILE* capture=fopen(fnit_statistics_path,"wb");
    if(!capture) abort();
    int neighbors=0;
    for(int vno=0;vno<mris->nvertices;vno++) neighbors+=mris->vertices_topology[vno].vnum;
    const char magic[8]={'F','N','I','T','F','I','T','1'};
    if(fwrite(magic,1,8,capture)!=8) abort();
    int header[14]={mris->nvertices,mris->nfaces,tp.nvertices,tp.nfaces,neighbors,
      h_dot->nbins,h_k1->nbins,h_k2->nbins,
      mri_k1_k2->width,mri_k1_k2->height,mri_k1_k2->depth,
      mri_gray_white->width,mri_gray_white->height,mri_gray_white->depth};
    if(fwrite(header,sizeof(int),14,capture)!=14) abort();
    #if MATRIX_ALLOCATION
    for(int row=1;row<=4;row++) for(int col=1;col<=4;col++) {
      float entry=*MATRIX_RELT(VoxelFromSRASmatrix,row,col);
      if(fwrite(&entry,sizeof(float),1,capture)!=1) abort();
    }
    #else
    abort(); // this recorder requires the fixed actual topology branch
    #endif
    for(int vno=0;vno<mris->nvertices;vno++) {
      const VERTEX* vv=&mris->vertices[vno];
      float fields[16]={vv->origx,vv->origy,vv->origz,vv->x,vv->y,vv->z,
        vv->nx,vv->ny,vv->nz,vv->area,vv->k1,vv->k2,
        vv->val2,vv->val2bak,(float)vv->marked,(float)vv->ripflag};
      if(fwrite(fields,sizeof(float),16,capture)!=16) abort();
    }
    for(int fno=0;fno<mris->nfaces;fno++) {
      if(fwrite(mris->faces[fno].v,sizeof(int),3,capture)!=3) abort();
    }
    for(int fno=0;fno<mris->nfaces;fno++) {
      const FaceNormCacheEntry* normal=getFaceNorm(mris,fno);
      float fields[4]={normal->nx,normal->ny,normal->nz,mris->faces[fno].area};
      if(fwrite(fields,sizeof(float),4,capture)!=4) abort();
    }
    int offset=0;
    for(int vno=0;vno<mris->nvertices;vno++) {
      if(fwrite(&offset,sizeof(int),1,capture)!=1) abort();
      offset+=mris->vertices_topology[vno].vnum;
    }
    if(fwrite(&offset,sizeof(int),1,capture)!=1) abort();
    for(int vno=0;vno<mris->nvertices;vno++) {
      const VERTEX_TOPOLOGY* vt=&mris->vertices_topology[vno];
      if(fwrite(vt->v,sizeof(int),vt->vnum,capture)!=(size_t)vt->vnum) abort();
    }
    if(fwrite(tp.vertices,sizeof(int),tp.nvertices,capture)!=(size_t)tp.nvertices) abort();
    if(fwrite(tp.faces,sizeof(int),tp.nfaces,capture)!=(size_t)tp.nfaces) abort();
    HISTOGRAM* histograms[3]={h_dot,h_k1,h_k2};
    for(int h=0;h<3;h++) {
      HISTOGRAM* histogram=histograms[h];
      float bin_size=histogram->bin_size;
      if(fwrite(&bin_size,sizeof(float),1,capture)!=1) abort();
      if(fwrite(histogram->bins,sizeof(float),histogram->nbins,capture)!=(size_t)histogram->nbins) abort();
      if(fwrite(histogram->counts,sizeof(float),histogram->nbins,capture)!=(size_t)histogram->nbins) abort();
    }
    float geometry[4]={mri_k1_k2->xstart,mri_k1_k2->ystart,mri_k1_k2->xsize,mri_k1_k2->ysize};
    if(fwrite(geometry,sizeof(float),4,capture)!=4) abort();
    MRI* volumes[2]={mri_k1_k2,mri_gray_white};
    for(int vi=0;vi<2;vi++) for(int x=0;x<volumes[vi]->width;x++)
      for(int y=0;y<volumes[vi]->height;y++) for(int z=0;z<volumes[vi]->depth;z++) {
        float value=MRIFvox(volumes[vi],x,y,z);
        if(fwrite(&value,sizeof(float),1,capture)!=1) abort();
      }
    double scores[8]={tp.face_ll,tp.vertex_ll,tp.curv_ll,tp.qcurv_ll,tp.fll,tp.vll,tp.cll,tp.qcll};
    if(fwrite(scores,sizeof(double),8,capture)!=8) abort();
    int score_sizes[8]={sizeof(tp.face_ll),sizeof(tp.vertex_ll),sizeof(tp.curv_ll),sizeof(tp.qcurv_ll),
      sizeof(tp.fll),sizeof(tp.vll),sizeof(tp.cll),sizeof(tp.qcll)};
    if(fwrite(score_sizes,sizeof(int),8,capture)!=8) abort();
    if(fclose(capture)!=0) abort();
    fprintf(stderr,"FNIT_TOPOLOGY_STATISTICS_CAPTURE_USED\n");
  }
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, native_build, base_observer, output):
    """固定build与已测active观察器→新编译观察器/源码程序SHA与完整argv。"""
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError("fresh benchmark build output required")
    receipt = json.loads((base_observer / "build.private.json").read_text())
    base, base_archive = base_observer / "mrisurf_defect.cpp", base_observer / "libutils.a"
    if sha(base) != receipt["patched_defect_sha256"] or sha(base_archive) != receipt["archive_sha256"]:
        raise ValueError("frozen active-transform observer source/archive changed")
    text = base.read_text()
    anchor = "  mrisDefectVertexMRILogLikelihood(mris, mri, &tp, NULL, NULL, NULL, mri_gray_white);\n"
    if text.count(anchor) != 1:
        raise ValueError("complete initial surface statistics capture site mismatch")
    output.mkdir(parents=True)
    patched = output / "mrisurf_defect.cpp"
    patched.write_text(text.replace(anchor, anchor + OBSERVE, 1))
    command = list(receipt["compile_argv"])
    obj = output / "mrisurf_defect.cpp.o"
    for flag, value in [("-o", str(obj)), ("-c", str(patched)), ("-MF", str(obj) + ".d"), ("-MT", str(obj))]:
        command[command.index(flag) + 1] = value
    archive = output / "libutils.a"
    shutil.copyfile(base_archive, archive)
    link = [str(archive) if token == str(base_archive) else token for token in receipt["link_argv"]]
    binary = output / "mris_fix_topology_statistics_observer"
    link[link.index("-o") + 1] = str(binary)
    compiler = Path(command[0]); started = time.perf_counter()
    with (output / "build.log").open("w") as log:
        subprocess.run(command, cwd=native_build / "build", stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ar")), "r", str(archive), str(obj)], check=True)
        subprocess.run([str(compiler.with_name("x86_64-conda-linux-gnu-ranlib")), str(archive)], check=True)
        subprocess.run(link, cwd=native_build / "build", stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(status="complete_benchmark_statistics_observer", scope="read-only full initial likelihood fields; not candidate fitness or GA completion",
        base_receipt_sha256=sha(base_observer / "build.private.json"), observer_builder_sha256=sha(__file__),
        patched_defect_sha256=sha(patched), archive_sha256=sha(archive), observer_program_sha256=sha(binary),
        compile_argv=command, link_argv=link, seconds=time.perf_counter() - started,
        vertex_fields=["origx","origy","origz","x","y","z","nx","ny","nz","area","k1","k2","white","gray","marked","rip"],
        score_fields=["face_ll","vertex_ll","curv_ll","qcurv_ll","fll","vll","cll","qcll"],
        binary_format="FNITFIT1; int32[14] header; float32 matrix; vertices(N,16); faces(F,3); face_normals_areas(F,4); CSR; selected vertices/faces; 3 histograms; kPDF geometry/data; gray_white_PDF; double[8] scores; int32[8] score sizes")
    (output / "build.private.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["native-build", "base-observer", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
