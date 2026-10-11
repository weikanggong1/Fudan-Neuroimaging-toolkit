"""只读捕获首个完整候选的平滑前、平滑后及MRI拟合后状态；仅benchmark。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from build_volume_observer import sha


CAPTURE = r'''
static DP* fnit_match_active_dp = NULL;
static void fnit_capture_candidate_match(MRIS* s, MRI* mri, DP* dp,
                                        int smooth, int match, int phase) {
  const char* base = getenv("FNIT_TOPOLOGY_CANDIDATE_MATCH_CAPTURE");
  static bool done[3] = {false, false, false};
  if (!base || fnit_match_active_dp != dp || done[phase]) { return; }
  done[phase] = true;
  char name[STRLEN];
  snprintf(name, sizeof(name), "%s.%s.bin", base,
           phase == 0 ? "before" : (phase == 1 ? "smooth" : "after"));
  FILE* f = fopen(name, "wb"); if (!f) abort();
  int neighbors = 0, incidents = 0;
  for (int v = 0; v < s->nvertices; v++) {
    neighbors += s->vertices_topology[v].vtotal;
    incidents += s->vertices_topology[v].num;
  }
  int h[10] = {s->nvertices, s->nfaces, neighbors, incidents,
               dp->tp.nvertices, dp->tp.nfaces, dp->tp.ninside,
               phase, smooth, match};
  fwrite("FNITMAT1", 1, 8, f); fwrite(h, sizeof(int), 10, f);
  float parameters[4] = {dp->defect->white_mean, dp->defect->gray_mean,
                         dp->defect->k1_mean, dp->defect->k2_mean};
  fwrite(parameters, sizeof(float), 4, f);
  MATRIX* map = GetSurfaceRASToVoxelMatrix(mri);
  for (int r = 1; r <= 4; r++) {
    for (int c = 1; c <= 4; c++) {
      double value = *MATRIX_RELT(map, r, c); fwrite(&value, sizeof(double), 1, f);
    }
  }
  MatrixFree(&map);
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX* p = &s->vertices[v];
    float fields[27] = {p->origx,p->origy,p->origz,p->x,p->y,p->z,
                       p->fx,p->fy,p->fz,p->nx,p->ny,p->nz,
                       p->e1x,p->e1y,p->e1z,p->e2x,p->e2y,p->e2z,
                       p->k1,p->k2,p->area,p->val,p->val2,p->val2bak,
                       p->curvbak,(float)p->undefval,(float)p->old_undefval};
    fwrite(fields, sizeof(float), 27, f);
  }
  for (int v = 0; v < s->nvertices; v++) {
    int fields[2] = {s->vertices[v].marked, s->vertices[v].ripflag};
    fwrite(fields, sizeof(int), 2, f);
  }
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX_TOPOLOGY* t = &s->vertices_topology[v];
    int fields[3] = {t->vnum, t->vtotal, t->nsizeCur};
    fwrite(fields, sizeof(int), 3, f);
  }
  for (int i = 0; i < s->nfaces; i++) {
    fwrite(s->faces[i].v, sizeof(int), 3, f);
    int rip = s->faces[i].ripflag; fwrite(&rip, sizeof(int), 1, f);
  }
  int offset = 0; fwrite(&offset, sizeof(int), 1, f);
  for (int v = 0; v < s->nvertices; v++) {
    offset += s->vertices_topology[v].vtotal; fwrite(&offset, sizeof(int), 1, f);
  }
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX_TOPOLOGY* t = &s->vertices_topology[v];
    fwrite(t->v, sizeof(int), t->vtotal, f);
  }
  offset = 0; fwrite(&offset, sizeof(int), 1, f);
  for (int v = 0; v < s->nvertices; v++) {
    offset += s->vertices_topology[v].num; fwrite(&offset, sizeof(int), 1, f);
  }
  for (int v = 0; v < s->nvertices; v++) {
    const VERTEX_TOPOLOGY* t = &s->vertices_topology[v];
    fwrite(t->f, sizeof(int), t->num, f);
    for (int i = 0; i < t->num; i++) {
      int corner = t->n[i]; fwrite(&corner, sizeof(int), 1, f);
    }
  }
  fwrite(dp->tp.vertices, sizeof(int), dp->tp.nvertices, f);
  fwrite(dp->tp.faces, sizeof(int), dp->tp.nfaces, f);
  if (fclose(f) != 0) abort();
}
'''


def build(*, native_build, base_observer, output):
    """在固定源码独立目录新增三phase观测，所有原运算保持不变。

    native_build 是固定Conda源码构建目录；base_observer是已哈希绑定的
    complete-fitness观察器；output必须不存在。写出独立观察器程序和
    私有编译收据，不改原程序/源码/结果。运行显式设置capture前缀，记录
    首次mrisDefectPatchFitness的实际orig/current/fx、完整原邻接和入射面。
    尚未完成编译/真实repeat的版本不能作为参考。目录/anchor/SHA失败报错。
    """
    native_build, base_observer, output = map(Path, (native_build, base_observer, output))
    if output.exists():
        raise FileExistsError('fresh candidate match observer build required')
    receipt = json.loads((base_observer/'build.private.json').read_text())
    base, old_archive = base_observer/'mrisurf_defect.cpp', base_observer/'libutils.a'
    if sha(base) != receipt['patched_defect_sha256'] or sha(old_archive) != receipt['archive_sha256']:
        raise ValueError('frozen complete fitness observer changed')
    text = base.read_text()
    declaration = 'static void defectMatch(MRI *mri, MRI_SURFACE *mris, DP *dp, int smooth, int match);'
    definition = 'static void defectMatch(MRI *mri, MRI_SURFACE *mris, DP *dp, int smooth, int match)\n{'
    call = '  defectMatch(mri, mris_corrected, dp, parms->smooth, parms->match);\n\n  computeDisplacement(mris_corrected, dp);'
    if any(text.count(a) != 1 for a in [declaration, definition, call]):
        raise ValueError('fixed candidate match anchors changed')
    text = text.replace(declaration, CAPTURE+'\n'+declaration, 1)
    text = text.replace(call,
        '  fnit_match_active_dp = dp;\n'
        '  fnit_capture_candidate_match(mris_corrected, mri, dp, parms->smooth, parms->match, 0);\n'
        '  defectMatch(mri, mris_corrected, dp, parms->smooth, parms->match);\n'
        '  fnit_capture_candidate_match(mris_corrected, mri, dp, parms->smooth, parms->match, 2);\n'
        '  fnit_match_active_dp = NULL;\n\n  computeDisplacement(mris_corrected, dp);', 1)
    first_smooth = '    defectSmooth(mris, dp, 25, 0.1, smooth);\n  }\n  if (match) {'
    if text.count(first_smooth) != 1:
        raise ValueError('fixed candidate smooth transition changed')
    text = text.replace(first_smooth,
        '    defectSmooth(mris, dp, 25, 0.1, smooth);\n  }\n'
        '  fnit_capture_candidate_match(mris, mri, dp, smooth, match, 1);\n  if (match) {', 1)
    output.mkdir(parents=True)
    patched = output/'mrisurf_defect.cpp'; patched.write_text(text)
    command = list(receipt['compile_argv']); obj = output/'mrisurf_defect.cpp.o'
    for flag, value in [('-o', str(obj)), ('-c', str(patched)), ('-MF', str(obj)+'.d'), ('-MT', str(obj))]:
        command[command.index(flag)+1] = value
    archive = output/'libutils.a'; shutil.copyfile(old_archive, archive)
    link = [str(archive) if x == str(old_archive) else x for x in receipt['link_argv']]
    binary = output/'mris_fix_topology_match_observer'; link[link.index('-o')+1] = str(binary)
    compiler = Path(command[0]); started = time.perf_counter()
    with (output/'build.log').open('w') as log:
        subprocess.run(command, cwd=native_build/'build', stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ar')), 'r', str(archive), str(obj)], check=True)
        subprocess.run([str(compiler.with_name('x86_64-conda-linux-gnu-ranlib')), str(archive)], check=True)
        subprocess.run(link, cwd=native_build/'build', stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(scope='first complete candidate before/smooth/after MRI match; not complete GA',
        status='complete_readonly_candidate_match_build', base_receipt_sha256=sha(base_observer/'build.private.json'),
        observer_builder_sha256=sha(__file__), patched_defect_sha256=sha(patched), archive_sha256=sha(archive),
        observer_program_sha256=sha(binary), compile_argv=command, link_argv=link,
        seconds=time.perf_counter()-started, format='FNITMAT1 header10; parameters4; matrix16D; vertices Nx27F; flags Nx2; topology Nx3; faces Fx4; vtotal CSR; incident face/corner CSR; selected V/F')
    (output/'build.private.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['native-build', 'base-observer', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    build(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
