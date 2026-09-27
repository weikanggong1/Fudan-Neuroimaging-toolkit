"""Build a copied-source first-step gradient probe against pinned libraries."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--main-object", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-iteration", type=int, default=0)
    parser.add_argument("--capture-through", type=int)
    parser.add_argument("--capture-white-repulsion", action="store_true")
    args = parser.parse_args()
    if args.capture_through is not None and args.capture_through < 1:
        parser.error("--capture-through must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    source = (args.source / "utils/mrisurf_mri.cpp").read_text()
    prefix, source = source.split("// #POS", 1)
    marker = "    MRISclearGradient(mris);"
    if source.count(marker) != 2:
        raise RuntimeError("expected two optimizer gradient loops")
    dump = r'''
    auto dump_first_gradient = [&](const char *stage) {
      const char *prefix = getenv("PLACE_GRAD_PREFIX");
      if (!prefix || CAPTURE_GUARD) return;
      char path[STRLEN];
      CAPTURE_PATH
      FILE *output = fopen(path, "wb");
      if (!output) exit(1);
      for (int vertex = 0; vertex < mris->nvertices; vertex++) {
        const VERTEX &v = mris->vertices[vertex];
        float values[] = {v.x, v.y, v.z, v.nx, v.ny, v.nz, v.dx, v.dy, v.dz};
        int flags[] = {v.ripflag, v.border, v.neg};
        fwrite(values, sizeof(values), 1, output);
        fwrite(flags, sizeof(flags), 1, output);
      }
      fclose(output);
      CAPTURE_CROPPED
      if (stage[0] == 'c') {
        snprintf(path, STRLEN, "%s.ps.mgz", prefix);
        MRIwrite(mri_brain, path);
        snprintf(path, STRLEN, "%s.intensity_input", prefix);
        FILE *intensity = fopen(path, "wb");
        if (!intensity) exit(1);
        for (int vertex = 0; vertex < mris->nvertices; vertex++) {
          const VERTEX &v = mris->vertices[vertex];
          float values[] = {v.val, v.val2};
          fwrite(values, sizeof(values), 1, intensity);
        }
        fclose(intensity);
        snprintf(path, STRLEN, "%s.orig_normals", prefix);
        FILE *original_normals = fopen(path, "wb");
        if (!original_normals) exit(1);
        for (int vertex = 0; vertex < mris->nvertices; vertex++) {
          float normal[3] = {0.0f, 0.0f, 0.0f};
          mrisComputeOrigNormal(mris, vertex, normal);
          fwrite(normal, sizeof(normal), 1, original_normals);
        }
        fclose(original_normals);
      }
      if (stage[4] == 'n') {
        snprintf(path, STRLEN, "%s.neighbors", prefix);
        FILE *topology = fopen(path, "wb");
        if (!topology) exit(1);
        for (int vertex = 0; vertex < mris->nvertices; vertex++) {
          const VERTEX_TOPOLOGY &vt = mris->vertices_topology[vertex];
          fwrite(&vt.vnum, sizeof(vt.vnum), 1, topology);
          fwrite(vt.v, sizeof(vt.v[0]), vt.vnum, topology);
        }
        fclose(topology);
        snprintf(path, STRLEN, "%s.neighbors_total", prefix);
        FILE *all_topology = fopen(path, "wb");
        if (!all_topology) exit(1);
        for (int vertex = 0; vertex < mris->nvertices; vertex++) {
          const VERTEX_TOPOLOGY &vt = mris->vertices_topology[vertex];
          fwrite(&vt.vtotal, sizeof(vt.vtotal), 1, all_topology);
          fwrite(vt.v, sizeof(vt.v[0]), vt.vtotal, all_topology);
        }
        fclose(all_topology);
      }
    };
    dump_first_gradient("clear");'''
    if args.capture_through is None:
        dump = dump.replace("CAPTURE_GUARD", f"n != {args.capture_iteration}")
        dump = dump.replace("CAPTURE_PATH", 'snprintf(path, STRLEN, "%s.%s", prefix, stage);')
        dump = dump.replace("CAPTURE_CROPPED", "")
    else:
        dump = dump.replace("CAPTURE_GUARD", f"n >= {args.capture_through}")
        dump = dump.replace("CAPTURE_PATH", 'snprintf(path, STRLEN, "%s.step%02d.%s", prefix, n + 1, stage);')
        crop = """
      if (strcmp(stage, "clear") == 0 || strcmp(stage, "after_collision") == 0) {
        snprintf(path, STRLEN, "%s.step%02d.%s.cropped", prefix, n + 1, stage);
        FILE *crops = fopen(path, "wb");
        if (!crops) exit(1);
        for (int vertex = 0; vertex < mris->nvertices; vertex++) {
          int cropped = mris->vertices[vertex].cropped;
          fwrite(&cropped, sizeof(cropped), 1, crops);
        }
        fclose(crops);
      }"""
        dump = dump.replace("CAPTURE_CROPPED", crop)
        end_step = "    mrisTrackTotalDistanceNew(mris); /* computes signed deformation amount */"
        if source.count(end_step) != 1:
            raise RuntimeError("missing first placement post-collision marker")
        source = source.replace(end_step, end_step + '\n    dump_first_gradient("after_collision");', 1)
        initial = "  showDtSSeRms(parms->fp, -1, 0.0, sse, rms, -1.0, __LINE__);"
        step = "    showDtSSeRms(parms->fp, n, delta_t, sse, rms, last_rms, __LINE__);"
        if source.count(initial) != 1 or source.count(step) != 2:
            raise RuntimeError("missing first placement objective markers")
        source = source.replace(initial, '  fprintf(stderr, "PY_OBJ_REF initial rms=%.17g sse=%.17g orig_area=%.17g total_area=%.17g\\n", rms, sse, mris->orig_area, mris->total_area);\n' + initial)
        source = source.replace(step, f'    if (n < {args.capture_through}) fprintf(stderr, "PY_OBJ_REF step%d rms=%.17g sse=%.17g orig_area=%.17g total_area=%.17g\\n", n + 1, rms, sse, mris->orig_area, mris->total_area);\n' + step, 1)
    source = source.replace(marker, marker + dump, 1)
    stages = (
        ("mrisComputeIntensityTerm(mris, l_intensity, mri_brain, mri_smooth, parms->sigma, parms);", "intensity"),
        ("mrisComputeSurfaceRepulsionTerm(mris, parms->l_surf_repulse, mht_v_orig);", "surface_repulsion"),
        ("mrisComputeNormalSpringTerm(mris, parms->l_nspring);", "normal_spring"),
        ("mrisComputeQuadraticCurvatureTerm(mris, parms->l_curv);", "curvature"),
        ("mrisComputeTangentialSpringTerm(mris, parms->l_tspring);", "tangential_spring"),
    )
    for needle, name in stages:
        if source.count(needle) < 1:
            raise RuntimeError(f"missing source optimizer term: {name}")
        diagnostic = needle + f'\n    dump_first_gradient("{name}");'
        if name in ("normal_spring", "tangential_spring"):
            diagnostic = f'dump_first_gradient("pre_{name}");\n    ' + diagnostic
        source = source.replace(needle, diagnostic, 1)
    if args.capture_white_repulsion:
        for needle, name in (
            ("mrisAverageSignedGradients(mris, avgs);", "after_signed_average"),
            ("mrisComputeRepulsiveTerm(mris, parms->l_repulse, mht_v_current, mht_f_current);",
             "after_self_repulsion"),
        ):
            if source.count(needle) < 1:
                raise RuntimeError(f"missing source optimizer term: {name}")
            source = source.replace(needle, needle + f'\n    dump_first_gradient("{name}");', 1)
    patched = args.out / "mrisurf_mri_gradient_probe.cpp"
    patched.write_text(prefix + "// #POS" + source)

    target = args.build / "utils/CMakeFiles/utils.dir"
    definitions = {}
    for line in (target / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            definitions[key] = shlex.split(value)
    definitions["CXX_INCLUDES"] = [
        token.replace("/tmp/fs_full_source_d932", str(args.source))
        for token in definitions["CXX_INCLUDES"]
    ]
    compiler = "/home1/gongwk/anaconda3/bin/x86_64-conda-linux-gnu-g++"
    compiled = args.out / "mrisurf_mri_gradient_probe.o"
    command = [compiler]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        command += definitions[key]
    command += ["-I", str(args.source / "include"), "-I", str(args.source / "utils"), "-c", str(patched), "-o", str(compiled)]
    subprocess.run(command, check=True)

    spring_source = (args.source / "utils/mrisurf_compute_dxyz.cpp").read_text()
    repulse_start = spring_source.index("int mrisComputeSurfaceRepulsionTerm(")
    repulse_end = spring_source.index("int mrisComputeWhichSurfaceRepulsionTerm(", repulse_start)
    repulse = spring_source[repulse_start:repulse_end]
    marker = "  for (vno = 0; vno < mris->nvertices; vno++) {"
    assert repulse.count(marker) == 1
    repulse = repulse.replace(marker, r"""  FILE *bucket_output = NULL;
  if (getenv("PLACE_REPULSE_BUCKETS")) {
    bucket_output = fopen(getenv("PLACE_REPULSE_BUCKETS"), "wb");
    if (!bucket_output) exit(1);
  }
""" + marker)
    marker = "    sx = sy = sz = 0.0;"
    assert repulse.count(marker) == 1
    repulse = repulse.replace(marker, r"""    if (bucket_output) {
      fwrite(&vno, sizeof(vno), 1, bucket_output);
      fwrite(&bucket->nused, sizeof(bucket->nused), 1, bucket_output);
      for (int bucket_index = 0; bucket_index < bucket->nused; bucket_index++) {
        fwrite(&bucket->bins[bucket_index].fno, sizeof(int), 1, bucket_output);
      }
    }
""" + marker)
    marker = "  return (NO_ERROR);"
    assert repulse.count(marker) == 2
    # The first return belongs to the FZERO guard; close only at function exit.
    index = repulse.rindex(marker)
    repulse = repulse[:index] + "  if (bucket_output) fclose(bucket_output);\n" + repulse[index:]
    spring_source = spring_source[:repulse_start] + repulse + spring_source[repulse_end:]
    curv_start = spring_source.index("int mrisComputeQuadraticCurvatureTerm(")
    curv_end = spring_source.index("int mrisComputeSurfaceNormalIntersectionTerm(", curv_start)
    curv = spring_source[curv_start:curv_end]
    marker = "  mrisComputeTangentPlanes(mris);"
    assert curv.count(marker) == 1
    curv = curv.replace(marker, marker + r"""
  if (getenv("PLACE_CURV_PREFIX")) {
    char path[STRLEN];
    snprintf(path, STRLEN, "%s.tangent_basis", getenv("PLACE_CURV_PREFIX"));
    FILE *basis = fopen(path, "wb");
    if (!basis) exit(1);
    for (int vertex = 0; vertex < mris->nvertices; vertex++) {
      const VERTEX &v = mris->vertices[vertex];
      float values[] = {v.e1x, v.e1y, v.e1z, v.e2x, v.e2y, v.e2z};
      fwrite(values, sizeof(values), 1, basis);
    }
    fclose(basis);
  }
""")
    marker = "  int vno;"
    assert curv.count(marker) == 1
    curv = curv.replace(marker, marker + r"""
  float *curvature_scalar = getenv("PLACE_CURV_PREFIX") ? (float*)calloc(mris->nvertices, sizeof(float)) : NULL;
""")
    marker = "    v_P = MatrixMultiply(m_X_inv, v_Y, v_P);"
    assert curv.count(marker) == 1
    curv = curv.replace(marker, marker + r"""
    if (getenv("PLACE_CURV_DEBUG") &&
        (vno == 42106 || vno == 51228 || vno == 101227 || vno == 43743 || vno == 66029)) {
      char path[STRLEN];
      snprintf(path, STRLEN, "%s.%d", getenv("PLACE_CURV_DEBUG"), vno);
      FILE *debug = fopen(path, "wb");
      if (!debug) exit(1);
      int count = vt->vtotal;
      fwrite(&count, sizeof(count), 1, debug);
      for (int row = 1; row <= count; row++)
        for (int col = 1; col <= 5; col++) {
          float value = *MATRIX_RELT(m_X, row, col);
          fwrite(&value, sizeof(value), 1, debug);
        }
      for (int row = 1; row <= count; row++) {
        float value = VECTOR_ELT(v_Y, row);
        fwrite(&value, sizeof(value), 1, debug);
      }
      MATRIX *transposed = MatrixTranspose(m_X, NULL);
      MATRIX *gram = MatrixMultiply(transposed, m_X, NULL);
      MATRIX *inverse = MatrixInverse(gram, NULL);
      for (int row = 1; row <= 5; row++)
        for (int col = 1; col <= 5; col++) {
          float value = *MATRIX_RELT(gram, row, col);
          fwrite(&value, sizeof(value), 1, debug);
        }
      for (int row = 1; row <= 5; row++)
        for (int col = 1; col <= 5; col++) {
          float value = *MATRIX_RELT(inverse, row, col);
          fwrite(&value, sizeof(value), 1, debug);
        }
      for (int row = 1; row <= 5; row++)
        for (int col = 1; col <= count; col++) {
          float value = *MATRIX_RELT(m_X_inv, row, col);
          fwrite(&value, sizeof(value), 1, debug);
        }
      for (int row = 1; row <= 5; row++) {
        float value = VECTOR_ELT(v_P, row);
        fwrite(&value, sizeof(value), 1, debug);
      }
      fclose(debug);
      MatrixFree(&inverse);
      MatrixFree(&gram);
      MatrixFree(&transposed);
    }
""")
    marker = "    e *= l_curv;"
    assert curv.count(marker) == 1
    curv = curv.replace(marker, marker + r"""
    if (curvature_scalar) curvature_scalar[vno] = e;
""")
    marker = "  return (NO_ERROR);"
    assert curv.count(marker) == 2
    index = curv.rindex(marker)
    curv = curv[:index] + r"""  if (curvature_scalar) {
    char path[STRLEN];
    snprintf(path, STRLEN, "%s.scalar", getenv("PLACE_CURV_PREFIX"));
    FILE *scalar = fopen(path, "wb");
    if (!scalar) exit(1);
    fwrite(curvature_scalar, sizeof(float), mris->nvertices, scalar);
    fclose(scalar);
    free(curvature_scalar);
  }
""" + curv[index:]
    spring_source = spring_source[:curv_start] + curv + spring_source[curv_end:]
    for inline_name, next_name, stage in (
        ("inline void vertexComputeNormalSpringTerm(", "void mrisComputeNormalSpringTerm(", "normal"),
        ("inline void vertexComputeTangentialSpringTerm(", "void mrisComputeTangentialSpringTerm(", "tangent"),
    ):
        start, end = spring_source.index(inline_name), spring_source.index(next_name)
        block = spring_source[start:end]
        scalar = "  float nc = sx * nx + sy * ny + sz * nz;"
        if block.count(scalar) != 1:
            raise RuntimeError(f"missing spring normal projection: {stage}")
        block = block.replace(scalar, scalar + f"""
  if (getenv("PLACE_SPRING_DEBUG") && (vno == 0 || vno == 1 || vno == 11 || vno == 31 || vno == 132)) {{
    fprintf(stderr, "{stage} vno=%d sx=%a sy=%a sz=%a nx=%a ny=%a nz=%a nc=%a\\n",
            vno, sx, sy, sz, nx, ny, nz, nc);
  }}""")
        spring_source = spring_source[:start] + block + spring_source[end:]
    for function, next_function, stage in (
        ("void mrisComputeNormalSpringTerm(", "inline void vertexComputeTangentialSpringTerm(", "normal"),
        ("void mrisComputeTangentialSpringTerm(", "int mrisComputeNonlinearTangentialSpringTerm(", "tangent"),
    ):
        start, end = spring_source.index(function), spring_source.index(next_function)
        block = spring_source[start:end]
        guard = "  if (FZERO(l_spring)) return;"
        if block.count(guard) != 1:
            raise RuntimeError(f"missing spring guard: {stage}")
        block = block.replace(guard, guard + f"""
  FILE *gradient_output = NULL;
  if (getenv("PLACE_SPRING_PREFIX")) {{
    char path[STRLEN];
    snprintf(path, STRLEN, "%s.{stage}_delta", getenv("PLACE_SPRING_PREFIX"));
    gradient_output = fopen(path, "wb");
    if (!gradient_output) exit(1);
  }}""")
        term = f"    vertexCompute{'Normal' if stage == 'normal' else 'Tangential'}SpringTerm(mris, vno, &dx, &dy, &dz, l_spring);"
        if block.count(term) != 1:
            raise RuntimeError(f"missing spring term: {stage}")
        block = block.replace(term, term + """
    if (gradient_output) {
      float values[] = {dx, dy, dz};
      fwrite(values, sizeof(values), 1, gradient_output);
    }""")
        close = "    vertex->dz += dz;\n  }\n}"
        if block.count(close) != 1:
            raise RuntimeError(f"missing spring close: {stage}")
        block = block.replace(close, "    vertex->dz += dz;\n  }\n  if (gradient_output) fclose(gradient_output);\n}")
        spring_source = spring_source[:start] + block + spring_source[end:]
    spring_patched = args.out / "mrisurf_compute_dxyz_spring_probe.cpp"
    spring_patched.write_text(spring_source)
    spring_object = args.out / "mrisurf_compute_dxyz_spring_probe.o"
    spring_command = [compiler]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        spring_command += definitions[key]
    spring_command += ["-I", str(args.source / "include"), "-I", str(args.source / "utils"), "-c", str(spring_patched), "-o", str(spring_object)]
    subprocess.run(spring_command, check=True)

    average_source = (args.source / "utils/mrisurf_metricProperties.cpp").read_text()
    begin = average_source.index("int mrisAverageSignedGradients(")
    end = average_source.index("\n}\n", begin)
    block = average_source[begin:end]
    marker = "      ROMP_PF_end\n    }\n  if (Gdiag_no >= 0) {"
    assert block.count(marker) == 1
    block = block.replace(marker, r"""      ROMP_PF_end
      if (getenv("PLACE_AVG_PREFIX")) {
        char path[STRLEN];
        snprintf(path, STRLEN, "%s.%d", getenv("PLACE_AVG_PREFIX"), i + 1);
        FILE *averaged = fopen(path, "wb");
        if (!averaged) exit(1);
        for (int vertex = 0; vertex < mris->nvertices; vertex++) {
          const VERTEX &v = mris->vertices[vertex];
          float values[] = {v.dx, v.dy, v.dz};
          fwrite(values, sizeof(values), 1, averaged);
        }
        fclose(averaged);
      }
    }
  if (Gdiag_no >= 0) {""")
    average_source = average_source[:begin] + block + average_source[end:]
    average_patched = args.out / "mrisurf_metricProperties_average_probe.cpp"
    average_patched.write_text(average_source)
    average_object = args.out / "mrisurf_metricProperties_average_probe.o"
    average_command = [compiler]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        average_command += definitions[key]
    average_command += ["-I", str(args.source / "include"), "-I", str(args.source / "utils"), "-c", str(average_patched), "-o", str(average_object)]
    subprocess.run(average_command, check=True)

    link = shlex.split((args.build / "mris_make_surfaces/CMakeFiles/mris_place_surface.dir/link.txt").read_text())
    link = [
        str(args.main_object) if token == "CMakeFiles/mris_place_surface.dir/mris_place_surface.cpp.o"
        else f"-Wl,-Map,{args.out / 'link.map'}" if token == "-Wl,-Map,ld_map.txt"
        else token
        for token in link
    ]
    link.insert(link.index("../utils/libutils.a"), str(compiled))
    link.insert(link.index("../utils/libutils.a"), str(spring_object))
    link.insert(link.index("../utils/libutils.a"), str(average_object))
    link[link.index("-o") + 1] = str(args.out / "mris_place_surface_gradient_probe")
    subprocess.run(link, cwd=args.build / "mris_make_surfaces", check=True)
    print(args.out / "mris_place_surface_gradient_probe")


if __name__ == "__main__":
    main()
