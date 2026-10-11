"""固定源积分的只观察探针：导出指定步输入/梯度/搜索，随后退出。"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError("fixed-source observer anchor absent or ambiguous")
    return text.replace(anchor, replacement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.native_build = args.native_build.resolve()
    args.reference_prefix = args.reference_prefix.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    recipe = json.loads((args.reference_prefix / "build.json").read_text())
    if recipe["source_commit"] != "d932c45b7941662ea380a05efef580568b98d41a":
        raise ValueError("fixed-source version mismatch")
    source = args.native_build / "source/utils/mrisurf_integrate.cpp"
    original = source.read_text()
    helper = r'''
static bool fnit_observe_current(INTEGRATION_PARMS* p) {
  const char* step = getenv("FNIT_CAPTURE_INTEGRATION_T");
  return step && p->t == atoi(step);
}
static void fnit_integration_observer(MRIS* mris, INTEGRATION_PARMS* p, const char* phase) {
  const char* directory = getenv("FNIT_INTEGRATION_OBSERVER_DIRECTORY");
  if (!directory || !fnit_observe_current(p)) return;
  char output[4096]; snprintf(output, sizeof(output), "%s/%s.vertices.f32", directory, phase);
  FILE* file = fopen(output, "wb");
  if (!file) exit(90);
  for (int index=0; index<mris->nvertices; index++) {
    auto v = &mris->vertices[index];
    float fields[9] = {v->x,v->y,v->z,v->nx,v->ny,v->nz,v->dx,v->dy,v->dz};
    if (fwrite(fields,sizeof(float),9,file)!=9) exit(91);
  }
  fclose(file);
  snprintf(output, sizeof(output), "%s/%s.faces.f32", directory, phase);
  file = fopen(output,"wb"); if (!file) exit(92);
  for (int index=0;index<mris->nfaces;index++) {
    float fields[2] = {mris->faces[index].area,mris->faceNormCacheEntries[index].orig_area};
    if (fwrite(fields,sizeof(float),2,file)!=2) exit(93);
  }
  fclose(file);
  printf("FNIT_INTEGRATION_OBSERVER phase=%s t=%d vertices=%d faces=%d total_area=%a orig_area=%a neg_area=%a l_dist=%a l_area=%a l_nlarea=%a\n",phase,p->t,mris->nvertices,mris->nfaces,(double)mris->total_area,(double)mris->orig_area,(double)mris->neg_area,(double)p->l_dist,(double)p->l_area,(double)p->l_nlarea);
  fflush(stdout);
}
'''
    signature = "int MRISintegrate(MRI_SURFACE *mris, INTEGRATION_PARMS *parms, int n_averages)"
    text = once(original, signature, helper + "\n" + signature)
    begin = text.index(signature)
    end = text.index("return (parms->t - parms->start_t);", begin)
    fragment = text[begin:end]
    anchor = "    mrisComputeDistanceTerm(mris, parms);"
    fragment = once(fragment, anchor, anchor + "\n    fnit_integration_observer(mris, parms, \"distance\");")
    anchor = "    MRISaverageGradients(mris, n_averages);"
    fragment = once(fragment, anchor, "    fnit_integration_observer(mris, parms, \"area-before-average\");\n" + anchor)
    anchor = "    if (mris->status == MRIS_PLANE && mris->neg_area > 4 * mris->total_area)"
    fragment = once(fragment, anchor, "    if (getenv(\"FNIT_INTEGRATION_OBSERVER_DIRECTORY\") && getenv(\"FNIT_CAPTURE_INTEGRATION_T\") && t == atoi(getenv(\"FNIT_CAPTURE_INTEGRATION_T\"))) { printf(\"FNIT_CAPTURE_PREFIX_COMPLETE updates=%d\\n\", t+1); fflush(stdout); exit(0); }\n" + anchor)
    text = text[:begin] + fragment + text[end:]
    signature = "static double mrisLineMinimize(MRI_SURFACE *mris, INTEGRATION_PARMS *parms)\n{"
    text = once(text, signature, signature + "\n  fnit_integration_observer(mris, parms, \"averaged\");")
    begin = text.index(signature)
    end = text.index("static double mrisLineMinimizeSearch(", begin)
    fragment = text[begin:end]
    anchor = "double sse = MRIScomputeSSE_asThoughGradientApplied(mris, delta_t, parms, sseCtx);"
    fragment = fragment.replace(anchor, anchor + "\n      if (fnit_observe_current(parms)) printf(\"FNIT_NATIVE_TRIAL dt=%a sse=%a\\n\",delta_t,sse);")
    anchor = "    double sse2 = MRIScomputeSSE_asThoughGradientApplied(mris, dt2, parms, sseCtx);"
    fragment = once(fragment, anchor, anchor + "\n    if (fnit_observe_current(parms)) printf(\"FNIT_NATIVE_BRACKET dt0=%a sse0=%a dt2=%a sse2=%a mean_delta=%a min_dt=%a max_dt=%a starting_sse=%a\\n\",dt0,sse0,dt2,sse2,mean_delta,min_dt,max_dt,starting_sse);")
    anchor = "      double c = RVECTOR_ELT(mP, 3);"
    fragment = once(fragment, anchor, anchor + "\n      if (fnit_observe_current(parms)) printf(\"FNIT_NATIVE_QUADRATIC a=%a b=%a c=%a\\n\",a,b,c);")
    text = text[:begin] + fragment + text[end:]
    instrumented = args.output / "integration_observer.cpp"
    instrumented.write_text(text)
    compiler = Path(recipe["compiler"])
    lines = subprocess.check_output([str(compiler.parent / "ninja"), "-t", "commands"],
                                    cwd=args.native_build / "build", text=True).splitlines()
    line = next(line for line in lines if " -c " in line and line.endswith(str(source)))
    compile_args = shlex.split(line)
    # Moving one translation unit to the private observer directory changes
    # quote-include resolution; retain its original sibling-header directory.
    compile_args.insert(1, "-I" + str(source.parent))
    obj = args.output / "integration_observer.o"
    compile_args[compile_args.index("-c") + 1] = str(instrumented)
    for flag, value in [("-o", str(obj)), ("-MF", str(obj) + ".d"), ("-MT", str(obj))]:
        compile_args[compile_args.index(flag) + 1] = value
    link_args = list(recipe["programs"]["mris_sphere"]["link_argv"])
    binary = args.output / "mris_sphere_integration_observer"
    link_args[link_args.index("-o") + 1] = str(binary)
    link_args.insert(1, str(obj))
    for index, token in enumerate(link_args):
        if token.startswith("-Wl,-Map,"):
            link_args[index] = "-Wl,-Map," + str(args.output / "integration_observer.map")
    started = time.perf_counter()
    with (args.output / "build.log").open("w") as log:
        subprocess.run(compile_args, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(link_args, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = {"status": "complete", "scope": "benchmark-only source-order force observer, exit after captured update",
              "original_source_sha256": sha(source), "private_instrumented_source_sha256": sha(instrumented),
              "builder_sha256": sha(Path(__file__)), "reference_recipe_sha256": sha(args.reference_prefix / "build.json"),
              "program_sha256": sha(binary), "compile_argv": compile_args, "link_argv": link_args,
              "build_seconds": time.perf_counter() - started,
              "vertex_fields": ["x", "y", "z", "nx", "ny", "nz", "dx", "dy", "dz"],
              "face_fields": ["area", "orig_area"], "array_dtype": "native endian float32"}
    (args.output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print("INTEGRATION_OBSERVER_BUILT", report["program_sha256"])


if __name__ == "__main__":
    main()
