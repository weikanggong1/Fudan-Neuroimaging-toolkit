"""Build an opt-in full native EM executable with FNIT single-input search cache.

Use the existing fixed-source Conda build only as read-only compile/link inputs.
All candidate source, dependency files, link map and executable go to output.
No upstream source or shared archive is modified; no system executable is copied.
"""
import argparse,hashlib,json,os,shlex,subprocess
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
UPSTREAM_COMMIT = 'd932c45b7941662ea380a05efef580568b98d41a'
UPSTREAM_EMUTILS_SHA256 = '132028d8e2c2ba3f6ab2b07380624fb22b8065450c089216f5aafbba3829019b'


def patch_source_text(text):
    patched=text
    anchor='#include "emregisterutils.h"'
    assert patched.count(anchor)==1
    patched=patched.replace(anchor,anchor+'\n#include "mri_em_register_native_score.hpp"')
    anchor='''      result = GCAcomputeLogSampleProbability( gca, gcas, mri,
\t\t\t\t\t       transform, nsamples, clamp );'''
    assert patched.count(anchor)==1
    patched=patched.replace(anchor,'''    {
      const char *backend = getenv("FNIT_GCA_SCORER");
      if (backend && !strcmp(backend, "cpu_cached") &&
          gca->ninputs == 1 && mri->type == MRI_UCHAR && mri->nframes == 1)
        result = fnit_gca_search::score(gca, gcas, mri, transform, nsamples, clamp);
      else
        result = GCAcomputeLogSampleProbability(gca, gcas, mri, transform, nsamples, clamp);
    }''')
    return patched


def apply_native_search_patch(source_root):
    """Apply only to installer-owned source copy, after fixed-source validation.

    Write emregisterutils.cpp, one FNIT header and an idempotence manifest in
    mri_em_register/. Reject altered source/header on repeat instead of guessing.
    """
    source_root=Path(source_root).resolve()
    module=source_root/'mri_em_register';upstream=module/'emregisterutils.cpp'
    marker=module/'.fnit-cached-gca.json'
    header=Path(__file__).with_name('mri_em_register_native_score.hpp')
    if marker.exists():
        report=json.loads(marker.read_text())
        if sha(upstream)!=report['patched_source_sha256'] or sha(module/header.name)!=sha(header):
            raise ValueError('modified installed cache patch or header version mismatch')
        return report
    if sha(upstream)!=UPSTREAM_EMUTILS_SHA256:
        raise ValueError('unverified emregisterutils source SHA-256')
    patched=patch_source_text(upstream.read_text())
    upstream.write_text(patched)
    (module/header.name).write_bytes(header.read_bytes())
    report={'upstream_commit':UPSTREAM_COMMIT,'upstream_source_sha256':UPSTREAM_EMUTILS_SHA256,
            'patched_source_sha256':sha(upstream),'fnit_header_sha256':sha(header),
            'patcher_sha256':sha(__file__),'activation':'FNIT_GCA_SCORER=cpu_cached',
            'capability_query':'FNIT_GCA_QUERY_CAPABILITIES=1 binary',
            'files':['mri_em_register/emregisterutils.cpp','mri_em_register/'+header.name]}
    marker.write_text(json.dumps(report,indent=2)+'\n')
    return report


def build_native_search(source_root, build_root, output, *, ninja):
    source_root=Path(source_root).resolve();build_root=Path(build_root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    upstream=source_root/'mri_em_register/emregisterutils.cpp'
    expected='132028d8e2c2ba3f6ab2b07380624fb22b8065450c089216f5aafbba3829019b'
    if sha(upstream)!=expected:raise ValueError('unverified emregisterutils source SHA-256')
    header=Path(__file__).with_name('mri_em_register_native_score.hpp')
    patched=patch_source_text(upstream.read_text())
    candidate=output/'emregisterutils.cpp';candidate.write_text(patched)
    (output/header.name).write_bytes(header.read_bytes())
    target='mri_em_register/CMakeFiles/mri_em_register.dir/emregisterutils.cpp.o'
    commands=subprocess.check_output([str(ninja),'-t','commands',target],cwd=build_root,text=True).splitlines()
    compile_line=next(line for line in reversed(commands) if '-c ' in line and str(upstream) in line)
    compile_args=shlex.split(compile_line)
    # Ninja's compile rule is one compiler invocation. Reject shell metacharacters.
    if any(x in compile_args for x in ('&&',';','|')):raise ValueError('unexpected compile shell command')
    for flag in ('-o','-MF'):
        index=compile_args.index(flag)+1;compile_args[index]=str(output/('emregisterutils.o' if flag=='-o' else 'emregisterutils.d'))
    if '-MT' in compile_args:compile_args[compile_args.index('-MT')+1]=str(output/'emregisterutils.o')
    compile_args[compile_args.index(str(upstream))]=str(candidate)
    compile_args.extend(['-I'+str(output),'-I'+str(source_root/'mri_em_register')])
    subprocess.run(compile_args,cwd=build_root,check=True)
    commands=subprocess.check_output([str(ninja),'-t','commands','mri_em_register/mri_em_register'],cwd=build_root,text=True).splitlines()
    link_line=next(line for line in reversed(commands) if '-o mri_em_register/mri_em_register' in line)
    parts=link_line.split(' && ')
    link_args=shlex.split(next(part for part in parts if '-o mri_em_register/mri_em_register' in part))
    link_args[link_args.index(target)]=str(output/'emregisterutils.o')
    executable=output/'mri_em_register_fnit_cached'
    link_args[link_args.index('-o')+1]=str(executable)
    link_args=[('-Wl,-Map,'+str(output/'ld_map.txt')) if x=='-Wl,-Map,ld_map.txt' else x for x in link_args]
    if any(x in link_args for x in ('&&',';','|')):raise ValueError('unexpected link shell command')
    subprocess.run(link_args,cwd=build_root,check=True)
    report={'upstream_commit':UPSTREAM_COMMIT,'upstream_source_sha256':sha(upstream),'patched_source_sha256':sha(candidate),'fnit_header_sha256':sha(header),'builder_sha256':sha(__file__),'binary_sha256':sha(executable),'compile_argv':compile_args,'link_argv':link_args,'optimizer':'unchanged complete native stable/all-sample EM','activation':'FNIT_GCA_SCORER=cpu_cached','baseline_fallback':'environment unset, or unsupported non-uint8/multi-input path', 'link_input_sha256': {str((build_root/x).resolve()):sha(build_root/x) for x in link_args if x.endswith(('.o','.a')) and (build_root/x).is_file()}, 'upstream_gca_sha256':sha(source_root/'utils/gca.cpp'), 'upstream_gca_header_sha256':sha(source_root/'include/gca.h')}
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    return executable

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root');p.add_argument('--build-root');p.add_argument('--output');p.add_argument('--ninja');p.add_argument('--apply-source-root');a=p.parse_args()
    if a.apply_source_root:
        print(json.dumps(apply_native_search_patch(a.apply_source_root),indent=2))
    elif all((a.source_root,a.build_root,a.output,a.ninja)):
        print(build_native_search(a.source_root,a.build_root,a.output,ninja=a.ninja))
    else: p.error('use --apply-source-root, or all four standalone build arguments')
if __name__=='__main__':main()
