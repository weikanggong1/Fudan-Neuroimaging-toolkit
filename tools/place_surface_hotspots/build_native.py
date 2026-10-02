"""固定Conda构建上独立编译表面放置热点；不修改原源码/静态库/安装。

去掉MRISpositionSurface中只传给未读参数的face hash构建。动态异步MHT、
顶点repulse表、所有参数和有序接受控制流保留。先构建unmodified control，
再构建候选；patched完整上游文件仅留指定构建目录，不复制进仓库。
"""
from __future__ import annotations
import argparse,difflib,hashlib,json,shlex,shutil,subprocess
from pathlib import Path

EXPECTED={
 'utils/mrisurf_mri.cpp':'16740fce04268f8d4e7841592acf3ff89e394bf211e6c0ab0a7b541527a7c092',
 'utils/mrisurf_compute_dxyz.cpp':'a6f71a38bc3de6e8251449d2372883c08f77131f454224fae2d072dc9ab51cd9',
 'mris_make_surfaces/mris_place_surface.cpp':'b57f6e18f2e3215fc99fb1a8720aa345459585e4618ed2b8fa2779c2ee0f9dcf',
 'utils/mrishash.cpp':'fbb99a00df194d82b1ca0c7965767a38c38d67c884ecc14a25bd8342cc79397c',
 'include/mrishash.h':'b2c6f7757c9b0a8aeabdd7e2dbe6c3a08ae880277214bc33affd0a473ae53428',
 'utils/mrisurf_metricProperties.cpp':'458f896e7d2bdcbcdd7c9e73b2a396823d2015c723117d604bb06e3e8f27ba17',
}
CAPABILITIES={'schema_version':1,'program':'mris_place_surface_fnit_hotspot',
 'features':['skip-unconsumed-repulse-face-table'],
 'upstream_commit':'d932c45b7941662ea380a05efef580568b98d41a','uses_cuda':False}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def remove_unused_face_hash(content,repulsive_source):
 start=repulsive_source.index('\nint mrisComputeRepulsiveTerm(')
 end=repulsive_source.index('\nint mrisComputeRepulsiveRatioTerm(',start)
 body=repulsive_source[start:end]
 # Only the declaration may mention this argument. Reject any source drift
 # or future implementation that consumes the face table.
 if body.count('mht_faces')!=1:raise ValueError('repulsive face parameter now consumed; refuse optimization')
 start=content.index('int MRISpositionSurface(MRI_SURFACE')
 end=content.index('int MRISpositionSurface_mef(',start)
 body=content[start:end]
 needle='      MHTfree(&mht_f_current); mht_f_current = MHTcreateFaceTable(mris);'
 if body.count(needle)!=1 or body.count('mht_f_current')!=6:
  raise ValueError('unexpected placement face-table lifetime')
 replacement='''      // FNIT: mrisComputeRepulsiveTerm does not read its mht_faces argument.
      // Keep the distinct dynamic self-intersection MHT below unchanged.
      // No face table is needed for this gradient; mht_f_current stays NULL.'''
 body=body.replace(needle,replacement)
 return content[:start]+body+content[end:]

def add_capability_query(content):
 marker='int main(int argc, char **argv) \n{'
 if content.count(marker)!=1:raise ValueError('unexpected placement program entry')
 query=json.dumps(json.dumps(CAPABILITIES,separators=(',',':')))
 hook='\n  if (argc == 2 && !strcmp(argv[1], "--fnit-placement-capabilities")) {\n    puts('+query+');\n    return 0;\n  }'
 return content.replace(marker,marker+hook,1)

def compile_source(build,original,own_source,obj,commands):
 line=next(x for x in commands if ' -c ' in x and x.rstrip().endswith(str(original)))
 command=shlex.split(line);command.insert(1,'-I'+str(original.parent));compiler=Path(command[0])
 if 'conda' not in compiler.name:raise ValueError('required original Conda compiler')
 command[command.index('-o')+1]=str(obj);command[command.index('-c')+1]=str(own_source)
 for flag,suffix in (('-MF','.d'),('-MT','')):
  if flag in command:command[command.index(flag)+1]=str(obj)+suffix
 subprocess.run(command,cwd=build,check=True)
 return compiler,command

def build_one(build,source,output,commands,patched):
 output.mkdir(parents=True,exist_ok=False)
 original=source/'utils/mrisurf_mri.cpp';content=original.read_text()
 candidate=remove_unused_face_hash(content,(source/'utils/mrisurf_compute_dxyz.cpp').read_text()) if patched else content
 own_source=output/'mrisurf_mri.cpp';own_source.write_text(candidate)
 (output/'source.patch').write_text(''.join(difflib.unified_diff(content.splitlines(True),candidate.splitlines(True),fromfile='utils/mrisurf_mri.cpp',tofile='utils/mrisurf_mri.cpp')))
 obj=output/'mrisurf_mri.cpp.o';compiler,command=compile_source(build,original,own_source,obj,commands)
 archive=output/'libutils.a';shutil.copy2(build/'utils/libutils.a',archive)
 ar=compiler.with_name('x86_64-conda-linux-gnu-ar');ranlib=compiler.with_name('x86_64-conda-linux-gnu-ranlib')
 subprocess.run([str(ar),'r',str(archive),str(obj)],check=True);subprocess.run([str(ranlib),str(archive)],check=True)
 tokens=shlex.split(commands[-1].split('&&')[1].strip());tokens=[str(archive) if x=='utils/libutils.a' else x for x in tokens]
 main_command=None
 if patched:
  original_main=source/'mris_make_surfaces/mris_place_surface.cpp';main_content=original_main.read_text();main_candidate=add_capability_query(main_content)
  own_main=output/'mris_place_surface.cpp';own_main.write_text(main_candidate)
  (output/'main.patch').write_text(''.join(difflib.unified_diff(main_content.splitlines(True),main_candidate.splitlines(True),fromfile='mris_make_surfaces/mris_place_surface.cpp',tofile='mris_make_surfaces/mris_place_surface.cpp')))
  main_obj=output/'mris_place_surface.cpp.o';_,main_command=compile_source(build,original_main,own_main,main_obj,commands)
  original_main_objects=[x for x in tokens if x.endswith('/mris_place_surface.cpp.o')]
  if len(original_main_objects)!=1:raise ValueError('unexpected placement main object')
  tokens=[str(main_obj) if x==original_main_objects[0] else x for x in tokens]
 binary=output/'mris_place_surface_fnit_hotspot';tokens[tokens.index('-o')+1]=str(binary);subprocess.run(tokens,cwd=build,check=True)
 capabilities=None
 if patched:
  capabilities=json.loads(subprocess.check_output([str(binary),'--fnit-placement-capabilities'],text=True))
  if capabilities!=CAPABILITIES:raise ValueError('capability query mismatch')
 report=dict(patched=patched,source_commit=CAPABILITIES['upstream_commit'],input_source_sha256={x:sha(source/x) for x in EXPECTED},original_archive_sha256=sha(build/'utils/libutils.a'),patched_source_sha256=sha(own_source),binary_sha256=sha(binary),object_sha256=sha(obj),compiler=str(compiler),compile_command=command,main_compile_command=main_command,capabilities=capabilities,link_command=tokens,builder_sha256=sha(__file__))
 (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
 return binary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ninja',type=Path,required=True);a=p.parse_args()
 build,source,output=a.build.resolve(),a.source.resolve(),a.output.resolve()
 for name,expected in EXPECTED.items():
  if sha(source/name)!=expected:raise ValueError('fixed source hash mismatch: '+name)
 if 'ninja' not in a.ninja.name or not a.ninja.is_file():raise ValueError('explicit completed Conda Ninja build required')
 commands=subprocess.check_output([str(a.ninja),'-t','commands','mris_place_surface'],cwd=build,text=True).splitlines()
 output.mkdir(parents=True,exist_ok=False)
 for patched,name in [(False,'control'),(True,'candidate')]:print(build_one(build,source,output/name,commands,patched),flush=True)
if __name__=='__main__':main()
