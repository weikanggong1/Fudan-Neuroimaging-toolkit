"""用固定 metricProperties TU 的前置头和实际目标编译参数核对表达式类型。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


PROBE = r'''
int main() {
  float a=0.5f;
  float v1[3]={0.1f,0.2f,0.3f},v2[3]={0.4f,0.5f,0.6f};
  printf("{\"float_bytes\":%d,\"double_bytes\":%d,\"sqrt_float_bytes\":%d,\"acos_float_bytes\":%d,\"dot_bytes\":%d,\"norm3_bytes\":%d,\"two_division_bytes\":%d,\"minmax_expression_bytes\":%d}\n",
    (int)sizeof(float),(int)sizeof(double),(int)sizeof(sqrt(a)),(int)sizeof(acos(a)),
    (int)sizeof(F_DOT(v1,v2)),(int)sizeof(NORM3(v1)),
    (int)sizeof(F_DOT(v1,v2)/NORM3(v1)/NORM3(v2)),
    (int)sizeof(MAX(0.0,MIN(1.0,F_DOT(v1,v2)/NORM3(v1)/NORM3(v2)))));
  float xyz[3][3]={{0.0f,0.0f,0.0f},{3.75f,-0.25f,0.3f},{0.4f,2.25f,-1.125f}};
  printf("{\"fixture_alpha_bits\":[");
  for (int corner=0;corner<3;++corner) {
    float e1[3],e2[3];
    for (int dim=0;dim<3;++dim) {
      e1[dim]=xyz[(corner+1)%3][dim]-xyz[corner][dim];
      e2[dim]=xyz[(corner+2)%3][dim]-xyz[corner][dim];
    }
    float alpha=MAX(0.0,MIN(1.0,F_DOT(e1,e2)/NORM3(e1)/NORM3(e2)));
    alpha=acos(alpha);
    unsigned int bits=0;
    static_assert(sizeof(bits)==sizeof(alpha), "float fixture raw bits width changed");
    memcpy(&bits,&alpha,sizeof(bits));
    printf("%s%u",corner ? "," : "",bits);
  }
  printf("]}\n");
  return 0;
}
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, native_build, output):
    """新只读表达式类型 probe，输入固定构建树和不存在的输出目录。

    保留实际 metricProperties 文件的两个 compiling define 与全部前置头；
    使用对应 CMake target 的 compiler/defines/includes/flags，不添加 cmath
    或 fallback 宏，不运行原拓扑算法。输出独立 probe、构建/类型 JSON。
    缺实际 Ninja recipe、源码前置头边界改变、目录已有或编译失败报错。
    sizeof 只证明本工具链的类型，不据此断言跨环境输出一致。
    """
    native_build, output = map(Path, (native_build, output))
    if output.exists():
        raise FileExistsError('new numeric type probe directory required')
    source_root = native_build / 'source'
    source = source_root / 'utils/mrisurf_metricProperties.cpp'
    text = source.read_text()
    anchor = 'static int int_compare('
    if text.count(anchor) != 1:
        raise ValueError('fixed metricProperties preamble anchor changed')
    recipe_file = native_build / 'build/build.ninja'
    lines = recipe_file.read_text().splitlines()
    matches = [i for i, line in enumerate(lines)
               if line.startswith('build utils/CMakeFiles/utils.dir/mrisurf_metricProperties.cpp.o:')]
    if len(matches) != 1:
        raise ValueError('actual metricProperties Ninja target recipe not identified')
    recipe_lines = [lines[matches[0]]]
    for line in lines[matches[0]+1:]:
        if not line.startswith('  '):
            break
        recipe_lines.append(line)
    fields = {}
    for line in recipe_lines[1:]:
        if ' = ' in line:
            name, value = line.split(' = ', 1)
            fields[name.strip()] = value
    if not all(name in fields for name in ('DEFINES', 'INCLUDES', 'FLAGS')):
        raise ValueError('metricProperties target defines/includes/flags missing')
    compiler = next(line.split('=', 1)[1] for line in (native_build / 'build/CMakeCache.txt').read_text().splitlines()
        if line.startswith('CMAKE_CXX_COMPILER:'))
    output.mkdir(parents=True)
    probe_source = output / 'metric_expression_type_probe.cpp'
    probe_source.write_text(text.split(anchor, 1)[0] + PROBE)
    obj = output / 'metric_expression_type_probe.o'
    binary = output / 'metric_expression_type_probe'
    compile_command = [compiler]
    for name in ('DEFINES', 'INCLUDES', 'FLAGS'):
        compile_command.extend(shlex.split(fields[name]))
    compile_command.extend(['-c', str(probe_source), '-o', str(obj)])
    link_command = [compiler, str(obj), '-o', str(binary)]
    started = time.perf_counter()
    with (output / 'build.log').open('w') as log:
        subprocess.run(compile_command, cwd=native_build / 'build', stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(link_command, cwd=native_build / 'build', stdout=log, stderr=subprocess.STDOUT, check=True)
    first = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    second = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    result = dict(scope='actual fixed target preamble/compiler expression type audit; not native stage timing',
        source_sha256=sha(source), macro_header_sha256=sha(source_root / 'include/mrisurf_base.h'),
        target_recipe_sha256=sha(recipe_file),
        builder_sha256=sha(__file__), probe_source_sha256=sha(probe_source), compiler_sha256=sha(compiler),
        program_sha256=sha(binary), compile_argv=compile_command, link_argv=link_command,
        actual_target_recipe='\n'.join(recipe_lines),
        types=json.loads(first.stdout.splitlines()[0]),
        fixture=json.loads(first.stdout.splitlines()[1]),
        repeat_stdout_exact=first.stdout == second.stdout, exit_codes=[first.returncode, second.returncode],
        build_and_probe_seconds=time.perf_counter() - started)
    (output / 'types.private.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
