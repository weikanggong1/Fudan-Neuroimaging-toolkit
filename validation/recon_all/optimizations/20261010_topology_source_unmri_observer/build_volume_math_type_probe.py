"""同固定TU的log/ceil overload与float位模式诊断，不覆盖原角度probe。"""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import build_numeric_type_probe as original

EXTRA=r'''
  float logarg=0.75f,ceilarg=1.125f;
  float logresult=log(logarg),ceilresult=ceil(ceilarg);
  unsigned int logbits=0,ceilbits=0;
  memcpy(&logbits,&logresult,sizeof(logbits));memcpy(&ceilbits,&ceilresult,sizeof(ceilbits));
  printf("{\"log_float_bytes\":%d,\"ceil_float_bytes\":%d,\"fixture_log_bits\":%u,\"fixture_ceil_bits\":%u}\n",(int)sizeof(log(logarg)),(int)sizeof(ceil(ceilarg)),logbits,ceilbits);
'''


def build(*,native_build,output):
    """保留固定target所有类型环境，另存新类型收据，默认原probe不改。"""
    if original.PROBE.count('  return 0;')!=1:raise ValueError('numeric probe anchor changed')
    original.PROBE=original.PROBE.replace('  return 0;',EXTRA+'  return 0;')
    result=original.build(native_build=native_build,output=output)
    binary=Path(output)/'metric_expression_type_probe'
    first=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    second=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    result['volume_math_types']=json.loads(first.stdout.splitlines()[2]);result['volume_math_repeat_exact']=first.stdout==second.stdout
    result['variant_builder_sha256']=original.sha(__file__)
    (Path(output)/'types.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    build(**vars(p.parse_args()))


if __name__=='__main__':main()
