"""Build an original, validation-only bridge to installed NEWMAT arithmetic."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE = r'''
// Own validation bridge; no upstream implementation is copied.
#include <algorithm>
#include <cmath>
#include "armawrap/newmat.h"
extern "C" double fnit_native_dot(const double* a,const double* b,int n) {
  NEWMAT::ColumnVector x(n),y(n);
  std::copy(a,a+n,x.Store());std::copy(b,b+n,y.Store());
  return NEWMAT::DotProduct(x,y);
}
extern "C" double fnit_native_norm(const double* a,int n) {
  NEWMAT::ColumnVector x(n);std::copy(a,a+n,x.Store());
  return x.NormFrobenius();
}
extern "C" int fnit_native_flags() {
  int flags=0;
#ifdef ARMA_USE_BLAS
  flags|=1;
#endif
#ifdef ARMA_USE_ATLAS
  flags|=2;
#endif
#if defined(__FINITE_MATH_ONLY__) && (__FINITE_MATH_ONLY__ > 0)
  flags|=4;
#endif
  return flags;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fsl', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cpp, lib = a.output/'native_arithmetic.cpp', a.output/'native_arithmetic.so'
    cpp.write_text(SOURCE)
    command = ['g++','-std=c++17','-O0','-fPIC','-shared',
               '-I'+str(a.fsl/'include'),'-I'+str(a.fsl/'include/newmat'),
               str(cpp), '-L'+str(a.fsl/'lib'),'-Wl,-rpath,'+str(a.fsl/'lib'),
               '-llapack','-lblas','-lpthread','-o',str(lib)]
    result = subprocess.run(command, capture_output=True, text=True)
    (a.output/'build.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    ldd = subprocess.run(['ldd',str(lib)],capture_output=True,text=True,check=True).stdout
    (a.output/'ldd.txt').write_text(ldd)
    def bound(path):
        return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    files = ['include/armawrap/newmat.h','include/armawrap/function_dotproduct.hpp',
             'include/armadillo_bits/op_dot_meat.hpp','include/armadillo_bits/fn_norm.hpp',
             'include/armadillo_bits/config.hpp','lib/libblas.so']
    meta = {'scope':'installed arithmetic only; never a production dependency',
            'command':command,'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
            'files':{x:bound(a.fsl/x) for x in files},'cpp':bound(cpp),'library':bound(lib),'ldd':ldd}
    (a.output/'binding.public.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'built':True,'library_sha256':meta['library']['sha256']}))


if __name__ == '__main__':
    main()
