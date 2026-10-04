"""Isolate ITK's float B-spline refinement solve; operator diagnostic, no MRI benchmark."""
import hashlib,json,os,shlex,subprocess
from pathlib import Path
ROOT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
SOURCE=r'''#include <itkCoxDeBoorBSplineKernelFunction.h>
#include <vnl/algo/vnl_svd.h>
#include <iostream>
#include <iomanip>
#include <cmath>
int main(){
 for(unsigned order=1;order<=5;order++){
 auto kernel=itk::CoxDeBoorBSplineKernelFunction<3,double>::New();
 kernel->SetSplineOrder(order);
 auto C=kernel->GetShapeFunctionsInZeroToOneInterval();
 vnl_matrix<float> R(C.rows(),C.cols()),S(C.rows(),C.cols());
 for(unsigned j=0;j<C.rows();j++)for(unsigned k=0;k<C.cols();k++)R(j,k)=S(j,k)=static_cast<float>(C(j,k));
 for(unsigned j=0;j<C.cols();j++){float c=std::pow(float(2),float(C.cols())-j-1);for(unsigned k=0;k<C.rows();k++)R(k,j)*=c;}
 R=R.transpose();R.flipud();S=S.transpose();S.flipud();
 auto result=vnl_svd<float>(R).solve(S).extract(2,S.cols());
 std::cout<<"order "<<order<<"\n"<<std::setprecision(17)<<"C\n"<<C<<"R\n"<<R<<"S\n"<<S<<"result\n"<<result;
 }
}
'''
source=ROOT/'refinement.cpp';source.write_text(SOURCE)
report={'kind':'operator_diagnostic_not_synthetic_mri_benchmark','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runs':[]}
for tag,build in [('modern','build4_float_conda'),('modern_oldlibraries','build4_oldlibs_conda'),('old','build4_gcc48_clean')]:
 directory=ROOT/build
 # Reuse exact tested compiler, flags and static libraries; replace only translation unit.
 flags={}
 for line in (directory/'CMakeFiles/fnit_n4_diagnostic.dir/flags.make').read_text().splitlines():
  if ' = ' in line:
   key,value=line.split(' = ',1);flags[key]=value
 link=shlex.split((directory/'CMakeFiles/fnit_n4_diagnostic.dir/link.txt').read_text())
 obj=ROOT/(tag+'.refinement.o');binary=ROOT/(tag+'.refinement')
 compile_command=[link[0],*shlex.split(flags.get('CXX_DEFINES','')),*shlex.split(flags.get('CXX_INCLUDES','')),*shlex.split(flags.get('CXX_FLAGS','')),'-c',str(source),'-o',str(obj)]
 subprocess.run(compile_command,check=True)
 link=[str(obj) if x.endswith('n4_diagnostic.cpp.o') else str(binary) if x=='fnit_n4_diagnostic' else x for x in link]
 subprocess.run(link,check=True)
 output=subprocess.check_output([str(binary)],text=True)
 (ROOT/(tag+'.refinement.txt')).write_text(output)
 report['runs'].append({'tag':tag,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'output':output,'compile_command':compile_command})
(ROOT/'refinement.json').write_text(json.dumps(report,indent=2))
print(json.dumps({x['tag']:x['output'] for x in report['runs']},indent=2))
