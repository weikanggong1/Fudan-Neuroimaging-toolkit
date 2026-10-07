"""One new strict CSC/division/native-reductions arm on archived solve3."""
import argparse
import csv
import json
import struct
import subprocess
import os
import socket
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.sparse import csc_matrix

from probe_common import load_vector, metrics, sha, write_json


@njit(fastmath=False, cache=True)
def column_matvec(indptr, indices, values, direction):
    result = np.zeros(direction.size, dtype=np.float64)
    for column in range(direction.size):
        weight = direction[column]
        for index in range(indptr[column], indptr[column+1]):
            result[indices[index]] += weight * values[index]
    return result


class NativeArithmetic:
    def __init__(self, executable):
        self.executable=executable.resolve()
        child_env=os.environ.copy()
        child_env['LD_LIBRARY_PATH']=os.environ['FNIT_NATIVE_ARITH_LIBRARY_PATH']
        self.identity=json.loads(subprocess.check_output([str(self.executable),'--identity'],text=True,env=child_env))
        self.process=subprocess.Popen([str(self.executable),'--stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=child_env)

    def request(self, op, first, second=None):
        if first.dtype!=np.float64 or not first.flags.c_contiguous:
            raise ValueError('protocol requires contiguous FP64')
        if second is not None and (second.dtype!=np.float64 or not second.flags.c_contiguous or first.shape!=second.shape):
            raise ValueError('protocol inputs differ')
        payload=struct.pack('<II',op,first.size)+first.tobytes()
        if second is not None:payload+=second.tobytes()
        self.process.stdin.write(payload);self.process.stdin.flush()
        result=self.process.stdout.read(8)
        if len(result)!=8:raise RuntimeError('isolated arithmetic bridge failed')
        return struct.unpack('<d',result)[0]

    def dot(self, first, second):return self.request(1,first,second)
    def norm(self, values):return self.request(2,values)

    def symbol_bindings(self):
        return {'isolated_executable':str(self.executable),'identity':self.identity,
                'compiled_link_context':'same4objects/headers/libs as original observer; no NumPy in this child'}

    def close(self):
        self.process.stdin.close()
        if self.process.wait(timeout=5):raise RuntimeError('arithmetic subprocess exit failure')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();native=NativeArithmetic(a.library)
    contracts=[]
    for solve in (2,3):
        prefix='solve'+str(solve)
        b=load_vector(a.oracle,prefix+'_rhs.f64');norm_b=native.norm(b)
        with (a.oracle/(prefix+'_trace.csv')).open() as stream:
            trace=list(csv.DictReader(stream))
        for item in trace:
            k=int(item['iteration'])
            vectors={key:load_vector(a.oracle,f'{prefix}_{key}_{k}.f64') for key in ('r','z','p','q','r_after')}
            rho=native.dot(vectors['r'],vectors['z']);den=native.dot(vectors['p'],vectors['q'])
            actual={'rho':rho,'denominator':den,'alpha':rho/den,
                    'relative_before':native.norm(vectors['r'])/norm_b,
                    'relative_after':native.norm(vectors['r_after'])/norm_b}
            differences={key:abs(actual[key]-float(item[key])) for key in actual}
            contracts.append({'solve':solve,'iteration':k,'exact':all(value==0 for value in differences.values()),
                              'absolute_difference':differences})
    contract_pass=all(row['exact'] for row in contracts)
    maps=Path('/proc/'+str(native.process.pid)+'/maps').read_text().splitlines()
    libraries=sorted(set(row.split()[-1] for row in maps if '.so' in row and any(x in row for x in ('blas','arithmetic_cli'))))
    contract_report={'scope':'same archived native vectors; no new native optimizer',
                     'rounds':len(contracts),'all_scalar_fields_exact':contract_pass,
                     'flags':native.identity['flags'],'loaded_arithmetic_libraries':libraries,
                     'library_sha256':sha(a.library),'blas_symbol_bindings':native.symbol_bindings(),
                     'protocol_signature':'LE uint32 operation,n then raw FP64 vectors, return FP64 scalar',
                     'loader':'Independent installed-context C++ child; no FSL DSO in FNIT/NumPy parent','rows':contracts}
    write_json(a.output/'contracts.public.json',contract_report)
    if not contract_pass:
        native.close()
        raise RuntimeError('native reduction bridge does not match archived scalar trace')
    dimension=1177;prefix='solve3'
    matrix=np.fromfile(a.oracle/(prefix+'_A.f64'),dtype='<f8').reshape(dimension,dimension)
    sparse=csc_matrix(matrix)
    product=lambda value:column_matvec(sparse.indptr,sparse.indices,sparse.data,value)
    b=load_vector(a.oracle,prefix+'_rhs.f64');d=load_vector(a.oracle,prefix+'_diagonal.f64')
    x=load_vector(a.oracle,prefix+'_x0.f64');r=b.copy()
    if np.count_nonzero(x) or np.any(d<=0):raise ValueError('unexpected original stopping/preconditioner state')
    # No floor, changed tolerance, exact solution, or prescribed stop count.
    z=r/d;pvec=z.copy();rho=native.dot(r,z);norm_b=native.norm(b)
    rows=[];first_vector_difference=None;trace=[]
    with (a.oracle/(prefix+'_trace.csv')).open() as stream:trace=list(csv.DictReader(stream))
    for k in range(1,501):
        q=product(pvec);den=native.dot(pvec,q)
        if not np.isfinite(den) or den<=0:raise RuntimeError('nonpositive denominator')
        alpha=rho/den;next_x=x+alpha*pvec;after=r-alpha*q
        relative=native.norm(after)/norm_b
        compare={}
        if k<=len(trace):
            for key,value in [('r',r),('z',z),('p',pvec),('q',q),('r_after',after)]:
                compare[key]=metrics(load_vector(a.oracle,f'{prefix}_{key}_{k}.f64'),value)
                if compare[key]['different_bits'] and first_vector_difference is None:
                    first_vector_difference={'iteration':k,'field':key,**compare[key]}
        record={'iteration':k,'rho':rho,'denominator':den,'alpha':alpha,'relative_after':relative,
                'native_vector_comparisons':compare}
        if k<=len(trace):
            record['native_scalar_exact']={key:record[key]==float(trace[k-1][key]) for key in ('rho','denominator','alpha','relative_after')}
        rows.append(record);x,r=next_x,after
        if relative<=1e-3:break
        z=r/d;new_rho=native.dot(r,z);pvec=z+(new_rho/rho)*pvec;rho=new_rho
    native.close()
    solution=metrics(load_vector(a.oracle,prefix+'_native_solution.f64'),x)
    x.tofile(a.output/'solution.private.f64')
    np.savetxt(a.output/'residuals.csv',np.array([[row['iteration'],row['relative_after']] for row in rows]),
               delimiter=',',header='iteration,relative_after',comments='',fmt=['%d','%.17g'])
    report={'scope':'one archived solve3 canonical A/RHS; not production assembly or stock registration',
            'host':socket.gethostname(),'affinity':sorted(os.sched_getaffinity(0)),
            'policy':'FP64 strict CSC, division, installed NEWMAT dot/norm, NumPy separate multiply/add',
            'tolerance':1e-3,'maximum_iterations':500,'zero_initial_value':True,
            'iterations':len(rows),'converged':relative<=1e-3,'relative_residual':relative,
            'true_relative_residual':float(np.linalg.norm(b-matrix@x)/np.linalg.norm(b)),
            'native_iterations':len(trace),'first_vector_difference':first_vector_difference,
            'solution_vs_native':solution,'all_native_rounds_exact':first_vector_difference is None and len(rows)==len(trace),
            'wall_seconds':time.monotonic()-start,'source_sha256':sha(__file__),'rows':rows}
    write_json(a.output/'summary.public.json',report)
    print({key:report[key] for key in ('iterations','converged','all_native_rounds_exact','solution_vs_native')})


if __name__=='__main__':main()
