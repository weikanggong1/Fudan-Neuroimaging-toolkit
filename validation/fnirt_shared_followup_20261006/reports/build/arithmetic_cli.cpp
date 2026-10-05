
// Original validation protocol. Installed implementations are only included.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <string>
#include "newimage/newimageall.h"
#include "miscmaths/nonlin.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/cg.h"
#include "fnirt_costfunctions.h"
#include "fnirtfns.h"
using namespace NEWMAT;
static ColumnVector readvec(const std::string& path) {
  ColumnVector value(1177);std::ifstream stream(path,std::ios::binary);
  if(!stream.read(reinterpret_cast<char*>(value.Store()),1177*sizeof(double)))
    throw std::runtime_error("invalid archived FP64 vector");
  return value;
}
static int flags() {
  int value=0;
#ifdef ARMA_USE_BLAS
  value|=1;
#endif
#ifdef ARMA_USE_ATLAS
  value|=2;
#endif
#if defined(__FINITE_MATH_ONLY__) && (__FINITE_MATH_ONLY__ > 0)
  value|=4;
#endif
#ifdef ARMA_BLAS_LONG
  value|=8;
#endif
#ifdef ARMA_BLAS_LONG_LONG
  value|=16;
#endif
  return value;
}
int main(int argc,char** argv) {
  if(argc<2)return 2;
  std::string mode=argv[1];
  if(mode=="--identity") {
    std::cout<<"{\"flags\":"<<flags()<<",\"blas_integer_bytes\":"<<sizeof(arma::blas_int)
             <<",\"double_bytes\":"<<sizeof(double)<<"}\n";return 0;
  }
  if(mode=="--archive") {
    if(argc!=3)return 2;
    const std::string root=argv[2];
    std::cout<<std::setprecision(17)<<"solve,iteration,rho,denominator,alpha,relative_before,relative_after\n";
    for(int solve=2;solve<=3;++solve) {
      std::string prefix=root+"/solve"+std::to_string(solve);
      ColumnVector b=readvec(prefix+"_rhs.f64");double norm=b.NormFrobenius();
      for(int k=1;k<=(solve==2?24:69);++k) {
        std::string suffix="_"+std::to_string(k)+".f64";
        ColumnVector r=readvec(prefix+"_r"+suffix),z=readvec(prefix+"_z"+suffix),
                     p=readvec(prefix+"_p"+suffix),q=readvec(prefix+"_q"+suffix),
                     after=readvec(prefix+"_r_after"+suffix);
        double rho=DotProduct(r,z),den=DotProduct(p,q);
        std::cout<<solve<<","<<k<<","<<rho<<","<<den<<","<<rho/den<<","
                 <<r.NormFrobenius()/norm<<","<<after.NormFrobenius()/norm<<"\n";
      }
    }
    return 0;
  }
  if(mode!="--stdio")return 2;
  std::ios::sync_with_stdio(false);
  while(true) {
    std::uint32_t op=0,n=0;
    if(!std::cin.read(reinterpret_cast<char*>(&op),sizeof(op)))return std::cin.eof()?0:3;
    if(!std::cin.read(reinterpret_cast<char*>(&n),sizeof(n)) || n<1 || n>10000000 || (op!=1&&op!=2))return 3;
    ColumnVector x(n),y(op==1?n:0);
    if(!std::cin.read(reinterpret_cast<char*>(x.Store()),n*sizeof(double)))return 3;
    if(op==1&&!std::cin.read(reinterpret_cast<char*>(y.Store()),n*sizeof(double)))return 3;
    double result=op==1?DotProduct(x,y):x.NormFrobenius();
    std::cout.write(reinterpret_cast<const char*>(&result),sizeof(result));std::cout.flush();
    if(!std::cout)return 3;
  }
}
