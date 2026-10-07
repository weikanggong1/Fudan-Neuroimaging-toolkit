// Independent oracle: compile original pinned VNL headers unchanged.
// The local shim implements matrix storage/access/scalar multiplication only.
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include "vnl/vnl_det.hxx"
#include "vnl/vnl_inverse.h"
static std::uint32_t bits(float x){std::uint32_t b;std::memcpy(&b,&x,4);return b;}
int main(){
 std::fesetround(FE_TONEAREST);
 unsigned count=0;std::cin>>count;
 for(unsigned k=0;k<count;++k){
  float data[16];for(float& x:data){std::uint32_t b;std::cin>>std::hex>>b;std::memcpy(&x,&b,4);}
  vnl_matrix_fixed<float,4,4> m(data);
  const float d=vnl_det(m);if(d==0){std::cerr<<"singular case";return 3;}
  const float reciprocal=1.0f/d;
  auto inverse=vnl_inverse(m);
  std::cout<<std::hex<<bits(d)<<' '<<bits(reciprocal);
  for(float x:inverse.values)std::cout<<' '<<bits(x);
  std::cout<<'\n';
 }
 return std::cin.fail()?4:0;
}
