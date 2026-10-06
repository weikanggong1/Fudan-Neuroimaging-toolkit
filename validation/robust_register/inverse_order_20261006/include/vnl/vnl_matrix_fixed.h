#pragma once
#include <cstddef>
template<class T> class vnl_matrix;
template<class T,unsigned R,unsigned C> struct vnl_matrix_fixed {
 T values[R*C]{};
 vnl_matrix_fixed()=default;
 explicit vnl_matrix_fixed(const T* p){for(unsigned i=0;i<R*C;++i)values[i]=p[i];}
 const T& operator()(unsigned r,unsigned c)const{return values[r*C+c];}
 T& operator()(unsigned r,unsigned c){return values[r*C+c];}
 const T* operator[](unsigned r)const{return values+r*C;}
 vnl_matrix_fixed operator*(T factor)const{vnl_matrix_fixed out;for(unsigned i=0;i<R*C;++i)out.values[i]=values[i]*factor;return out;}
};
