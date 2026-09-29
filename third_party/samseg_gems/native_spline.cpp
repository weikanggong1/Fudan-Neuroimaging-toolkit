// Adapted from FreeSurfer 8.2 MRItoBSpline/MRIresample for FNIT.
// FreeSurfer Software License: licenses/FreeSurfer.txt.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_inverse.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <cmath>
#include <cfloat>
#include <vector>
#include <array>
#include <algorithm>
namespace py = pybind11;

static void coeff_line(std::vector<double>& c) {
  const int n = c.size();
  if (n <= 1) return;
  const double z = std::sqrt(3.) - 2.;
  const double lambda = (1. - z) * (1. - 1. / z);
  for (double& v : c) v *= lambda;
  int horizon = static_cast<int>(std::ceil(std::log(DBL_EPSILON) / std::log(std::abs(z))));
  double sum = c[0], zn = z;
  if (horizon < n) {
    for (int k = 1; k < horizon; ++k, zn *= z) sum += zn * c[k];
  } else {
    double iz = 1. / z, z2n = std::pow(z, n - 1.);
    sum = c[0] + z2n * c[n-1];
    z2n *= z2n * iz;
    for (int k = 1; k <= n-2; ++k) {
      sum += (zn + z2n) * c[k];
      zn *= z;
      z2n *= iz;
    }
    sum /= 1. - zn * zn;
  }
  c[0] = sum;
  for (int k = 1; k < n; ++k) c[k] += z * c[k-1];
  c[n-1] = (z / (z*z - 1.)) * (z*c[n-2] + c[n-1]);
  for (int k = n-2; k >= 0; --k) c[k] = z * (c[k+1] - c[k]);
}

static void filter_axis(std::vector<float>& a, std::array<int,3> shape, int axis) {
  const int sx=shape[0], sy=shape[1], sz=shape[2];
  int n=shape[axis];
  std::vector<double> line(n);
  for (int x=0;x<(axis==0?1:sx);++x)
    for (int y=0;y<(axis==1?1:sy);++y)
      for (int z=0;z<(axis==2?1:sz);++z) {
        for (int k=0;k<n;++k) {
          int ix=axis==0?k:x, iy=axis==1?k:y, iz=axis==2?k:z;
          line[k]=a[(ix*sy+iy)*sz+iz];
        }
        coeff_line(line);
        for (int k=0;k<n;++k) {
          int ix=axis==0?k:x, iy=axis==1?k:y, iz=axis==2?k:z;
          a[(ix*sy+iy)*sz+iz]=static_cast<float>(line[k]);
        }
      }
}

static std::array<double,4> weights(double x, int& base) {
  base=static_cast<int>(std::floor(x))-1;
  double w=x-static_cast<double>(base+1);
  std::array<double,4> t;
  t[3]=(1./6.)*w*w*w;
  t[0]=(1./6.)+(1./2.)*w*(w-1.)-t[3];
  t[2]=w+t[0]-2.*t[3];
  t[1]=1.-t[0]-t[2]-t[3];
  return t;
}
static int mirror(int k,int n) {
  if (n==1) return 0;
  int span=2*n-2;
  if (k<0) k=-k-span*((-k)/span);
  else k=k-span*(k/span);
  return k>=n ? span-k : k;
}

static std::array<std::array<float,4>,4> vox2ras(
  const std::array<int,3>& shape,
  py::array_t<float,py::array::c_style|py::array::forcecast> zoom,
  py::array_t<float,py::array::c_style|py::array::forcecast> mdc,
  py::array_t<float,py::array::c_style|py::array::forcecast> center) {
  auto z=zoom.unchecked<1>(); auto d=mdc.unchecked<2>(); auto c=center.unchecked<1>();
  std::array<std::array<float,4>,4> a{};
  for(int i=0;i<4;++i) a[i][i]=1.f;
  for(int i=0;i<3;++i) {
    double offset=0.;
    for(int j=0;j<3;++j) {
      a[i][j]=static_cast<float>(static_cast<double>(d(j,i))*z(j));
      offset+=static_cast<double>(a[i][j])*static_cast<float>(shape[j]/2.);
    }
    a[i][3]=static_cast<float>(static_cast<double>(c(i))-static_cast<float>(offset));
  }
  return a;
}

py::tuple resample(
  py::array_t<float,py::array::c_style|py::array::forcecast> src,
  py::array_t<float,py::array::c_style|py::array::forcecast> src_zoom,
  py::array_t<float,py::array::c_style|py::array::forcecast> src_mdc,
  py::array_t<float,py::array::c_style|py::array::forcecast> src_center,
  std::array<int,3> dst_shape,
  py::array_t<float,py::array::c_style|py::array::forcecast> dst_zoom,
  py::array_t<float,py::array::c_style|py::array::forcecast> dst_mdc,
  py::array_t<float,py::array::c_style|py::array::forcecast> dst_center) {
  auto ss=src.shape();
  std::array<int,3> shape{static_cast<int>(ss[0]),static_cast<int>(ss[1]),static_cast<int>(ss[2])};
  auto a=vox2ras(shape,src_zoom,src_mdc,src_center);
  auto b=vox2ras(dst_shape,dst_zoom,dst_mdc,dst_center);
  vnl_matrix_fixed<float,4,4> va;
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)va(i,j)=a[i][j];
  auto vi=vnl_inverse(va);
  std::array<std::array<float,4>,4> m{};
  for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
    float v=0.f; for(int k=0;k<4;++k)v+=vi(i,k)*b[k][j];
    m[i][j]=v;
  }
  std::vector<float> co(src.data(),src.data()+src.size());
  const bool srcneg=std::any_of(co.begin(),co.end(),[](float v){return v<0.f;});
  for(int axis=0;axis<3;++axis)filter_axis(co,shape,axis);
  py::array_t<float> out({dst_shape[0],dst_shape[1],dst_shape[2]});
  float* q=out.mutable_data();
  for(int x=0;x<dst_shape[0];++x)for(int y=0;y<dst_shape[1];++y)for(int z=0;z<dst_shape[2];++z) {
    float pos[3];
    for(int i=0;i<3;++i) {
      float v=0.f; v+=m[i][0]*x; v+=m[i][1]*y; v+=m[i][2]*z; v+=m[i][3];
      pos[i]=v;
    }
    int idx=(x*dst_shape[1]+y)*dst_shape[2]+z;
    if (std::nearbyint(pos[0])<0||std::nearbyint(pos[0])>=shape[0]||
        std::nearbyint(pos[1])<0||std::nearbyint(pos[1])>=shape[1]||
        std::nearbyint(pos[2])<0||std::nearbyint(pos[2])>=shape[2]) {q[idx]=0.f;continue;}
    int bas[3]; std::array<double,4> w[3];
    for(int i=0;i<3;++i)w[i]=weights(pos[i],bas[i]);
    double v=0.;
    for(int k=0;k<4;++k) {
      double v2=0.;
      int iz=mirror(bas[2]+k,shape[2]);
      for(int j=0;j<4;++j) {
        double v1=0.;
        int iy=mirror(bas[1]+j,shape[1]);
        for(int i=0;i<4;++i) {
          int ix=mirror(bas[0]+i,shape[0]);
          v1+=w[0][i]*co[(ix*shape[1]+iy)*shape[2]+iz];
        }
        v2+=w[1][j]*v1;
      }
      v+=w[2][k]*v2;
    }
    q[idx]=static_cast<float>(srcneg ? v : std::max(v,0.));
  }
  py::array_t<float> mat({4,4});
  auto mm=mat.mutable_unchecked<2>();
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)mm(i,j)=m[i][j];
  return py::make_tuple(out,mat);
}

py::tuple resample_simple(
  py::array_t<float,py::array::c_style|py::array::forcecast> src,
  py::array_t<float,py::array::c_style|py::array::forcecast> src_zoom,
  py::array_t<float,py::array::c_style|py::array::forcecast> src_mdc,
  py::array_t<float,py::array::c_style|py::array::forcecast> src_center,
  std::array<int,3> dst_shape,
  py::array_t<float,py::array::c_style|py::array::forcecast> dst_zoom,
  py::array_t<float,py::array::c_style|py::array::forcecast> dst_mdc,
  py::array_t<float,py::array::c_style|py::array::forcecast> dst_center,
  bool nearest) {
  const std::array<int,3> shape{int(src.shape(0)),int(src.shape(1)),int(src.shape(2))};
  auto a=vox2ras(shape,src_zoom,src_mdc,src_center);
  auto b=vox2ras(dst_shape,dst_zoom,dst_mdc,dst_center);
  vnl_matrix_fixed<float,4,4> va;
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)va(i,j)=a[i][j];
  auto inv=vnl_inverse(va);
  std::array<std::array<float,4>,4> m{};
  for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
    float v=0.f;for(int k=0;k<4;++k)v+=inv(i,k)*b[k][j];m[i][j]=v;
  }
  auto data=src.unchecked<3>();
  py::array_t<float> out({dst_shape[0],dst_shape[1],dst_shape[2]});
  auto dst=out.mutable_unchecked<3>();
  auto sample=[&](int x,int y,int z) {
    return x<0||y<0||z<0||x>=shape[0]||y>=shape[1]||z>=shape[2] ? 0.f : data(x,y,z);
  };
  for(int x=0;x<dst_shape[0];++x)for(int y=0;y<dst_shape[1];++y)for(int z=0;z<dst_shape[2];++z) {
    float p[3];
    for(int i=0;i<3;++i) {
      float v=0.f;v+=m[i][0]*x;v+=m[i][1]*y;v+=m[i][2]*z;v+=m[i][3];p[i]=v;
    }
    const int x0=int(std::floor(p[0])),y0=int(std::floor(p[1])),z0=int(std::floor(p[2]));
    const float dx=p[0]-x0,dy=p[1]-y0,dz=p[2]-z0;
    if(nearest) dst(x,y,z)=sample(x0+(dx>=0.5f),y0+(dy>=0.5f),z0+(dz>=0.5f));
    else {
      const float ax=1.f-dx,ay=1.f-dy,az=1.f-dz;
      dst(x,y,z)=ax*ay*az*sample(x0,y0,z0)+ax*ay*dz*sample(x0,y0,z0+1)
        +ax*dy*az*sample(x0,y0+1,z0)+ax*dy*dz*sample(x0,y0+1,z0+1)
        +dx*ay*az*sample(x0+1,y0,z0)+dx*ay*dz*sample(x0+1,y0,z0+1)
        +dx*dy*az*sample(x0+1,y0+1,z0)+dx*dy*dz*sample(x0+1,y0+1,z0+1);
    }
  }
  py::array_t<float> mat({4,4});auto mm=mat.mutable_unchecked<2>();
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)mm(i,j)=m[i][j];
  return py::make_tuple(out,mat);
}

PYBIND11_MODULE(gems_resample,m) {
  m.def("resample_cubic",&resample);
  m.def("resample_simple",&resample_simple);
}
