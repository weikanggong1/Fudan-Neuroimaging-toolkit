// FreeSurfer-compatible float-matrix trilinear warp for robust registration.
// FreeSurfer Software License: licenses/FreeSurfer.txt.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_inverse.h>
#include <vnl/algo/vnl_qr.h>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;

py::array_t<float> warp_linear(
    py::array_t<float, py::array::c_style | py::array::forcecast> src,
    py::array_t<float, py::array::c_style | py::array::forcecast> transform,
    std::array<int, 3> dst_shape) {
  auto dims = src.shape();
  const int sx=dims[0], sy=dims[1], sz=dims[2];
  auto mat=transform.unchecked<2>();
  vnl_matrix_fixed<float,4,4> m;
  for (int r=0;r<4;++r) for (int c=0;c<4;++c) m(r,c)=mat(r,c);
  auto inv=vnl_inverse(m);
  py::array_t<float> dst({dst_shape[0],dst_shape[1],dst_shape[2]});
  auto a=src.unchecked<3>();
  auto o=dst.mutable_unchecked<3>();
  for (int z=0;z<dst_shape[2];++z) for (int y=0;y<dst_shape[1];++y) for (int x=0;x<dst_shape[0];++x) {
    float p[3];
    for (int r=0;r<3;++r) {
      p[r]=inv(r,0)*x+inv(r,1)*y+inv(r,2)*z+inv(r,3);
    }
    if (std::rint(p[0])<0 || std::rint(p[0])>=sx ||
        std::rint(p[1])<0 || std::rint(p[1])>=sy ||
        std::rint(p[2])<0 || std::rint(p[2])>=sz) {
      o(x,y,z)=0;
      continue;
    }
    const double xx=std::max(0.0,std::min(static_cast<double>(p[0]),static_cast<double>(sx-1)));
    const double yy=std::max(0.0,std::min(static_cast<double>(p[1]),static_cast<double>(sy-1)));
    const double zz=std::max(0.0,std::min(static_cast<double>(p[2]),static_cast<double>(sz-1)));
    const int x0=static_cast<int>(xx),y0=static_cast<int>(yy),z0=static_cast<int>(zz);
    const int x1=std::min(sx-1,x0+1),y1=std::min(sy-1,y0+1),z1=std::min(sz-1,z0+1);
    const double dx=xx-static_cast<float>(x0),dy=yy-static_cast<float>(y0),dz=zz-static_cast<float>(z0);
    const double ax=1.0f-dx,ay=1.0f-dy,az=1.0f-dz;
    const double v=ax*ay*az*a(x0,y0,z0)+ax*ay*dz*a(x0,y0,z1)
      +ax*dy*az*a(x0,y1,z0)+ax*dy*dz*a(x0,y1,z1)
      +dx*ay*az*a(x1,y0,z0)+dx*ay*dz*a(x1,y0,z1)
      +dx*dy*az*a(x1,y1,z0)+dx*dy*dz*a(x1,y1,z1);
    o(x,y,z)=static_cast<float>(v);
  }
  return dst;
}

static std::vector<float> filter_axis(
    const std::vector<float>& src, const std::array<int,3>& shape,
    const std::array<float,5>& kernel, int axis) {
  std::vector<float> dst(src.size());
  const int nx=shape[0],ny=shape[1],nz=shape[2];
  for (int z=0;z<nz;++z) for (int y=0;y<ny;++y) for (int x=0;x<nx;++x) {
    float sum=0;
    for (int i=0;i<5;++i) {
      const int xx=axis==0?std::max(0,std::min(nx-1,x+i-2)):x;
      const int yy=axis==1?std::max(0,std::min(ny-1,y+i-2)):y;
      const int zz=axis==2?std::max(0,std::min(nz-1,z+i-2)):z;
      sum+=kernel[i]*src[(xx*ny+yy)*nz+zz];
    }
    dst[(x*ny+y)*nz+z]=sum;
  }
  return dst;
}

py::tuple partials(
    py::array_t<float, py::array::c_style | py::array::forcecast> src,
    py::array_t<float, py::array::c_style | py::array::forcecast> trg) {
  const std::array<int,3> shape{int(src.shape(0)),int(src.shape(1)),int(src.shape(2))};
  const std::array<float,5> pre{0.03504f,0.24878f,0.43234f,0.24878f,0.03504f};
  const std::array<float,5> der{-0.10689f,-0.28461f,0.f,0.28461f,0.10689f};
  std::vector<float> avg(src.size()),diff(src.size());
  for (ssize_t i=0;i<src.size();++i) {
    avg[i]=(src.data()[i]+trg.data()[i])*0.5f;
    diff[i]=src.data()[i]-trg.data()[i];
  }
  auto fx=filter_axis(filter_axis(filter_axis(avg,shape,der,0),shape,pre,1),shape,pre,2);
  auto by=filter_axis(avg,shape,pre,0);
  auto fy=filter_axis(filter_axis(by,shape,der,1),shape,pre,2);
  auto bz=filter_axis(by,shape,pre,1);
  auto fz=filter_axis(bz,shape,der,2);
  auto b=filter_axis(filter_axis(filter_axis(diff,shape,pre,0),shape,pre,1),shape,pre,2);
  auto array=[&](const std::vector<float>& v) {
    py::array_t<float> a({shape[0],shape[1],shape[2]});
    std::copy(v.begin(),v.end(),a.mutable_data());
    return a;
  };
  return py::make_tuple(array(fx),array(fy),array(fz),array(b));
}

py::tuple construct_affine(
    py::array_t<float,py::array::c_style|py::array::forcecast> src,
    py::array_t<float,py::array::c_style|py::array::forcecast> trg,
    float eps=1e-5f) {
  const std::array<int,3> shape{int(src.shape(0)),int(src.shape(1)),int(src.shape(2))};
  const std::array<float,5> pre{0.03504f,0.24878f,0.43234f,0.24878f,0.03504f};
  const std::array<float,5> der{-0.10689f,-0.28461f,0.f,0.28461f,0.10689f};
  std::vector<float> avg(src.size()),diff(src.size());
  for (ssize_t i=0;i<src.size();++i) {
    avg[i]=(src.data()[i]+trg.data()[i])*0.5f;
    diff[i]=src.data()[i]-trg.data()[i];
  }
  auto fx=filter_axis(filter_axis(filter_axis(avg,shape,der,0),shape,pre,1),shape,pre,2);
  auto by=filter_axis(avg,shape,pre,0);
  auto fy=filter_axis(filter_axis(by,shape,der,1),shape,pre,2);
  auto fz=filter_axis(filter_axis(by,shape,pre,1),shape,der,2);
  auto b=filter_axis(filter_axis(filter_axis(diff,shape,pre,0),shape,pre,1),shape,pre,2);
  std::vector<std::array<float,12>> rows;
  std::vector<float> residuals;
  py::array_t<bool> valid(src.size());
  std::fill(valid.mutable_data(),valid.mutable_data()+src.size(),false);
  for (int z=0;z<shape[2];++z) for (int x=0;x<shape[0];++x) for (int y=0;y<shape[1];++y) {
    const int idx=(x*shape[1]+y)*shape[2]+z;
    if (std::fabs(src.data()[idx])<=eps || std::fabs(trg.data()[idx])<=eps) continue;
    const float dx=fx[idx],dy=fy[idx],dz=fz[idx];
    if (!std::isfinite(dx)||!std::isfinite(dy)||!std::isfinite(dz)||!std::isfinite(b[idx])) continue;
    if (std::fabs(dx)<eps && std::fabs(dy)<eps && std::fabs(dz)<eps) continue;
    rows.push_back({dx*x,dx*y,dx*z,dx,dy*x,dy*y,dy*z,dy,dz*x,dz*y,dz*z,dz});
    residuals.push_back(b[idx]);
    valid.mutable_data()[idx]=true;
  }
  py::array_t<float> aa({static_cast<ssize_t>(rows.size()),static_cast<ssize_t>(12)});
  py::array_t<float> bb(residuals.size());
  for (size_t i=0;i<rows.size();++i) std::copy(rows[i].begin(),rows[i].end(),aa.mutable_data()+12*i);
  std::copy(residuals.begin(),residuals.end(),bb.mutable_data());
  return py::make_tuple(aa,bb,valid);
}

py::tuple construct_rigid(
    py::array_t<float,py::array::c_style|py::array::forcecast> src,
    py::array_t<float,py::array::c_style|py::array::forcecast> trg,
    float eps=1e-5f) {
  const std::array<int,3> shape{int(src.shape(0)),int(src.shape(1)),int(src.shape(2))};
  const std::array<float,5> pre{0.03504f,0.24878f,0.43234f,0.24878f,0.03504f};
  const std::array<float,5> der{-0.10689f,-0.28461f,0.f,0.28461f,0.10689f};
  std::vector<float> avg(src.size()),diff(src.size());
  for (ssize_t i=0;i<src.size();++i) {
    avg[i]=(src.data()[i]+trg.data()[i])*0.5f;
    diff[i]=src.data()[i]-trg.data()[i];
  }
  auto fx=filter_axis(filter_axis(filter_axis(avg,shape,der,0),shape,pre,1),shape,pre,2);
  auto by=filter_axis(avg,shape,pre,0);
  auto fy=filter_axis(filter_axis(by,shape,der,1),shape,pre,2);
  auto bz=filter_axis(by,shape,pre,1);
  auto fz=filter_axis(bz,shape,der,2);
  auto b=filter_axis(filter_axis(filter_axis(diff,shape,pre,0),shape,pre,1),shape,pre,2);
  std::vector<std::array<float,6>> rows;
  std::vector<float> residuals;
  py::array_t<bool> valid(src.size());
  std::fill(valid.mutable_data(),valid.mutable_data()+src.size(),false);
  for (int z=0;z<shape[2];++z) for (int x=0;x<shape[0];++x) for (int y=0;y<shape[1];++y) {
    const int idx=(x*shape[1]+y)*shape[2]+z;
    if (std::fabs(src.data()[idx])<=eps || std::fabs(trg.data()[idx])<=eps) continue;
    const float dx=fx[idx],dy=fy[idx],dz=fz[idx];
    if (!std::isfinite(dx)||!std::isfinite(dy)||!std::isfinite(dz)||!std::isfinite(b[idx])) continue;
    if (std::fabs(dx)<eps && std::fabs(dy)<eps && std::fabs(dz)<eps) continue;
    rows.push_back({dx,dy,dz,dz*y-dy*z,dx*z-dz*x,dy*x-dx*y});
    residuals.push_back(b[idx]);
    valid.mutable_data()[idx]=true;
  }
  py::array_t<float> aa({static_cast<ssize_t>(rows.size()),static_cast<ssize_t>(6)});
  py::array_t<float> bb(residuals.size());
  for (size_t i=0;i<rows.size();++i) std::copy(rows[i].begin(),rows[i].end(),aa.mutable_data()+6*i);
  std::copy(residuals.begin(),residuals.end(),bb.mutable_data());
  return py::make_tuple(aa,bb,valid);
}

static std::pair<std::vector<float>,std::array<int,3>> reduce_axis(
    const std::vector<float>& src,std::array<int,3> shape,int axis) {
  const int len=shape[axis],n=len/2;
  if (len<=1) return {src,shape};
  const std::array<double,21> g{
    0.708792,0.328616,-0.165157,-0.114448,0.0944036,0.0543881,
    -0.05193,-0.0284868,0.0281854,0.0152877,-0.0152508,-0.00825077,
    0.00824629,0.00445865,-0.0044582,-0.00241009,0.00241022,
    0.00130278,-0.00130313,-0.000704109,0.000704784};
  auto output_shape=shape;output_shape[axis]=n;
  std::vector<float> out(size_t(output_shape[0])*output_shape[1]*output_shape[2]);
  for (int x=0;x<(axis==0?1:shape[0]);++x)
    for (int y=0;y<(axis==1?1:shape[1]);++y)
      for (int z=0;z<(axis==2?1:shape[2]);++z) {
        std::vector<double> line(2*n),filtered(2*n);
        for (int k=0;k<2*n;++k) {
          int xx=axis==0?k:x,yy=axis==1?k:y,zz=axis==2?k:z;
          line[k]=src[(xx*shape[1]+yy)*shape[2]+zz];
        }
        for (int k=0;k<2*n;++k) {
          double value=line[k]*g[0];
          for (int i=1;i<21;++i) {
            int left=k-i,right=k+i;
            if (left<0) {left=(4*n-1-left)%(4*n);if(left>=2*n)left=4*n-left-1;}
            if (right>=2*n) {right%=4*n;if(right>=2*n)right=4*n-right-1;}
            value+=g[i]*(line[left]+line[right]);
          }
          filtered[k]=value;
        }
        for (int k=0;k<n;++k) {
          int xx=axis==0?k:x,yy=axis==1?k:y,zz=axis==2?k:z;
          out[(xx*output_shape[1]+yy)*output_shape[2]+zz]=
            static_cast<float>((filtered[2*k]+filtered[2*k+1])/2.);
        }
      }
  return {out,output_shape};
}

py::array_t<float> pyramid_step(
    py::array_t<float,py::array::c_style|py::array::forcecast> image) {
  std::array<int,3> shape{int(image.shape(0)),int(image.shape(1)),int(image.shape(2))};
  const std::array<float,5> smooth{0.0625f,0.25f,0.375f,0.25f,0.0625f};
  std::vector<float> data(image.data(),image.data()+image.size());
  for (int axis=0;axis<3;++axis) data=filter_axis(data,shape,smooth,axis);
  for (int axis=0;axis<3;++axis) {
    auto result=reduce_axis(data,shape,axis);
    data=std::move(result.first);shape=result.second;
  }
  py::array_t<float> out({shape[0],shape[1],shape[2]});
  std::copy(data.begin(),data.end(),out.mutable_data());
  return out;
}

std::array<double,3> weighted_centroid(
    py::array_t<float,py::array::c_style|py::array::forcecast> image) {
  const int nx=image.shape(0),ny=image.shape(1),nz=image.shape(2);
  const float* data=image.data();
  double sum=0,xsum=0,ysum=0,zsum=0;
  for (int z=0;z<nz;++z) for (int y=0;y<ny;++y) for (int x=0;x<nx;++x) {
    const double v=data[(x*ny+y)*nz+z];
    sum+=v;xsum+=(x+1)*v;ysum+=(y+1)*v;zsum+=(z+1)*v;
  }
  return {xsum/sum,ysum/sum,zsum/sum};
}

py::array_t<double> solve_wls_vnl(
    py::array_t<double, py::array::c_style | py::array::forcecast> aa,
    py::array_t<double, py::array::c_style | py::array::forcecast> bb,
    py::array_t<double, py::array::c_style | py::array::forcecast> ww) {
  const int rows=aa.shape(0), cols=aa.shape(1);
  auto a=aa.unchecked<2>(); auto b=bb.unchecked<1>(); auto w=ww.unchecked<1>();
  vnl_matrix<float> weighted(rows,cols);
  vnl_vector<float> rhs(rows);
  for (int r=0;r<rows;++r) {
    for (int c=0;c<cols;++c) weighted(r,c)=static_cast<float>(a(r,c)*w(r));
    rhs[r]=static_cast<float>(b(r)*w(r));
  }
  vnl_qr<float> qr(weighted);
  auto p=qr.solve(rhs);
  py::array_t<double> out(cols);
  for (int c=0;c<cols;++c) out.mutable_data()[c]=p[c];
  return out;
}

static float median(std::vector<float> values) {
  const size_t n=values.size();
  std::nth_element(values.begin(),values.begin()+n/2,values.end());
  const float hi=values[n/2];
  if (n%2) return hi;
  std::nth_element(values.begin(),values.begin()+n/2-1,values.end());
  return static_cast<float>(0.5*(values[n/2-1]+hi));
}

py::tuple irls_float(
    py::array_t<float, py::array::c_style | py::array::forcecast> aa,
    py::array_t<float, py::array::c_style | py::array::forcecast> bb,
    double sat) {
  const int rows=aa.shape(0),cols=aa.shape(1);
  auto a0=aa.unchecked<2>();auto b0=bb.unchecked<1>();
  vnl_matrix<float> a(rows,cols);
  vnl_vector<float> b(rows),r(rows),w(rows),p(cols),lastp(cols),lastw(rows);
  for (int i=0;i<rows;++i) {
    for (int j=0;j<cols;++j) a(i,j)=a0(i,j);
    b[i]=b0(i);r[i]=b[i];
  }
  float prev_err=std::numeric_limits<float>::infinity();
  float err=prev_err,sigma=0;
  int count=0;
  do {
    ++count;
    if (count>1) {lastp=p;lastw=w;}
    std::vector<float> rv(r.data_block(),r.data_block()+rows);
    const float center=median(rv);
    for (float& v:rv) v=std::fabs(v-center);
    sigma=1.4826f*median(rv);
    if (sigma<2e-12) w.fill(1.f);
    else {
      r*=static_cast<float>(1.0/sigma);
      for (int i=0;i<rows;++i) {
        if (std::fabs(r[i])>=sat) w[i]=0.f;
        else {
          const double t=static_cast<double>(r[i])/sat;
          w[i]=static_cast<float>(1.0-t*t);
        }
      }
    }
    vnl_matrix<float> wa(rows,cols);
    vnl_vector<float> wb(rows);
    for (int i=0;i<rows;++i) {
      for (int j=0;j<cols;++j) wa(i,j)=a(i,j)*w[i];
      wb[i]=b[i]*w[i];
    }
    vnl_qr<float> qr(wa);
    p=qr.solve(wb);
    r=b-a*p;
    float sw=0,swr=0;
    for (int i=0;i<rows;++i) {
      const float wi=w[i]*w[i],ri=r[i]*r[i];
      sw+=wi;swr+=wi*ri;
    }
    err=swr/sw;
    if (prev_err<=err) break;
    prev_err=err;
  } while (count<20 && err>2e-12);
  if (err>prev_err) {p=lastp;w=lastw;err=prev_err;}
  py::array_t<float> out(cols);
  for (int j=0;j<cols;++j) out.mutable_data()[j]=p[j];
  double mean=0;
  for (int i=0;i<rows;++i) mean+=w[i];
  return py::make_tuple(out,static_cast<double>(err),mean/rows,static_cast<double>(sigma),count);
}

PYBIND11_MODULE(gems_warp,m) {
  m.def("warp_linear",&warp_linear);
  m.def("partials",&partials);
  m.def("construct_affine",&construct_affine,py::arg("source"),py::arg("target"),py::arg("eps")=1e-5f);
  m.def("construct_rigid",&construct_rigid,py::arg("source"),py::arg("target"),py::arg("eps")=1e-5f);
  m.def("pyramid_step",&pyramid_step);
  m.def("weighted_centroid",&weighted_centroid);
  m.def("solve_wls_vnl",&solve_wls_vnl);
  m.def("irls_float",&irls_float);
}
