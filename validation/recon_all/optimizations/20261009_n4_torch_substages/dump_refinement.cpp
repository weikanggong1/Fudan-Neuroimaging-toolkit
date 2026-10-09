// Independent fixed-ITK numerical constants diagnostic, no image inputs.
#include <cstring>
#include <iostream>
#include "itkCoxDeBoorBSplineKernelFunction.h"
#include "vnl/algo/vnl_svd.h"

int main() {
  auto kernel = itk::CoxDeBoorBSplineKernelFunction<3>::New();
  kernel->SetSplineOrder(3);
  const auto c = kernel->GetShapeFunctionsInZeroToOneInterval();
  vnl_matrix<float> r(c.rows(), c.cols()), s(c.rows(), c.cols());
  for (unsigned int row = 0; row < c.rows(); ++row)
    for (unsigned int col = 0; col < c.cols(); ++col)
      r(row,col) = s(row,col) = static_cast<float>(c(row,col));
  for (unsigned int col = 0; col < c.cols(); ++col) {
    const float factor = std::pow(2.0f, static_cast<float>(c.cols()-col-1));
    for (unsigned int row = 0; row < c.rows(); ++row) r(row,col) *= factor;
  }
  r = r.transpose(); r.flipud(); s = s.transpose(); s.flipud();
  const auto coefficients = vnl_svd<float>(r).solve(s).extract(2, s.cols());
  std::cout << "{\"dtype\":\"float32\",\"shape\":[2,4],\"row_major_bits\":[";
  for (unsigned int i=0; i<8; ++i) {
    unsigned int bits; const float value = coefficients(i/4, i%4);
    std::memcpy(&bits, &value, 4); if(i) std::cout << ','; std::cout << bits;
  }
  std::cout << "]}\n";
}
