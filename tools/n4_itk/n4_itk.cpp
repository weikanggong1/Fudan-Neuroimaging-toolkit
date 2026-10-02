#include <fstream>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include "itkBSplineControlPointImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkExpImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkMultiThreaderBase.h"
#include "itkN4BiasFieldCorrectionImageFilter.h"
#include "itkShrinkImageFilter.h"

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--capabilities") {
    std::cout << "{\"reconstruction_threads\":true,\"profile\":true,\"fitting_threads\":1}\n";
    return 0;
  }
  if (argc < 9 || argc > 11) {
    std::cerr << "usage: fnit_n4_itk INPUT_F32 OUTPUT_F32 NX NY NZ SX SY SZ [RECONSTRUCTION_THREADS [PROFILE_JSON]]\n";
    return 2;
  }
  try {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b-a).count(); };
    const int reconstruction_threads = argc >= 10 ? std::stoi(argv[9]) : 1;
    if (reconstruction_threads < 1) throw std::runtime_error("reconstruction threads must be positive");
    using Image = itk::Image<float, 3>;
    Image::SizeType size;
    Image::SpacingType spacing;
    for (int axis = 0; axis < 3; ++axis) {
      size[axis] = std::stoul(argv[axis + 3]);
      spacing[axis] = std::stod(argv[axis + 6]);
    }
    const auto count = size[0] * size[1] * size[2];
    Image::RegionType region;
    region.SetSize(size);
    auto input = Image::New();
    input->SetRegions(region);
    input->SetSpacing(spacing);
    input->Allocate();
    std::ifstream stream(argv[1], std::ios::binary);
    stream.read(reinterpret_cast<char *>(input->GetBufferPointer()), count * sizeof(float));
    if (!stream || stream.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("input byte count does not match the image shape");

    using Mask = itk::Image<unsigned char, 3>;
    auto mask = Mask::New();
    mask->CopyInformation(input);
    mask->SetRegions(region);
    mask->Allocate();
    mask->FillBuffer(1);
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);

    using Shrink = itk::ShrinkImageFilter<Image, Image>;
    auto small = Shrink::New();
    small->SetInput(input);
    small->SetShrinkFactors(4);
    using MaskShrink = itk::ShrinkImageFilter<Mask, Mask>;
    auto small_mask = MaskShrink::New();
    small_mask->SetInput(mask);
    small_mask->SetShrinkFactors(4);

    using N4 = itk::N4BiasFieldCorrectionImageFilter<Image, Mask, Image>;
    auto corrector = N4::New();
    N4::VariableSizeArrayType iterations(4);
    for (int index = 0; index < 4; ++index) iterations[index] = 50;
    corrector->SetMaximumNumberOfIterations(iterations);
    corrector->SetNumberOfFittingLevels(4);
    corrector->SetConvergenceThreshold(0.0);
    corrector->SetUseMaskLabel(true);
    corrector->SetMaskLabel(1);
    corrector->SetInput(small->GetOutput());
    corrector->SetMaskImage(small_mask->GetOutput());
    const auto fit_start = Clock::now();
    corrector->Update();
    const auto fit_end = Clock::now();

    // Only spatially independent reconstruction is threaded. N4 fitting stays 1.
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(reconstruction_threads);

    using BSpline = itk::BSplineControlPointImageFilter<
        N4::BiasFieldControlPointLatticeType, N4::ScalarImageType>;
    auto bspline = BSpline::New();
    bspline->SetInput(corrector->GetLogBiasFieldControlPointLattice());
    bspline->SetSplineOrder(corrector->GetSplineOrder());
    bspline->SetSize(input->GetLargestPossibleRegion().GetSize());
    bspline->SetOrigin(input->GetOrigin());
    bspline->SetDirection(input->GetDirection());
    bspline->SetSpacing(input->GetSpacing());
    bspline->Update();
    const auto spline_end = Clock::now();

    auto log_field = Image::New();
    log_field->CopyInformation(input);
    log_field->SetRegions(region);
    log_field->Allocate();
    itk::ImageRegionIterator<N4::ScalarImageType> src(
        bspline->GetOutput(), bspline->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<Image> dst(log_field, region);
    for (src.GoToBegin(), dst.GoToBegin(); !src.IsAtEnd(); ++src, ++dst)
      dst.Set(src.Get()[0]);

    using Exp = itk::ExpImageFilter<Image, Image>;
    auto exp_field = Exp::New();
    exp_field->SetInput(log_field);
    using Divide = itk::DivideImageFilter<Image, Image, Image>;
    auto divide = Divide::New();
    divide->SetInput1(input);
    divide->SetInput2(exp_field->GetOutput());
    using Extract = itk::ExtractImageFilter<Image, Image>;
    auto crop = Extract::New();
    crop->SetInput(divide->GetOutput());
    crop->SetExtractionRegion(region);
    crop->SetDirectionCollapseToSubmatrix();
    crop->Update();
    const auto reconstruction_end = Clock::now();

    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char *>(crop->GetOutput()->GetBufferPointer()),
              count * sizeof(float));
    if (!out) throw std::runtime_error("could not write corrected image");
    out.close();
    if (argc == 11) {
      std::ofstream profile(argv[10]);
      profile.precision(17);
      profile << "{\"fitting_threads\":1,\"reconstruction_threads\":" << reconstruction_threads
              << ",\"fitting_seconds\":" << seconds(fit_start, fit_end)
              << ",\"bspline_seconds\":" << seconds(fit_end, spline_end)
              << ",\"exp_divide_copy_seconds\":" << seconds(spline_end, reconstruction_end)
              << ",\"read_prepare_seconds\":" << seconds(start, fit_start)
              << ",\"write_seconds\":" << seconds(reconstruction_end, Clock::now()) << "}\n";
      if (!profile) throw std::runtime_error("could not write N4 profile");
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
