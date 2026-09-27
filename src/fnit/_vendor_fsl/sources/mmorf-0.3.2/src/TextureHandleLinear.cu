//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief RAII wrapper for dealing with CUDA Linear Texture object
/// \details Use this helper class to deal with the hassle of loading image volumes into
///          linear texture memory, and ensure proper memory acquisition and release
/// \author Frederik Lange
/// \date May 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#include "TextureHandleLinear.cuh"
#include "helper_cuda.h"

#include <thrust/device_vector.h>

#include <vector>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
  /// Ctor
  TextureHandleLinear::TextureHandleLinear(
      thrust::device_vector<float>& ima,
      const std::vector<int>& ima_sz)
    : texture_(0)
  {
    // Create resource descriptor
    cudaResourceDesc res_desc;
    memset(&res_desc,0,sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = thrust::raw_pointer_cast(ima.data());
    res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
    res_desc.res.linear.desc.x = 32;
    res_desc.res.linear.sizeInBytes = ima_sz[0]*ima_sz[1]*ima_sz[2]*sizeof(float);
    // Create texture descriptor
    cudaTextureDesc tex_desc;
    memset(&tex_desc,0,sizeof(tex_desc));
    tex_desc.readMode = cudaReadModeElementType;
    // Create texture object based on channel and resource descriptors
    checkCudaErrors(cudaCreateTextureObject(&texture_,&res_desc,&tex_desc,NULL));
  }
  /// Dtor
  TextureHandleLinear::~TextureHandleLinear()
  {
    // Release memory
    checkCudaErrors(cudaDestroyTextureObject(texture_));
  }
  /// Return created texture
  /// \todo Check if this should potentially return a const& instead?
  cudaTextureObject_t TextureHandleLinear::get_texture()
  {
    return texture_;
  }
} // MMORF
