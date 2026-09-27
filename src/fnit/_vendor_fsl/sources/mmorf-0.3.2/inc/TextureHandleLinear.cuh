//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief RAII wrapper for dealing with CUDA Linear Texture object
/// \details Use this helper class to deal with the hassle of loading image volumes into
///          linear texture memory, and ensure proper memory acquisition and release
/// \author Frederik Lange
/// \date May 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef TEXTURE_HANDLE_LINEAR_CUH
#define TEXTURE_HANDLE_LINEAR_CUH

#include <thrust/device_vector.h>

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class TextureHandleLinear
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5 (Moveable but not copyable)
////////////////////////////////////////////////////////////////////////////////
      /// Move ctor
      TextureHandleLinear(TextureHandleLinear&& rhs) = default;
      /// Move assignment operator
      TextureHandleLinear& operator=(TextureHandleLinear&& rhs) = default;
      /// Copy ctor
      TextureHandleLinear(const TextureHandleLinear& rhs) = delete;
      /// Copy assignment operator
      TextureHandleLinear& operator=(const TextureHandleLinear& rhs) = delete;
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Ctor
      TextureHandleLinear(
          thrust::device_vector<float>& ima,
          const std::vector<int>& ima_sz);
      TextureHandleLinear();
      /// Dtor
      ~TextureHandleLinear();
      /// Return created texture
      /// \todo Check if this should potentially return a const& instead?
      cudaTextureObject_t get_texture();
    private:
      cudaTextureObject_t texture_;
  }; // TextureHandleLinear
} // MMORF
#endif // TEXTURE_HANDLE_LINEAR_CUH
