// FNIT diagnostic: ordered FP32 averaging with one persistent OpenMP team.
// This file is not called by the production package. No upstream code is bundled.
#include <algorithm>
#include <cstdint>
#include <new>
#include <vector>

extern "C" int fnit_average_persistent(
    const float* gradient, const std::int64_t* neighbors,
    const std::int64_t* degrees, const float* reciprocals,
    std::int64_t count, std::int64_t width, std::int64_t iterations,
    int threads, float* output) {
  if (count < 0 || width < 0 || iterations < 0 || threads < 1) return 1;
  try {
    std::vector<float> first(gradient, gradient + count * 3);
    std::vector<float> second(count * 3);
    float* a = first.data();
    float* b = second.data();
    // A barrier at the end of omp for completes every full Jacobi round.
    // Each component retains the source's neighbor order and FP32 rounding.
    #pragma omp parallel num_threads(threads) shared(a, b)
    {
      for (std::int64_t round = 0; round < iterations; ++round) {
        const float* current = (round % 2 == 0) ? a : b;
        float* following = (round % 2 == 0) ? b : a;
        #pragma omp for schedule(static)
        for (std::int64_t vertex = 0; vertex < count; ++vertex) {
          float x = current[3 * vertex];
          float y = current[3 * vertex + 1];
          float z = current[3 * vertex + 2];
          for (std::int64_t column = 0; column < degrees[vertex]; ++column) {
            const auto offset = 3 * neighbors[vertex * width + column];
            x = x + current[offset];
            y = y + current[offset + 1];
            z = z + current[offset + 2];
          }
          following[3 * vertex] = x * reciprocals[vertex];
          following[3 * vertex + 1] = y * reciprocals[vertex];
          following[3 * vertex + 2] = z * reciprocals[vertex];
        }
      }
    }
    const float* result = (iterations % 2 == 0) ? a : b;
    std::copy(result, result + count * 3, output);
    return 0;
  } catch (const std::bad_alloc&) {
    return 2;
  }
}
