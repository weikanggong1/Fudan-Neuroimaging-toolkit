// FNIT-owned copy glue. No upstream implementation is embedded.
// Only an already-loaded, separately validated Torch LP64 SGEMM pointer is used.
#include <ATen/Parallel.h>
#include <cstdint>
#include <cstring>
#include <limits>

using SgemmLP64 = void (*)(char*, char*, int*, int*, int*, float*,
                          const float*, int*, const float*, int*, float*, float*, int*);
static_assert(sizeof(float) == 4 && sizeof(int) == 4, "LP64 SGEMM requires F32/int32");

extern "C" int fnit_columns_abi_description() noexcept {
    // Interface-only probe may call this; it does not touch any tensor or BLAS.
    return 10000 + 100 * static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(int));
}

extern "C" int fnit_copy_columns_f32(const float* input, float* columns,
                                    std::int64_t input_depth, std::int64_t height,
                                    std::int64_t width, std::int64_t output_depth,
                                    std::int64_t columns_items) noexcept {
    constexpr std::int64_t channels = 72, kernel_items = 27, k = channels * kernel_items;
    if (!input || !columns || input == columns || output_depth < 1 || output_depth > 14
            || input_depth != output_depth + 2 || height < 1 || width < 1) return 1;
    if (height > std::numeric_limits<int>::max() / width) return 2;
    const std::int64_t plane = height * width;
    if (output_depth > std::numeric_limits<int>::max() / plane) return 3;
    const std::int64_t m = output_depth * plane;
    if (columns_items < k * m) return 4;
    try {
        // Each task only writes its own K-major row; no floating point arithmetic.
        // The loaded Torch parallel helper respects the caller's existing budget.
        at::parallel_for(0, k, 0, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t q = begin; q < end; ++q) {
                const std::int64_t channel = q / kernel_items;
                const std::int64_t kd = (q / 9) % 3;
                const std::int64_t kh = (q / 3) % 3;
                const std::int64_t kw = q % 3;
                float* destination = columns + q * m;
                for (std::int64_t d = 0; d < output_depth; ++d) {
                    for (std::int64_t h = 0; h < height; ++h) {
                        float* row = destination + (d * height + h) * width;
                        const std::int64_t source_h = h + kh - 1;
                        if (source_h < 0 || source_h >= height) {
                            std::memset(row, 0, static_cast<std::size_t>(width) * sizeof(float));
                            continue;
                        }
                        const float* source = input + ((channel * input_depth + d + kd) * height + source_h) * width;
                        if (kw == 1) {
                            std::memcpy(row, source, static_cast<std::size_t>(width) * sizeof(float));
                        } else if (kw == 0) {
                            std::memset(row, 0, sizeof(float));
                            if (width > 1) std::memcpy(row + 1, source, static_cast<std::size_t>(width - 1) * sizeof(float));
                        } else {
                            if (width > 1) std::memcpy(row, source + 1, static_cast<std::size_t>(width - 1) * sizeof(float));
                            std::memset(row + width - 1, 0, sizeof(float));
                        }
                    }
                }
            }
        });
        return 0;
    } catch (...) { return 5; }
}

extern "C" int fnit_same_provider_sgemm_f32(void* approved_sgemm,
                                           const float* columns, const float* weight,
                                           float* bias_prefilled_output,
                                           std::int64_t m64) noexcept {
    if (!approved_sgemm || !columns || !weight || !bias_prefilled_output
            || m64 < 1 || m64 > std::numeric_limits<int>::max()) return 1;
    int m = static_cast<int>(m64), n = 24, k = 1944, lda = m, ldb = k, ldc = m;
    char transa = 'N', transb = 'N';
    float alpha = 1.0f, beta = 1.0f;
    // This is the public SGEMM provider, not the hidden ATen CPUBlas wrapper.
    reinterpret_cast<SgemmLP64>(approved_sgemm)(
        &transa, &transb, &m, &n, &k, &alpha,
        columns, &lda, weight, &ldb, &beta, bias_prefilled_output, &ldc);
    return 0;
}
