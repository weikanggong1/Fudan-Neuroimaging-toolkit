// FNIT's small CPU matrix-function adapter. Eigen is a Conda dependency.
// No image, registration, or original-software runtime is used here.
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <cstddef>

extern "C" int fnit_cpu_affine_sqrt(const float* input, float* output, std::size_t count) {
    using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    try {
        for (std::size_t batch = 0; batch < count; ++batch) {
            for (int index = 0; index < 16; ++index) {
                if (!std::isfinite(input[16 * batch + index])) return 2;
            }
            const Eigen::Map<const Matrix> matrix(input + 16 * batch, 4, 4);
            const Matrix value = matrix;
            const Matrix root = value.sqrt();
            for (int index = 0; index < 16; ++index) {
                if (!std::isfinite(root.data()[index])) return 2;
                output[16 * batch + index] = root.data()[index];
            }
        }
        return 0;
    } catch (...) {
        return 1;
    }
}
