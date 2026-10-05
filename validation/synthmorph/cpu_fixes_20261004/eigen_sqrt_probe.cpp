// Diagnostic adapter for a bound, independently supplied Eigen header tree.
// This program is never loaded by the FNIT runtime.
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    std::ofstream output(argv[2], std::ios::binary);
    using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Matrix matrix(4, 4);
    int count = 0;
    while (input.read(reinterpret_cast<char*>(matrix.data()), 16 * sizeof(float))) {
        const Eigen::RealSchur<Matrix> schur(matrix);
        const Matrix root = matrix.sqrt();
        for (const Matrix* value : {&root, &schur.matrixT(), &schur.matrixU()}) {
            output.write(reinterpret_cast<const char*>(value->data()), 16 * sizeof(float));
        }
        ++count;
    }
    std::cout << "matrices=" << count << " Eigen=" << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    return input.bad() || !output ? 1 : 0;
}
