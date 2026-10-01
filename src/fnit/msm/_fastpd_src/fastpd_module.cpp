#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <map>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "FastPD.h"

namespace {

struct Face {
    std::array<int, 3> ids;
    std::array<double, 8> cost;
};

struct Term {
    int first, second;
    double coefficient;
};

struct Terms {
    std::vector<Term> values;
    std::unordered_map<std::uint64_t, std::size_t> index;

    void add(int first, int second, double coefficient) {
        if (first > second) std::swap(first, second);
        const auto key = (std::uint64_t(std::uint32_t(first)) << 32) |
                         std::uint32_t(second);
        auto found = index.find(key);
        if (found == index.end()) {
            index[key] = values.size();
            values.push_back({first, second, coefficient});
        } else {
            values[found->second].coefficient += coefficient;
        }
    }
};

std::string fuse(const char* face_bytes, const char* cost_bytes,
                 int vertices, int face_count) {
    std::vector<Face> faces(face_count);
    Terms unary, pair, reduced_unary, reduced_pair;
    // The source creates zero unary terms before adding higher-order cliques.
    // This also preserves unconnected original variables in the reduction.
    for (int node = 0; node < vertices; ++node) unary.add(node, node, 0.0);
    std::vector<std::pair<std::array<int, 3>, double>> cubic;
    std::map<std::array<int, 3>, std::size_t> cubic_index;
    double constant = 0.0;
    for (int f = 0; f < face_count; ++f) {
        Face& face = faces[f];
        for (int j = 0; j < 3; ++j) {
            std::int32_t value;
            std::memcpy(&value, face_bytes + (3*f+j)*sizeof(value), sizeof(value));
            face.ids[j] = value;
        }
        for (int j = 0; j < 8; ++j)
            std::memcpy(&face.cost[j], cost_bytes + (8*f+j)*sizeof(double), sizeof(double));
        // Moebius coefficients, in the reference's ascending energy-table
        // summation order. Hand-written differences change rounding near ties.
        for (int bits = 0; bits < 8; ++bits) {
            double coefficient = 0.0;
            for (int state = 0; state < 8; ++state) {
                int sign = 0;
                if ((~bits & state) == 0) {
                    unsigned parity = static_cast<unsigned>(bits ^ state);
                    int odd = 0;
                    while (parity) { odd ^= parity & 1; parity >>= 1; }
                    sign = odd ? -1 : 1;
                }
                coefficient += face.cost[state] * sign;
            }
            if (coefficient == 0.0) continue;
            if (bits == 0) { constant += coefficient; continue; }
            std::array<int, 3> variables{};
            int degree = 0;
            for (int j = 0; j < 3; ++j)
                if (bits & (4 >> j)) variables[degree++] = face.ids[j];
            if (degree == 1) unary.add(variables[0], variables[0], coefficient);
            else if (degree == 2) pair.add(variables[0], variables[1], coefficient);
            else {
                auto found = cubic_index.find(variables);
                if (found == cubic_index.end()) {
                    cubic_index[variables] = cubic.size();
                    cubic.push_back({variables, coefficient});
                } else cubic[found->second].second += coefficient;
            }
        }
    }

    int nodes = vertices;
    // Aggregate repeated cubic monomials before introducing their auxiliary
    // variables, as HOCR does. One variable per input face is not equivalent
    // to one variable per monomial under approximate FastPD optimization.
    for (const auto& term : cubic) {
        const double c = term.second;
        const int extra = nodes++;
        for (int id : term.first)
            reduced_pair.add(id, extra, c < 0.0 ? c : -c);
        reduced_unary.add(extra, extra, c < 0.0 ? -2*c : c);
        if (c >= 0.0) {
            const auto& ids = term.first;
            reduced_pair.add(ids[0], ids[1], c);
            reduced_pair.add(ids[0], ids[2], c);
            reduced_pair.add(ids[1], ids[2], c);
        }
    }
    for (const auto& term : pair.values)
        reduced_pair.add(term.first, term.second, term.coefficient);
    for (const auto& term : unary.values)
        reduced_unary.add(term.first, term.second, term.coefficient);

    auto model = std::make_shared<newmeshreg::DiscreteModel>();
    model->AddNode(nodes);
    for (const auto& term : reduced_pair.values)
        model->AddPairwiseTerm(term.first, term.second, 0, 0, 0,
                               term.coefficient);
    for (const auto& term : reduced_unary.values)
        model->AddUnaryTerm(term.first, 0, term.coefficient);
    model->AddUnaryTerm(0, constant, constant);

    FPD::FastPD optimizer(model, 5);
    optimizer.run();
    std::vector<int> labels(nodes);
    optimizer.getLabeling(labels.data());
    std::string result(vertices, '\0');
    for (int i = 0; i < vertices; ++i) result[i] = static_cast<char>(labels[i]);
    return result;
}

PyObject* optimize(PyObject*, PyObject* args) {
    Py_buffer face_buffer{}, cost_buffer{};
    int vertices;
    if (!PyArg_ParseTuple(args, "y*y*i", &face_buffer, &cost_buffer, &vertices))
        return nullptr;
    constexpr Py_ssize_t face_width = 3*sizeof(std::int32_t);
    constexpr Py_ssize_t cost_width = 8*sizeof(double);
    const Py_ssize_t face_count = face_buffer.len / face_width;
    if (vertices <= 0 || face_buffer.len == 0 ||
        face_buffer.len % face_width != 0 ||
        face_count > std::numeric_limits<Py_ssize_t>::max()/cost_width ||
        cost_buffer.len != face_count * cost_width) {
        PyBuffer_Release(&face_buffer);
        PyBuffer_Release(&cost_buffer);
        PyErr_SetString(PyExc_ValueError, "expected sorted int32 faces [F,3] and float64 costs [F,8]");
        return nullptr;
    }
    if (face_count > std::numeric_limits<int>::max() ||
        face_count > std::numeric_limits<int>::max() - vertices) {
        PyBuffer_Release(&face_buffer);
        PyBuffer_Release(&cost_buffer);
        PyErr_SetString(PyExc_ValueError, "too many faces or vertices for FastPD");
        return nullptr;
    }
    // Buffers can start at arbitrary byte offsets. Reading through typed
    // pointers is undefined on unaligned input, so validation also uses memcpy.
    const auto* face_data = static_cast<const char*>(face_buffer.buf);
    const auto* cost_data = static_cast<const char*>(cost_buffer.buf);
    for (Py_ssize_t f = 0; f < face_count; ++f) {
        std::int32_t ids[3];
        std::memcpy(ids, face_data + f*3*sizeof(std::int32_t), sizeof(ids));
        if (ids[0] < 0 || ids[2] >= vertices ||
            !(ids[0] < ids[1] && ids[1] < ids[2])) {
            PyBuffer_Release(&face_buffer);
            PyBuffer_Release(&cost_buffer);
            PyErr_SetString(PyExc_ValueError, "face IDs must be sorted, distinct and in range");
            return nullptr;
        }
        for (int state = 0; state < 8; ++state) {
            double value;
            std::memcpy(&value, cost_data + (8*f+state)*sizeof(double), sizeof(value));
            if (!std::isfinite(value)) {
                PyBuffer_Release(&face_buffer);
                PyBuffer_Release(&cost_buffer);
                PyErr_SetString(PyExc_ValueError, "face costs must be finite");
                return nullptr;
            }
        }
    }
    PyObject* answer = nullptr;
    try {
        const auto labels = fuse(static_cast<const char*>(face_buffer.buf),
                                 static_cast<const char*>(cost_buffer.buf),
                                 vertices, static_cast<int>(face_count));
        answer = PyBytes_FromStringAndSize(labels.data(), labels.size());
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    }
    PyBuffer_Release(&face_buffer);
    PyBuffer_Release(&cost_buffer);
    return answer;
}

PyObject* source_wls_cost(PyObject*, PyObject* args) {
    // Independently written scalar reduction matching MIT-licensed newMSM's
    // rigid_costfunction.cpp WLS arithmetic (260718953547743c028a45f8c885d163441df87a).
    // Geometry and pair similarity remain on the GPU. This runs at the
    // existing optimizer host boundary: one packed buffer, one scalar result.
    Py_buffer buffer{};
    Py_ssize_t rows, width;
    double sigma;
    if (!PyArg_ParseTuple(args, "y*nnd", &buffer, &rows, &width, &sigma))
        return nullptr;
    constexpr Py_ssize_t sample_bytes = 3*sizeof(double);
    const Py_ssize_t maximum = std::numeric_limits<Py_ssize_t>::max();
    const double denominator = (2*sigma)*sigma;
    if (rows < 0 || width <= 0 || width > maximum/sample_bytes ||
        rows > maximum/(width*sample_bytes) ||
        buffer.len != rows*width*sample_bytes || !std::isfinite(sigma) ||
        sigma <= 0.0 || !std::isfinite(denominator) || denominator <= 0.0) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError,
            "expected float64 [rows,width,3] buffer and finite positive sigma");
        return nullptr;
    }
    const char* data = static_cast<const char*>(buffer.buf);
    double cost = 0.0;
    for (Py_ssize_t row = 0; row < rows; ++row) {
        double weight_sum = 0.0, value_sum = 0.0;
        for (Py_ssize_t column = 0; column < width; ++column) {
            // A buffer may be a byte-offset memoryview. Avoid undefined
            // unaligned double loads, and retain the GIL for mutable buffers.
            double sample[3];
            std::memcpy(sample, data + (row*width+column)*sample_bytes,
                        sizeof(sample));
            const double distance = sample[0], similarity = sample[1];
            const double valid = sample[2];
            if (!std::isfinite(distance) || distance < 0.0 ||
                !std::isfinite(similarity) || (valid != 0.0 && valid != 1.0)) {
                PyBuffer_Release(&buffer);
                PyErr_SetString(PyExc_ValueError,
                    "WLS distance and similarity must be finite; distance nonnegative and valid zero or one");
                return nullptr;
            }
            if (valid == 0.0 || distance == 0.0) continue;
            const double weight = std::exp(-distance/denominator);
            weight_sum += weight;
            value_sum += similarity*weight;
        }
        if (weight_sum > 0.0) value_sum /= weight_sum;
        cost += value_sum;
    }
    PyBuffer_Release(&buffer);
    if (!std::isfinite(cost)) {
        PyErr_SetString(PyExc_ValueError, "WLS reduction must remain finite");
        return nullptr;
    }
    return PyFloat_FromDouble(cost);
}

PyMethodDef methods[] = {
    {"optimize", optimize, METH_VARARGS, "HOCR and FastPD fusion for triangle costs."},
    {"source_wls_cost", source_wls_cost, METH_VARARGS,
     "Ordered source WLS reduction: float64 distance/similarity/valid buffer, rows, width, sigma."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_fastpd_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__fastpd_native() { return PyModule_Create(&module); }
