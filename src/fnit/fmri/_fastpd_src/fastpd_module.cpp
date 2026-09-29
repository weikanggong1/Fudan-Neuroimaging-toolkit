#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cstdint>
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
        if (coefficient == 0.0) return;
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
    std::vector<double> cubic(face_count);
    Terms unary, pair, reduced_unary, reduced_pair;
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
        const auto& c = face.cost;
        const auto a = face.ids[0], b = face.ids[1], d = face.ids[2];
        constant += c[0];
        unary.add(d, d, c[1]-c[0]);
        unary.add(b, b, c[2]-c[0]);
        pair.add(b, d, c[3]-c[2]-c[1]+c[0]);
        unary.add(a, a, c[4]-c[0]);
        pair.add(a, d, c[5]-c[4]-c[1]+c[0]);
        pair.add(a, b, c[6]-c[4]-c[2]+c[0]);
        cubic[f] = c[7]-c[6]-c[5]-c[3]+c[4]+c[2]+c[1]-c[0];
    }

    int nodes = vertices;
    for (int f = 0; f < face_count; ++f) {
        const double c = cubic[f];
        if (c == 0.0) continue;
        const int extra = nodes++;
        for (int id : faces[f].ids)
            reduced_pair.add(id, extra, c < 0.0 ? c : -c);
        reduced_unary.add(extra, extra, c < 0.0 ? -2*c : c);
        if (c > 0.0) {
            const auto& ids = faces[f].ids;
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
        if (term.coefficient != 0.0)
            model->AddPairwiseTerm(term.first, term.second, 0, 0, 0,
                                   term.coefficient);
    for (const auto& term : reduced_unary.values)
        if (term.coefficient != 0.0)
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
    const auto face_count = face_buffer.len / (3*sizeof(std::int32_t));
    if (vertices <= 0 || face_buffer.len == 0 ||
        face_buffer.len % (3*sizeof(std::int32_t)) != 0 ||
        cost_buffer.len != face_count * 8 * sizeof(double)) {
        PyBuffer_Release(&face_buffer);
        PyBuffer_Release(&cost_buffer);
        PyErr_SetString(PyExc_ValueError, "expected sorted int32 faces [F,3] and float64 costs [F,8]");
        return nullptr;
    }
    const auto* ids = static_cast<const std::int32_t*>(face_buffer.buf);
    for (Py_ssize_t f = 0; f < face_count; ++f) {
        if (ids[3*f] < 0 || ids[3*f+2] >= vertices ||
            !(ids[3*f] < ids[3*f+1] && ids[3*f+1] < ids[3*f+2])) {
            PyBuffer_Release(&face_buffer);
            PyBuffer_Release(&cost_buffer);
            PyErr_SetString(PyExc_ValueError, "face IDs must be sorted, distinct and in range");
            return nullptr;
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

PyMethodDef methods[] = {
    {"optimize", optimize, METH_VARARGS, "HOCR and FastPD fusion for triangle costs."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_fastpd_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__fastpd_native() { return PyModule_Create(&module); }
