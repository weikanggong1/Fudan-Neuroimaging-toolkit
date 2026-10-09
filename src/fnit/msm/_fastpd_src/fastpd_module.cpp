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
#include <stdexcept>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include "FastPD.h"
#include "ordered_face_octree.h"
#include "source_unfold.h"

namespace {

// Every graph/geometry computation below owns its state. Immutable snapshots
// let independent hemisphere workers run without the Python GIL, including
// when a caller supplied a writable or read-only view of a mutable array.
// Snapshot references and Python exceptions are handled only while holding it.
struct BufferSnapshot {
    PyObject* object;
    const char* pointer;
    explicit BufferSnapshot(const Py_buffer& buffer) {
        if (PyBytes_Check(buffer.obj)) { object = buffer.obj; Py_INCREF(object); }
        else object = PyBytes_FromStringAndSize(static_cast<const char*>(buffer.buf), buffer.len);
        pointer = object ? PyBytes_AS_STRING(object) : nullptr;
    }
    ~BufferSnapshot() { Py_XDECREF(object); }
    const char* data() const { return pointer; }
};
struct ReleasedGIL {
    PyThreadState* state;
    ReleasedGIL() : state(PyEval_SaveThread()) {}
    ~ReleasedGIL() { PyEval_RestoreThread(state); }
    ReleasedGIL(const ReleasedGIL&) = delete;
    ReleasedGIL& operator=(const ReleasedGIL&) = delete;
};

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
    // Validate the immutable copies, so the checked bytes are exactly those
    // consumed after releasing the GIL, even for mutable caller buffers.
    BufferSnapshot face_copy(face_buffer), cost_copy(cost_buffer);
    if (!face_copy.object || !cost_copy.object) {
        PyBuffer_Release(&face_buffer); PyBuffer_Release(&cost_buffer); return nullptr;
    }
    // Buffers can start at arbitrary byte offsets. Reading through typed
    // pointers is undefined on unaligned input, so validation also uses memcpy.
    const auto* face_data = face_copy.data();
    const auto* cost_data = cost_copy.data();
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
        std::string labels;
        {
            ReleasedGIL unlocked;
            labels = fuse(face_copy.data(), cost_copy.data(), vertices, static_cast<int>(face_count));
        }
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
    BufferSnapshot snapshot(buffer);
    if (!snapshot.object) { PyBuffer_Release(&buffer); return nullptr; }
    const char* data = snapshot.data();
    double cost = 0.0;
    try {
      ReleasedGIL unlocked;
      for (Py_ssize_t row = 0; row < rows; ++row) {
        double weight_sum = 0.0, value_sum = 0.0;
        for (Py_ssize_t column = 0; column < width; ++column) {
            // A buffer may be a byte-offset memoryview. Avoid undefined
            // unaligned double loads. The immutable snapshot stays valid
            // while the GIL is released.
            double sample[3];
            std::memcpy(sample, data + (row*width+column)*sample_bytes,
                        sizeof(sample));
            const double distance = sample[0], similarity = sample[1];
            const double valid = sample[2];
            if (!std::isfinite(distance) || distance < 0.0 ||
                !std::isfinite(similarity) || (valid != 0.0 && valid != 1.0)) {
                throw std::invalid_argument(
                    "WLS distance and similarity must be finite; distance nonnegative and valid zero or one");
            }
            if (valid == 0.0 || distance == 0.0) continue;
            const double weight = std::exp(-distance/denominator);
            weight_sum += weight;
            value_sum += similarity*weight;
        }
        if (weight_sum > 0.0) value_sum /= weight_sum;
        cost += value_sum;
    }
    } catch (const std::exception& error) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    PyBuffer_Release(&buffer);
    if (!std::isfinite(cost)) {
        PyErr_SetString(PyExc_ValueError, "WLS reduction must remain finite");
        return nullptr;
    }
    return PyFloat_FromDouble(cost);
}

using Point3 = std::array<double, 3>;

double point_norm(const Point3& point) {
    return std::sqrt((point[0]*point[0] + point[1]*point[1]) +
                     point[2]*point[2]);
}

void normalize_point(Point3& point) {
    const double length = point_norm(point);
    if (length > 1e-8) {
        point[0] /= length;
        point[1] /= length;
        point[2] /= length;
    }
}

PyObject* source_rotation_matrices(PyObject*, PyObject* args) {
    // Independently authored scalar Point/Rodrigues arithmetic following
    // MIT-licensed newMSM point.cpp at the pinned source revision above.
    // Cache once per iteration at the existing prior-coordinate host copy;
    // application to all proposed labels remains a batched GPU operation.
    Py_buffer prior_buffer{}, centre_buffer{};
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "y*y*n", &prior_buffer, &centre_buffer, &count))
        return nullptr;
    constexpr Py_ssize_t point_bytes = 3*sizeof(double);
    constexpr Py_ssize_t matrix_bytes = 9*sizeof(double);
    const auto maximum = std::numeric_limits<Py_ssize_t>::max();
    const auto release = [&]() {
        PyBuffer_Release(&prior_buffer);
        PyBuffer_Release(&centre_buffer);
    };
    if (count < 0 || count > maximum/matrix_bytes ||
        prior_buffer.len != count*point_bytes || centre_buffer.len != point_bytes) {
        release();
        PyErr_SetString(PyExc_ValueError,
            "expected float64 prior [count,3] and centre [3] buffers");
        return nullptr;
    }
    Point3 centre;
    std::memcpy(centre.data(), centre_buffer.buf, point_bytes);
    for (double value : centre) if (!std::isfinite(value)) {
        release();
        PyErr_SetString(PyExc_ValueError, "rotation coordinates must be finite");
        return nullptr;
    }
    normalize_point(centre);
    PyObject* output = PyBytes_FromStringAndSize(nullptr, count*matrix_bytes);
    if (!output) { release(); return nullptr; }
    char* matrices = PyBytes_AS_STRING(output);
    BufferSnapshot snapshot(prior_buffer);
    if (!snapshot.object) { release(); Py_DECREF(output); return nullptr; }
    const char* points = snapshot.data();
    try {
      ReleasedGIL unlocked;
      for (Py_ssize_t index = 0; index < count; ++index) {
        Point3 point;
        std::memcpy(point.data(), points+index*point_bytes, point_bytes);
        for (double value : point) if (!std::isfinite(value)) {
            throw std::invalid_argument("rotation coordinates must be finite");
        }
        normalize_point(point);
        const double cosine = (centre[0]*point[0]+centre[1]*point[1]) +
                               centre[2]*point[2];
        // Do not clamp the cosine or alter the source EPS branches. A same
        // direction cosine slightly above one still selects identity below.
        const double angle = std::acos(cosine);
        Point3 axis{{centre[1]*point[2]-centre[2]*point[1],
                     centre[2]*point[0]-centre[0]*point[2],
                     centre[0]*point[1]-centre[1]*point[0]}};
        normalize_point(axis);
        double matrix[9];
        if (std::fabs(1-cosine) < 1e-8) {
            const double identity[9] = {1,0,0,0,1,0,0,0,1};
            std::memcpy(matrix, identity, sizeof(matrix));
        } else if (point_norm(axis) < 1e-8) {
            const double opposite[9] = {-1,0,0,0,-1,0,0,0,-1};
            std::memcpy(matrix, opposite, sizeof(matrix));
        } else if (std::fabs(-1-cosine) < 1e-8) {
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    matrix[3*row+column] = 2*(axis[row]*axis[column]) -
                                           (row == column ? 1.0 : 0.0);
        } else {
            const double skew[9] = {0,-axis[2],axis[1],
                                    axis[2],0,-axis[0],
                                    -axis[1],axis[0],0};
            const double sine = std::sin(angle);
            const double one_minus_cosine = 1-std::cos(angle);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    const double squared =
                        (skew[3*row]*skew[column] +
                         skew[3*row+1]*skew[3+column]) +
                         skew[3*row+2]*skew[6+column];
                    matrix[3*row+column] =
                        ((row == column ? 1.0 : 0.0) +
                         skew[3*row+column]*sine) + one_minus_cosine*squared;
                }
            }
        }
        std::memcpy(matrices+index*matrix_bytes, matrix, sizeof(matrix));
    }
    } catch (const std::exception& error) {
        release(); Py_DECREF(output); PyErr_SetString(PyExc_ValueError, error.what()); return nullptr;
    }
    release();
    return output;
}

Point3 subtract_point(const Point3& first, const Point3& second) {
    return {{first[0]-second[0], first[1]-second[1], first[2]-second[2]}};
}

Point3 cross_point(const Point3& first, const Point3& second) {
    return {{first[1]*second[2]-first[2]*second[1],
             second[0]*first[2]-second[2]*first[0],
             first[0]*second[1]-second[0]*first[1]}};
}

double dot_point(const Point3& first, const Point3& second) {
    return (first[0]*second[0]+first[1]*second[1])+first[2]*second[2];
}

bool same_triangle_side(const Point3& point, const Point3& opposite,
                        const Point3& first, const Point3& second) {
    const auto edge = subtract_point(second, first);
    return dot_point(cross_point(edge, subtract_point(point, first)),
                     cross_point(edge, subtract_point(opposite, first))) > -1e-8;
}

double finite_triangle_distance(const Point3& point, const std::array<Point3, 3>& triangle) {
    double best = std::numeric_limits<double>::max();
    const int edges[3][2] = {{0,1},{0,2},{1,2}};
    for (const auto& pair : edges) {
        const auto first_delta = subtract_point(point, triangle[pair[0]]);
        const auto second_delta = subtract_point(point, triangle[pair[1]]);
        const auto edge = subtract_point(triangle[pair[1]], triangle[pair[0]]);
        if (dot_point(first_delta, edge) > 0 && dot_point(second_delta, edge) < 0) {
            const double distance = point_norm(cross_point(first_delta, second_delta))/point_norm(edge);
            if (distance < best) best = distance;
        }
    }
    for (const auto& corner : triangle) {
        const double distance = point_norm(subtract_point(point, corner));
        if (distance < best) best = distance;
    }
    return best;
}

PyObject* source_radial_selection(PyObject*, PyObject* args) {
    // Independent scalar implementation of the pinned Point projection and
    // finite-edge comparisons. Resolve ambiguous containing candidates;
    // retain GPU evaluation for ordinary points and all feature sampling.
    Py_buffer vertices{}, faces{}, queries{}, candidates{};
    Py_ssize_t vertex_count, face_count, query_count, width;
    auto release = [&]() {
        for (auto* buffer : {&vertices, &faces, &queries, &candidates})
            if (buffer->obj) PyBuffer_Release(buffer);
    };
    if (!PyArg_ParseTuple(args, "y*y*y*y*nnnn", &vertices, &faces, &queries, &candidates,
                          &vertex_count, &face_count, &query_count, &width)) {
        release(); return nullptr;
    }
    if (vertex_count < 1 || face_count < 1 || query_count < 0 || width < 1 ||
        vertices.len != vertex_count*3*Py_ssize_t(sizeof(double)) ||
        faces.len != face_count*3*Py_ssize_t(sizeof(std::int64_t)) ||
        queries.len != query_count*3*Py_ssize_t(sizeof(double)) ||
        candidates.len != query_count*width*Py_ssize_t(sizeof(std::int64_t))) {
        release(); PyErr_SetString(PyExc_ValueError, "invalid radial-selection buffer dimensions");
        return nullptr;
    }
    auto* output = PyBytes_FromStringAndSize(nullptr, query_count*4*sizeof(double));
    if (!output) { release(); return nullptr; }
    char* output_data = PyBytes_AS_STRING(output);
    BufferSnapshot vertices_copy(vertices);
    BufferSnapshot faces_copy(faces);
    BufferSnapshot queries_copy(queries);
    BufferSnapshot candidates_copy(candidates);
    if (!vertices_copy.object || !faces_copy.object || !queries_copy.object || !candidates_copy.object) { release(); Py_DECREF(output); return nullptr; }
    auto read_point = [](const char* data, Py_ssize_t index) {
        Point3 point;
        std::memcpy(point.data(), data+index*3*sizeof(double), 3*sizeof(double));
        return point;
    };
    try {
      ReleasedGIL unlocked;
    for (Py_ssize_t row = 0; row < query_count; ++row) {
        const Point3 point = read_point(queries_copy.data(), row);
        double best = std::numeric_limits<double>::max();
        double result[4] = {-1, 0, 0, 0};
        for (Py_ssize_t column = 0; column < width; ++column) {
            std::int64_t face;
            std::memcpy(&face, candidates_copy.data()+(row*width+column)*sizeof(face), sizeof(face));
            if (face < 0 || face >= face_count) {
                throw std::invalid_argument("radial candidate is outside the mesh");
            }
            std::array<Point3, 3> triangle;
            for (int corner = 0; corner < 3; ++corner) {
                std::int64_t vertex;
                std::memcpy(&vertex, faces_copy.data()+(face*3+corner)*sizeof(vertex), sizeof(vertex));
                if (vertex < 0 || vertex >= vertex_count) {
                    throw std::invalid_argument("triangle vertex is outside the mesh");
                }
                triangle[corner] = read_point(vertices_copy.data(), vertex);
            }
            auto first = subtract_point(triangle[2], triangle[0]); normalize_point(first);
            auto second = subtract_point(triangle[1], triangle[0]); normalize_point(second);
            auto normal = cross_point(first, second); normalize_point(normal);
            const double scale = dot_point(normal, triangle[0])/dot_point(normal, point);
            const Point3 projected{{point[0]*scale, point[1]*scale, point[2]*scale}};
            if (!same_triangle_side(projected, triangle[0], triangle[1], triangle[2]) ||
                !same_triangle_side(projected, triangle[1], triangle[2], triangle[0]) ||
                !same_triangle_side(projected, triangle[2], triangle[0], triangle[1])) continue;
            const double distance = finite_triangle_distance(projected, triangle);
            if (distance < best) {
                best = distance; result[0] = static_cast<double>(face);
                for (int dimension = 0; dimension < 3; ++dimension) result[dimension+1] = projected[dimension];
            }
        }
        std::memcpy(output_data+row*4*sizeof(double), result, sizeof(result));
    }
    } catch (const std::exception& error) {
        release(); Py_DECREF(output); PyErr_SetString(PyExc_ValueError, error.what()); return nullptr;
    }
    release(); return output;
}

struct OrderedScalarTriangle {
    std::array<Point3, 3> corners;
    Point3 normal;
    double plane;
    std::array<Point3, 3> edges, side_origins, side_normals;
    explicit OrderedScalarTriangle(const std::array<Point3, 3>& values): corners(values) {
        auto first = subtract_point(corners[2], corners[0]); normalize_point(first);
        auto second = subtract_point(corners[1], corners[0]); normalize_point(second);
        normal = cross_point(first, second); normalize_point(normal);
        plane = dot_point(normal, corners[0]);
        const int order[3][3] = {{0,1,2},{1,2,0},{2,0,1}};
        for (int side = 0; side < 3; ++side) {
            const auto& ids = order[side];
            side_origins[side] = corners[ids[1]];
            edges[side] = subtract_point(corners[ids[2]], corners[ids[1]]);
            side_normals[side] = cross_point(edges[side], subtract_point(corners[ids[0]], corners[ids[1]]));
        }
    }
    Point3 project(const Point3& point) const {
        const double scale = plane / dot_point(normal, point);
        return {{point[0] * scale, point[1] * scale, point[2] * scale}};
    }
    bool contains(const Point3& point) const {
        for (int side = 0; side < 3; ++side)
            if (!(dot_point(cross_point(edges[side], subtract_point(point, side_origins[side])), side_normals[side]) > -1e-8)) return false;
        return true;
    }
};

PyObject* source_ordered_selection(PyObject*, PyObject* args) {
    // Independent literal-double ordered leaf, direct-sibling and nearest
    // sibling-corner selection. All input arrays are immutable snapshots;
    // per-row work has independent state and a caller-controlled CPU budget.
    Py_buffer vertices{}, faces{}, queries{}, node_ids{}, leaf_offsets{}, leaf_faces{}, fallback_offsets{}, fallback_faces{};
    Py_ssize_t vertex_count, face_count, query_count, worker_count;
    auto release = [&]() {
        for (auto* buffer: {&vertices, &faces, &queries, &node_ids, &leaf_offsets, &leaf_faces, &fallback_offsets, &fallback_faces})
            if (buffer->obj) PyBuffer_Release(buffer);
    };
    if (!PyArg_ParseTuple(args, "y*y*y*y*y*y*y*y*nnnn", &vertices, &faces, &queries, &node_ids,
                         &leaf_offsets, &leaf_faces, &fallback_offsets, &fallback_faces,
                         &vertex_count, &face_count, &query_count, &worker_count)) { release(); return nullptr; }
    const auto maximum = std::numeric_limits<Py_ssize_t>::max();
    const Py_ssize_t node_count = leaf_offsets.len / Py_ssize_t(sizeof(std::int64_t)) - 1;
    if (vertex_count < 1 || face_count < 1 || query_count < 0 || worker_count < 1 || worker_count > 1024 || node_count < 1 ||
        vertex_count > maximum / (3 * Py_ssize_t(sizeof(double))) || face_count > maximum / (3 * Py_ssize_t(sizeof(std::int64_t))) ||
        query_count > maximum / (4 * Py_ssize_t(sizeof(double))) ||
        vertices.len != vertex_count * 3 * Py_ssize_t(sizeof(double)) || faces.len != face_count * 3 * Py_ssize_t(sizeof(std::int64_t)) ||
        queries.len != query_count * 3 * Py_ssize_t(sizeof(double)) || node_ids.len != query_count * Py_ssize_t(sizeof(std::int32_t)) ||
        leaf_offsets.len % sizeof(std::int64_t) || fallback_offsets.len != leaf_offsets.len ||
        leaf_faces.len % sizeof(std::int32_t) || fallback_faces.len % sizeof(std::int32_t)) {
        release(); PyErr_SetString(PyExc_ValueError, "invalid ordered-selection buffer dimensions"); return nullptr;
    }
    auto* output = PyBytes_FromStringAndSize(nullptr, query_count * 4 * sizeof(double));
    if (!output) { release(); return nullptr; }
    char* output_data = PyBytes_AS_STRING(output);
    BufferSnapshot v_copy(vertices), f_copy(faces), q_copy(queries), n_copy(node_ids),
        lo_copy(leaf_offsets), lf_copy(leaf_faces), fo_copy(fallback_offsets), ff_copy(fallback_faces);
    if (!v_copy.object || !f_copy.object || !q_copy.object || !n_copy.object || !lo_copy.object || !lf_copy.object || !fo_copy.object || !ff_copy.object) {
        release(); Py_DECREF(output); return nullptr;
    }
    const auto read_point = [](const char* bytes, Py_ssize_t index) {
        Point3 point; std::memcpy(point.data(), bytes + index * 3 * sizeof(double), 3 * sizeof(double)); return point;
    };
    const auto read_i64 = [](const char* bytes, Py_ssize_t index) {
        std::int64_t value; std::memcpy(&value, bytes + index * sizeof(value), sizeof(value)); return value;
    };
    const auto read_i32 = [](const char* bytes, Py_ssize_t index) {
        std::int32_t value; std::memcpy(&value, bytes + index * sizeof(value), sizeof(value)); return value;
    };
    try {
        ReleasedGIL unlocked;
        // Reject malformed offsets/indices before starting worker threads.
        const struct { const char* offsets; const char* faces; Py_ssize_t entries; } pools[] = {
            {lo_copy.data(), lf_copy.data(), leaf_faces.len / Py_ssize_t(sizeof(std::int32_t))},
            {fo_copy.data(), ff_copy.data(), fallback_faces.len / Py_ssize_t(sizeof(std::int32_t))}};
        for (const auto& pool: pools) {
            const auto* offsets = pool.offsets;
            const auto entries = pool.entries;
            if (read_i64(offsets, 0) != 0 || read_i64(offsets, node_count) != entries)
                throw std::invalid_argument("ordered face offsets do not span their pool");
            for (Py_ssize_t node = 0; node < node_count; ++node)
                if (read_i64(offsets, node) < 0 || read_i64(offsets, node + 1) < read_i64(offsets, node))
                    throw std::invalid_argument("ordered face offsets are not monotonic");
            for (Py_ssize_t index = 0; index < entries; ++index) {
                const auto face = read_i32(pool.faces, index);
                if (face < 0 || face >= face_count) throw std::invalid_argument("ordered face candidate is outside the mesh");
            }
        }
        std::vector<OrderedScalarTriangle> triangles;
        triangles.reserve(static_cast<std::size_t>(face_count));
        for (Py_ssize_t face = 0; face < face_count; ++face) {
            std::array<Point3, 3> corners;
            for (int corner = 0; corner < 3; ++corner) {
                const auto vertex = read_i64(f_copy.data(), face * 3 + corner);
                if (vertex < 0 || vertex >= vertex_count) throw std::invalid_argument("ordered triangle vertex is outside the mesh");
                corners[corner] = read_point(v_copy.data(), vertex);
                for (double value: corners[corner]) if (!std::isfinite(value)) throw std::invalid_argument("ordered sphere coordinates must be finite");
            }
            triangles.emplace_back(corners);
        }
        const auto select_rows = [&](Py_ssize_t begin, Py_ssize_t end) {
            for (Py_ssize_t row = begin; row < end; ++row) {
                const Point3 point = read_point(q_copy.data(), row);
                for (double value: point) if (!std::isfinite(value) || value < -101.0 || value > 101.0)
                    throw std::invalid_argument("ordered query must be finite and within [-101,101] root");
                const auto node = read_i32(n_copy.data(), row);
                if (node < 0 || node >= node_count) throw std::invalid_argument("ordered flat node is outside the tree");
                double result[4] = {-1, 0, 0, 0};
                const auto select_pool = [&](const char* offsets, const char* pool) {
                    double best = std::numeric_limits<double>::max();
                    for (auto entry = read_i64(offsets, node); entry < read_i64(offsets, node + 1); ++entry) {
                        const auto face = read_i32(pool, static_cast<Py_ssize_t>(entry));
                        const auto& triangle = triangles[face];
                        const auto projected = triangle.project(point);
                        if (!triangle.contains(projected)) continue;
                        const auto distance = finite_triangle_distance(projected, triangle.corners);
                        if (distance < best) {
                            best = distance; result[0] = static_cast<double>(face);
                            for (int axis = 0; axis < 3; ++axis) result[axis + 1] = projected[axis];
                        }
                    }
                };
                select_pool(lo_copy.data(), lf_copy.data());
                if (result[0] < 0) select_pool(fo_copy.data(), ff_copy.data());
                if (result[0] < 0) {
                    // Source's final fallback scans direct siblings only,
                    // retaining face/corner order and strict distance ties.
                    double best = std::numeric_limits<double>::max();
                    std::int32_t chosen = -1;
                    for (auto entry = read_i64(fo_copy.data(), node); entry < read_i64(fo_copy.data(), node + 1); ++entry) {
                        const auto face = read_i32(ff_copy.data(), static_cast<Py_ssize_t>(entry));
                        for (const auto& corner: triangles[face].corners) {
                            const auto chord = point_norm(subtract_point(corner, point));
                            const auto distance = 200.0 * std::asin(chord / 200.0);
                            if (distance < best) { best = distance; chosen = face; }
                        }
                    }
                    if (chosen < 0) throw std::invalid_argument("ordered leaf/direct-sibling fallback has no selected triangle");
                    const auto projected = triangles[chosen].project(point);
                    result[0] = static_cast<double>(chosen);
                    for (int axis = 0; axis < 3; ++axis) result[axis + 1] = projected[axis];
                }
                std::memcpy(output_data + row * 4 * sizeof(double), result, sizeof(result));
            }
        };
        const Py_ssize_t workers = std::min(worker_count, std::max(Py_ssize_t(1), query_count));
        if (workers == 1) select_rows(0, query_count);
        else {
            std::vector<std::thread> threads;
            std::vector<std::exception_ptr> errors(static_cast<std::size_t>(workers));
            threads.reserve(static_cast<std::size_t>(workers));
            try {
                for (Py_ssize_t worker = 0; worker < workers; ++worker)
                    threads.emplace_back([&, worker]() {
                        try {
                            const auto chunk = query_count / workers, remainder = query_count % workers;
                            const auto begin = chunk * worker + std::min(worker, remainder);
                            const auto end = begin + chunk + (worker < remainder ? 1 : 0);
                            select_rows(begin, end);
                        }
                        catch (...) { errors[worker] = std::current_exception(); }
                    });
            } catch (...) { for (auto& thread: threads) thread.join(); throw; }
            for (auto& thread: threads) thread.join();
            for (const auto& error: errors) if (error) std::rethrow_exception(error);
        }
    } catch (const std::exception& error) {
        release(); Py_DECREF(output); PyErr_SetString(PyExc_ValueError, error.what()); return nullptr;
    }
    release(); return output;
}

PyObject* source_sphere_warp(PyObject*, PyObject* args) {
    // Rebuild a sphere point in scalar order at the warp boundary. The
    // face lookup remains batched; no host crossings occur per point.
    Py_buffer vertices{}, faces{}, target{}, queries{}, patches{};
    Py_ssize_t vertex_count, face_count, query_count;
    auto release = [&]() {
        for (auto* buffer : {&vertices, &faces, &target, &queries, &patches})
            if (buffer->obj) PyBuffer_Release(buffer);
    };
    if (!PyArg_ParseTuple(args, "y*y*y*y*y*nnn", &vertices, &faces, &target,
                         &queries, &patches, &vertex_count, &face_count, &query_count)) {
        release(); return nullptr;
    }
    const Py_ssize_t maximum = std::numeric_limits<Py_ssize_t>::max();
    if (vertex_count < 1 || face_count < 1 || query_count < 0 ||
        vertex_count > maximum/(3*sizeof(double)) ||
        face_count > maximum/(3*sizeof(std::int64_t)) ||
        query_count > maximum/(3*sizeof(double)) ||
        vertices.len != vertex_count*3*Py_ssize_t(sizeof(double)) ||
        target.len != vertices.len ||
        faces.len != face_count*3*Py_ssize_t(sizeof(std::int64_t)) ||
        queries.len != query_count*3*Py_ssize_t(sizeof(double)) ||
        patches.len != query_count*Py_ssize_t(sizeof(std::int64_t))) {
        release(); PyErr_SetString(PyExc_ValueError, "invalid sphere-warp buffer dimensions");
        return nullptr;
    }
    auto* output = PyBytes_FromStringAndSize(nullptr, queries.len);
    if (!output) { release(); return nullptr; }
    char* output_data = PyBytes_AS_STRING(output);
    BufferSnapshot vertices_copy(vertices);
    BufferSnapshot faces_copy(faces);
    BufferSnapshot target_copy(target);
    BufferSnapshot queries_copy(queries);
    BufferSnapshot patches_copy(patches);
    if (!vertices_copy.object || !faces_copy.object || !target_copy.object || !queries_copy.object || !patches_copy.object) { release(); Py_DECREF(output); return nullptr; }
    auto read_point = [](const char* data, Py_ssize_t index) {
        Point3 point;
        std::memcpy(point.data(), data+index*3*sizeof(double), 3*sizeof(double));
        return point;
    };
    try {
      ReleasedGIL unlocked;
    for (Py_ssize_t row = 0; row < query_count; ++row) {
        std::int64_t face;
        std::memcpy(&face, patches_copy.data()+row*sizeof(face), sizeof(face));
        if (face < 0 || face >= face_count) {
            throw std::invalid_argument("warp face is outside the mesh");
        }
        std::array<Point3, 3> triangle;
        std::array<std::int64_t, 3> ids;
        for (int corner = 0; corner < 3; ++corner) {
            std::memcpy(&ids[corner], faces_copy.data()+(face*3+corner)*sizeof(std::int64_t), sizeof(std::int64_t));
            if (ids[corner] < 0 || ids[corner] >= vertex_count) {
                throw std::invalid_argument("warp vertex is outside the mesh");
            }
            triangle[corner] = read_point(vertices_copy.data(), ids[corner]);
        }
        auto first = subtract_point(triangle[2], triangle[0]); normalize_point(first);
        auto second = subtract_point(triangle[1], triangle[0]); normalize_point(second);
        auto normal = cross_point(first, second); normalize_point(normal);
        const auto point = read_point(queries_copy.data(), row);
        const double scale = dot_point(normal, triangle[0])/dot_point(normal, point);
        const Point3 projected{{point[0]*scale, point[1]*scale, point[2]*scale}};
        const auto area = [&](int first_corner, int second_corner) {
            return 0.5*point_norm(cross_point(subtract_point(triangle[first_corner], projected),
                                             subtract_point(triangle[second_corner], projected)));
        };
        std::array<double, 3> weights{{area(1, 2), area(0, 2), area(0, 1)}};
        const double total = (weights[0]+weights[1])+weights[2];
        std::array<int, 3> order{{0, 1, 2}};
        std::sort(order.begin(), order.end(), [&](int a, int b) { return ids[a] < ids[b]; });
        Point3 result{{0, 0, 0}};
        for (int corner : order) {
            const auto destination = read_point(target_copy.data(), ids[corner]);
            const double weight = weights[corner]/total;
            for (int dimension = 0; dimension < 3; ++dimension)
                result[dimension] += destination[dimension]*weight;
        }
        normalize_point(result);
        for (double& value : result) value *= 100;
        if (!std::isfinite(result[0]) || !std::isfinite(result[1]) || !std::isfinite(result[2])) {
            throw std::invalid_argument("sphere warp produced a nonfinite point");
        }
        std::memcpy(output_data+row*3*sizeof(double), result.data(), 3*sizeof(double));
    }
    } catch (const std::exception& error) {
        release(); Py_DECREF(output); PyErr_SetString(PyExc_ValueError, error.what()); return nullptr;
    }
    release(); return output;
}

PyObject* source_triangle_nearest(PyObject*, PyObject* args) {
    Py_buffer vertices{}, faces{}, queries{}, patches{};
    Py_ssize_t vertex_count, face_count, query_count;
    auto release = [&]() {
        for (auto* buffer : {&vertices, &faces, &queries, &patches})
            if (buffer->obj) PyBuffer_Release(buffer);
    };
    if (!PyArg_ParseTuple(args, "y*y*y*y*nnn", &vertices, &faces, &queries,
                         &patches, &vertex_count, &face_count, &query_count)) {
        release(); return nullptr;
    }
    const Py_ssize_t maximum = std::numeric_limits<Py_ssize_t>::max();
    if (vertex_count < 1 || face_count < 1 || query_count < 0 ||
        vertex_count > maximum/(3*sizeof(double)) ||
        face_count > maximum/(3*sizeof(std::int64_t)) ||
        query_count > maximum/(3*sizeof(double)) ||
        vertices.len != vertex_count*3*Py_ssize_t(sizeof(double)) ||
        faces.len != face_count*3*Py_ssize_t(sizeof(std::int64_t)) ||
        queries.len != query_count*3*Py_ssize_t(sizeof(double)) ||
        patches.len != query_count*Py_ssize_t(sizeof(std::int64_t))) {
        release(); PyErr_SetString(PyExc_ValueError, "invalid triangle-nearest buffer dimensions");
        return nullptr;
    }
    auto* output = PyBytes_FromStringAndSize(nullptr, patches.len);
    if (!output) { release(); return nullptr; }
    char* output_data = PyBytes_AS_STRING(output);
    BufferSnapshot vertices_copy(vertices);
    BufferSnapshot faces_copy(faces);
    BufferSnapshot queries_copy(queries);
    BufferSnapshot patches_copy(patches);
    if (!vertices_copy.object || !faces_copy.object || !queries_copy.object || !patches_copy.object) { release(); Py_DECREF(output); return nullptr; }
    try {
      ReleasedGIL unlocked;
    for (Py_ssize_t row = 0; row < query_count; ++row) {
        std::int64_t face;
        std::memcpy(&face, patches_copy.data()+row*sizeof(face), sizeof(face));
        if (face < 0 || face >= face_count) {
            throw std::invalid_argument("nearest face is outside the mesh");
        }
        Point3 point;
        std::memcpy(point.data(), queries_copy.data()+row*3*sizeof(double), 3*sizeof(double));
        double best = std::numeric_limits<double>::max();
        std::int64_t selected = -1;
        for (int corner = 0; corner < 3; ++corner) {
            std::int64_t vertex;
            std::memcpy(&vertex, faces_copy.data()+(face*3+corner)*sizeof(vertex), sizeof(vertex));
            if (vertex < 0 || vertex >= vertex_count) {
                throw std::invalid_argument("nearest vertex is outside the mesh");
            }
            Point3 coordinates;
            std::memcpy(coordinates.data(), vertices_copy.data()+vertex*3*sizeof(double), 3*sizeof(double));
            const double distance = point_norm(subtract_point(point, coordinates));
            if (distance < best) { best = distance; selected = vertex; }
        }
        if (selected < 0) {
            throw std::invalid_argument("nearest point distances are nonfinite");
        }
        std::memcpy(output_data+row*sizeof(selected), &selected, sizeof(selected));
    }
    } catch (const std::exception& error) {
        release(); Py_DECREF(output); PyErr_SetString(PyExc_ValueError, error.what()); return nullptr;
    }
    release(); return output;
}

PyObject* source_unfold(PyObject*,PyObject* args) {
    Py_buffer vertices{},faces{};Py_ssize_t count,face_count,maximum_sweeps;
    if(!PyArg_ParseTuple(args,"y*y*nnn",&vertices,&faces,&count,&face_count,&maximum_sweeps)) {
        if(vertices.obj)PyBuffer_Release(&vertices);
        if(faces.obj)PyBuffer_Release(&faces);
        return nullptr;
    }
    const auto release=[&](){PyBuffer_Release(&vertices);PyBuffer_Release(&faces);};
    const auto maximum=std::numeric_limits<Py_ssize_t>::max();
    if(count<1 || face_count<1 || count>maximum/24 || face_count>maximum/24 ||
       maximum_sweeps<0 || maximum_sweeps>1000000 || vertices.len!=count*24 || faces.len!=face_count*24) {
        release();PyErr_SetString(PyExc_ValueError,"invalid float64/int64 unfolding buffers or sweep limit");return nullptr;
    }
    BufferSnapshot points_copy(vertices),faces_copy(faces);
    if(!points_copy.object || !faces_copy.object){release();return nullptr;}
    std::vector<fnit_source_unfold::Point> points;
    std::vector<fnit_source_unfold::Face> triangles;
    std::uint64_t updates=0;
    try {
        ReleasedGIL unlocked;
        points.resize(static_cast<std::size_t>(count));
        triangles.resize(static_cast<std::size_t>(face_count));
        std::memcpy(points.data(),points_copy.data(),vertices.len);
        std::memcpy(triangles.data(),faces_copy.data(),faces.len);
        std::vector<std::vector<std::int64_t>> incident(static_cast<std::size_t>(count));
        for(const auto& point:points)for(double value:point)
            if(!std::isfinite(value))throw std::invalid_argument("unfolding coordinates must be finite");
        for(std::int64_t face=0;face<face_count;++face)for(const auto vertex:triangles[face]) {
            if(vertex<0 || vertex>=count)throw std::invalid_argument("unfolding face index outside mesh");
            incident[vertex].push_back(face);
        }
        for(const auto& neighbours:incident)
            if(neighbours.empty())throw std::invalid_argument("unfolding requires an incident face for every vertex");
        updates=fnit_source_unfold::unfold(points,triangles,incident,maximum_sweeps);
        for(const auto& point:points)for(double value:point)
            if(!std::isfinite(value))throw std::invalid_argument("unfolding produced nonfinite coordinates");
    }catch(const std::exception& error) {
        release();PyErr_SetString(PyExc_ValueError,error.what());return nullptr;
    }
    release();
    return Py_BuildValue("(y#K)",reinterpret_cast<const char*>(points.data()),count*24,
                         static_cast<unsigned long long>(updates));
}

PyMethodDef methods[] = {
    {"source_unfold",source_unfold,METH_VARARGS,
     "Sequential scalar source unfolding: float64 vertices, int64 faces, V,F,sweep_limit -> coordinates, update count."},
    {"build_ordered_face_octree", fnit_ordered_octree::build, METH_VARARGS,
     "Build source-ordered octree node/leaf/direct-sibling buffers."},
    {"source_ordered_selection", source_ordered_selection, METH_VARARGS,
     "Strict scalar ordered leaf/direct-sibling sphere face selection."},
    {"optimize", optimize, METH_VARARGS, "HOCR and FastPD fusion for triangle costs."},
    {"source_wls_cost", source_wls_cost, METH_VARARGS,
     "Ordered source WLS reduction: float64 distance/similarity/valid buffer, rows, width, sigma."},
    {"source_rotation_matrices", source_rotation_matrices, METH_VARARGS,
     "Source Point rotations: float64 prior [count,3], centre [3], count -> row-major float64 [count,3,3] bytes."},
    {"source_radial_selection", source_radial_selection, METH_VARARGS,
     "Ordered radial projection: float64 vertices/points, int64 faces/candidates, V,F,Q,K -> float64 [Q,4] face/projection."},
    {"source_sphere_warp", source_sphere_warp, METH_VARARGS,
     "Ordered sphere warp: float64 source/target/queries, int64 faces/patches, V,F,Q -> float64 [Q,3]."},
    {"source_triangle_nearest", source_triangle_nearest, METH_VARARGS,
     "Closest original triangle corner: float64 vertices/queries, int64 faces/patches, V,F,Q -> int64 [Q]."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_fastpd_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit__fastpd_native() { return PyModule_Create(&module); }
