#pragma once
/*
Copyright (c) 2022 King's College London, MeTrICS Lab, Renato Besenczi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Octree search code used with the permission of Tim Coalson under the same licence as above.
Copyright (C) Washington University School of Medicine
Original: https://github.com/Washington-University/workbench/blob/master/src/Common/OctTree.h
Related Octree selector source:
https://github.com/Washington-University/workbench/blob/master/src/Files/SignedDistanceHelper.h
https://github.com/Washington-University/workbench/blob/master/src/Files/SignedDistanceHelper.cxx
*/
// FNIT independently written ordered octree builder for sphere interpolation.
// Algorithm provenance: newMSM 260718953547743c028a45f8c885d163441df87a,
// libraries/msm-newresampler/src/node.cpp and octree.cpp, MIT licensed.
// It builds source-ordered leaf/fallback pools without linking upstream code.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fnit_ordered_octree {
using Point = std::array<double, 3>;
using Face = std::array<std::int32_t, 3>;
struct Cell {
    Point low, high, mid;
    std::int32_t parent;
    std::array<std::int32_t, 8> children;
    std::vector<std::int32_t> faces;
    Cell(const Point& l, const Point& h, std::int32_t p): low(l), high(h), parent(p) {
        for (int axis = 0; axis < 3; ++axis) mid[axis] = (l[axis] + h[axis]) / 2.0;
        children.fill(-1);
    }
};
struct Tree {
    std::vector<Point> face_low, face_high;
    std::vector<Cell> nodes;
    Tree(const std::vector<Point>& vertices, const std::vector<Face>& triangles) {
        face_low.reserve(triangles.size()); face_high.reserve(triangles.size());
        for (const auto& ids: triangles) {
            Point low, high;
            for (int axis = 0; axis < 3; ++axis) {
                low[axis] = std::min(std::min(vertices[ids[0]][axis], vertices[ids[1]][axis]), vertices[ids[2]][axis]);
                high[axis] = std::max(std::max(vertices[ids[0]][axis], vertices[ids[1]][axis]), vertices[ids[2]][axis]);
            }
            face_low.push_back(low); face_high.push_back(high);
        }
        nodes.emplace_back(Point{-101.0, -101.0, -101.0}, Point{101.0, 101.0, 101.0}, -1);
        for (std::size_t face = 0; face < triangles.size(); ++face) add(0, static_cast<std::int32_t>(face));
    }
    bool intersects(std::int32_t node, std::int32_t face) const {
        for (int axis = 0; axis < 3; ++axis)
            if (face_high[face][axis] < nodes[node].low[axis] || face_low[face][axis] > nodes[node].high[axis]) return false;
        return true;
    }
    void add(std::int32_t node, std::int32_t face) {
        if (nodes[node].children[0] >= 0) {
            const auto children = nodes[node].children;
            for (const auto child: children) if (intersects(child, face)) add(child, face);
            return;
        }
        nodes[node].faces.push_back(face);
        const std::size_t count = nodes[node].faces.size();
        if (count < 50) return;
        std::size_t total_sizes = 0;
        bool any_non_eight = false;
        for (const auto id: nodes[node].faces) {
            int same_octant_axes = 0;
            for (int axis = 0; axis < 3; ++axis)
                same_octant_axes += (face_low[id][axis] < nodes[node].mid[axis]) == (face_high[id][axis] < nodes[node].mid[axis]);
            const std::size_t size = std::size_t(1) << (3 - same_octant_axes);
            total_sizes += size; any_non_eight = any_non_eight || size != 8;
        }
        if (!any_non_eight || total_sizes >= 3 * count) return;
        const Point low = nodes[node].low, high = nodes[node].high, mid = nodes[node].mid;
        // Finite double bounds cannot be subdivided after midpoint saturation.
        // This only guards pathological zero-size duplicate input triangles.
        for (int axis = 0; axis < 3; ++axis) if (mid[axis] == low[axis] || mid[axis] == high[axis]) return;
        if (nodes.size() > std::size_t(std::numeric_limits<std::int32_t>::max()) - 8)
            throw std::overflow_error("ordered octree node count exceeds int32");
        std::array<std::int32_t, 8> children;
        for (int x = 0; x < 2; ++x) for (int y = 0; y < 2; ++y) for (int z = 0; z < 2; ++z) {
            const std::array<int, 3> upper{x, y, z};
            Point child_low, child_high;
            for (int axis = 0; axis < 3; ++axis) {
                child_low[axis] = upper[axis] ? mid[axis] : low[axis];
                child_high[axis] = upper[axis] ? high[axis] : mid[axis];
            }
            children[x * 4 + y * 2 + z] = static_cast<std::int32_t>(nodes.size());
            nodes.emplace_back(child_low, child_high, node);
        }
        nodes[node].children = children;
        auto old_faces = std::move(nodes[node].faces);
        nodes[node].faces.clear();
        for (const auto old_face: old_faces)
            for (const auto child: children) if (intersects(child, old_face)) add(child, old_face);
    }
    void ordered_nodes(std::int32_t node, std::vector<std::int32_t>& order) const {
        order.push_back(node);
        if (nodes[node].children[0] >= 0)
            for (const auto child: nodes[node].children) ordered_nodes(child, order);
    }
};
struct FlatTree {
    std::vector<Point> low, high, mid;
    std::vector<std::array<std::int32_t, 8>> children;
    std::vector<std::int32_t> parent, leaf_index, leaf_faces, fallback_faces;
    std::vector<std::int64_t> leaf_offsets{0}, fallback_offsets{0};
    std::int32_t leaves = 0;
    explicit FlatTree(const Tree& tree) {
        std::vector<std::int32_t> order; tree.ordered_nodes(0, order);
        std::vector<std::int32_t> inverse(order.size());
        for (std::size_t i = 0; i < order.size(); ++i) inverse[order[i]] = static_cast<std::int32_t>(i);
        for (const auto id: order) {
            const auto& node = tree.nodes[id];
            low.push_back(node.low); high.push_back(node.high); mid.push_back(node.mid);
            parent.push_back(node.parent < 0 ? -1 : inverse[node.parent]);
            auto remapped_children = node.children;
            for (auto& child: remapped_children) if (child >= 0) child = inverse[child];
            children.push_back(remapped_children);
            const bool leaf = node.children[0] < 0;
            leaf_index.push_back(leaf ? leaves++ : -1);
            if (leaf) {
                leaf_faces.insert(leaf_faces.end(), node.faces.begin(), node.faces.end());
                if (node.parent >= 0) {
                    for (const auto sibling: tree.nodes[node.parent].children) {
                        const auto& pool = tree.nodes[sibling].faces;
                        fallback_faces.insert(fallback_faces.end(), pool.begin(), pool.end());
                    }
                }
            }
            leaf_offsets.push_back(static_cast<std::int64_t>(leaf_faces.size()));
            fallback_offsets.push_back(static_cast<std::int64_t>(fallback_faces.size()));
        }
    }
};
template<class T>
bool set_buffer(PyObject* result, const char* name, const std::vector<T>& values) {
    PyObject* object = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(values.data()), static_cast<Py_ssize_t>(values.size() * sizeof(T)));
    if (!object) return false;
    const int status = PyDict_SetItemString(result, name, object); Py_DECREF(object);
    return status == 0;
}
bool set_number(PyObject* result, const char* name, std::size_t value) {
    PyObject* object = PyLong_FromSize_t(value); if (!object) return false;
    const int status = PyDict_SetItemString(result, name, object); Py_DECREF(object);
    return status == 0;
}
PyObject* build(PyObject*, PyObject* args) {
    Py_buffer vertices_buffer{}, faces_buffer{};
    Py_ssize_t vertex_count, face_count;
    if (!PyArg_ParseTuple(args, "y*y*nn", &vertices_buffer, &faces_buffer, &vertex_count, &face_count)) {
        if (vertices_buffer.obj) PyBuffer_Release(&vertices_buffer);
        if (faces_buffer.obj) PyBuffer_Release(&faces_buffer);
        return nullptr;
    }
    bool dimensions = vertex_count > 0 && face_count > 0 &&
        vertex_count <= std::numeric_limits<std::int32_t>::max() && face_count <= std::numeric_limits<std::int32_t>::max() &&
        vertex_count <= PY_SSIZE_T_MAX / Py_ssize_t(sizeof(Point)) && face_count <= PY_SSIZE_T_MAX / Py_ssize_t(sizeof(Face)) &&
        vertices_buffer.len == vertex_count * Py_ssize_t(sizeof(Point)) && faces_buffer.len == face_count * Py_ssize_t(sizeof(Face));
    if (!dimensions) {
        PyBuffer_Release(&vertices_buffer); PyBuffer_Release(&faces_buffer);
        PyErr_SetString(PyExc_ValueError, "expected finite float64 vertices[V,3] and int32 faces[F,3]"); return nullptr;
    }
    try {
        std::vector<Point> vertices(static_cast<std::size_t>(vertex_count));
        std::vector<Face> faces(static_cast<std::size_t>(face_count));
        std::memcpy(vertices.data(), vertices_buffer.buf, static_cast<std::size_t>(vertices_buffer.len));
        std::memcpy(faces.data(), faces_buffer.buf, static_cast<std::size_t>(faces_buffer.len));
        PyBuffer_Release(&vertices_buffer); PyBuffer_Release(&faces_buffer);
        for (const auto& point: vertices) for (const auto value: point)
            if (!std::isfinite(value)) throw std::invalid_argument("non-finite sphere coordinate");
        for (const auto& face: faces) for (const auto id: face)
            if (id < 0 || id >= vertex_count) throw std::invalid_argument("invalid sphere face index");
        PyThreadState* state = PyEval_SaveThread();
        FlatTree* flat = nullptr;
        try { Tree tree(vertices, faces); flat = new FlatTree(tree); }
        catch (...) { PyEval_RestoreThread(state); throw; }
        PyEval_RestoreThread(state);
        PyObject* result = PyDict_New();
        bool success = result && set_buffer(result, "low", flat->low) && set_buffer(result, "high", flat->high) &&
            set_buffer(result, "mid", flat->mid) && set_buffer(result, "children", flat->children) &&
            set_buffer(result, "parent", flat->parent) && set_buffer(result, "leaf_index", flat->leaf_index) &&
            set_buffer(result, "leaf_offsets", flat->leaf_offsets) && set_buffer(result, "leaf_faces", flat->leaf_faces) &&
            set_buffer(result, "fallback_offsets", flat->fallback_offsets) && set_buffer(result, "fallback_faces", flat->fallback_faces) &&
            set_number(result, "n_nodes", flat->low.size()) && set_number(result, "n_leaves", static_cast<std::size_t>(flat->leaves));
        delete flat;
        if (!success) { Py_XDECREF(result); return nullptr; }
        return result;
    } catch (const std::exception& error) {
        if (vertices_buffer.obj) PyBuffer_Release(&vertices_buffer);
        if (faces_buffer.obj) PyBuffer_Release(&faces_buffer);
        PyErr_SetString(PyExc_ValueError, error.what()); return nullptr;
    }
}
} // namespace fnit_ordered_octree
