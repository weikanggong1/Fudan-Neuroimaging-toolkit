#pragma once
/*
Copyright (c) 2023 King's College London, MeTrICS Lab, Renato Besenczi
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
*/
// Independently written scalar unfolding following pinned newMSM reg_tools.cpp
// and Point/Triangle arithmetic. No original library is linked at runtime.
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fnit_source_unfold {
using Point = std::array<double,3>;
using Face = std::array<std::int64_t,3>;
inline Point subtract(const Point& a,const Point& b) {
    return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};
}
inline Point cross(const Point& a,const Point& b) {
    return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};
}
inline double dot(const Point& a,const Point& b) {
    return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2];
}
inline double norm(const Point& a) { return std::sqrt(dot(a,a)); }
inline void normalize(Point& a) {
    const double length=norm(a);
    if(length>1e-8)for(double& value:a)value/=length;
}
inline Point normal(const std::vector<Point>& points,const Face& face) {
    auto value=cross(subtract(points[face[2]],points[face[0]]),
                     subtract(points[face[1]],points[face[0]]));
    normalize(value);return value;
}
inline bool intersects(std::int64_t vertex,const std::vector<Point>& points,
                       const std::vector<Face>& faces,
                       const std::vector<std::vector<std::int64_t>>& incident) {
    const auto first=normal(points,faces[incident[vertex][0]]);
    for(const auto face:incident[vertex])
        if(dot(first,normal(points,faces[face]))<=0.5)return true;
    return false;
}
inline Point area_gradient(const Point& a,const Point& b,const Point& c) {
    auto first=subtract(c,a),second=subtract(b,a);
    const double base=norm(second);
    if(norm(first)>1e-10)normalize(first);else first={{0,0,0}};
    if(norm(second)>1e-10)normalize(second);else second={{0,0,0}};
    auto triangle_normal=cross(first,second);
    if(norm(triangle_normal)>1e-10)normalize(triangle_normal);else triangle_normal={{0,0,0}};
    auto edge_normal=cross(second,triangle_normal);
    if(dot(first,edge_normal)<0)for(double& value:edge_normal)value*=-1;
    // Point * 0.5 * base has two separately rounded multiplications.
    for(double& value:edge_normal){value*=0.5;value*=base;}
    return edge_normal;
}
inline std::uint64_t unfold(std::vector<Point>& points,const std::vector<Face>& faces,
                            const std::vector<std::vector<std::int64_t>>& incident,
                            std::int64_t maximum_sweeps) {
    std::uint64_t updates=0;
    std::vector<std::int64_t> selected;
    std::vector<Point> gradients;
    for(std::int64_t sweep=0;sweep<maximum_sweeps;++sweep) {
        selected.clear();gradients.clear();
        for(std::size_t vertex=0;vertex<points.size();++vertex)
            if(intersects(vertex,points,faces,incident))selected.push_back(vertex);
        if(selected.empty())break;
        for(const auto vertex:selected) {
            Point gradient{{0,0,0}};
            for(const auto face:incident[vertex]) {
                const auto& ids=faces[face];
                const auto& a=points[ids[0]];const auto& b=points[ids[1]];const auto& c=points[ids[2]];
                Point value;
                // Upstream identifies coincident corners by exact coordinates.
                if(norm(subtract(points[vertex],a))==0)value=area_gradient(b,c,a);
                else if(norm(subtract(points[vertex],b))==0)value=area_gradient(c,a,b);
                else value=area_gradient(a,b,c);
                for(int axis=0;axis<3;++axis)gradient[axis]+=value[axis];
            }
            gradients.push_back(gradient);
        }
        for(std::size_t row=0;row<selected.size();++row) {
            const auto vertex=selected[row];const auto start=points[vertex];
            double step=1.0;
            do {
                Point proposed;
                for(int axis=0;axis<3;++axis)proposed[axis]=start[axis]-gradients[row][axis]*step;
                normalize(proposed);
                for(double& value:proposed)value*=100;
                points[vertex]=proposed;
                step*=0.5;
            }while(intersects(vertex,points,faces,incident) && step>1e-3);
            ++updates;
        }
    }
    return updates;
}
} // namespace fnit_source_unfold
