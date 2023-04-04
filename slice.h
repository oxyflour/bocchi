#pragma once

#include <numeric>
#include <map>

#include "utils.h"

namespace bocchi {

struct slice_out_t {
    int i;
    double2 p;
};

struct slice_conn_t {
    int i, j;
    double2 p;
};

struct slice_poly_t {
    poly_t p;
    int s;
};

struct slice_shape_t {
    vector<vector<shape_t>> x, y, z;
};

struct poly_builder_t {
    std::map<int, slice_conn_t> conns;
    inline auto add(int i, int j, double2 p) {
        if (!conns.count(i)) {
            conns[i] = { j, -1, p };
        } else {
            conns[i].j = j;
        }
    }
    inline auto add(slice_out_t &u, slice_out_t &v) {
        add(u.i, v.i, u.p);
        add(v.i, u.i, v.p);
    }
    auto get() {
        vector<slice_poly_t> polys;
        while (conns.size()) {
            poly_t poly;
            auto begin = conns.begin();
            auto idx = begin->first;
            auto conn = begin->second;
            while (conns.count(idx)) {
                auto conn = conns[idx];
                conns.erase(idx);
                poly.push_back(conn.p);
                idx = conns.count(conn.i) ? conn.i : conn.j;
            }
            if (idx < 0) {
                idx = conns.count(conn.i) ? conn.i : conn.j;
            } else {
                poly.push_back(poly.front());
            }
            while (conns.count(idx)) {
                auto conn = conns[idx];
                conns.erase(idx);
                poly.insert(poly.begin(), conn.p);
                idx = conns.count(conn.i) ? conn.i : conn.j;
            }
            polys.push_back({ poly, max(conn.i, conn.j) });
        }
        return polys;
    }
};

__inline__ __device__ auto add_joint(double x, int i, double3 p0, double3 p1) {
    auto f = (x - p0.x) / (p1.x - p0.x);
    auto p = p0 * (1 - f) + p1 * f;
    return slice_out_t { i, { p.y, p.z } };
}
__inline__ __device__ auto reorder_xyz(double3 p, int dir) {
    return dir == 0 ? p :
           dir == 1 ? double3 { p.y, p.z, p.x } :
                      double3 { p.z, p.x, p.y };
}
__inline__ __device__ auto index_of(int a, int b, size_t n) {
    return min(a, b) + max(a, b) * (int) n;
}
__global__ void kernel_slice(
        double3 *verts, size_t vertNum,
        int3 *faces, size_t faceNum,
        double *pos, size_t posNum,
        int dir, double tol,
        int *num, slice_out_t *out) {
    for (int i = cuIdx(x); i < faceNum; i += cuDim(x)) {
        auto &f = faces[i];
        auto &a = verts[f.x], &b = verts[f.y], &c = verts[f.z];
        auto p0 = fmin(a, b, c), p1 = fmax(a, b, c);
        auto m = dir == 0 ? double2 { p0.x, p1.x } :
                 dir == 1 ? double2 { p0.y, p1.y } :
                            double2 { p0.z, p1.z };
        for (int j = 0; j < posNum; j ++) {
            auto v = pos[j] + tol / 2.;
            if (m.x < v && v < m.y) {
                auto n = atomicAdd(num + j, 2);
                if (out) {
                    auto ix = index_of(f.y, f.z, vertNum),
                         iy = index_of(f.z, f.x, vertNum),
                         iz = index_of(f.x, f.y, vertNum);
                    auto px = reorder_xyz(a, dir),
                         py = reorder_xyz(b, dir),
                         pz = reorder_xyz(c, dir);
                    if ((px.x - v) * (v - py.x) <= 0) {
                        out[n ++] = add_joint(v, ix, py, pz);
                        out[n ++] = add_joint(v, iy, pz, px);
                    } else if ((py.x - v) * (v - pz.x) <= 0) {
                        out[n ++] = add_joint(v, iy, pz, px);
                        out[n ++] = add_joint(v, iz, px, py);
                    } else {
                        out[n ++] = add_joint(v, iz, px, py);
                        out[n ++] = add_joint(v, ix, py, pz);
                    }
                }
            }
        }
    }
}

struct slice_options_t {
    double tol = 1e-6;
    bool verbose = false;
};

auto slice(vector<mesh_t> &list, grid_t &grid, slice_options_t &&opts) {
    mesh_t merged;
    vector<int> faceIdToMeshId;
    for (int s = 0; s < list.size(); s ++) {
        auto &mesh = list[s];
        auto start = merged.verts.size();
        for (auto vert : mesh.verts) {
            merged.verts.push_back(round_by(vert, opts.tol));
        }
        for (auto face : mesh.faces) {
            merged.faces.push_back(face + start);
            faceIdToMeshId.push_back(s);
        }
    }
    device_vector verts(merged.verts);
    device_vector faces(merged.faces);

    slice_shape_t ret;
    for (int dir = 0; dir < 3; dir ++) {
        device_vector pos(dir == 0 ? grid.xs : dir == 1 ? grid.ys : grid.zs);
        device_vector len(vector<int>(pos.len));
        kernel_slice CU_DIM(256, 64) (
            verts.ptr, verts.len, faces.ptr, faces.len, pos.ptr, pos.len,
            dir, opts.tol, len.ptr, NULL);
        CUDA_ASSERT(cudaGetLastError());

        auto vec = from_device(len);
        auto num = accumulate(vec.begin(), vec.end(), 0);
        exclusive_scan(vec.begin(), vec.end(), vec.begin(), 0);
        to_device(vec, len);
        if (opts.verbose) {
            printf("INFO: got %d joints for %zu verts and %zu faces at dir %s\n",
                num, verts.len, faces.len, dir == 0 ? "x" : dir == 1 ? "y" : "z");
        }

        device_vector<slice_out_t> casted(num);
        kernel_slice CU_DIM(256, 64) (
            verts.ptr, verts.len, faces.ptr, faces.len, pos.ptr, pos.len,
            dir, opts.tol, len.ptr, casted.ptr);
        CUDA_ASSERT(cudaGetLastError());

        auto &ref = dir == 0 ? ret.x : dir == 1 ? ret.y : ret.z;
        auto out = from_device(casted);
        for (int i = 0; i < vec.size(); i ++) {
            ref.push_back({ });
            auto &shapes = ref.back();
            shapes.resize(list.size());
            poly_builder_t builder;
            for (int b = vec[i], e = i < vec.size() - 1 ? vec[i + 1] : num; b < e; b += 2) {
                builder.add(out[b], out[b + 1]);
            }
            for (auto &item : builder.get()) {
                auto s = faceIdToMeshId[item.s % verts.len];
                shapes[s].push_back(item.p);
            }
        }
    }
    return ret;
}

};
