#pragma once

#include <numeric>

#include "utils.h"

namespace lycoris {

struct slice_out_t {
    int i, j;
    double u, v;
};

__inline__ __device__ auto check_joint(double x,
        int i0, double3 p0, int i1, double3 p1,
        int* num, slice_out_t *out) {
    if ((p0.x - x) * (x - p1.x) > 0) {
        auto i = atomicAdd(num, 1);
        if (out) {
            auto f = (x - p0.x) / (p1.x - p0.x);
            auto p = p0 * (1 - f) + p1 * f;
            out[i] = { i0, i1, p.y, p.z };
        }
    }
}
__inline__ __device__ auto reorder_xyz(double3 p, int dir) {
    return dir == 0 ? p :
           dir == 1 ? double3 { p.y, p.z, p.x } :
                      double3 { p.z, p.x, p.y };
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
                if (out) {
                    auto px = reorder_xyz(a, dir),
                         py = reorder_xyz(b, dir),
                         pz = reorder_xyz(c, dir);
                    //check_joint(v, f.x, px, f.y, py, num, out);
                    //check_joint(v, f.y, py, f.z, pz, num, out);
                    //check_joint(v, f.z, pz, f.x, px, num, out);
                } else {
                    atomicAdd(num + j, 2);
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
    for (auto &mesh : list) {
        auto start = merged.verts.size();
        for (auto vert : mesh.verts) {
            merged.verts.push_back(round_by(vert, opts.tol));
        }
        for (auto face : mesh.faces) {
            merged.faces.push_back(face + start);
        }
    }
    device_vector verts(merged.verts);
    device_vector faces(merged.faces);

    for (int dir = 0; dir < 3; dir ++) {
        device_vector pos(dir == 0 ? grid.xs : dir == 1 ? grid.ys : grid.zs);
        device_vector len(vector<int>(pos.len));
        kernel_slice CU_DIM(256, 64) (
            verts.ptr, verts.len, faces.ptr, faces.len, pos.ptr, pos.len,
            dir, opts.tol, len.ptr, NULL);

        auto vec = from_device(len);
        auto num = accumulate(vec.begin(), vec.end(), 0);
        exclusive_scan(vec.begin(), vec.end(), vec.begin(), 0);
        to_device(vec, len);
        if (opts.verbose) {
            printf("got %d joints for %zu verts and %zu faces at dir %s\n",
                num, verts.len, faces.len, dir == 0 ? "x" : dir == 1 ? "y" : "z");
        }

        device_vector<slice_out_t> casted(num);
        kernel_slice CU_DIM(256, 64) (
            verts.ptr, verts.len, faces.ptr, faces.len, pos.ptr, pos.len,
            dir, opts.tol, len.ptr, casted.ptr);
    }
}

};
