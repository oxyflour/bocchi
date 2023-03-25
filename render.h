#pragma once

#include "cast.h"

namespace lycoris {

struct render_pixel_t {
    int s;
};

struct render_range_t {
    size_t i0, j0, i1, j1;
};

__global__ void kernel_render(
    cast_output_t *jnt, size_t nj, int *base,
    double *xs, size_t nx, double *ys, size_t ny,
    render_range_t range, render_pixel_t *out) {
    for (int j = cuIdx(y) + range.j0; j < range.j1; j += cuDim(y)) {
        for (int i = cuIdx(x) + range.i0; i < range.i1; i += cuDim(x)) {
            for (auto b = base[j], e = j < nx + ny - 1 ? base[j + 1] : (int) nj; b + 1 < e; b ++) {
                auto &t0 = jnt[b], &t1 = jnt[b + 1];
                if (t0.s == t1.s) {
                    if (t0.v < xs[i] && xs[i] < t1.v) {
                        auto &p = out[i + j * (range.i1 - range.i0)];
                        p.s = t0.s;
                    }
                    b ++;
                }
            }
        }
    }
}

__global__ void kernel_fill(render_pixel_t *out, int len) {
    for (int i = cuIdx(x); i < len; i += cuDim(x)) {
        out[i].s = -1;
    }
}

auto render(casted_t &casted, render_range_t &range, device_vector<render_pixel_t> &out) {
    device_vector jnt(casted.jnt);
    device_vector len(casted.len);
    device_vector xs(casted.xs), ys(casted.ys);
    kernel_fill CU_DIM(256, 64) (out.ptr, out.len);
    kernel_render CU_DIM(dim3(16, 16, 1), dim3(4, 4, 1)) (
        jnt.ptr, jnt.len, len.ptr,
        xs.ptr, xs.len, ys.ptr, ys.len,
        range, out.ptr);
}

};
