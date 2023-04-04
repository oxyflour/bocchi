#pragma once

#include "deps/lodepng/lodepng.h"

#include "cast.h"

namespace bocchi {

struct render_pixel_t {
    int s;
};

struct render_range_t {
    size_t i0, j0, i1, j1;
    __host__ __device__ auto width() {
        return i1 - i0;
    }
    __host__ __device__ auto height() {
        return j1 - j0;
    }
    __host__ __device__ auto size() {
        return (i1 - i0) * (j1 - j0);
    }
};

__global__ void kernel_render(
    cast_output_t *jnt, size_t nj, int *base,
    double *xs, size_t nx, double *ys, size_t ny,
    render_range_t range, render_pixel_t *out) {
    auto w = range.width();
    for (int j = cuIdx(y) + range.j0; j < range.j1; j += cuDim(y)) {
        for (int i = cuIdx(x) + range.i0; i < range.i1; i += cuDim(x)) {
            for (auto b = base[j], e = j + 1 < nx + ny ? base[j + 1] : (int) nj; b + 1 < e; b ++) {
                auto &t0 = jnt[b], &t1 = jnt[b + 1];
                if (t0.s == t1.s) {
                    if (t0.v < xs[i] && xs[i] < t1.v) {
                        out[i + j * w].s = t0.s;
                    }
                    b ++;
                }
            }
            int k = ny + i;
            for (auto b = base[k], e = k + 1 < nx + ny ? base[k + 1] : (int) nj; b + 1 < e; b ++) {
                auto &t0 = jnt[b], &t1 = jnt[b + 1];
                if (t0.s == t1.s) {
                    if (t0.v < ys[j] && ys[j] < t1.v) {
                        out[i + j * w].s = t0.s;
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

auto render(casted_t &casted, render_range_t &range, render_pixel_t *ptr) {
    device_vector jnt(casted.jnt);
    device_vector len(casted.len);
    device_vector xs(casted.xs), ys(casted.ys);
    kernel_fill CU_DIM(256, 64) (ptr, range.size());
    kernel_render CU_DIM(dim3(16, 16, 1), dim3(4, 4, 1)) (
        jnt.ptr, jnt.len, len.ptr,
        xs.ptr, xs.len, ys.ptr, ys.len,
        range, ptr);
}

auto rand_int3() {
    return int3 { rand() % 256, rand() % 256, rand() % 256 };
}

auto dump_png(string file, vector<render_pixel_t> &vec, size_t width, size_t height, map<int, int3> colors = { }) {
    vector<unsigned char> buf(width * height * 4);
    for (int i = 0; i < width; i ++) {
        for (int j = 0; j < height; j ++) {
            auto k = i + j * width;
            auto p = buf.data() + k * 4;
            auto s = vec[k].s;
            auto c = colors.count(s) ? colors[s] : (colors[s] = rand_int3());
            p[0] = c.x;
            p[1] = c.y;
            p[2] = c.z;
            p[3] = 255;
        }
    }
    lodepng::encode(file, buf, width, height);
}

auto dump_png(string file, casted_t &casted, render_range_t &range) {
    device_vector<render_pixel_t> img(range.size());
    render(casted, range, img.ptr);
    auto buf = from_device(img);
    dump_png(file, buf, range.width(), range.height());
}

auto dump(string file, casted_t &casted) {
    if (ends_width(file, ".png")) {
        render_range_t range { 0, 0, casted.xs.size(), casted.ys.size() };
        dump_png(file, casted, range);
    } else {
        dump_svg(file, casted);
    }
}

};
