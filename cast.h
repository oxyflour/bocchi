#pragma once

#include <map>
#include <algorithm>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "utils.h"

namespace bocchi {
using namespace std;

struct cast_input_t {
    int s, p;
    double x, y;
};

struct cast_output_t {
    int s;
    double v;
};

__global__ void kernel_cast(
        cast_input_t *inputs, size_t inputNum,
        double *pos, size_t posNum,
        int dir, double tol,
        int *len, cast_output_t *out) {
    for (int i = cuIdx(x); i < inputNum - 1; i += cuDim(x)) {
        auto &a = inputs[i], &b = inputs[i + 1];
        if (a.s == b.s && a.p == b.p) {
            auto p0 = dir == 0 ? double2 { a.x, a.y } : double2 { a.y, a.x },
                 p1 = dir == 0 ? double2 { b.x, b.y } : double2 { b.y, b.x };
            for (int j = 0; j < posNum; j ++) {
                auto x = pos[j] + tol / 2;
                if ((x - p0.x) * (p1.x - x) > 0) {
                    auto i = atomicAdd(len + j, 1);
                    if (out) {
                        auto f = (x - p0.x) / (p1.x - p0.x);
                        out[i] = { a.s, p0.y * (1 - f) + p1.y * f };
                    }
                }
            }
        }
    }
}

struct cast_options_t {
    double tol = 1e-6;
    bool verbose = false;
};

static auto rand_rgb() {
    return "rgb(" +
        to_string(rand() % 256) + ", " +
        to_string(rand() % 256) + ", " +
        to_string(rand() % 256) + ")";
}

struct casted_t {
    vector<cast_output_t> jnt;
    vector<int> len;
    vector<double> xs, ys;
};

casted_t cast(vector<cast_input_t> &inputs,
        device_vector<double> &xs, device_vector<double> &ys, cast_options_t &&opts) {
    casted_t casted;
    from_device(xs, casted.xs);
    from_device(ys, casted.ys);
    casted.len = vector<int>(ys.len + xs.len);
    if (!inputs.size()) {
        return casted;
    }
    auto cast_start = clock_now();

    device_vector inp(inputs);
    device_vector len(casted.len);
    kernel_cast CU_DIM(512, 256) (inp.ptr, inp.len, ys.ptr, ys.len, 0, opts.tol, len.ptr, NULL);
    kernel_cast CU_DIM(512, 256) (inp.ptr, inp.len, xs.ptr, xs.len, 1, opts.tol, len.ptr + ys.len, NULL);

    auto &vec = from_device(len, casted.len);
    auto num = accumulate(vec.begin(), vec.end(), 0);
    exclusive_scan(vec.begin(), vec.end(), vec.begin(), 0);
    to_device(vec, len);

    device_vector<cast_output_t> out(num);
    kernel_cast CU_DIM(512, 256) (inp.ptr, inp.len, ys.ptr, ys.len, 0, opts.tol, len.ptr, out.ptr);
    kernel_cast CU_DIM(512, 256) (inp.ptr, inp.len, xs.ptr, xs.len, 1, opts.tol, len.ptr + ys.len, out.ptr);
    if (opts.verbose) {
        printf("PERF: got %d joints for %zu segments at %zu x %zu grids in %f seconds\n", num, inp.len, xs.len, ys.len, seconds_since(cast_start));
    }

    auto sort_start = clock_now();
    auto sort_joints = [] __host__ __device__ (cast_output_t a, cast_output_t b) { return a.s != b.s ? a.s < b.s : a.v < b.v; };
#ifdef USE_THRUST_SORTING
    for (int i = 0; i < vec.size(); i ++) {
        auto begin = vec[i], end = i < vec.size() - 1 ? vec[i + 1] : num;
        thrust::sort(thrust::device, out.ptr + begin, out.ptr + end, sort_joints);
    }
    if (opts.verbose) {
        printf("PERF: thrust::sorted %d joints in %f seconds\n", num, seconds_since(sort_start));
    }
#endif
    auto casted_ptr = from_device(out, casted.jnt).data();
#ifndef USE_THRUST_SORTING
    for (int i = 0; i < vec.size(); i ++) {
        auto begin = vec[i], end = i < vec.size() - 1 ? vec[i + 1] : num;
        std::sort(casted_ptr + begin, casted_ptr + end, sort_joints);
    }
    if (opts.verbose) {
        printf("PERF: std::sorted %d joints in %f seconds\n", num, seconds_since(sort_start));
    }
#endif

    return casted;
}

casted_t cast(std::map<int, shape_t> &shapes,
        device_vector<double> &xs, device_vector<double> &ys, cast_options_t &&opts) {
    vector<cast_input_t> inputs;
    for (auto &[s, shape] : shapes) {
        for (int p = 0; p < shape.size(); p ++) { auto &poly = shape[p];
            for (int i = 0; i < poly.size(); i ++) { auto &pt = poly[i];
                inputs.push_back({ s, p, pt.x, pt.y });
            }
        }
    }
    return cast(inputs, xs, ys, move(opts));
}

casted_t cast(vector<shape_t> &shapes,
        device_vector<double> &xs, device_vector<double> &ys, cast_options_t &&opts) {
    vector<cast_input_t> inputs;
    for (int s = 0; s < shapes.size(); s ++) {  auto &shape = shapes[s];
        for (int p = 0; p < shape.size(); p ++) { auto &poly = shape[p];
            for (int i = 0; i < poly.size(); i ++) { auto &pt = poly[i];
                inputs.push_back({ s, p, pt.x, pt.y });
            }
        }
    }
    return cast(inputs, xs, ys, move(opts));
}

auto dump_svg(string file, casted_t &casted, map<int, string> colors = { }) {
    ofstream fn(file);
    if (ends_width(file, ".html")) {
        fn << "<html><body>" << endl;
    }
    auto y0 = casted.ys.front(), x0 = casted.xs.front(), y1 = casted.ys.back(), x1 = casted.xs.back();
    auto s = 2. / max((y1 - y0) / casted.ys.size(), (x1 - x0) / casted.xs.size());
    fn << "<svg width=\"" << (x1 - x0) * s << "\" height=\"" << (y1 - y0) * s << "\">" << endl;
    for (int j = 0; j < casted.len.size(); j ++) {
        for (int b = casted.len[j], e = j < casted.len.size() - 1 ? casted.len[j + 1] : casted.jnt.size(); b + 1 < e; b ++) {
            auto &t0 = casted.jnt[b], &t1 = casted.jnt[b + 1];
            if (t0.s == t1.s) {
                auto c = colors.count(t0.s) ? colors[t0.s] : (colors[t0.s] = rand_rgb());
                fn << "<path fill=\"none\" stroke=\"" << c << "\" stroke-width=\"1\" d=\"";
                if (j < casted.ys.size()) {
                    fn << "M " << t0.v * s - x0 << " " << casted.ys[j] * s - y0 << " ";
                    if (b + 1 < e) {
                        fn << "H " << t1.v * s - x0 << " ";
                    }
                } else {
                    auto i = j - casted.ys.size();
                    fn << "M " << casted.xs[i] * s - x0 << " " << t0.v * s - y0 << " ";
                    if (b + 1 < e) {
                        fn << "V " << t1.v * s - y0 << " ";
                    }
                }
                fn << "\" />" << endl;
                b ++;
            }
        }
    }
    fn << "</svg>" << endl;
    if (ends_width(file, ".html")) {
        fn << "</body></html>" << endl;
    }
}

};
