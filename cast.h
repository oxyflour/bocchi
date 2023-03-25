#pragma once

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "utils.h"

namespace lycoris {
using namespace std;

struct cast_input_t {
    int s, l;
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
        if (a.s == b.s && a.l == b.l) {
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

typedef vector<double2> loop_t;
typedef vector<loop_t> loops_t;

struct cast_options_t {
    double tol = 1e-6;
    bool verbose = false;
};

struct casted_t {
    vector<cast_output_t> jnt;
    vector<int> len;
    vector<double> xs, ys;
    auto dump(string file) {
        ofstream fn(file);
        if (ends_width(file, ".html")) {
            fn << "<html><body>" << endl;
        }
        fn << "<svg viewBox=\"" <<
            xs.front() << " " << ys.front() << " " <<
            xs.back()  << " " << ys.back()  << "\">" <<
        endl;
        auto w = max((ys.back() - ys.front()) / ys.size(), (xs.back() - xs.front()) / xs.size()) * 0.01;
        for (int j = 0; j < len.size(); j ++) {
            fn << "<path fill=\"none\" stroke=\"black\" stroke-width=\"" << w << "\" d=\"";
            for (int b = len[j], e = j < len.size() - 1 ? len[j + 1] : jnt.size(); b + 1 < e; b ++) {
                auto &t0 = jnt[b], &t1 = jnt[b + 1];
                if (t0.s == t1.s) {
                    if (j < ys.size()) {
                        fn << "M " << t0.v << " " << ys[j] << " ";
                        if (b + 1 < e) {
                            fn << "H " << t1.v << " ";
                        }
                    } else {
                        auto i = j - ys.size();
                        fn << "M " << xs[i] << " " << t0.v << " ";
                        if (b + 1 < e) {
                            fn << "V " << t1.v << " ";
                        }
                    }
                    b ++;
                }
            }
            fn << "\" />" << endl;
        }
        fn << "</svg>" << endl;
        if (ends_width(file, ".html")) {
            fn << "</body></html>" << endl;
        }
    }
    auto dump(ofstream &fn) {
    }
};

casted_t cast(vector<loops_t> &shapes,
        device_vector<double> &xs, device_vector<double> &ys, cast_options_t &&opts) {
    vector<cast_input_t> inputs;
    for (int s = 0; s < shapes.size(); s ++) {  auto &loops = shapes[s];
        for (int l = 0; l < loops.size(); l ++) { auto &loop = loops[l];
            for (int i = 0; i < loop.size(); i ++) { auto &pt = loop[i];
                inputs.push_back({ s, l, pt.x, pt.y });
            }
        }
    }

    device_vector inp(inputs);
    device_vector len(vector<int>(ys.len + xs.len));
    kernel_cast CU_DIM(512, 256) (inp.ptr, inp.len, ys.ptr, ys.len, 0, opts.tol, len.ptr, NULL);
    kernel_cast CU_DIM(512, 256) (inp.ptr, inp.len, xs.ptr, xs.len, 1, opts.tol, len.ptr + ys.len, NULL);

    auto vec = len.to_host();
    auto num = accumulate(vec.begin(), vec.end(), 0);
    exclusive_scan(vec.begin(), vec.end(), vec.begin(), 0);
    len.from_host(vec);
    if (opts.verbose) {
        printf("INFO: got %d joints for %zu segments at %zu x %zu grids\n", num, inp.len, xs.len, ys.len);
    }

    device_vector<cast_output_t> out(num);
    kernel_cast CU_DIM(512, 256) (inp.ptr, inp.len, ys.ptr, ys.len, 0, opts.tol, len.ptr, out.ptr);
    kernel_cast CU_DIM(512, 256) (inp.ptr, inp.len, xs.ptr, xs.len, 1, opts.tol, len.ptr + ys.len, out.ptr);

    auto sort_start = clock_now();
    auto sort_joints = [] __host__ __device__ (cast_output_t a, cast_output_t b) { return a.s < b.s || a.v < b.v; };
    for (int i = 0; i < vec.size(); i ++) {
        auto begin = vec[i],
            end = i < vec.size() - 1 ? vec[i + 1] : num;
        thrust::sort(thrust::device, out.ptr + begin, out.ptr + end, sort_joints);
    }
    if (opts.verbose) {
        printf("PERF: sorted %d joints in %f seconds\n", num, seconds_since(sort_start));
    }

    return casted_t { out.to_host(), vec, xs.to_host(), ys.to_host() };
}

};
