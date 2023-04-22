#include <stdio.h>

#include "slice.h"
#include "cast.h"
#include "render.h"
#include "trace.h"

using namespace bocchi;

struct bound_t {
    double3 min = {  INFINITY,  INFINITY,  INFINITY },
            max = { -INFINITY, -INFINITY, -INFINITY };
    auto extend(double3 vert) {
        min = fmin(min, vert);
        max = fmax(max, vert);
    }
};

auto get_bound(vector<mesh_t> &shapes, double padding = 0.1) {
    bound_t b;
    for (auto &shape : shapes) {
        for (auto &vert : shape.verts) {
            b.extend(vert);
        }
    }
    auto d = b.max - b.min;
    b.min = b.min - d * 0.1;
    b.max = b.max + d * 0.1;
    printf("shape bound (%f %f %f) ~ (%f %f %f)\n", b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z);
    return b;
}

auto get_grid(vector<mesh_t> &shapes) {
    auto b = get_bound(shapes);
    auto p0 = b.min, p1 = b.max, d = p1 - p0;
    auto res = fmin(fmin(d.x, d.y), d.z) / 100.;
    return grid_t {
        range(p0.x, p1.x, res),
        range(p0.y, p1.y, res),
        range(p0.z, p1.z, res),
    };
}

auto refine_array(vector<double> &arr, int refine, double ext, double tol) {
    vector<double> out;
    for (int i = 0; i < arr.size() - 1; i ++) {
        double b = arr[i] + ext,
            s = (arr[i + 1] - arr[i] - ext * 2) / (refine - 1);
        for (int j = 0; j < refine; j ++, b += s) {
            out.push_back(round_by(b, tol));
        }
    }
    return out;
}

auto test_slice(vector<mesh_t> &shapes, grid_t &g, double ext = 1e-4, double tol = 1e-6) {
    grid_t grid {
        round_vector_by(g.xs, tol),
        round_vector_by(g.ys, tol),
        round_vector_by(g.zs, tol),
    };
    auto slice_start = clock_now();
    auto sliced = slice(shapes, grid, { tol, 1e-3, true });
    printf("PERF: sliced in %f s\n", seconds_since(slice_start));

    cast_options_t opts { tol, false };
    for (int dir = 0; dir < 3; dir ++) {
        auto render_start = clock_now();
        auto &s = dir == 0 ? sliced.x : dir == 1 ? sliced.y : sliced.z;
        auto  a = dir == 0 ? "x" : dir == 1 ? "y" : "z";
        device_vector
            u(refine_array(dir == 0 ? g.zs : dir == 1 ? g.xs : g.ys, 16, ext, tol)),
            v(refine_array(dir == 0 ? g.ys : dir == 1 ? g.zs : g.xs, 16, ext, tol));
        render_range_t range { 0, 0, u.len, v.len };
        device_vector<render_pixel_t> pixels(range.size());
        vector<render_pixel_t> buf(range.size());
        map<int, int3> colors;
        for (int i = 0; i < s.size(); i ++) {
            auto casted = cast(s[i], u, v, move(opts));
            // DEBUG
            if (casted.jnt.size()) {
                dump_png("build\\slice-" + string(a) + to_string(i) + ".png", casted, range, pixels, buf, colors);
            } else {
                render(casted, range, pixels.ptr);
            }
        }
        printf("PERF: render dir %s (%zu x %zu) in %f s\n", a, u.len, v.len, seconds_since(render_start));
    }
}

int main() {
    /*
    vector<mesh_t> shapes;
    // https://github.com/alecjacobson/common-3d-test-models/raw/master/data/armadillo.obj
    auto armadillo = mesh_t::load_obj("build\\armadillo.obj");
    for (int i = 0; i < 1; i ++) {
        //shapes.push_back(armadillo);
        shapes.push_back(bimba);
    }
    auto grid = get_grid(shapes);
    test_slice(shapes, grid);
     */
    auto bimba = mesh_t::load_obj("build\\bimba.obj");
    trace_t tracer(bimba);
    tracer.render("build/trace.png");
    printf("ok\n");
    return 0;
}
