#include <stdio.h>

#include "slice.h"
#include "cast.h"
#include "render.h"

using namespace bocchi;

auto test_slice(vector<mesh_t> &shapes, double tol = 1e-6) {
    double3 p0 = {  INFINITY,  INFINITY,  INFINITY },
            p1 = { -INFINITY, -INFINITY, -INFINITY };
    for (auto &shape : shapes) {
        for (auto &vert : shape.verts) {
            p0 = fmin(p0, vert);
            p1 = fmax(p1, vert);
        }
    }
    auto d = p1 - p0;
    p0 = p0 - d * 0.1;
    p1 = p1 + d * 0.1;
    auto res = fmin(fmin(d.x, d.y), d.z) / 100.;
    printf("shape bound (%f %f %f) ~ (%f %f %f)\n", p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);

    grid_t grid {
        round_vector_by(range(p0.x, p1.x, res), tol),
        round_vector_by(range(p0.y, p1.y, res), tol),
        round_vector_by(range(p0.z, p1.z, res), tol),
    };
    auto slice_start = clock_now();
    auto sliced = slice(shapes, grid, { tol, 1e-3, true });
    printf("PERF: sliced in %f s\n", seconds_since(slice_start));
    device_vector
        xs(round_vector_by(range(p0.x, p1.x, res / 16), tol)),
        ys(round_vector_by(range(p0.y, p1.y, res / 16), tol)),
        zs(round_vector_by(range(p0.z, p1.z, res / 16), tol));

    cast_options_t opts { tol, false };
    for (int d = 0; d < 3; d ++) {
        auto render_start = clock_now();
        auto &s = d == 0 ? sliced.x : d == 1 ? sliced.y : sliced.z;
        auto &u = d == 0 ? zs : d == 1 ? xs : ys,
             &v = d == 0 ? ys : d == 1 ? zs : xs;
        auto  a = d == 0 ? "x" : d == 1 ? "y" : "z";
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
    vector<mesh_t> shapes;
    // https://github.com/alecjacobson/common-3d-test-models/raw/master/data/armadillo.obj
    auto armadillo = mesh_t::load_obj("build\\armadillo.obj");
    auto bimba = mesh_t::load_obj("build\\bimba.obj");
    for (int i = 0; i < 1; i ++) {
        //shapes.push_back(armadillo);
        shapes.push_back(bimba);
    }
    test_slice(shapes);
    printf("ok\n");
    return 0;
}
