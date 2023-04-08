#include <stdio.h>

#include "slice.h"
#include "cast.h"
#include "render.h"

using namespace bocchi;

auto armadillo_mesh = mesh_t::load_obj("build\\armadillo.obj");
vector<mesh_t> TEST_SLICE_SHAPES {
    armadillo_mesh,
};

auto test_slice() {
    grid_t grid {
        range(-100., 100., 200. / 500.),
        range(-100., 100., 200. / 500.),
        range(-100., 100., 200. / 500.),
    };
    auto sliced = slice(TEST_SLICE_SHAPES, grid, { 1e-6, true });
    device_vector
        xs(range(grid.xs.front(), grid.xs.back(), 200. / 5000.)),
        ys(range(grid.ys.front(), grid.ys.back(), 200. / 5000.)),
        zs(range(grid.zs.front(), grid.zs.back(), 200. / 5000.));

    auto render_start = clock_now();
    cast_options_t opts { 1e-6, true };
    for (int d = 0; d < 3; d ++) {
        auto &s = d == 0 ? sliced.x : d == 1 ? sliced.y : sliced.z;
        auto &u = d == 0 ? ys : d == 1 ? zs : xs,
             &v = d == 0 ? zs : d == 1 ? xs : ys;
        auto  a = d == 0 ? "x" : d == 1 ? "y" : "z";
        render_range_t range { 0, 0, u.len, v.len };
        device_vector<render_pixel_t> pixels(range.size());
        vector<render_pixel_t> buf(range.size());
        for (int i = 0; i < s.size(); i ++) {
            auto casted = cast(s[i], u, v, move(opts));
            if (0) {
                render(casted, range, pixels.ptr);
            } else if (casted.jnt.size()) {
                dump_png("build\\slice-" + string(a) + to_string(i) + ".png", casted, range, pixels, buf);
            }
        }
        printf("PERF: render %zu x %zu in %f s\n", u.len, v.len, seconds_since(render_start));
    }
}

auto test_cast() {
    vector<shape_t> shapes = {
        {
            // shape1
            {
                { 0, 0 },
                { 0, 1 },
                { 1, 1 },
                { 1, 0 },
                { 0, 0 },
            }, {
                { .2, .2 },
                { .8, .2 },
                { .8, .8 },
                { .2, .2 },
            }
        }, {
            // shape2
            {
                { .1, .1 },
                { .1, .9 },
                { .9, .9 },
                { .1, .1 },
            }
        }
    };
    device_vector
        xs(range(-0.1, 1.1, 0.002)),
        ys(range(-0.1, 1.1, 0.002));
    auto casted = cast(shapes, xs, ys, { 1e-6, true });
    dump("build\\test.html", casted);

    auto render_start = clock_now();
    dump("build\\test.png", casted);
    printf("PERF: render %zu x %zu in %f s\n", casted.xs.size(), casted.ys.size(), seconds_since(render_start));
}

int main() {
    test_slice();
    printf("ok\n");
    return 0;
}
