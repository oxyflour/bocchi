#include <stdio.h>

#include "slice.h"
#include "cast.h"
#include "render.h"

using namespace lycoris;

auto test_slice() {
    vector<mesh_t> list {
        {
            {
                { 0., 0., 0. },
                { 1., 0., 0. },
                { 0., 1., 0. },
                { 0., 0., 1. },
            }, {
                { 0, 1, 2 },
                { 0, 1, 3 },
                { 0, 2, 3 },
                { 1, 2, 3 },
            }
        /*
        }, {
            {
                { 0., 0., 0. },
                { 2., 0., 0. },
                { 0., 2., 0. },
                { 0., 0., 2. },
            }, {
                { 1, 2, 3 },
            }
         */
        }
    };
    grid_t grid {
        { 0.5 },
        { },
        { },
    };
    slice(list, grid, { 1e-6, true });
}

auto test_cast() {
    vector<polys_t> shapes = {
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
    casted.dump("build\\test.html");

    render_range_t range { 0, 0, casted.xs.size(), casted.ys.size() };
    auto render_start = clock_now();
    dump("build\\test.png", casted, range);
    printf("PERF: render %zu x %zu in %f s\n", range.width(), range.height(), seconds_since(render_start));
}

int main() {
    test_slice();
    printf("ok\n");
    return 0;
}
