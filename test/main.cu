#include <stdio.h>

#include "slice.h"
#include "cast.h"
#include "render.h"

using namespace bocchi;

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
                { 0, 1, 2 },
                { 1, 2, 3 },
            }
             */
        }
    };
    grid_t grid {
        { 0.4 },
        { },
        { },
    };
    auto ret = slice(list, grid, { 1e-6, true });
    device_vector
        xs(range(-0.1, 1.1, 0.002)),
        ys(range(-0.1, 1.1, 0.002));
    auto casted = cast(ret.x[0], xs, ys, { 1e-6, true });
    dump("build\\test.png", casted);
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
