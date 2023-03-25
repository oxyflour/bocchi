#include <stdio.h>

#include "slice.h"
#include "cast.h"
#include "render.h"

#include "deps/lodepng/lodepng.h"

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
            }
        }, {
            {
                { 0., 0., 0. },
                { 2., 0., 0. },
                { 0., 2., 0. },
                { 0., 0., 2. },
            }, {
                { 1, 2, 3 },
            }
        }
    };
    grid_t grid {
        { 0, 1, 2 },
        { 0, 1, 2 },
        { 0, 1, 2 },
    };
    slice(list, grid, { 1e-6, true });
}

auto test_cast() {
    vector<loops_t> shapes = {
        {
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
        }
    };
    device_vector
        xs(range(-0.1, 1.1, 0.0002)),
        ys(range(-0.1, 1.1, 0.0002));
    auto casted = cast(shapes, xs, ys, { 1e-6, true });
    casted.dump("build\\test.html");

    render_range_t range { 0, 0, casted.xs.size(), casted.ys.size() };
    auto width = range.i1 - range.i0, height = range.j1 - range.j0;
    device_vector<render_pixel_t> img(width * height);
    auto render_start = clock_now();
    render(casted, range, img);
    printf("PERF: render %zu x %zu in %f s\n", width, height, seconds_since(render_start));

    auto vec = img.to_host();
    vector<unsigned char> buf(width * height * 4);
    for (int i = 0; i < width; i ++) {
        for (int j = 0; j < height; j ++) {
            auto c = i + j * width;
            auto p = buf.data() + c * 4;
            p[0] = p[1] = p[2] = vec[c].s ? 255 : 0;
            p[3] = 255;
        }
    }
    lodepng::encode("build\\test.png", buf, width, height);
}

int main() {
    test_cast();
    printf("ok\n");
    return 0;
}
