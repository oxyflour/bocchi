#pragma once

#include "utils.h"

namespace lycoris {

struct Params {
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};
struct RayGenData {
    // No data needed
};
struct MissData {
    float3 bg_color;
};
struct HitGroupData {
    // No data needed
};

class Trace {
public:
    OptixDeviceContext ctx;
    Trace() {
        OPTIX_ASSERT(optixInit());
        OptixDeviceContextOptions opts;
        OPTIX_ASSERT(optixDeviceContextCreate(0, &opts, &ctx));
    }
    ~Trace() {
        OPTIX_ASSERT(optixDeviceContextDestroy(ctx));
    }
};

};
