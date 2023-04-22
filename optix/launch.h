#include <optix.h>
#include <vector_types.h>

#ifdef __INTELLISENSE__
#include <internal/optix_7_device_impl.h>
unsigned int __float_as_uint(float in);
float __uint_as_float(unsigned int in);
float3 make_float3(float x, float y, float z);
float2 make_float2(float x, float y);
#endif

inline __host__ __device__ float2 operator*(float a, float2 b) {
    return float2 { a * b.x, a * b.y };
}
inline __host__ __device__ float2 operator-(float2 a, float b) {
    return float2 { a.x - b, a.y - b };
}
inline __host__ __device__ float3 operator*(float a, float3 b) {
    return float3 { a * b.x, a * b.y, a * b.z };
}
inline __host__ __device__ float3 operator*(float3 b, float a) {
    return float3 { a * b.x, a * b.y, a * b.z };
}
inline __host__ __device__ float3 operator/(float3 b, float a) {
    return float3 { a / b.x, a / b.y, a / b.z };
}
inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return float3 { a.x + b.x, a.y + b.y, a.z + b.z };
}
inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return float3 { a.x - b.x, a.y - b.y, a.z - b.z };
}
inline __host__ __device__ float3 operator+(float3 a, float b) {
    return float3 { a.x + b, a.y + b, a.z + b };
}
inline __host__ __device__ float3 normalize(float3 a) {
    float len = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    return len ? a * 0.f :  a / len;
}
inline __host__ __device__ float3 make_float3(float4 a) {
    return float3 { a.x, a.y, a.z };
}
inline __host__ __device__ uchar4 make_color(float3 rgb) {
    return uchar4 {
        (unsigned char) (rgb.x * 255),
        (unsigned char) (rgb.y * 255),
        (unsigned char) (rgb.z * 255),
        255
    };
}

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};

