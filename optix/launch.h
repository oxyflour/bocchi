#include <optix.h>
#include <vector_types.h>
#include <vector_functions.h>

#ifdef __INTELLISENSE__
#include <internal/optix_7_device_impl.h>
unsigned int __float_as_uint(float in);
float __uint_as_float(unsigned int in);
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
inline __host__ __device__ float3 &operator*=(float3 &b, float a) {
    b.x *= a;
    b.y *= a;
    b.z *= a;
    return b;
}
inline __host__ __device__ float3 operator/(float3 a, float b) {
    return float3 { a.x / b, a.y / b, a.z / b };
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
inline __host__ __device__ float length(float3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
inline __host__ __device__ float3 cross(const float3 a, const float3 b) {
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline __host__ __device__ float3 normalize(float3 a) {
    float len = length(a);
    return len ? a / len : a * 0.f;
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

struct Params {
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

#define M_PIf       3.14159265358979323846f
struct Camera {
    float3 eye, lookat, up;
    float fov, aspect;
    __host__ __device__ void init(int width, int height) {
        eye = { 0, 0, 1 };
        lookat = { 0, 0, 0 };
        up = { 0, 1, 0 };
        fov = 15;
        aspect = 1. * width / height;
    }
    __host__ __device__ void setup(Params &params) {
        init(params.image_width, params.image_height);

        float3 u, v, w;
        w = lookat - eye; // Do not normalize W -- it implies focal length
        float wlen = length(w);
        u = normalize(cross(w, up));
        v = normalize(cross(u, w));

        float vlen = wlen * tanf(0.5f * fov * M_PIf / 180.0f);
        v *= vlen;
        float ulen = vlen * aspect;
        u *= ulen;

        params.cam_eye = eye;
        params.cam_u = u;
        params.cam_v = v;
        params.cam_w = w;
    }
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

