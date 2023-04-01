#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <fstream>
#include <chrono>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

namespace lycoris {

__device__ __host__ inline auto operator+(int3 a, int b) {
    return int3 { a.x + b, a.y + b, a.z + b };
}
__device__ __host__ inline auto operator/(double3 a, double b) {
    return double3 { a.x / b, a.y / b, a.z / b };
}
__device__ __host__ inline auto operator*(double3 a, double b) {
    return double3 { a.x * b, a.y * b, a.z * b };
}
__device__ __host__ inline auto operator+(double3 a, double3 b) {
    return double3 { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ __host__ inline auto floor(double3 v) {
    return double3 { ::floor(v.x), ::floor(v.y), ::floor(v.z) };
}
template <typename T>
__device__ __host__ inline auto round_by(T val, double tol) {
    return floor(val / tol) * tol;
}

__device__ __host__ inline auto fmin(double3 a, double3 b) {
    return double3 { ::fmin(a.x, b.x), ::fmin(a.y, b.y), ::fmin(a.z, b.z) };
}
__device__ __host__ inline auto fmin(double3 a, double3 b, double3 c) {
    return fmin(fmin(a, b), c);
}
__device__ __host__ inline auto fmax(double3 a, double3 b) {
    return double3 { ::fmax(a.x, b.x), ::fmax(a.y, b.y), ::fmax(a.z, b.z) };
}
__device__ __host__ inline auto fmax(double3 a, double3 b, double3 c) {
    return fmax(fmax(a, b), c);
}
// https://stackoverflow.com/a/27992604
#ifdef __INTELLISENSE__
struct dim3 {
    int x;
    int y;
    int z;
};
dim3 blockIdx;
dim3 blockDim;
dim3 threadIdx;
dim3 gridDim;
#define CU_DIM(grid, block)
#define CU_DIM_MEM(grid, block, bytes)
#define CU_DIM_MEM_STREAM(grid, block, bytes, stream)
extern int atomicAdd(int *, int);
extern size_t atomicAdd(size_t *, size_t);
#else
#define CU_DIM(grid, block) <<<(grid), (block)>>>
#define CU_DIM_MEM(grid, block, bytes) <<<(grid), (block), (bytes)>>>
#define CU_DIM_MEM_STREAM(grid, block, bytes, stream) <<<(grid), (block), (bytes), (stream)>>>
#endif

#define cuIdx(D) (threadIdx.D + blockIdx.D * blockDim.D)
#define cuDim(D) (blockDim.D * gridDim.D)

#define CALL_AND_ASSERT(call, success, message) do { \
    auto code = call;       \
    if (code != success) { \
        fprintf(stderr, "call %s failed with message %s\n at %s:%d\n", #call, message(code), __FILE__, __LINE__); \
        exit(-1);           \
    }                       \
} while (0)

#define  CUDA_ASSERT(call) CALL_AND_ASSERT(call, cudaSuccess, cudaGetErrorString)
#define OPTIX_ASSERT(call) CALL_AND_ASSERT(call, OPTIX_SUCCESS, optixGetErrorString)

auto _cuda_assert(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA FATAL: %s\n at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

template <typename T>
auto _malloc_device(size_t size, int retry, const char* file, int line) {
    T *ptr = NULL;
    cudaError_t err;
    while (retry > 0) {
        cudaMalloc(&ptr, size * sizeof(T));
        err = cudaGetLastError();
        if (err == cudaErrorMemoryAllocation) {
            retry --;
        } else {
            break;
        }
    }
    _cuda_assert(err, file, line);
    return ptr;
}
template <typename T>
auto _malloc_device(size_t size, const char* file, int line) {
    return _malloc_device<T>(size, (int) 10, file, line);
}
#define malloc_device(type, ...) _malloc_device<type>(##__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
auto _to_device(T *ptr, size_t size, T *out, const char *file, int line) {
    if (out == NULL) {
        out = malloc_device(T, size, 10);
    }
    if (size) {
        _cuda_assert(cudaMemcpy(out, ptr, size * sizeof(T), cudaMemcpyDefault), file, line);
    }
    return out;
}
template <typename T>
auto _to_device(T *ptr, size_t size, const char *file, int line) {
    return _to_device(ptr, size, (T *) NULL, file, line);
}

template <typename T>
auto _from_device(T *ptr, size_t size, T *out, const char *file, int line) {
    if (out == NULL) {
        out = (T *) malloc(size * sizeof(T));
    }
    if (size) {
        _cuda_assert(cudaMemcpy(out, ptr, size * sizeof(T), cudaMemcpyDefault), file, line);
    }
    return out;
}
template <typename T>
auto _from_device(T *ptr, size_t size, const char *file, int line) {
    return _from_device(ptr, size, NULL, file, line);
}

using std::vector;
using std::string;
using std::string_view;
using std::ifstream;
using std::ofstream;
using std::endl;

template <typename T>
auto range(T from, T to, T step = 1) {
    vector<T> ret;
    for (T val = from; val < to; val += step) {
        ret.push_back(val);
    }
    return ret;
}

using std::chrono::high_resolution_clock;
using std::chrono::steady_clock;
auto clock_now() {
    return high_resolution_clock::now();
}
auto seconds_since(steady_clock::time_point &start) {
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::seconds>(clock_now() - start);
    return duration.count();
}

bool ends_width(string_view str, string_view suffix) {
    return str.size() >= suffix.size() &&
        0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

template <typename T>
struct device_vector {
    T *ptr = NULL;
    size_t len = 0;
    device_vector(vector<T> &vec) {
        resize(vec.size());
        _to_device(vec.data(), vec.size(), ptr, __FILE__, __LINE__);
    }
    device_vector(vector<T> &&vec) {
        resize(vec.size());
        _to_device(vec.data(), vec.size(), ptr, __FILE__, __LINE__);
    }
    device_vector(size_t len) : len(len) {
        ptr = _malloc_device<T>(len, __FILE__, __LINE__);
    }
    ~device_vector() {
        _cuda_assert(cudaFree(ptr), __FILE__, __LINE__);
    }
    auto resize(size_t size) {
        if (len != size) {
            if (ptr != NULL) {
                _cuda_assert(cudaFree(ptr), __FILE__, __LINE__);
            }
            ptr = malloc_device(T, size);
            len = size;
        }
    }
private:
    device_vector(device_vector &vec) {
    }
};

template <typename T>
auto &_from_device(device_vector<T> &vec, vector<T> &out, const char *file, int line) {
    out.resize(vec.len);
    _from_device(vec.ptr, vec.len, out.data(), file, line);
    return out;
}
template <typename T>
auto _from_device(device_vector<T> &vec, const char *file, int line) {
    vector<T> out;
    _from_device(vec, out, file, line);
    return out;
}
#define from_device(...) _from_device(##__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
auto &_to_device(vector<T> &vec, device_vector<T> &out, const char *file, int line) {
    out.resize(vec.size());
    _to_device(vec.data(), vec.size(), out.ptr, file, line);
    return out;
}
#define to_device(...) _to_device(##__VA_ARGS__, __FILE__, __LINE__)

struct grid_t {
    vector<double> xs, ys, zs;
    static auto load(ifstream &fn, vector<double> &arr) {
        int num;
        fn >> num;
        arr.resize(num);
        for (int i = 0; i < num; i ++) {
            double v;
            fn >> v;
            arr[i] = v;
        }
    }
    static auto load(string file) {
        grid_t grid;
        ifstream fn(file);
        load(fn, grid.xs);
        load(fn, grid.ys);
        load(fn, grid.zs);
    }
    static auto save(ofstream &fn, vector<double> &arr) {
        fn << arr.size();
        for (auto v : arr) {
            fn << v << endl;
        }
    }
    static auto save(string file, grid_t &grid) {
        ofstream fn(file);
        save(fn, grid.xs);
        save(fn, grid.ys);
        save(fn, grid.zs);
    }
};

struct mesh_t {
    vector<double3> verts;
    vector<int3> faces;
    static auto load(ifstream &fn) {
        mesh_t mesh;
        int num;
        fn >> num;
        for (int i = 0; i < num; i ++) {
            double3 vert;
            fn >> vert.x >> vert.y >> vert.z;
            mesh.verts.push_back(vert);
        }
        fn >> num;
        for (int i = 0; i < num; i ++) {
            int3 face;
            fn >> face.x >> face.y >> face.z;
            mesh.faces.push_back(face);
        }
        return mesh;
    }
    static auto load(string file) {
        ifstream fn(file);
        load(fn);
    }
    static auto save(ofstream &fn, mesh_t &mesh) {
        fn << mesh.verts.size();
        for (auto vert : mesh.verts) {
            fn << vert.x << vert.y << vert.z << endl;
        }
        fn << mesh.faces.size();
        for (auto face : mesh.faces) {
            fn << face.x << face.y << face.z << endl;
        }
    }
    static auto save(string file, mesh_t &mesh) {
        ofstream fn(file);
        save(fn, mesh);
    }
    static auto load_list(string file) {
        ifstream fn(file);
        int num;
        fn >> num;
        vector<mesh_t> list;
        for (int i = 0; i < num; i ++) {
            auto mesh = load(fn);
            list.push_back(mesh);
        }
        return list;
    }
    static auto save_list(string file, vector<mesh_t> &list) {
        ofstream fn(file);
        fn << list.size();
        for (auto &mesh : list) {
            save(fn, mesh);
        }
    }
};

};
