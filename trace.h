#pragma once

#include <optix_device.h>
#include <optix_stack_size.h>

#include "utils.h"
#include "optix/launch.h"

namespace bocchi {

#ifdef USE_OPTIX

template <typename T> struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

auto compile_ptx() {
    string src;
    {
        ifstream fn("C:\\Projects\\bocchi\\optix\\launch.cu");
        stringstream code;
        fn >> code.rdbuf();
        src = code.str();
    }
    nvrtcProgram prog;
    NVRTC_ASSERT(nvrtcCreateProgram(&prog, src.c_str(), "main.cu", 0, NULL, NULL));
    const char* options[] = {
        "-IC:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.6.0\\include",
        "-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include",
        "-IC:\\Projects\\bocchi",
        "--std=c++17",
    };
    auto ret = nvrtcCompileProgram(prog, sizeof(options) / sizeof(char *), options);
    size_t log_size;
    NVRTC_ASSERT(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
        vector<char> log(log_size);
        NVRTC_ASSERT(nvrtcGetProgramLog(prog, log.data()));
        fprintf(stderr, log.data());
    }
    NVRTC_ASSERT(ret);
    size_t ptx_size;
    NVRTC_ASSERT(nvrtcGetPTXSize(prog, &ptx_size));
    vector<char> ptx(ptx_size);
    NVRTC_ASSERT(nvrtcGetPTX(prog, ptx.data()));
    NVRTC_ASSERT(nvrtcDestroyProgram(&prog));
    return ptx;
}

struct trace_t {
    OptixDeviceContext ctx;
    OptixModule mod;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt = { };

    device_vector<Params> dev_params;
    device_vector<RayGenSbtRecord>   dev_raygen;
    device_vector<MissSbtRecord>     dev_miss;
    device_vector<HitGroupSbtRecord> dev_hitgroup;
    device_vector<uchar4> dev_buffer;
    device_vector<char> dev_accel;
    Params params;
    CUstream stream;

    static void log_cb(unsigned int level, const char *tag, const char *message, void *) {
        printf("[%d][%s] %s\n", level, tag, message);
    }
    static char log_buf[2048];
    static size_t log_size;
    static void check_log(const char *tag) {
        if (log_size > 1) {
            printf("[%s] %s\n", tag, log_buf);
        }
    }

    auto create_program(OptixProgramGroupKind kind, string entry) {
        OptixProgramGroupOptions opts = { };
        OptixProgramGroupDesc desc = { };
        desc.kind = kind;
        if (kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN) {
            desc.raygen.module = mod;
            desc.raygen.entryFunctionName = entry.c_str();
        } else if (kind == OPTIX_PROGRAM_GROUP_KIND_MISS) {
            desc.miss.module = mod;
            desc.miss.entryFunctionName = entry.c_str();
        } else if (kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP) {
            auto idx = entry.find_first_of('/');
            auto &hg = desc.hitgroup;
            hg.moduleCH = mod;
            hg.entryFunctionNameCH = entry.substr(0, idx).c_str();
        }
        OptixProgramGroup group;
        log_size = sizeof(log_buf);
        OPTIX_ASSERT(optixProgramGroupCreate(ctx, &desc, 1, &opts, log_buf, &log_size, &group));
        check_log("PG");
        return group;
    }
    auto build_accel(mesh_t &mesh) {
        OptixTraversableHandle handle;
        vector<float3> verts_f32; for (auto &vert : mesh.verts) {
            verts_f32.push_back(float3 { (float) vert.x, (float) vert.y, (float) vert.z });
        }
        device_vector verts(verts_f32);
        vector<uint32_t> faces_u32; for (auto &face : mesh.faces) {
            faces_u32.push_back((uint32_t) face.x);
            faces_u32.push_back((uint32_t) face.y);
            faces_u32.push_back((uint32_t) face.z);
        };
        device_vector faces(faces_u32);

        OptixAccelBuildOptions opts = { };
        opts.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        opts.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixBuildInput input = { };
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto &arr = input.triangleArray;
        arr.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        arr.vertexStrideInBytes = sizeof(float3);
        arr.numVertices   = verts.len;
        arr.vertexBuffers = (CUdeviceptr *) &verts.ptr;
        uint32_t flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        arr.flags         = flags;
        arr.numSbtRecords = sizeof(flags) / sizeof(uint32_t);
        arr.sbtIndexOffsetBuffer        = (CUdeviceptr) faces.ptr;
        arr.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
        arr.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

        OptixAccelBufferSizes size;
        OPTIX_ASSERT(optixAccelComputeMemoryUsage(ctx, &opts, &input, 1, &size));
        device_vector<char> tmpBuf(size.tempSizeInBytes), outBuf(size.outputSizeInBytes);

        OptixAccelEmitDesc emitDesc = { };
        device_vector<uint64_t> compactedSize(1);
        emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = (CUdeviceptr) compactedSize.ptr;
        OPTIX_ASSERT(optixAccelBuild(ctx, 0, &opts, &input, 1,
            (CUdeviceptr) tmpBuf.ptr, tmpBuf.len,
            (CUdeviceptr) outBuf.ptr, outBuf.len,
            &handle, &emitDesc, 1));
        CUDA_ASSERT(cudaDeviceSynchronize());

        auto compactedSizeVec = from_device(compactedSize);
        dev_accel.resize(compactedSizeVec[0]);
        OPTIX_ASSERT(optixAccelCompact(ctx, 0, handle, (CUdeviceptr) dev_accel.ptr, dev_accel.len, &handle));
        CUDA_ASSERT(cudaDeviceSynchronize());
        return handle;
    }
    trace_t(mesh_t &mesh) : dev_params(1), dev_buffer(1), dev_accel(1),
            dev_raygen(1), dev_miss(1), dev_hitgroup(1) {
        CUDA_ASSERT(cudaFree(NULL));
        OPTIX_ASSERT(optixInit());

        CUcontext cuCtx = 0;
        OptixDeviceContextOptions opts = { };
        opts.logCallbackFunction = &log_cb;
        opts.logCallbackLevel = 4;
        OPTIX_ASSERT(optixDeviceContextCreate(cuCtx, &opts, &ctx));

        OptixModuleCompileOptions modCompileOpts = { };
        modCompileOpts.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        modCompileOpts.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        OptixPipelineCompileOptions pipCompileOpts = { };
        pipCompileOpts.usesMotionBlur                   = false;
        pipCompileOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipCompileOpts.numPayloadValues                 = 3;
        pipCompileOpts.numAttributeValues               = 3;
        pipCompileOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        pipCompileOpts.pipelineLaunchParamsVariableName = "params";
        pipCompileOpts.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        auto ptx = compile_ptx();
        log_size = sizeof(log_buf);
        OPTIX_ASSERT(optixModuleCreateFromPTX(ctx, &modCompileOpts, &pipCompileOpts, ptx.data(), ptx.size(), log_buf, &log_size, &mod));
        check_log("MODULE");

        auto raygenPg  = create_program(OPTIX_PROGRAM_GROUP_KIND_RAYGEN,   "__raygen__rg"),
            missPg     = create_program(OPTIX_PROGRAM_GROUP_KIND_MISS,     "__miss__ms"),
            hitgroupPg = create_program(OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "__closesthit__ch");

        OptixPipelineLinkOptions pipLinkOpts = { };
        pipLinkOpts.maxTraceDepth = 2;
        pipLinkOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        vector<OptixProgramGroup> group({ raygenPg, missPg, hitgroupPg });
        log_size = sizeof(log_buf);
        OPTIX_ASSERT(optixPipelineCreate(ctx, &pipCompileOpts, &pipLinkOpts,
            group.data(), group.size(), log_buf, &log_size, &pipeline));
        check_log("PIPELINE");

        OptixStackSizes stackSizes = { };
        for (auto &item : group) {
            OPTIX_ASSERT(optixUtilAccumulateStackSizes(item, &stackSizes));
        }
        uint32_t sizeFromTraversal, sizeFromState, sizeForContinuation;
        OPTIX_ASSERT(optixUtilComputeStackSizes(&stackSizes, pipLinkOpts.maxTraceDepth, 0, 0,
            &sizeFromTraversal, &sizeFromState, &sizeForContinuation));
        OPTIX_ASSERT(optixPipelineSetStackSize(pipeline, sizeFromTraversal, sizeFromState, sizeForContinuation, 1));;

        {
            vector<RayGenSbtRecord> rec(1);
            OPTIX_ASSERT(optixSbtRecordPackHeader(raygenPg, rec.data()));
            to_device(rec, dev_raygen);
        }
        {
            vector<MissSbtRecord> rec(1);
            rec[0].data = { .3, .1, .2 };
            OPTIX_ASSERT(optixSbtRecordPackHeader(missPg, rec.data()));
            to_device(rec, dev_miss);
        }
        {
            vector<HitGroupSbtRecord> rec(1);
            OPTIX_ASSERT(optixSbtRecordPackHeader(hitgroupPg, rec.data()));
            to_device(rec, dev_hitgroup);
        }
        sbt.raygenRecord        = (CUdeviceptr) dev_raygen.ptr;
        sbt.missRecordBase      = (CUdeviceptr) dev_miss.ptr;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount     = 1;
        sbt.hitgroupRecordBase  = (CUdeviceptr) dev_hitgroup.ptr;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1;

        dev_buffer.resize(params.image_width * params.image_height);
        CUDA_ASSERT(cudaStreamCreate(&stream));
        params.image = dev_buffer.ptr;
        params.image_width  = 1024;
        params.image_height = 768;
        params.handle = build_accel(mesh);
        to_device(&params, 1, dev_params.ptr);
    }
    auto render() {
        OPTIX_ASSERT(optixLaunch(pipeline, stream,
            (CUdeviceptr) dev_params.ptr, sizeof(Params), &sbt,
            params.image_width, params.image_height, 1));
        CUDA_ASSERT(cudaDeviceSynchronize());
    }
    ~trace_t() {
        OPTIX_ASSERT(optixDeviceContextDestroy(ctx));
        OPTIX_ASSERT(optixModuleDestroy(mod));
    }
};

char trace_t::log_buf[2048];
size_t trace_t::log_size;

#endif

};
