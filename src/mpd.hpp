#ifndef mpd_hpp
#define mpd_hpp

#define PI 3.14159265358979323846

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadMPDLibrary.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <tuple>

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;


class GPUCalculateMPD {
    static const int dataSize = sizeof(float);

    // NS::AutoreleasePool* autoreleasePool;
    MTL::Device* device;
    MTL::Library* library;
    MTL::Function* fn;
    MTL::ComputePipelineState* computePipelineState;

    MTL::Buffer* obsBuffer;
    MTL::Buffer* predBuffer;
    MTL::Buffer* diffsBuffer;
    MTL::Buffer* diffsMaskBuffer;

    MTL::CommandQueue* commandQueue;
    MTL::CommandBuffer* commandBuffer;

    MTL::ComputeCommandEncoder* computeCommandEncoder;

    MTL::Size threadgroupSize;

    // Parameters.
    int N;

    // Misc.
    bool didRelease = false;

public:
    GPUCalculateMPD(int N) {
        this->N = N;

        // autoreleasePool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();

        NS::Error* error = nullptr;

        library = loadMPDLibrary(device);

        fn = library->newFunction( NS::String::string("calculate_mpd", NS::UTF8StringEncoding) );

        computePipelineState = device->newComputePipelineState( fn, &error );
        if ( !computePipelineState )
        {
            __builtin_printf( "%s", error->localizedDescription()->utf8String() );
            assert(false);
        }

        threadgroupSize = MTL::Size(computePipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);

        obsBuffer = device->newBuffer(12 * N * dataSize, MTL::ResourceOptions());
        predBuffer = device->newBuffer(12 * N * dataSize, MTL::ResourceOptions());
        diffsBuffer = device->newBuffer(N * dataSize, MTL::ResourceOptions());
        diffsMaskBuffer = device->newBuffer(N * sizeof(bool), MTL::ResourceOptions());
    }

    void release() {
        obsBuffer->release();
        predBuffer->release();
        diffsBuffer->release();
        diffsMaskBuffer->release();
        computePipelineState->release();
        commandQueue->release();
        device->release();

        // autoreleasePool->release();

        didRelease = true;
    }

    ~GPUCalculateMPD() {
        if (!(didRelease)) {
            release();
        }
    }

    std::tuple<float, unsigned int> run(pyArray obs, pyArray pred) {
        float mpdVal = 0.0f;
        unsigned int ignored = 0, count = 0, inputN;
        float* diffs;
        bool* diffsMask;
        void * obsPtr, * predPtr;

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        py::buffer_info obsInfo = obs.request();
        py::buffer_info predInfo = pred.request();
        if (obsInfo.shape != predInfo.shape)
            throw std::runtime_error("Incompatible buffer shapes!");
        if ((obsInfo.ndim != 2) || (predInfo.ndim != 2))
            throw std::runtime_error("Incompatible buffer dimensions!");
        if ((obsInfo.shape[0] != 12) || (predInfo.shape[0] != 12))
            throw std::runtime_error("Expected shapes (12, N)!");
        if ((obsInfo.shape[1] != N) || (predInfo.shape[1] != N))
            throw std::runtime_error(
                "Number of points along 2nd dimension do not match number given in class setup!"
            );
        inputN = obsInfo.size;
        obsPtr = obsInfo.ptr;
        predPtr = predInfo.ptr;

        // Release Python GIL after dealing with the input Python (NumPy)
        // arrays and getting the underlying data pointer.
        py::gil_scoped_release release;

        memcpy(obsBuffer->contents(), obsPtr, inputN * dataSize);
        memcpy(predBuffer->contents(), predPtr, inputN * dataSize);

        obsBuffer->didModifyRange(NS::Range::Make(0, obsBuffer->length()));
        predBuffer->didModifyRange(NS::Range::Make(0, predBuffer->length()));

        commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        computeCommandEncoder = commandBuffer->computeCommandEncoder();

        computeCommandEncoder->setComputePipelineState(computePipelineState);

        // Inputs.
        computeCommandEncoder->setBuffer(obsBuffer, 0, 0);
        computeCommandEncoder->setBuffer(predBuffer, 0, 1);

        // N.
        computeCommandEncoder->setBytes(&N, sizeof(int), 2);

        // Outputs.
        computeCommandEncoder->setBuffer(diffsBuffer, 0, 3);
        computeCommandEncoder->setBuffer(diffsMaskBuffer, 0, 4);

        MTL::Size gridSize = MTL::Size(N, 1, 1);

        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();
        commandBuffer->commit();

        commandBuffer->waitUntilCompleted();

        diffs = (float*)diffsBuffer->contents();
        diffsMask = (bool*)diffsMaskBuffer->contents();

        for (int i = 0; i < N; i++) {
             if (diffsMask[i]) ignored++;
             else {
                 mpdVal += diffs[i];
                 count++;
            }
        }

        pool->release();

        py::gil_scoped_acquire acquire;

        return std::make_tuple(mpdVal / (PI * count), ignored);
    }
};

#endif
