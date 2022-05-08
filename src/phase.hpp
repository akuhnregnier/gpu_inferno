#ifndef phase_hpp
#define phase_hpp

#define PI 3.14159265358979323846

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadPhaseLibrary.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;


class GPUCalculatePhase {
    static const int dataSize = sizeof(float);

    // NS::AutoreleasePool* autoreleasePool;
    MTL::Device* device;
    MTL::Library* library;
    MTL::Function* fn;
    MTL::ComputePipelineState* computePipelineState;

    MTL::Buffer* inputBuffer;
    MTL::Buffer* outputBuffer;

    MTL::CommandQueue* commandQueue;
    MTL::CommandBuffer* commandBuffer;

    MTL::ComputeCommandEncoder* computeCommandEncoder;

    MTL::Size threadgroupSize;

    // Parameters.
    int N;

    // Misc.
    bool didRelease = false;

public:
    GPUCalculatePhase(int N) {
        this->N = N;

        // autoreleasePool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();

        NS::Error* error = nullptr;

        library = loadPhaseLibrary(device);

        fn = library->newFunction( NS::String::string("calculate_phase", NS::UTF8StringEncoding) );

        computePipelineState = device->newComputePipelineState( fn, &error );
        if ( !computePipelineState )
        {
            __builtin_printf( "%s", error->localizedDescription()->utf8String() );
            assert(false);
        }

        threadgroupSize = MTL::Size(computePipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);

        inputBuffer = device->newBuffer(12 * N * dataSize, MTL::ResourceOptions());
        outputBuffer = device->newBuffer(N * dataSize, MTL::ResourceOptions());
    }

    void release() {
        inputBuffer->release();
        outputBuffer->release();
        computePipelineState->release();
        commandQueue->release();
        device->release();

        // NOTE - calling this here causes issues in certain scenarios. Is it not needed at all?
        // autoreleasePool->release();

        didRelease = true;
    }

    ~GPUCalculatePhase() {
        if (!(didRelease)) {
            release();
        }
    }

    pyArray run(pyArray x) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        py::buffer_info xBuf = x.request();
        if (xBuf.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");
        if (xBuf.shape[0] != 12)
            throw std::runtime_error("Expected shape (12, N)!");
        if (xBuf.shape[1] != N)
            throw std::runtime_error(
                "Number of points along 2nd dimension does not match number given in class setup!"
            );

        memcpy(inputBuffer->contents(), xBuf.ptr, xBuf.size * dataSize);
        inputBuffer->didModifyRange(NS::Range::Make(0, inputBuffer->length()));

        commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        computeCommandEncoder = commandBuffer->computeCommandEncoder();

        computeCommandEncoder->setComputePipelineState(computePipelineState);

        // Output.
        computeCommandEncoder->setBuffer(outputBuffer, 0, 0);

        // Input.
        computeCommandEncoder->setBuffer(inputBuffer, 0, 1);

        // N.
        computeCommandEncoder->setBytes(&N, sizeof(int), 2);

        MTL::Size gridSize = MTL::Size(N, 1, 1);

        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();
        commandBuffer->commit();

        commandBuffer->waitUntilCompleted();

        pool->release();

        return pyArray(N, (float*)outputBuffer->contents());
    }
};


pyArray calculate_phase(pyArray x) {
    py::buffer_info xBuf = x.request();
    if (xBuf.shape[0] != 12)
        throw std::runtime_error("Expected shape (12, N)!");
    if (xBuf.ndim != 2)
        throw std::runtime_error("Incompatible buffer dimension!");

    unsigned int flat_index;
    float* xPtr = static_cast<float*>(xBuf.ptr);
    // About the same speed.
    // auto xr = x.unchecked<2>();

    unsigned int outputCount = xBuf.shape[1];

    float cosThetas[12];
    float sinThetas[12];
    float theta;
    float step = 1.0f / 12.0f;
    for (unsigned int i = 0; i < 12; i++) {
        theta = 2 * PI * i * step;
        sinThetas[i] = sin(theta);
        cosThetas[i] = cos(theta);
    }

    float phaseArr[outputCount];
    float lx, ly;
    for (unsigned int i = 0; i < outputCount; i++) {
        lx = 0.0f;
        ly = 0.0f;
        for (unsigned int j = 0; j < 12; j++) {
            flat_index = (j * outputCount) + i;
            lx += xPtr[flat_index] * cosThetas[j];
            ly += xPtr[flat_index] * sinThetas[j];
            // lx += xr(j, i) * cosThetas[j];
            // ly += xr(j, i) * sinThetas[j];
        }
        phaseArr[i] = atan2(lx, ly);
    }

    return pyArray(outputCount, phaseArr);
}

#endif
