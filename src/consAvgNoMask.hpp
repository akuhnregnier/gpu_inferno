#ifndef consAvgNoMask_hpp
#define consAvgNoMask_hpp

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadConsAvgNoMaskLibrary.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;
using pyBoolArray = py::array_t<bool, py::array::c_style>;

class GPUConsAvgNoMask {
    static const int dataSize = sizeof(float);
    static const int boolSize = sizeof(bool);

    MTL::Device* device;
    MTL::Library* library;
    MTL::Function* fn;
    MTL::ComputePipelineState* computePipelineState;

    MTL::CommandQueue* commandQueue;
    MTL::CommandBuffer* commandBuffer;

    MTL::ComputeCommandEncoder* computeCommandEncoder;

    MTL::Size threadgroupSize;

    // Buffers.
    MTL::Buffer* inDataBuffer;
    MTL::Buffer* weightsBuffer;
    MTL::Buffer* outDataBuffer;

    // Parameters.
    int M, N, L;

    // Misc.
    bool didRelease = false;

public:
    GPUConsAvgNoMask(int L, pyArray weights) {
        this->L = L;

        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();

        NS::Error* error = nullptr;

        library = loadConsAvgNoMaskLibrary(device);

        fn = library->newFunction( NS::String::string("cons_avg_no_mask", NS::UTF8StringEncoding) );

        computePipelineState = device->newComputePipelineState( fn, &error );
        if ( !computePipelineState )
        {
            __builtin_printf( "%s", error->localizedDescription()->utf8String() );
            assert(false);
        }

        threadgroupSize = MTL::Size(computePipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);

        // Store weights in buffer.
        py::buffer_info weightsInfo = weights.request();

        if (weightsInfo.ndim != 2)
            throw std::runtime_error("Incompatible weights dimensions!");

        M = weightsInfo.shape[0];
        N = weightsInfo.shape[1];

        inDataBuffer = device->newBuffer(M * L * dataSize, MTL::ResourceOptions());
        weightsBuffer = device->newBuffer(M * N * dataSize, MTL::ResourceOptions());
        outDataBuffer = device->newBuffer(N * L * dataSize, MTL::ResourceOptions());

        memcpy(weightsBuffer->contents(), weightsInfo.ptr, weightsInfo.size * dataSize);
        weightsBuffer->didModifyRange(NS::Range::Make(0, weightsBuffer->length()));
    }

    void release() {
        inDataBuffer->release();
        weightsBuffer->release();
        outDataBuffer->release();
        computePipelineState->release();
        commandQueue->release();
        device->release();

        didRelease = true;
    }

    ~GPUConsAvgNoMask() {
        if (!(didRelease)) {
            release();
        }
    }

    pyArray run(pyArray inData) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        py::buffer_info inDataInfo = inData.request();
        if (inDataInfo.shape[0] != M)
            throw std::runtime_error("Expected input shape (M, L)!");
        if (inDataInfo.shape[1] != L)
            throw std::runtime_error("Expected input shape (M, L)!");

        void* inDataPtr = inDataInfo.ptr;

        // Release Python GIL after dealing with the input Python (NumPy)
        // arrays and getting the underlying data pointer.
        py::gil_scoped_release release;

        memcpy(inDataBuffer->contents(), inDataPtr, M * L * dataSize);
        inDataBuffer->didModifyRange(NS::Range::Make(0, inDataBuffer->length()));

        commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        computeCommandEncoder = commandBuffer->computeCommandEncoder();

        computeCommandEncoder->setComputePipelineState(computePipelineState);

        // Outputs.
        computeCommandEncoder->setBuffer(outDataBuffer, 0, 0);

        // Inputs.
        computeCommandEncoder->setBuffer(weightsBuffer, 0, 1);
        computeCommandEncoder->setBuffer(inDataBuffer, 0, 2);

        // Parameters.
        computeCommandEncoder->setBytes(&M, sizeof(int), 3);
        computeCommandEncoder->setBytes(&N, sizeof(int), 4);
        computeCommandEncoder->setBytes(&L, sizeof(int), 5);

        MTL::Size gridSize = MTL::Size(N * L, 1, 1);

        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();
        commandBuffer->commit();

        commandBuffer->waitUntilCompleted();

        pool->release();

        py::gil_scoped_acquire acquire;

        return pyArray(N * L, (float*)outDataBuffer->contents());
    }

};

#endif
