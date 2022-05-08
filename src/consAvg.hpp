#ifndef consAvg_hpp
#define consAvg_hpp

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadConsAvgLibrary.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;
using pyBoolArray = py::array_t<bool, py::array::c_style>;

class GPUConsAvg {
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
    MTL::Buffer* inMaskBuffer;
    MTL::Buffer* weightsBuffer;
    MTL::Buffer* outDataBuffer;
    MTL::Buffer* outMaskBuffer;

    // Parameters.
    int M, N, L;

    // Misc.
    bool didRelease = false;

public:
    GPUConsAvg(int L, pyArray weights) {
        this->L = L;

        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();

        NS::Error* error = nullptr;

        library = loadConsAvgLibrary(device);

        fn = library->newFunction( NS::String::string("cons_avg", NS::UTF8StringEncoding) );

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
        inMaskBuffer = device->newBuffer(M * L * boolSize, MTL::ResourceOptions());
        weightsBuffer = device->newBuffer(M * N * dataSize, MTL::ResourceOptions());
        outDataBuffer = device->newBuffer(N * L * dataSize, MTL::ResourceOptions());
        outMaskBuffer = device->newBuffer(N * L * boolSize, MTL::ResourceOptions());

        memcpy(weightsBuffer->contents(), weightsInfo.ptr, weightsInfo.size * dataSize);
        weightsBuffer->didModifyRange(NS::Range::Make(0, weightsBuffer->length()));
    }

    void release() {
        inDataBuffer->release();
        inMaskBuffer->release();
        weightsBuffer->release();
        outDataBuffer->release();
        outMaskBuffer->release();
        computePipelineState->release();
        commandQueue->release();
        device->release();

        didRelease = true;
    }

    ~GPUConsAvg() {
        if (!(didRelease)) {
            release();
        }
    }

    std::tuple<pyArray, pyBoolArray> run(pyArray inData, pyBoolArray inMask) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        py::buffer_info inDataInfo = inData.request();
        py::buffer_info inMaskInfo = inMask.request();
        if (inDataInfo.shape != inMaskInfo.shape)
            throw std::runtime_error("Data and mask shapes do not match!");
        if (inDataInfo.shape[0] != M)
            throw std::runtime_error("Expected input shape (M, L)!");
        if (inDataInfo.shape[1] != L)
            throw std::runtime_error("Expected input shape (M, L)!");

        void* inDataPtr = inDataInfo.ptr;
        void* inMaskPtr = inMaskInfo.ptr;

        // Release Python GIL after dealing with the input Python (NumPy)
        // arrays and getting the underlying data pointer.
        py::gil_scoped_release release;

        memcpy(inDataBuffer->contents(), inDataPtr, M * L * dataSize);
        inDataBuffer->didModifyRange(NS::Range::Make(0, inDataBuffer->length()));

        memcpy(inMaskBuffer->contents(), inMaskPtr, M * L * boolSize);
        inMaskBuffer->didModifyRange(NS::Range::Make(0, inMaskBuffer->length()));

        commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        computeCommandEncoder = commandBuffer->computeCommandEncoder();

        computeCommandEncoder->setComputePipelineState(computePipelineState);

        // Outputs.
        computeCommandEncoder->setBuffer(outDataBuffer, 0, 0);
        computeCommandEncoder->setBuffer(outMaskBuffer, 0, 1);

        // Inputs.
        computeCommandEncoder->setBuffer(weightsBuffer, 0, 2);
        computeCommandEncoder->setBuffer(inDataBuffer, 0, 3);
        computeCommandEncoder->setBuffer(inMaskBuffer, 0, 4);

        // Parameters.
        computeCommandEncoder->setBytes(&M, sizeof(int), 5);
        computeCommandEncoder->setBytes(&N, sizeof(int), 6);
        computeCommandEncoder->setBytes(&L, sizeof(int), 7);

        MTL::Size gridSize = MTL::Size(N * L, 1, 1);

        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();
        commandBuffer->commit();

        commandBuffer->waitUntilCompleted();

        pool->release();

        py::gil_scoped_acquire acquire;

        return std::make_tuple(
            pyArray(N * L, (float*)outDataBuffer->contents()),
            pyBoolArray(N * L, (bool*)outMaskBuffer->contents())
        );
    }

};

#endif
