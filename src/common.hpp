#ifndef common_hpp
#define common_hpp

#include <stdexcept>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibrary.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <algorithm>
#include <string>
#include <vector>
#include <functional>


constexpr int dataSize = sizeof(float);
constexpr int paramSize = sizeof(int);
constexpr int boolSize = sizeof(bool);

const float ARCSINH_FACTOR = 1e6;

const int landPts = 7771;
const int nPFTGroups = 3;
const int nPFT = 13;
const int nTotalPFT = 17;


MTL::ComputePipelineState* getComputePipelineState(MTL::Device* device, MTL::Function* fn) {
    NS::Error* error = nullptr;

    MTL::ComputePipelineState* computePipelineState = device->newComputePipelineState( fn, &error );
    if ( !computePipelineState )
    {
        __builtin_printf( "%s", error->localizedDescription()->utf8String() );
        assert(false);
    }

    return computePipelineState;
}


struct RunParams {
    NS::AutoreleasePool* pool;
    MTL::CommandBuffer* commandBuffer;
    MTL::ComputeCommandEncoder* computeCommandEncoder;
};


template<class T>
void releaseObj (T* x) {
    x->release();
}


class GPUBase {

protected:
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::Function* fn;
    MTL::ComputePipelineState* computePipelineState;
    MTL::Size threadgroupSize;

    std::vector<std::function<void()>> toRelease;
    bool didRelease = false;

    template <class T>
    void releaseLater(T* x) {
        toRelease.push_back(std::bind(releaseObj<T>, x));
    }

public:
    GPUBase(
        MTL::Library* (*loadLibraryFunc)(MTL::Device*),
        const std::string& kernelName
    ) : GPUBase(
        MTL::CreateSystemDefaultDevice(),
        loadLibraryFunc,
        kernelName
    ) { }

    void release() {
        if (didRelease) return;

        for (auto& releaseFunc: toRelease)
            releaseFunc();

        didRelease = true;
    }

    ~GPUBase() {
        if (!(didRelease))
            release();
    }

    RunParams getRunParams() {
        checkNotReleased();

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        MTL::ComputeCommandEncoder* computeCommandEncoder = commandBuffer->computeCommandEncoder();

        computeCommandEncoder->setComputePipelineState(computePipelineState);

        return RunParams { pool, commandBuffer, computeCommandEncoder };
    }

    void submit(unsigned int nGrid, RunParams runParams) {
        checkNotReleased();

        MTL::Size gridSize = MTL::Size(nGrid, 1, 1);

        runParams.computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        runParams.computeCommandEncoder->endEncoding();
        runParams.commandBuffer->commit();

        runParams.commandBuffer->waitUntilCompleted();

        runParams.pool->release();
    }

private:
    // Chain of constructors to create temporary variables needed for member
    // initialisation.
    GPUBase(
        MTL::Device* device,
        MTL::Library* (*loadLibraryFunc)(MTL::Device*),
        const std::string& kernelName
    ) : GPUBase(
        device,
        device->newCommandQueue(),
        loadLibraryFunc(device)->newFunction( NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding) )
    ) { }

    GPUBase(
        MTL::Device* device,
        MTL::CommandQueue* commandQueue,
        MTL::Function* fn
    ) : GPUBase(
        device,
        commandQueue,
        fn,
        getComputePipelineState(device, fn)
    ) { }

    GPUBase(
        MTL::Device* device,
        MTL::CommandQueue* commandQueue,
        MTL::Function* fn,
        MTL::ComputePipelineState* computePipelineState
    ) :
        device(device),
        commandQueue(commandQueue),
        fn(fn),
        computePipelineState(computePipelineState),
        threadgroupSize(MTL::Size(computePipelineState->maxTotalThreadsPerThreadgroup(), 1, 1))
    {
        releaseLater(computePipelineState);
        releaseLater(commandQueue);
        releaseLater(device);
    }

    void checkNotReleased() {
        if (didRelease)
            throw std::runtime_error("Already released resources!");
    }
};

#endif
