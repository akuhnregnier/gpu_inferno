#ifndef inferno_hpp
#define inferno_hpp

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibrary.hpp"

class GPUCompute {
    NS::AutoreleasePool* autoreleasePool;
    MTL::Device* device;
    MTL::Library* library;
    MTL::Function* fn;
    MTL::ComputePipelineState* computePSO;

    static const int N = 5;
    static const int dataSize = sizeof(float);

    MTL::Buffer* pBuffer;
    MTL::CommandQueue* pCommandQueue;
    MTL::CommandBuffer* pCommandBuffer;

    MTL::ComputeCommandEncoder* pComputeEncoder;

    NS::UInteger threadGroupSize;

public:
    GPUCompute() {
        std::cout << "Init" << std::endl;
        autoreleasePool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();

        NS::Error* error = nullptr;

        library = loadLibrary(device);

        fn = library->newFunction( NS::String::string("test_fn", NS::UTF8StringEncoding) );

        computePSO = device->newComputePipelineState( fn, &error );
        if ( !computePSO )
        {
            __builtin_printf( "%s", error->localizedDescription()->utf8String() );
            assert(false);
        }
        threadGroupSize = computePSO->maxTotalThreadsPerThreadgroup();

        pBuffer = device->newBuffer(dataSize * N, MTL::ResourceStorageModeManaged);
    }

    ~GPUCompute() {
        std::cout << "Destructor called" << std::endl;
        autoreleasePool->release();
    }

    void run(float x = 0.0f) {
        pCommandQueue = device->newCommandQueue();
        pCommandBuffer = pCommandQueue->commandBuffer();
        assert(pCommandBuffer);
        pComputeEncoder = pCommandBuffer->computeCommandEncoder();

        pComputeEncoder->setComputePipelineState( computePSO );

        pComputeEncoder->setBuffer( pBuffer, 0, 0 );

        pComputeEncoder->setBytes( &x, sizeof(float), 1);

        MTL::Size gridSize = MTL::Size( N, 1, 1 );

        MTL::Size threadgroupSize( threadGroupSize, 1, 1 );

        pComputeEncoder->dispatchThreads( gridSize, threadgroupSize );
        pComputeEncoder->endEncoding();
        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();

        float* bufferContents = (float*)pBuffer->contents();

        std::cout << "Contents:" << "\n";
        for (int i = 0 ; i < N ; ++i) {
            std::cout << *bufferContents << "\n";
            bufferContents++;
        }
    }
};

void use_metal() {
    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    std::cout << "Starting!\n";

    MTL::Device* _pDevice = MTL::CreateSystemDefaultDevice();

    NS::Error* pError = nullptr;

    MTL::Library* pLibrary = loadLibrary(_pDevice);

    MTL::Function* testFn = pLibrary->newFunction( NS::String::string("test_fn", NS::UTF8StringEncoding) );
    MTL::ComputePipelineState* _pComputePSO = _pDevice->newComputePipelineState( testFn, &pError );
    if ( !_pComputePSO )
    {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert(false);
    }

    int N = 5;
    int dataSize = sizeof(float);

    MTL::Buffer* _pBuffer = _pDevice->newBuffer(dataSize * N, MTL::ResourceStorageModeManaged);

    MTL::CommandQueue* _pCommandQueue = _pDevice->newCommandQueue();

    MTL::CommandBuffer* pCommandBuffer = _pCommandQueue->commandBuffer();
    assert(pCommandBuffer);

    MTL::ComputeCommandEncoder* pComputeEncoder = pCommandBuffer->computeCommandEncoder();

    pComputeEncoder->setComputePipelineState( _pComputePSO );

    pComputeEncoder->setBuffer( _pBuffer, 0, 0 );

    MTL::Size gridSize = MTL::Size( N, 1, 1 );

    NS::UInteger threadGroupSize = _pComputePSO->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize( threadGroupSize, 1, 1 );

    pComputeEncoder->dispatchThreads( gridSize, threadgroupSize );
    pComputeEncoder->endEncoding();
    pCommandBuffer->commit();
    pCommandBuffer->waitUntilCompleted();

    float* bufferContents = (float*)_pBuffer->contents();

    std::cout << "Contents:" << "\n";
    for (int i = 0 ; i < N ; ++i) {
        std::cout << *bufferContents << "\n";
        bufferContents++;
    }

    pAutoreleasePool->release();
}

#endif
