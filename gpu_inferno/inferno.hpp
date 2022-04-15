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
