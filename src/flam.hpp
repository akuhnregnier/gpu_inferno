#ifndef flam_hpp
#define flam_hpp

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadFlam2Library.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <tuple>

class GPUFlam2 {
    static const int dataSize = sizeof(float);
    static const int intSize = sizeof(int);

    MTL::Device* device;
    MTL::Library* library;
    MTL::Function* fn;
    MTL::ComputePipelineState* computePipelineState;

    MTL::Buffer* outBuffer;
    MTL::Buffer* inFloatBuffer;
    MTL::Buffer* inIntBuffer;

    MTL::CommandQueue* commandQueue;
    MTL::CommandBuffer* commandBuffer;

    MTL::ComputeCommandEncoder* computeCommandEncoder;

    MTL::Size threadgroupSize;

    // Misc.
    bool didRelease = false;

public:
    GPUFlam2() {
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();

        NS::Error* error = nullptr;

        library = loadFlam2Library(device);

        fn = library->newFunction( NS::String::string("calc_flam_flam2_kernel", NS::UTF8StringEncoding) );

        computePipelineState = device->newComputePipelineState( fn, &error );
        if ( !computePipelineState )
        {
            __builtin_printf( "%s", error->localizedDescription()->utf8String() );
            assert(false);
        }

        threadgroupSize = MTL::Size(computePipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);

        outBuffer = device->newBuffer(1 * dataSize, MTL::ResourceOptions());
        inFloatBuffer = device->newBuffer(28 * dataSize, MTL::ResourceOptions());
        inIntBuffer = device->newBuffer(3 * intSize, MTL::ResourceOptions());
    }

    void release() {
        outBuffer->release();
        inFloatBuffer->release();
        inIntBuffer->release();

        computePipelineState->release();
        commandQueue->release();
        device->release();

        didRelease = true;
    }

    ~GPUFlam2() {
        if (!(didRelease)) {
            release();
        }
    }

    float run(
        float temp_l,
        float fuel_build_up,
        float fapar,
        float dry_days,
        int dryness_method,
        int fuel_build_up_method,
        float fapar_factor,
        float fapar_centre,
        float fapar_shape,
        float fuel_build_up_factor,
        float fuel_build_up_centre,
        float fuel_build_up_shape,
        float temperature_factor,
        float temperature_centre,
        float temperature_shape,
        float dry_day_factor,
        float dry_day_centre,
        float dry_day_shape,
        float dry_bal,
        float dry_bal_factor,
        float dry_bal_centre,
        float dry_bal_shape,
        float litter_pool,
        float litter_pool_factor,
        float litter_pool_centre,
        float litter_pool_shape,
        int include_temperature,
        float fapar_weight,
        float dryness_weight,
        float temperature_weight,
        float fuel_weight
    ) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        // Update parameters.

        float* inFloatBufferPtr = (float*)inFloatBuffer->contents();
        int* inIntBufferPtr = (int*)inIntBuffer->contents();

        inFloatBufferPtr[0] = temp_l;
        inFloatBufferPtr[1] = fuel_build_up;
        inFloatBufferPtr[2] = fapar;
        inFloatBufferPtr[3] = dry_days;
        inFloatBufferPtr[4] = fapar_factor;
        inFloatBufferPtr[5] = fapar_centre;
        inFloatBufferPtr[6] = fapar_shape;
        inFloatBufferPtr[7] = fuel_build_up_factor;
        inFloatBufferPtr[8] = fuel_build_up_centre;
        inFloatBufferPtr[9] = fuel_build_up_shape;
        inFloatBufferPtr[10] = temperature_factor;
        inFloatBufferPtr[11] = temperature_centre;
        inFloatBufferPtr[12] = temperature_shape;
        inFloatBufferPtr[13] = dry_day_factor;
        inFloatBufferPtr[14] = dry_day_centre;
        inFloatBufferPtr[15] = dry_day_shape;
        inFloatBufferPtr[16] = dry_bal;
        inFloatBufferPtr[17] = dry_bal_factor;
        inFloatBufferPtr[18] = dry_bal_centre;
        inFloatBufferPtr[19] = dry_bal_shape;
        inFloatBufferPtr[20] = litter_pool;
        inFloatBufferPtr[21] = litter_pool_factor;
        inFloatBufferPtr[22] = litter_pool_centre;
        inFloatBufferPtr[23] = litter_pool_shape;
        inFloatBufferPtr[24] = fapar_weight;
        inFloatBufferPtr[25] = dryness_weight;
        inFloatBufferPtr[26] = temperature_weight;
        inFloatBufferPtr[27] = fuel_weight;

        inIntBufferPtr[0] = dryness_method;
        inIntBufferPtr[1] = fuel_build_up_method;
        inIntBufferPtr[2] = include_temperature;

        inFloatBuffer->didModifyRange(NS::Range::Make(0, inFloatBuffer->length()));
        inIntBuffer->didModifyRange(NS::Range::Make(0, inIntBuffer->length()));

        // Run.

        commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);
        computeCommandEncoder = commandBuffer->computeCommandEncoder();
        computeCommandEncoder->setComputePipelineState(computePipelineState);

        computeCommandEncoder->setBuffer(outBuffer, 0, 0);
        computeCommandEncoder->setBuffer(inFloatBuffer, 0, 1);
        computeCommandEncoder->setBuffer(inIntBuffer, 0, 2);

        // Run for the single set of parameters given.
        MTL::Size gridSize = MTL::Size(1, 1, 1);

        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();
        commandBuffer->commit();

        commandBuffer->waitUntilCompleted();
        pool->release();
        return ((float*)outBuffer->contents())[0];
    }
};

#endif
