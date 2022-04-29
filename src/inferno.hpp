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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;

class GPUCompute {
    static const int dataSize = sizeof(float);
    static const int paramSize = sizeof(int);

    static const int landPts = 7771;
    static const int nPFTGroups = 3;
    static const int nPFT = 13;
    static const int nTotalPFT = 17;

    NS::AutoreleasePool* autoreleasePool;
    MTL::Device* device;
    MTL::Library* library;
    MTL::Function* fn;
    MTL::ComputePipelineState* computePipelineState;

    MTL::Buffer* outputBuffer;

    static const int nData = 17;
    static const int nParam = 22;
    std::array<MTL::Buffer*, nData> dataBuffers;
    std::array<MTL::Buffer*, nParam> paramBuffers;

    MTL::ArgumentEncoder* argumentEncoder;
    MTL::Buffer* argumentBuffer;

    MTL::CommandQueue* commandQueue;
    MTL::CommandBuffer* commandBuffer;

    MTL::ComputeCommandEncoder* computeCommandEncoder;

    MTL::Size threadgroupSize;

    // Parameters.
    int ignitionMethod;
    int flammabilityMethod;
    int drynessMethod;
    int fuelBuildUpMethod;
    int includeTemperature;
    int Nt;
    long outputCount;

    // Misc.
    bool didSetData = false;
    bool didSetParams = false;
    bool didRelease = false;

    MTL::Buffer* createBufferFromPyArray(pyArray array) {
        py::buffer_info buf = array.request();
        return device->newBuffer(static_cast<float *>(buf.ptr), buf.shape[0] * dataSize, MTL::ResourceOptions());
    }

public:
    GPUCompute() {
        autoreleasePool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();

        NS::Error* error = nullptr;

        library = loadLibrary(device);

        fn = library->newFunction( NS::String::string("multi_timestep_inferno", NS::UTF8StringEncoding) );

        argumentEncoder = fn->newArgumentEncoder(7);
        argumentBuffer = device->newBuffer(argumentEncoder->encodedLength(), MTL::ResourceOptions());
        argumentEncoder->setArgumentBuffer(argumentBuffer, 0);

        computePipelineState = device->newComputePipelineState( fn, &error );
        if ( !computePipelineState )
        {
            __builtin_printf( "%s", error->localizedDescription()->utf8String() );
            assert(false);
        }

        threadgroupSize = MTL::Size(computePipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);
    }

    void release() {
        autoreleasePool->release();
        didRelease = true;
    }

    ~GPUCompute() {
        if (!(didRelease)) {
            autoreleasePool->release();
        }
    }

    void set_data(
        int _ignitionMethod,
        int _flammabilityMethod,
        int _drynessMethod,
        int _fuelBuildUpMethod,
        int _includeTemperature,
        int _Nt,  // i.e. <data>.shape[0].
        // -------------- Start data arrays --------------
        // Input arrays.
        pyArray t1p5m_tile,
        pyArray q1p5m_tile,
        pyArray pstar,
        // XXX NOTE - This is with a single soil layer selected!
        pyArray sthu_soilt_single,
        pyArray frac,
        pyArray c_soil_dpm_gb,
        pyArray c_soil_rpm_gb,
        pyArray canht,
        pyArray ls_rain,
        pyArray con_rain,
        pyArray pop_den,
        pyArray flash_rate,
        pyArray fuel_build_up,
        pyArray fapar_diag_pft,
        pyArray grouped_dry_bal,
        pyArray litter_pool,
        pyArray dry_days
        // -------------- End data arrays --------------
    ) {
        // Set parameters.
        ignitionMethod = _ignitionMethod;
        flammabilityMethod = _flammabilityMethod;
        drynessMethod = _drynessMethod;
        fuelBuildUpMethod = _fuelBuildUpMethod;
        includeTemperature = _includeTemperature;
        Nt = _Nt;

        // Create buffers.
        std::array<pyArray, nData> dataArrays = {
            t1p5m_tile,
            q1p5m_tile,
            pstar,
            sthu_soilt_single,
            frac,
            c_soil_dpm_gb,
            c_soil_rpm_gb,
            canht,
            ls_rain,
            con_rain,
            pop_den,
            flash_rate,
            fuel_build_up,
            fapar_diag_pft,
            grouped_dry_bal,
            litter_pool,
            dry_days,
        };

        for (unsigned long i = 0; i < nData; i++) {
            dataBuffers[i] = createBufferFromPyArray(dataArrays[i]);
            argumentEncoder->setBuffer(dataBuffers[i], 0, i);
        }

        outputCount = Nt * nPFT * landPts;
        outputBuffer = device->newBuffer(outputCount * dataSize, MTL::ResourceOptions());

        didSetData = true;
    }

    void set_params(
        pyArray fapar_factor,
        pyArray fapar_centre,
        pyArray fapar_shape,
        pyArray fuel_build_up_factor,
        pyArray fuel_build_up_centre,
        pyArray fuel_build_up_shape,
        pyArray temperature_factor,
        pyArray temperature_centre,
        pyArray temperature_shape,
        pyArray dry_day_factor,
        pyArray dry_day_centre,
        pyArray dry_day_shape,
        pyArray dry_bal_factor,
        pyArray dry_bal_centre,
        pyArray dry_bal_shape,
        pyArray litter_pool_factor,
        pyArray litter_pool_centre,
        pyArray litter_pool_shape,
        pyArray fapar_weight,
        pyArray dryness_weight,
        pyArray temperature_weight,
        pyArray fuel_weight
    ) {
        // Create buffers.
        std::array<pyArray, nParam> paramArrays = {
            fapar_factor,
            fapar_centre,
            fapar_shape,
            fuel_build_up_factor,
            fuel_build_up_centre,
            fuel_build_up_shape,
            temperature_factor,
            temperature_centre,
            temperature_shape,
            dry_day_factor,
            dry_day_centre,
            dry_day_shape,
            dry_bal_factor,
            dry_bal_centre,
            dry_bal_shape,
            litter_pool_factor,
            litter_pool_centre,
            litter_pool_shape,
            fapar_weight,
            dryness_weight,
            temperature_weight,
            fuel_weight,
        };

        if (didSetParams) {
            for (unsigned long i = 0; i < nParam; i++) {
                paramBuffers[i]->release();
            }
        }

        for (unsigned long i = 0; i < nParam; i++) {
            paramBuffers[i] = createBufferFromPyArray(paramArrays[i]);
        }
        didSetParams = true;
    }

    pyArray run() {
        if (!(didSetParams && didSetData)) {
            if (!(didSetData)) __builtin_printf("Did not set data.");
            if (!(didSetParams)) __builtin_printf("Did not set params.");
            assert(false);
        }

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        computeCommandEncoder = commandBuffer->computeCommandEncoder();

        computeCommandEncoder->setComputePipelineState(computePipelineState);

        // Output.
        computeCommandEncoder->setBuffer(outputBuffer, 0, 0);

        // Parameters.
        computeCommandEncoder->setBytes(&ignitionMethod, paramSize, 1);
        computeCommandEncoder->setBytes(&flammabilityMethod, paramSize, 2);
        computeCommandEncoder->setBytes(&drynessMethod, paramSize, 3);
        computeCommandEncoder->setBytes(&fuelBuildUpMethod, paramSize, 4);
        computeCommandEncoder->setBytes(&includeTemperature, paramSize, 5);
        computeCommandEncoder->setBytes(&Nt, paramSize, 6);

        // Data arrays.
        computeCommandEncoder->setBuffer(argumentBuffer, 0, 7);
        for (unsigned long i = 0; i < nData; i++) {
            computeCommandEncoder->useResource(dataBuffers[i], MTL::ResourceUsageRead);
        }

        // Parameter arrays.
        for (unsigned long i = 0; i < nParam; i++) {
            computeCommandEncoder->setBuffer(paramBuffers[i], 0, 8 + i);
        }

        MTL::Size gridSize = MTL::Size(landPts * nPFT, 1, 1);

        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();
        commandBuffer->commit();

        commandBuffer->waitUntilCompleted();

        pool->release();

        return pyArray(
            outputCount,
            (float*)outputBuffer->contents()
        );
    }
};

#endif
