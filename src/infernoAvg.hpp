#ifndef infernoAvg_hpp
#define infernoAvg_hpp

#include <stdexcept>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadInfAvgLibrary.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <algorithm>

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;
using pyBoolArray = py::array_t<bool, py::array::c_style>;


class GPUInfernoAvg {
    static const int dataSize = sizeof(float);
    static const int paramSize = sizeof(int);
    static const int boolSize = sizeof(bool);

    static const int landPts = 7771;
    static const int nPFTGroups = 3;
    static const int nPFT = 13;
    static const int nTotalPFT = 17;

    MTL::Device* device;
    MTL::Library* library;
    MTL::Function* fn;
    MTL::ComputePipelineState* computePipelineState;

    MTL::Buffer* outputBuffer;

    static const int nData = 17;
    static const int nParam = 22;
    std::array<MTL::Buffer*, nData> dataBuffers;
    std::array<MTL::Buffer*, nParam> paramBuffers;
    MTL::Buffer* checksFailedBuffer;

    // Cons Avg.
    MTL::Buffer* weightsBuffer;
    int M, N, L;
    MTL::Buffer* consAvgParamBuffer;  // stores M, N, L

    MTL::ArgumentEncoder* argumentEncoder;
    MTL::ArgumentEncoder* consAvgArgumentEncoder;
    MTL::Buffer* argumentBuffer;
    MTL::Buffer* consAvgArgumentBuffer;

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

    // Misc.
    bool didSetData = false;
    bool didSetParams = false;
    bool didRelease = false;

    MTL::Buffer* createBufferFromPyArray(pyArray array) {
        py::buffer_info buf = array.request();
        return device->newBuffer(static_cast<float *>(buf.ptr), buf.shape[0] * dataSize, MTL::ResourceOptions());
    }

public:
    GPUInfernoAvg(int L, pyArray weights) {
        if (L != landPts)
            throw std::runtime_error("Not implemented!");

        this->L = L;

        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();

        NS::Error* error = nullptr;

        library = loadInfAvgLibrary(device);

        fn = library->newFunction( NS::String::string("inferno_cons_avg_ig1_flam2", NS::UTF8StringEncoding) );

        argumentEncoder = fn->newArgumentEncoder(5);
        argumentBuffer = device->newBuffer(argumentEncoder->encodedLength(), MTL::ResourceOptions());
        argumentEncoder->setArgumentBuffer(argumentBuffer, 0);

        consAvgArgumentEncoder = fn->newArgumentEncoder(29);
        consAvgArgumentBuffer = device->newBuffer(consAvgArgumentEncoder->encodedLength(), MTL::ResourceOptions());
        consAvgArgumentEncoder->setArgumentBuffer(consAvgArgumentBuffer, 0);

        computePipelineState = device->newComputePipelineState( fn, &error );
        if ( !computePipelineState )
        {
            __builtin_printf( "%s", error->localizedDescription()->utf8String() );
            assert(false);
        }

        threadgroupSize = MTL::Size(computePipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);

        // Create parameter buffers for later reuse.
        for (unsigned long i = 0; i < nParam; i++) {
            paramBuffers[i] = device->newBuffer(nPFTGroups * dataSize, MTL::ResourceOptions());
        }

        // Store weights in buffer.
        py::buffer_info weightsInfo = weights.request();

        if (weightsInfo.ndim != 2)
            throw std::runtime_error("Incompatible weights dimensions!");

        M = weightsInfo.shape[0];
        N = weightsInfo.shape[1];

        weightsBuffer = device->newBuffer(M * N * dataSize, MTL::ResourceOptions());

        memcpy(weightsBuffer->contents(), weightsInfo.ptr, weightsInfo.size * dataSize);
        weightsBuffer->didModifyRange(NS::Range::Make(0, weightsBuffer->length()));

        consAvgParamBuffer = device->newBuffer(3 * paramSize, MTL::ResourceOptions());
        int * consAvgParams = (int*)consAvgParamBuffer->contents();
        consAvgParams[0] = M;
        consAvgParams[1] = N;
        consAvgParams[2] = L;
        consAvgParamBuffer->didModifyRange(NS::Range::Make(0, consAvgParamBuffer->length()));

        consAvgArgumentEncoder->setBuffer(weightsBuffer, 0, 0);
        consAvgArgumentEncoder->setBuffer(consAvgParamBuffer, 0, 1);
    }

    void release() {
        for (unsigned long i = 0; i < nParam; i++) {
            paramBuffers[i]->release();
        }
        weightsBuffer->release();
        consAvgParamBuffer->release();

        if (didSetData) {
            for (unsigned long i = 0; i < nData; i++) {
                dataBuffers[i]->release();
            }
            outputBuffer->release();
            checksFailedBuffer->release();
        }
        argumentBuffer->release();
        argumentEncoder->release();
        computePipelineState->release();
        commandQueue->release();
        device->release();

        didRelease = true;
    }

    ~GPUInfernoAvg() {
        if (!(didRelease)) {
            release();
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
        pyArray dry_days,
        pyBoolArray checks_failed
        // -------------- End data arrays --------------
    ) {
        // Set parameters.
        ignitionMethod = _ignitionMethod;
        flammabilityMethod = _flammabilityMethod;
        drynessMethod = _drynessMethod;
        fuelBuildUpMethod = _fuelBuildUpMethod;
        includeTemperature = _includeTemperature;
        Nt = _Nt;

        if (Nt != M)
            throw std::runtime_error("Nt != M");
        if (ignitionMethod != 1)
            throw std::runtime_error("Ignition method has to be 1!");
        if (flammabilityMethod != 2)
            throw std::runtime_error("Flammability method has to be 2!");

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

        outputBuffer = device->newBuffer(N * landPts * dataSize, MTL::ResourceOptions());

        py::buffer_info checksFailedInfo = checks_failed.request();
        if (checksFailedInfo.size != Nt * nPFT * landPts)
            throw std::runtime_error("Wrong checks_failed size!");

        checksFailedBuffer = device->newBuffer(
            (bool*)checksFailedInfo.ptr,
            Nt * nPFT * landPts * sizeof(bool),
            MTL::ResourceOptions()
        );

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

        for (unsigned long i = 0; i < nParam; i++) {
            py::buffer_info paramBuf = paramArrays[i].request();

            assert(paramBuf.shape[0] == nPFTGroups);

            memcpy(paramBuffers[i]->contents(), paramBuf.ptr, nPFTGroups * dataSize);
            paramBuffers[i]->didModifyRange(NS::Range::Make(0, paramBuffers[i]->length()));
        }

        didSetParams = true;
    }

    void run(pyArray out) {
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
        computeCommandEncoder->setBytes(&drynessMethod, paramSize, 1);
        computeCommandEncoder->setBytes(&fuelBuildUpMethod, paramSize, 2);
        computeCommandEncoder->setBytes(&includeTemperature, paramSize, 3);
        computeCommandEncoder->setBytes(&Nt, paramSize, 4);

        // Data arrays.
        computeCommandEncoder->setBuffer(argumentBuffer, 0, 5);
        for (unsigned long i = 0; i < nData; i++) {
            computeCommandEncoder->useResource(dataBuffers[i], MTL::ResourceUsageRead);
        }

        // Parameter arrays.
        for (unsigned long i = 0; i < nParam; i++) {
            computeCommandEncoder->setBuffer(paramBuffers[i], 0, 6 + i);
        }

        computeCommandEncoder->setBuffer(checksFailedBuffer, 0, 28);

        computeCommandEncoder->setBuffer(consAvgArgumentBuffer, 0, 29);
        computeCommandEncoder->useResource(weightsBuffer, MTL::ResourceUsageRead);
        computeCommandEncoder->useResource(consAvgParamBuffer, MTL::ResourceUsageRead);

        MTL::Size gridSize = MTL::Size(landPts, 1, 1);

        computeCommandEncoder->dispatchThreads(gridSize, threadgroupSize);
        computeCommandEncoder->endEncoding();
        commandBuffer->commit();

        commandBuffer->waitUntilCompleted();

        pool->release();

        // Copy output.

        py::buffer_info outInfo = out.request();
        if (outInfo.ndim != 2)
            throw std::runtime_error("Incompatible out dimension!");
        if ((outInfo.shape[0] != N) || (outInfo.shape[1] != landPts))
            throw std::runtime_error("Expected out dimensions (N, landPts)!");

        memcpy(outInfo.ptr, outputBuffer->contents(), outInfo.size * dataSize);
    }
};

#endif
