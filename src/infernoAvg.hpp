#ifndef infernoAvg_hpp
#define infernoAvg_hpp

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <simd/simd.h>
#include <stdexcept>

#include "common.hpp"
#include "inferno.hpp"
#include "loadInfAvgLibrary.hpp"

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;
using pyBoolArray = py::array_t<bool, py::array::c_style>;


class GPUInfernoAvg final : public GPUInfernoBase {

    // Cons Avg.
    int M, N, L;
    MTL::Buffer* weightsBuffer;
    MTL::Buffer* consAvgParamBuffer;  // stores M, N, L
    MTL::ArgumentEncoder* consAvgArgumentEncoder;
    MTL::Buffer* consAvgArgumentBuffer;

public:
    GPUInfernoAvg(int L, pyArray weights) : GPUInfernoBase(
            loadInfAvgLibrary,
            "inferno_cons_avg_ig1_flam2"
    ) {
        if (L != landPts)
            throw std::runtime_error("Not implemented!");

        this->L = L;

        argumentEncoder = fn->newArgumentEncoder(5);
        argumentBuffer = device->newBuffer(argumentEncoder->encodedLength(), MTL::ResourceOptions());
        argumentEncoder->setArgumentBuffer(argumentBuffer, 0);

        consAvgArgumentEncoder = fn->newArgumentEncoder(29);
        consAvgArgumentBuffer = device->newBuffer(consAvgArgumentEncoder->encodedLength(), MTL::ResourceOptions());
        consAvgArgumentEncoder->setArgumentBuffer(consAvgArgumentBuffer, 0);

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

        releaseLater(weightsBuffer);
        releaseLater(consAvgParamBuffer);

        releaseLater(argumentBuffer);
        releaseLater(argumentEncoder);

        releaseLater(consAvgArgumentEncoder);
        releaseLater(consAvgArgumentBuffer);
    }

    void setOutputBuffer() override {
        if (N == 0)
            throw std::runtime_error("N is not set (=0).");
        outputBuffer = device->newBuffer(N * landPts * dataSize, MTL::ResourceOptions());
    }

    void checkNt() override {
        if (Nt != M)
            throw std::runtime_error("Nt != M");
    }

    void run(pyArray out) {
        if (!(didSetData))
            throw std::runtime_error("Did not set data.");
        if (!(didSetParams))
            throw std::runtime_error("Did not set params.");

        py::buffer_info outInfo = out.request();
        if (outInfo.ndim != 2)
            throw std::runtime_error("Incompatible out dimension!");
        if ((outInfo.shape[0] != N) || (outInfo.shape[1] != landPts))
            throw std::runtime_error("Expected out dimensions (N, landPts)!");

        RunParams runParams = getRunParams();
        auto computeCommandEncoder = runParams.computeCommandEncoder;

        py::gil_scoped_release release;

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

        submit(landPts, runParams);

        py::gil_scoped_acquire acquire;

        // Copy output.
        memcpy(outInfo.ptr, outputBuffer->contents(), outInfo.size * dataSize);
    }
};

#endif
