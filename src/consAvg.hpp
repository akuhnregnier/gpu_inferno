#ifndef consAvg_hpp
#define consAvg_hpp

#include <array>
#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <simd/simd.h>
#include <tuple>

#include "common.hpp"
#include "loadConsAvgLibrary.hpp"


namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;
using pyBoolArray = py::array_t<bool, py::array::c_style>;


class GPUConsAvg : public GPUBase {

    // Buffers.
    MTL::Buffer* inDataBuffer;
    MTL::Buffer* inMaskBuffer;
    MTL::Buffer* weightsBuffer;
    MTL::Buffer* outDataBuffer;
    MTL::Buffer* outMaskBuffer;

    // Parameters.
    int M, N, L;

public:
    GPUConsAvg(int L, pyArray weights) : GPUBase(
        loadConsAvgLibrary,
        "cons_avg"
    ) {
        this->L = L;

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

        releaseLater(inDataBuffer);
        releaseLater(inMaskBuffer);
        releaseLater(weightsBuffer);
        releaseLater(outDataBuffer);
        releaseLater(outMaskBuffer);
    }

    std::tuple<pyArray, pyBoolArray> run(pyArray inData, pyBoolArray inMask) {
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

        RunParams runParams = getRunParams();
        auto computeCommandEncoder = runParams.computeCommandEncoder;

        // Release Python GIL after dealing with the input Python (NumPy)
        // arrays and getting the underlying data pointer.
        py::gil_scoped_release release;

        memcpy(inDataBuffer->contents(), inDataPtr, M * L * dataSize);
        inDataBuffer->didModifyRange(NS::Range::Make(0, inDataBuffer->length()));

        memcpy(inMaskBuffer->contents(), inMaskPtr, M * L * boolSize);
        inMaskBuffer->didModifyRange(NS::Range::Make(0, inMaskBuffer->length()));

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

        submit(N * L, runParams);

        py::gil_scoped_acquire acquire;

        return std::make_tuple(
            pyArray(N * L, (float*)outDataBuffer->contents()),
            pyBoolArray(N * L, (bool*)outMaskBuffer->contents())
        );
    }
};

#endif
