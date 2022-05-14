#ifndef consAvgNoMask_hpp
#define consAvgNoMask_hpp

#include <array>
#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <simd/simd.h>

#include "common.hpp"
#include "loadConsAvgNoMaskLibrary.hpp"


namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;
using pyBoolArray = py::array_t<bool, py::array::c_style>;


class GPUConsAvgNoMask : public GPUBase {

    // Buffers.
    MTL::Buffer* inDataBuffer;
    MTL::Buffer* weightsBuffer;
    MTL::Buffer* outDataBuffer;

    // Parameters.
    int M, N, L;

public:
    GPUConsAvgNoMask(int L, pyArray weights) : GPUBase(
        loadConsAvgNoMaskLibrary,
        "cons_avg_no_mask"
    ) {
        this->L = L;

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

        releaseLater(inDataBuffer);
        releaseLater(weightsBuffer);
        releaseLater(outDataBuffer);
    }

    pyArray run(pyArray inData) {
        py::buffer_info inDataInfo = inData.request();
        if (inDataInfo.shape[0] != M)
            throw std::runtime_error("Expected input shape (M, L)!");
        if (inDataInfo.shape[1] != L)
            throw std::runtime_error("Expected input shape (M, L)!");

        void* inDataPtr = inDataInfo.ptr;

        RunParams runParams = getRunParams();
        auto computeCommandEncoder = runParams.computeCommandEncoder;

        // Release Python GIL after dealing with the input Python (NumPy)
        // arrays and getting the underlying data pointer.
        py::gil_scoped_release release;

        memcpy(inDataBuffer->contents(), inDataPtr, M * L * dataSize);
        inDataBuffer->didModifyRange(NS::Range::Make(0, inDataBuffer->length()));

        // Outputs.
        computeCommandEncoder->setBuffer(outDataBuffer, 0, 0);

        // Inputs.
        computeCommandEncoder->setBuffer(weightsBuffer, 0, 1);
        computeCommandEncoder->setBuffer(inDataBuffer, 0, 2);

        // Parameters.
        computeCommandEncoder->setBytes(&M, sizeof(int), 3);
        computeCommandEncoder->setBytes(&N, sizeof(int), 4);
        computeCommandEncoder->setBytes(&L, sizeof(int), 5);

        submit(N * L, runParams);

        py::gil_scoped_acquire acquire;

        return pyArray(N * L, (float*)outDataBuffer->contents());
    }
};

#endif
