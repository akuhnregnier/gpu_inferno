#ifndef phase_hpp
#define phase_hpp

#define PI 3.14159265358979323846

#include <array>
#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <simd/simd.h>

#include "common.hpp"
#include "loadPhaseLibrary.hpp"

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;


class GPUCalculatePhase : GPUBase {

    MTL::Buffer* inputBuffer;
    MTL::Buffer* outputBuffer;

    // Parameters.
    int N;

public:
    GPUCalculatePhase(int N) : GPUBase (
        loadPhaseLibrary,
        "calculate_phase"
    ) {
        this->N = N;

        inputBuffer = device->newBuffer(12 * N * dataSize, MTL::ResourceOptions());
        outputBuffer = device->newBuffer(N * dataSize, MTL::ResourceOptions());

        releaseLater(inputBuffer);
        releaseLater(outputBuffer);
    }

    pyArray run(pyArray x) {
        RunParams runParams = getRunParams();
        auto computeCommandEncoder = runParams.computeCommandEncoder;

        py::buffer_info xBuf = x.request();
        if (xBuf.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");
        if (xBuf.shape[0] != 12)
            throw std::runtime_error("Expected shape (12, N)!");
        if (xBuf.shape[1] != N)
            throw std::runtime_error(
                "Number of points along 2nd dimension does not match number given in class setup!"
            );

        memcpy(inputBuffer->contents(), xBuf.ptr, xBuf.size * dataSize);
        inputBuffer->didModifyRange(NS::Range::Make(0, inputBuffer->length()));

        // Output.
        computeCommandEncoder->setBuffer(outputBuffer, 0, 0);

        // Input.
        computeCommandEncoder->setBuffer(inputBuffer, 0, 1);

        // N.
        computeCommandEncoder->setBytes(&N, sizeof(int), 2);

        submit(N, runParams);

        return pyArray(N, (float*)outputBuffer->contents());
    }
};


pyArray calculate_phase(pyArray x) {
    py::buffer_info xBuf = x.request();
    if (xBuf.shape[0] != 12)
        throw std::runtime_error("Expected shape (12, N)!");
    if (xBuf.ndim != 2)
        throw std::runtime_error("Incompatible buffer dimension!");

    unsigned int flat_index;
    float* xPtr = static_cast<float*>(xBuf.ptr);
    // About the same speed.
    // auto xr = x.unchecked<2>();

    unsigned int outputCount = xBuf.shape[1];

    float cosThetas[12];
    float sinThetas[12];
    float theta;
    float step = 1.0f / 12.0f;
    for (unsigned int i = 0; i < 12; i++) {
        theta = 2 * PI * i * step;
        sinThetas[i] = sin(theta);
        cosThetas[i] = cos(theta);
    }

    float phaseArr[outputCount];
    float lx, ly;
    for (unsigned int i = 0; i < outputCount; i++) {
        lx = 0.0f;
        ly = 0.0f;
        for (unsigned int j = 0; j < 12; j++) {
            flat_index = (j * outputCount) + i;
            lx += xPtr[flat_index] * cosThetas[j];
            ly += xPtr[flat_index] * sinThetas[j];
            // lx += xr(j, i) * cosThetas[j];
            // ly += xr(j, i) * sinThetas[j];
        }
        phaseArr[i] = atan2(lx, ly);
    }

    return pyArray(outputCount, phaseArr);
}

#endif
