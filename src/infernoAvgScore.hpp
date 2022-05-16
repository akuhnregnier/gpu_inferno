#ifndef infernoAvgScore_hpp
#define infernoAvgScore_hpp

#define PI 3.14159265358979323846

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <simd/simd.h>
#include <stdexcept>
#include <tuple>

#include "infernoAvg.hpp"
#include "loadInfAvgScoreLibrary.hpp"

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;
using pyBoolArray = py::array_t<bool, py::array::c_style>;

class GPUInfernoAvgScore final : public GPUInfernoAvgBase {

    // How many results will be produced per land point.
    static const int scoreUnitWidth = 4;

    MTL::Buffer* obsBuffer;  // BA Observations.
    MTL::Buffer* cropBuffer;  // Crop Observations.
    MTL::Buffer* obsParamBuffer;  // overall_scale, crop_f.

    void checkData(const py::buffer_info& x) {
        if (x.ndim != 2)
            throw std::runtime_error("Incompatible obs dimensions!");
        if ((x.shape[0] != 12) || (x.shape[1] != L))
            throw std::runtime_error("Expected obs shape (12, L).");
    }

public:
    GPUInfernoAvgScore(int L, pyArray weights, pyArray obsData, pyArray obsCrop) : GPUInfernoAvgBase(
            L,
            weights,
            loadInfAvgScoreLibrary,
            "inferno_cons_avg_score_ig1_flam2"
    ) {
        py::buffer_info obsInfo = obsData.request();
        checkData(obsInfo);

        py::buffer_info cropInfo = obsCrop.request();
        checkData(cropInfo);

        obsBuffer = device->newBuffer(12 * L * dataSize, MTL::ResourceOptions());
        cropBuffer = device->newBuffer(12 * L * dataSize, MTL::ResourceOptions());
        obsParamBuffer = device->newBuffer(2 * dataSize, MTL::ResourceOptions());

        memcpy(obsBuffer->contents(), obsInfo.ptr, obsInfo.size * dataSize);
        memcpy(cropBuffer->contents(), cropInfo.ptr, cropInfo.size * dataSize);
        consAvgArgumentEncoder->setBuffer(obsBuffer, 0, 2);
        consAvgArgumentEncoder->setBuffer(cropBuffer, 0, 3);
        consAvgArgumentEncoder->setBuffer(obsParamBuffer, 0, 4);  // Will be modified later.
        obsBuffer->didModifyRange(NS::Range::Make(0, obsBuffer->length()));
        cropBuffer->didModifyRange(NS::Range::Make(0, cropBuffer->length()));

        releaseLater(obsBuffer);
        releaseLater(cropBuffer);
        releaseLater(obsParamBuffer);
    }

    void setOutputBuffer() override {
        if (N == 0)
            throw std::runtime_error("N is not set (=0).");
        if (L == 0)
            throw std::runtime_error("L is not set (=0).");
        // 4 values for each land point:
        //  0 - Mean value at land point (for arcsinh NME)
        //  1 - Sum abs(pred - obs) at land point (for arcsinh NME)
        //  2 - phase diff (float) (for MPD)
        //  3 - phase diff ignored (1 / 0) (for MPD)
        // All in a single float buffer for convenience - will be aggregated in run().
        outputBuffer = device->newBuffer(scoreUnitWidth * L * dataSize, MTL::ResourceOptions());
        releaseLater(outputBuffer);
    }

    std::tuple<float, float, int> run(float overall_scale, float crop_f) {
        // Return:
        // - Arcsinh NME
        // - MPD
        // - N. ignored for MPD
        if (!(didSetData))
            throw std::runtime_error("Did not set data.");
        if (!(didSetParams))
            throw std::runtime_error("Did not set params.");

        float* obsParamPtr = (float*)obsParamBuffer->contents();
        obsParamPtr[0] = overall_scale;
        obsParamPtr[1] = crop_f;
        obsParamBuffer->didModifyRange(NS::Range::Make(0, obsParamBuffer->length()));

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
        computeCommandEncoder->useResource(obsBuffer, MTL::ResourceUsageRead);
        computeCommandEncoder->useResource(cropBuffer, MTL::ResourceUsageRead);
        computeCommandEncoder->useResource(obsParamBuffer, MTL::ResourceUsageRead);

        submit(landPts, runParams);

        // Aggregate output into arcsinh NME, MPD.
        float* outPtr = (float*)outputBuffer->contents();
        // Arcsinh NME
        float meanObs = 0.0f;
        float denom = 0.0f;
        float sumAbsDiff = 0.0f;
        // MPD
        float mpd = 0.0f;
        int nIgnored = 0;
        for (int i = 0; i < L; i++) {
            int offset = scoreUnitWidth * i;

            // Arcsinh NME
            meanObs += outPtr[offset];
            sumAbsDiff += outPtr[offset + 1];

            // MPD
            if (outPtr[offset + 3] > 0.5) {
                // If ignored.
                nIgnored += 1;
            } else {
                mpd += outPtr[offset + 2];
            }
        }
        meanObs /= L;
        mpd /= (L - nIgnored);
        mpd /= PI;

        // Finish arcsinh NME calc.
        float* obsVals = (float*)obsBuffer->contents();
        for (int i = 0; i < 12 * L; i++) {
            denom += abs(asinh(ARCSINH_FACTOR * obsVals[i]) - meanObs);
        }

        float arcsinh_nme = sumAbsDiff / denom;

        py::gil_scoped_acquire acquire;

        return std::make_tuple(arcsinh_nme, mpd, nIgnored);
    }
};

#endif
