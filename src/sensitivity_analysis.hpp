#ifndef sensitivity_analysis_hpp
#define sensitivity_analysis_hpp

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <simd/simd.h>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "common.hpp"
#include "loadSALibrary.hpp"

namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;

class GPUSACompute final : public GPUBase {
public:
    static const int nData = 30;
    static const int maxNSamples = 10000;
    constexpr static const std::array<int, 5> samples_Nt_npft_indices = { 0, 1, 2, 3, 5};

    MTL::Buffer* outputBuffer;
    std::array<MTL::Buffer*, nData> saBuffers;

    // Parameters.
    // NOTE ignition 1, flammability 2 always.
    int drynessMethod;
    int fuelBuildUpMethod;
    int includeTemperature;
    int Nt;
    int nSamples;
    float overallScale;
    float crop_f;

    std::array<int, 3> outShape3d;

    MTL::ArgumentEncoder* argumentEncoder;
    MTL::Buffer* argumentBuffer;

    MTL::Buffer* createBufferFromPyArray(pyArray array) {
        py::buffer_info buf = array.request();
        return device->newBuffer(static_cast<float *>(buf.ptr), buf.shape[0] * dataSize, MTL::ResourceOptions());
    }

    GPUSACompute(int Nt) : GPUBase(
        loadSALibrary,
        "sa_inferno_ig1_flam2"
    ) {
        this->Nt = Nt;

        argumentEncoder = fn->newArgumentEncoder(8);
        argumentBuffer = device->newBuffer(argumentEncoder->encodedLength(), MTL::ResourceOptions());
        argumentEncoder->setArgumentBuffer(argumentBuffer, 0);

        releaseLater(argumentBuffer);
        releaseLater(argumentEncoder);

        // Buffers for later reuse.
        for (const int& i : samples_Nt_npft_indices) {
            saBuffers[i] = device->newBuffer(maxNSamples * Nt * nPFT * dataSize, MTL::ResourceOptions());
        }
        saBuffers[4] = device->newBuffer(maxNSamples * Nt * nPFTGroups * dataSize, MTL::ResourceOptions());
        saBuffers[6] = device->newBuffer(maxNSamples * Nt * dataSize, MTL::ResourceOptions());
        saBuffers[7] = device->newBuffer(maxNSamples * Nt * dataSize, MTL::ResourceOptions());
        for (unsigned long i = 8; i < nData; i++) {
            saBuffers[i] = device->newBuffer(maxNSamples * nPFTGroups * dataSize, MTL::ResourceOptions());
        }

        for (unsigned long i = 0; i < nData; i++) {
            argumentEncoder->setBuffer(saBuffers[i], 0, i);
            releaseLater(saBuffers[i]);
        }

        outShape3d = {maxNSamples, Nt, nPFT};
        outputBuffer = device->newBuffer(maxNSamples * Nt * nPFT * dataSize, MTL::ResourceOptions());
        releaseLater(outputBuffer);
    }

    int getMaxNSamples() {
        return maxNSamples;
    }

    void setData(
        int drynessMethod,
        int fuelBuildUpMethod,
        int includeTemperature,
        int nSamples,
        float overallScale,
        float crop_f,
        // Data.
        pyArray t1p5m_tile,
        pyArray frac,
        pyArray fuel_build_up,
        pyArray fapar_diag_pft,
        pyArray grouped_dry_bal,
        pyArray litter_pool,
        pyArray dry_days,
        pyArray obs_pftcrop_1d,
        // Params.
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
        if (nSamples > maxNSamples)
            throw std::runtime_error("Too many samples requested.");

        this->nSamples = nSamples;
        this->drynessMethod = drynessMethod;
        this->fuelBuildUpMethod = fuelBuildUpMethod;
        this->includeTemperature = includeTemperature;
        this->overallScale = overallScale;
        this->crop_f = crop_f;

        std::array<pyArray, nData> saArrays = {
            t1p5m_tile,
            frac,
            fuel_build_up,
            fapar_diag_pft,
            grouped_dry_bal,
            litter_pool,
            dry_days,
            obs_pftcrop_1d,
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

        for (unsigned long i = 0; i < nData; i++) {
            py::buffer_info paramBuf = saArrays[i].request();

            // Shape checks.

            if (!(paramBuf.shape[0] == maxNSamples))
                throw std::runtime_error("Parameter buffer wrong shape.");

            if (
                std::any_of(
                    std::begin(samples_Nt_npft_indices),
                    std::end(samples_Nt_npft_indices),
                    [=](const unsigned long n) { return n == i; }
                )
            ) {
                if (!((paramBuf.shape[1] == Nt) && (paramBuf.shape[2] == nPFT)))
                    throw std::runtime_error("Parameter buffer wrong shape.");
            }
            else if ((i == 4) && ((paramBuf.shape[1] != Nt) || (paramBuf.shape[2] != nPFTGroups)))
                throw std::runtime_error("Parameter buffer wrong shape.");
            else if (((i == 6) || (i == 7)) && (paramBuf.shape[1] != Nt))
                throw std::runtime_error("Parameter buffer wrong shape.");
            else if (i >= 8) {
                if (!(paramBuf.shape[1] == nPFTGroups)) {
                    throw std::runtime_error("Parameter buffer wrong shape.");
                }
            }

            if ((saBuffers[i]->length() / dataSize) != paramBuf.size) {
                throw std::runtime_error("Wrong size.");
            }

            // Copy to buffers.

            memcpy(saBuffers[i]->contents(), paramBuf.ptr, paramBuf.size * dataSize);
            saBuffers[i]->didModifyRange(NS::Range::Make(0, saBuffers[i]->length()));
        }
    }

    void run(pyArray out) {
        py::buffer_info outInfo = out.request();
        if (outInfo.ndim != 2)
            throw std::runtime_error("Incompatible out dimension!");
        if ((outInfo.shape[0] != maxNSamples) || (outInfo.shape[1] != Nt))
            throw std::runtime_error("Expected out dimensions (maxNSamples, Nt)!");

        RunParams runParams = getRunParams();
        auto computeCommandEncoder = runParams.computeCommandEncoder;

        py::gil_scoped_release release;

        // Output.
        computeCommandEncoder->setBuffer(outputBuffer, 0, 0);

        computeCommandEncoder->setBytes(&drynessMethod, paramSize, 1);
        computeCommandEncoder->setBytes(&fuelBuildUpMethod, paramSize, 2);
        computeCommandEncoder->setBytes(&includeTemperature, paramSize, 3);
        computeCommandEncoder->setBytes(&Nt, paramSize, 4);
        computeCommandEncoder->setBytes(&nSamples, paramSize, 5);
        computeCommandEncoder->setBytes(&overallScale, dataSize, 6);
        computeCommandEncoder->setBytes(&crop_f, dataSize, 7);

        computeCommandEncoder->setBuffer(argumentBuffer, 0, 8);
        for (unsigned long i = 0; i < nData; i++) {
            computeCommandEncoder->useResource(saBuffers[i], MTL::ResourceUsageRead);
        }

        submit(nSamples * Nt * nPFT, runParams);

        // Acquire GIL before using Python array below.
        py::gil_scoped_acquire acquire;

        // Perform sum over PFT axis.
        //
        // NOTE - The below would be more correct, but does not currently correspond to the
        // NUMBA Python implementation.
        // return np.sum(data * frac[:, : data.shape[1]], axis=1) / np.sum(
        //     frac[:, : data.shape[1]], axis=1
        // )
        // return np.sum(data * frac[:, : data.shape[1]], axis=1)

        // Note that multiplication by frac already occurs within the kernel!
        float* ba_frac = (float*)outputBuffer->contents();
        auto outR = out.mutable_unchecked<2>();

        // Sum over PFT axis.
        for (int sample_i = 0; sample_i < nSamples; sample_i++) {
            for (int ti = 0; ti < Nt; ti++) {
                float pft_sum = 0.0f;
                for (int pft_i = 0; pft_i < nPFT; pft_i++) {
                    pft_sum += ba_frac[
                        get_index_3d(sample_i, ti, pft_i, outShape3d.data())
                    ];
                }
                outR(sample_i, ti) = pft_sum;
            }
        }
    }

};

#endif
