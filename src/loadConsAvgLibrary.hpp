#ifndef loadConsAvgLibrary_hpp
#define loadConsAvgLibrary_hpp

#include <stdio.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>


MTL::Library* loadConsAvgLibrary(MTL::Device* device) {
    const char* shaderSrc = R"(
#include <metal_stdlib>
#include <metal_math>

using namespace metal;

kernel void cons_avg(
    // Output buffers.
    device float* outData [[ buffer(0) ]],
    device bool* outMask [[ buffer(1) ]],
    // Input buffers.
    const device float* weights [[ buffer(2) ]],
    const device float* inData [[ buffer(3) ]],
    const device bool* inMask [[ buffer(4) ]],
    // Size of input dims.
    const device int& M [[ buffer(5) ]],
    const device int& N [[ buffer(6) ]],
    const device int& L [[ buffer(7) ]],
    // Thread index.
    uint n_land_i [[ thread_position_in_grid ]]
) {
    int land_i, sel_val, n, m, in_flat_i, out_flat_i, weight_flat_i;
    float cum_weight, cum_sum, weight, weight_val;
    bool out_mask_val;

    land_i = n_land_i / N;
    n = n_land_i - (land_i * N);

    cum_sum = 0.0f;
    cum_weight = 0.0f;

    in_flat_i = land_i - L;
    weight_flat_i = n - N;

    for (m = 0; m < M; m++) {
        in_flat_i += L;
        weight_flat_i += N;

        weight = weights[weight_flat_i];

        if (weight < 1e-9) continue;

        sel_val = !inMask[in_flat_i];  // Invert and promote to int.
        weight_val = weight * sel_val;

        cum_sum += inData[in_flat_i] * weight_val;
        cum_weight += weight_val;
    }

    out_mask_val = (cum_weight < 1e-15);
    out_flat_i = n * L + land_i;
    outMask[out_flat_i] = out_mask_val;

    if (out_mask_val) return;

    outData[out_flat_i] = cum_sum / cum_weight;
}
    )";

    NS::Error* error = nullptr;

    MTL::Library* library = device->newLibrary(
        NS::String::string(shaderSrc, NS::UTF8StringEncoding),
        nullptr,
        &error
    );
    if ( !library ) {
        __builtin_printf( "%s", error->localizedDescription()->utf8String() );
        std::cout << "Failed!" << std::endl;
        assert(false);
    }

    return library;
}

#endif

