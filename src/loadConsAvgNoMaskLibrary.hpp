#ifndef loadConsAvgNoMaskLibrary_hpp
#define loadConsAvgNoMaskLibrary_hpp

#include <stdio.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>


MTL::Library* loadConsAvgNoMaskLibrary(MTL::Device* device) {
    const char* shaderSrc = R"(
#include <metal_stdlib>
#include <metal_math>

using namespace metal;

kernel void cons_avg_no_mask(
    // Output buffers.
    device float* outData [[ buffer(0) ]],
    // Input buffers.
    const device float* weights [[ buffer(1) ]],
    const device float* inData [[ buffer(2) ]],
    // Size of input dims.
    const device int& M [[ buffer(3) ]],
    const device int& N [[ buffer(4) ]],
    const device int& L [[ buffer(5) ]],
    // Thread index.
    uint n_land_i [[ thread_position_in_grid ]]
) {
    int land_i, sel_val, n, m, in_flat_i, out_flat_i, weight_flat_i;
    float cum_weight, cum_sum, weight;

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

        cum_sum += inData[in_flat_i] * weight;
        cum_weight += weight;
    }

    out_flat_i = n * L + land_i;

    if (cum_weight < 1e-15)
        outData[out_flat_i] = 0.0f;
    else
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

