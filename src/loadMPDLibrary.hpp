#ifndef loadMPDLibrary_hpp
#define loadMPDLibrary_hpp

#include <stdio.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>


MTL::Library* loadMPDLibrary(MTL::Device* device) {
    const char* shaderSrc = R"(
#include <metal_stdlib>
#include <metal_math>

using namespace metal;

constant float sinThetas[12] = {
    0.0f,
    0.49999999999999994f,
    0.8660254037844386f,
    1.0f,
    0.8660254037844387f,
    0.49999999999999994f,
    1.2246467991473532e-16f,
    -0.4999999999999998f,
    -0.8660254037844384f,
    -1.0f,
    -0.8660254037844386f,
    -0.5000000000000004f,
};
constant float cosThetas[12] = {
    1.0f,
    0.8660254037844387f,
    0.5000000000000001f,
    6.123233995736766e-17f,
    -0.49999999999999983f,
    -0.8660254037844387f,
    -1.0f,
    -0.8660254037844388f,
    -0.5000000000000004f,
    -1.8369701987210297e-16f,
    0.5000000000000001f,
    0.8660254037844384f,
};

kernel void calculate_mpd(
    // Input data.
    const device float* obs [[ buffer(0) ]],
    const device float* pred [[ buffer(1) ]],
    // Size of input dim 2.
    const device int& N [[ buffer(2) ]],
    // Output data.
    device float* diffs [[ buffer(3) ]],
    device bool* diffsMask [[ buffer(4) ]],
    // Thread index.
    uint id [[ thread_position_in_grid ]]
) {
    float obsVal, predVal, obsPhase, predPhase;
    float lxObs = 0.0f;
    float lyObs = 0.0f;
    float lxPred = 0.0f;
    float lyPred = 0.0f;
    unsigned int flat_index;
    unsigned int obsCloseZeroCount = 0;
    unsigned int predCloseZeroCount = 0;
    bool masked = false;

    for (unsigned int i = 0; i < 12; i++) {
        flat_index = (i * N) + id;
        obsVal = obs[flat_index];
        predVal = pred[flat_index];

        lxObs += obsVal * cosThetas[i];
        lyObs += obsVal * sinThetas[i];

        lxPred += predVal * cosThetas[i];
        lyPred += predVal * sinThetas[i];

        if (abs(obsVal) < 1e-15) obsCloseZeroCount += 1;
        if (abs(predVal) < 1e-15) predCloseZeroCount += 1;
    }
    // TODO - test if statement preventing certain calculations if masked?

    if ((obsCloseZeroCount == 12) || (predCloseZeroCount == 12)) masked = true;

    obsPhase = atan2(lxObs, lyObs);
    predPhase = atan2(lxPred, lyPred);

    diffs[id] = acos(cos(predPhase - obsPhase));
    diffsMask[id] = masked;
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
