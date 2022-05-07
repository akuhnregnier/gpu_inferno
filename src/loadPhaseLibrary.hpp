#ifndef loadPhaseLibrary_hpp
#define loadPhaseLibrary_hpp

#include <stdio.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>


MTL::Library* loadPhaseLibrary(MTL::Device* device) {
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

kernel void calculate_phase(
    // Output buffer.
    device float* out [[ buffer(0) ]],
    // Input buffer.
    const device float* input [[ buffer(1) ]],
    // Size of input dim 2.
    const device int& N [[ buffer(2) ]],
    // Thread index.
    uint id [[ thread_position_in_grid ]]
) {
    float lx = 0.0f;
    float ly = 0.0f;
    unsigned int flat_index;

    for (unsigned int i = 0; i < 12; i++) {
        flat_index = (i * N) + id;
        lx += input[flat_index] * cosThetas[i];
        ly += input[flat_index] * sinThetas[i];
    }

    out[id] = atan2(lx, ly);
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
