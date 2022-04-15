#ifndef loadLibrary_hpp
#define loadLibrary_hpp

#include <stdio.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>

MTL::Library* loadLibrary(MTL::Device* device) {
    const char* shaderSrc = R"(
    #include <metal_stdlib>
    using namespace metal;

    kernel void test_fn(
            device float* buffer [[buffer(0)]],
            uint index [[thread_position_in_grid]],
            uint gridSize [[threads_per_grid]]
        ) {
            buffer[index] = index + 1.2f;
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
