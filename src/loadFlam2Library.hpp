#ifndef loadFlam2Library_hpp
#define loadFlam2Library_hpp

#include <stdio.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>


MTL::Library* loadFlam2Library(MTL::Device* device) {
    const char* shaderSrc = R"(
#include <metal_stdlib>
#include <metal_math>

using namespace metal;


inline float sigmoid(float x, float factor, float centre, float shape) {
    // Apply generalised sigmoid with slope determine by `factor`, position by
    // `centre`, and shape by `shape`, with the result being in [0, 1].
    return pow((1.0 + exp(factor * shape * (centre - x))), (-1.0 / shape));
}


float calc_flam_flam2_only(
    float temp_l,
    float fuel_build_up,
    float fapar,
    float dry_days,
    int dryness_method,
    int fuel_build_up_method,
    float fapar_factor,
    float fapar_centre,
    float fapar_shape,
    float fuel_build_up_factor,
    float fuel_build_up_centre,
    float fuel_build_up_shape,
    float temperature_factor,
    float temperature_centre,
    float temperature_shape,
    float dry_day_factor,
    float dry_day_centre,
    float dry_day_shape,
    float dry_bal,
    float dry_bal_factor,
    float dry_bal_centre,
    float dry_bal_shape,
    float litter_pool,
    float litter_pool_factor,
    float litter_pool_centre,
    float litter_pool_shape,
    int include_temperature,
    float fapar_weight,
    float dryness_weight,
    float temperature_weight,
    float fuel_weight
) {
    // Only flammability_method 2.

    float dry_factor, fuel_factor, fapar_sigmoid, weighted_temperature_sigmoid;

    // New calculation, based on FAPAR (and derived fuel_build_up).

    if (dryness_method == 1) {
        dry_factor = sigmoid(dry_days, dry_day_factor, dry_day_centre, dry_day_shape);
    }
    else if (dryness_method == 2) {
        dry_factor = sigmoid(dry_bal, dry_bal_factor, dry_bal_centre, dry_bal_shape);
    }
    else {
        // raise ValueError("Unknown 'dryness_method'.");
        dry_factor = -1;
    }

    if (fuel_build_up_method == 1) {
        fuel_factor = sigmoid(
            fuel_build_up,
            fuel_build_up_factor,
            fuel_build_up_centre,
            fuel_build_up_shape
        );
    }
    else if (fuel_build_up_method == 2) {
        fuel_factor = sigmoid(
            litter_pool, litter_pool_factor, litter_pool_centre, litter_pool_shape
        );
    }
    else {
        // raise ValueError("Unknown 'fuel_build_up_method'.")
        fuel_factor = -1.0;
    }

    if (include_temperature == 1) {
        float temperature_sigmoid = sigmoid(
            temp_l, temperature_factor, temperature_centre, temperature_shape
        );
        weighted_temperature_sigmoid = (1 + temperature_weight * (temperature_sigmoid - 1));
    }
    else if (include_temperature == 0) {
        weighted_temperature_sigmoid = 1.0;
    }
    else {
        // raise ValueError("Unknown 'include_temperature'.")
        weighted_temperature_sigmoid = -1.0;
    }

    fapar_sigmoid = sigmoid(fapar, fapar_factor, fapar_centre, fapar_shape);

    // Convert fuel build-up index to flammability factor.
    return (
        (1 + dryness_weight * (dry_factor - 1))
        * weighted_temperature_sigmoid
        * (1 + fuel_weight * (fuel_factor - 1))
        * (1 + fapar_weight * (fapar_sigmoid - 1))
    );
}


kernel void calc_flam_flam2_kernel(
    device float* out [[ buffer(0) ]],
    device float* inFloat [[ buffer(1) ]],
    device int* inInt [[ buffer(2) ]]
) {
    float temp_l = inFloat[0];
    float fuel_build_up = inFloat[1];
    float fapar = inFloat[2];
    float dry_days = inFloat[3];
    float fapar_factor = inFloat[4];
    float fapar_centre = inFloat[5];
    float fapar_shape = inFloat[6];
    float fuel_build_up_factor = inFloat[7];
    float fuel_build_up_centre = inFloat[8];
    float fuel_build_up_shape = inFloat[9];
    float temperature_factor = inFloat[10];
    float temperature_centre = inFloat[11];
    float temperature_shape = inFloat[12];
    float dry_day_factor = inFloat[13];
    float dry_day_centre = inFloat[14];
    float dry_day_shape = inFloat[15];
    float dry_bal = inFloat[16];
    float dry_bal_factor = inFloat[17];
    float dry_bal_centre = inFloat[18];
    float dry_bal_shape = inFloat[19];
    float litter_pool = inFloat[20];
    float litter_pool_factor = inFloat[21];
    float litter_pool_centre = inFloat[22];
    float litter_pool_shape = inFloat[23];
    float fapar_weight = inFloat[24];
    float dryness_weight = inFloat[25];
    float temperature_weight = inFloat[26];
    float fuel_weight = inFloat[27];

    int dryness_method = inInt[0];
    int fuel_build_up_method = inInt[1];
    int include_temperature = inInt[2];

    out[0] = calc_flam_flam2_only(
        temp_l,
        fuel_build_up,
        fapar,
        dry_days,
        dryness_method,
        fuel_build_up_method,
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
        dry_bal,
        dry_bal_factor,
        dry_bal_centre,
        dry_bal_shape,
        litter_pool,
        litter_pool_factor,
        litter_pool_centre,
        litter_pool_shape,
        include_temperature,
        fapar_weight,
        dryness_weight,
        temperature_weight,
        fuel_weight
    );
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

