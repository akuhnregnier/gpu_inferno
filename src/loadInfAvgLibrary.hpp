#ifndef loadInfAvgLibrary_hpp
#define loadInfAvgLibrary_hpp

#include <stdio.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <simd/simd.h>
#include <iostream>
#include <fstream>


MTL::Library* loadInfAvgLibrary(MTL::Device* device) {
    const char* shaderSrc = R"(
#include <metal_stdlib>
#include <metal_math>


using namespace metal;


// Indexing convention is time, pft, land

// Seconds in a month.
// Note that this is approx. 365 days/12months but is slightly larger.
// This should be changed in a future update.
constant float s_in_month = 2.6280288e6;
constant float m2_in_km2 = 1.0e6;
// constant float rsec_per_day = 86400.0;
constant float s_in_day = 86400.0;

constant int npft = 13;
constant int n_pft_groups = 3;
constant int n_total_pft = 17;
constant int land_pts = 7771;

constant float avg_ba[13] = { 1.7e6, 1.7e6, 1.7e6, 1.7e6, 1.7e6, 3.2e6, 0.4e6, 3.2e6, 3.2e6, 0.4e6, 3.2e6, 2.7e6, 2.7e6 };


// NOTE Ignition method 1.
//
// Assume a multi-year annual mean of 2.7/km2/yr
// (Huntrieser et al. 2007) 75% are Cloud to Ground flashes
// (Prentice and Mackerras 1977)
constant float nat_ign_l = 2.7 / s_in_month / m2_in_km2 / 12.0 * 0.75;

// We parameterised 1.5 ignitions/km2/month globally from GFED
constant float man_ign_l = 1.5 / s_in_month / m2_in_km2;

// Total
constant float total_ignition_1 = man_ign_l + nat_ign_l;


struct DataArrays {
    // Input arrays.
    const device float* t1p5m_tile [[ id(0) ]];
    const device float* q1p5m_tile [[ id(1) ]];
    const device float* pstar [[ id(2) ]];
    // XXX NOTE - This is with a single soil layer selected!
    const device float* sthu_soilt_single [[ id(3) ]];
    const device float* frac [[ id(4) ]];
    const device float* c_soil_dpm_gb [[ id(5) ]];
    const device float* c_soil_rpm_gb [[ id(6) ]];
    const device float* canht [[ id(7) ]];
    const device float* ls_rain [[ id(8) ]];
    const device float* con_rain [[ id(9) ]];
    const device float* pop_den [[ id(10) ]];
    const device float* flash_rate [[ id(11) ]];
    const device float* fuel_build_up [[ id(12) ]];
    const device float* fapar_diag_pft [[ id(13) ]];
    const device float* grouped_dry_bal [[ id(14) ]];
    const device float* litter_pool [[ id(15) ]];
    const device float* dry_days [[ id(16) ]];
};


struct ConsAvgData {
    // Input arrays.
    const device float* weights [[ id(0) ]];
    const device int* params [[ id(1) ]];  // M, N, L
};


int get_pft_group_index(int pft_i) {
    if (pft_i <= 4) {
        return 0;
    }
    else if (pft_i <= 10) {
        return 1;
    }
    else {
        // 11, 12
        return 2;
    }
}


inline int get_index_3d(int x, int y, int z, const thread int* shape_3d) {
    return (x * shape_3d[1] * shape_3d[2]) + (y * shape_3d[2]) + z;
}


inline int get_index_2d(int x, int y, const thread int* shape_2d) {
    return (x * shape_2d[1]) + y;
}


inline float calc_burnt_area(float flam_l, float ignitions_l, float avg_ba_i) {
    return flam_l * ignitions_l * avg_ba_i;
}


// XXX The sigmoid function here does not behave exactly like the python / numba equivalent, with small output values, e.g. 1e-13 being truncated.

inline float sigmoid(float x, float factor, float centre, float shape) {
    // Apply generalised sigmoid with slope determine by `factor`, position by
    // `centre`, and shape by `shape`, with the result being in [0, 1].
    return pow((1.0 + exp(factor * shape * (centre - x))), (-1.0 / shape));
}


float calc_flam_flam2(
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


inline float calculate_weighted_ba(
    int pft_i,
    float temperature,
    float fuel_build_up_val,
    float fapar_diag_pft_val,
    float dry_days_val,
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
    float grouped_dry_bal_val,
    float dry_bal_factor,
    float dry_bal_centre,
    float dry_bal_shape,
    float litter_pool_val,
    float litter_pool_factor,
    float litter_pool_centre,
    float litter_pool_shape,
    float include_temperature,
    float fapar_weight,
    float dryness_weight,
    float temperature_weight,
    float fuel_weight,
    float frac_val
) {
    float flammability_ft_i_l = calc_flam_flam2(
        temperature,
        fuel_build_up_val,
        fapar_diag_pft_val,
        dry_days_val,
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
        grouped_dry_bal_val,
        dry_bal_factor,
        dry_bal_centre,
        dry_bal_shape,
        litter_pool_val,
        litter_pool_factor,
        litter_pool_centre,
        litter_pool_shape,
        include_temperature,
        fapar_weight,
        dryness_weight,
        temperature_weight,
        fuel_weight
    );

    float burnt_area_ft_i_l = calc_burnt_area(
        flammability_ft_i_l,
        // NOTE OPT ignition mode 1 only
        total_ignition_1,
        avg_ba[pft_i]
    );

    // Record the pft-specific BA weighted by frac.
    return burnt_area_ft_i_l * frac_val;
}

kernel void inferno_cons_avg_ig1_flam2(
    // Optimised for only:
    // - ignition mode 1
    // - flammability_method 2

    // Output buffer.
    device float* out [[ buffer(0) ]],
    // Params.
    const device int& dryness_method [[ buffer(1) ]],
    const device int& fuel_build_up_method [[ buffer(2) ]],
    const device int& include_temperature [[ buffer(3) ]],
    const device int& Nt [[ buffer(4) ]],  // i.e. <data>.shape[0].
    // Input arrays.
    const device DataArrays& data_arrays [[ buffer(5) ]],
    // Parameters.
    const device float* fapar_factor [[ buffer(6) ]],
    const device float* fapar_centre [[ buffer(7) ]],
    const device float* fapar_shape [[ buffer(8) ]],
    const device float* fuel_build_up_factor [[ buffer(9) ]],
    const device float* fuel_build_up_centre [[ buffer(10) ]],
    const device float* fuel_build_up_shape [[ buffer(11) ]],
    const device float* temperature_factor [[ buffer(12) ]],
    const device float* temperature_centre [[ buffer(13) ]],
    const device float* temperature_shape [[ buffer(14) ]],
    const device float* dry_day_factor [[ buffer(15) ]],
    const device float* dry_day_centre [[ buffer(16) ]],
    const device float* dry_day_shape [[ buffer(17) ]],
    const device float* dry_bal_factor [[ buffer(18) ]],
    const device float* dry_bal_centre [[ buffer(19) ]],
    const device float* dry_bal_shape [[ buffer(20) ]],
    const device float* litter_pool_factor [[ buffer(21) ]],
    const device float* litter_pool_centre [[ buffer(22) ]],
    const device float* litter_pool_shape [[ buffer(23) ]],
    const device float* fapar_weight [[ buffer(24) ]],
    const device float* dryness_weight [[ buffer(25) ]],
    const device float* temperature_weight [[ buffer(26) ]],
    const device float* fuel_weight [[ buffer(27) ]],
    const device bool* checks_failed [[ buffer(28) ]],
    const device ConsAvgData& cons_avg_data [[ buffer(29) ]],
    // Thread index.
    uint land_i [[ thread_position_in_grid ]]
) {
    // Input arrays.
    const device float* t1p5m_tile = data_arrays.t1p5m_tile;
    const device float* q1p5m_tile = data_arrays.q1p5m_tile;
    const device float* pstar = data_arrays.pstar;
    // XXX NOTE - This is with a single soil layer selected!
    const device float* sthu_soilt_single = data_arrays.sthu_soilt_single;
    const device float* frac = data_arrays.frac;
    const device float* c_soil_dpm_gb = data_arrays.c_soil_dpm_gb;
    const device float* canht = data_arrays.canht;
    const device float* ls_rain = data_arrays.ls_rain;
    const device float* con_rain = data_arrays.con_rain;
    const device float* fuel_build_up = data_arrays.fuel_build_up;
    const device float* fapar_diag_pft = data_arrays.fapar_diag_pft;
    const device float* grouped_dry_bal = data_arrays.grouped_dry_bal;
    const device float* litter_pool = data_arrays.litter_pool;
    const device float* dry_days = data_arrays.dry_days;

    const device float* weights = cons_avg_data.weights;
    const int M = cons_avg_data.params[0];  // `m` input index, aka `ti`.
    const int Nout = cons_avg_data.params[1];  // `n` output index.
    const int L = cons_avg_data.params[2];  // Number of land points.

    const int out_shape_2d[2] = { Nout, land_pts };
    const int shape_2d[2] = { Nt, land_pts };
    const int weights_shape_2d[2] = { Nt, Nout };
    const int total_pft_shape_3d[3] = { Nt, n_total_pft, land_pts };
    const int nat_pft_shape_3d[3] = { Nt, npft, land_pts };
    const int grouped_pft_shape_3d[3] = { Nt, n_pft_groups, land_pts };

    // Temporary variables used to compute the conservative average.
    float cum_sum[12] = { };
    float cum_weight[12] = { };

    for (int ti = 0; ti < Nt; ti++) {

        float pft_weighted_ba_ti = 0.0f;

        for (int pft_i = 0; pft_i < npft; pft_i++) {
            int nat_pft_3d_flat_i = get_index_3d(ti, pft_i, land_i, nat_pft_shape_3d);

            if (checks_failed[nat_pft_3d_flat_i]) continue;

            // If all the checks were passes, start fire calculations

            const int pft_group_i = get_pft_group_index(pft_i);

            int total_pft_3d_flat_i = get_index_3d(ti, pft_i, land_i, total_pft_shape_3d);
            int grouped_pft_3d_flat_i = get_index_3d(ti, pft_group_i, land_i, grouped_pft_shape_3d);
            int flat_2d = get_index_2d(ti, land_i, shape_2d);

            float temperature = t1p5m_tile[total_pft_3d_flat_i];
            float fuel_build_up_val = fuel_build_up[nat_pft_3d_flat_i];
            float fapar_diag_pft_val = fapar_diag_pft[nat_pft_3d_flat_i];
            float dry_days_val = dry_days[flat_2d];
            float litter_pool_val = litter_pool[nat_pft_3d_flat_i];
            float grouped_dry_bal_val = grouped_dry_bal[grouped_pft_3d_flat_i];
            float frac_val = frac[total_pft_3d_flat_i];

            pft_weighted_ba_ti +=  calculate_weighted_ba(
                pft_i,
                temperature,
                fuel_build_up_val,
                fapar_diag_pft_val,
                dry_days_val,
                dryness_method,
                fuel_build_up_method,
                fapar_factor[pft_group_i],
                fapar_centre[pft_group_i],
                fapar_shape[pft_group_i],
                fuel_build_up_factor[pft_group_i],
                fuel_build_up_centre[pft_group_i],
                fuel_build_up_shape[pft_group_i],
                temperature_factor[pft_group_i],
                temperature_centre[pft_group_i],
                temperature_shape[pft_group_i],
                dry_day_factor[pft_group_i],
                dry_day_centre[pft_group_i],
                dry_day_shape[pft_group_i],
                grouped_dry_bal_val,
                dry_bal_factor[pft_group_i],
                dry_bal_centre[pft_group_i],
                dry_bal_shape[pft_group_i],
                litter_pool_val,
                litter_pool_factor[pft_group_i],
                litter_pool_centre[pft_group_i],
                litter_pool_shape[pft_group_i],
                include_temperature,
                fapar_weight[pft_group_i],
                dryness_weight[pft_group_i],
                temperature_weight[pft_group_i],
                fuel_weight[pft_group_i],
                frac_val
            );
        }

        for (int n = 0; n < Nout; n++) {
            float weight = weights[get_index_2d(ti, n, weights_shape_2d)];
            cum_sum[n] += pft_weighted_ba_ti * weight;
            cum_weight[n] += weight;
        }
    }

    for (int n = 0; n < Nout; n++) {
        if (cum_weight[n] > 1e-15) {
            float cons_avg_val = cum_sum[n] / cum_weight[n];
            out[get_index_2d(n, land_i, out_shape_2d)] = cons_avg_val;
        } else {
            out[get_index_2d(n, land_i, out_shape_2d)] = 0.0f;
        }
    }
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
