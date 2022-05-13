#include "inferno.hpp"
#include "infernoAvg.hpp"
#include "phase.hpp"
#include "mpd.hpp"
#include "flam.hpp"
#include "consAvg.hpp"
#include "consAvgNoMask.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <stdexcept>


namespace py = pybind11;

using pyArray = py::array_t<float, py::array::c_style>;


float nme(pyArray obs, pyArray pred) {
    py::buffer_info obsInfo = obs.request();
    py::buffer_info predInfo = pred.request();
    if (obsInfo.shape != predInfo.shape)
        throw std::runtime_error("Incompatible buffer shapes!");

    int i;
    int N = obsInfo.size;
    float* obsPtr = (float*)obsInfo.ptr;
    float* predPtr = (float*)predInfo.ptr;

    // Release Python GIL after dealing with the input Python (NumPy)
    // arrays and getting the underlying data pointer.
    py::gil_scoped_release release;

    float meanObs = 0.0f;
    float denom = 0.0f;
    float meanAbsDiff = 0.0f;

    for (i = 0; i < N; i++) {
        meanObs += obsPtr[i];
    }
    meanObs /= N;

    for (i = 0; i < N; i++) {
        float obsVal;
        obsVal = obsPtr[i];
        denom += abs(obsVal - meanObs);
        meanAbsDiff += abs(predPtr[i] - obsVal);
    }
    // No need to divide both by N since they will be divided later on anyway.
    // denom /= N;
    // meanAbsDiff /= N;

    py::gil_scoped_acquire acquire;

    return meanAbsDiff / denom;
}


void cons_avg_no_mask(pyArray weights, pyArray data, pyArray out) {
    py::buffer_info weightsInfo = weights.request();
    py::buffer_info dataInfo = data.request();
    py::buffer_info outInfo = out.request();
    if (weightsInfo.ndim != 2)
        throw std::runtime_error("Expected weights array with 2 dimensions!");
    if (dataInfo.ndim != 2)
        throw std::runtime_error("Expected data array with 2 dimensions!");
    if (outInfo.ndim != 2)
        throw std::runtime_error("Expected out array with 2 dimensions!");

    int M = weightsInfo.shape[0];
    int N = weightsInfo.shape[1];
    int L = dataInfo.shape[1];

    if (dataInfo.shape[0] != M)
        throw std::runtime_error("Expected data dim 0 = N!");
    if (outInfo.shape[0] != N)
        throw std::runtime_error("Expected out dim 0 = N!");
    if (outInfo.shape[1] != L)
        throw std::runtime_error("Expected out dim 1 = L!");

    float* weightsPtr = (float*)weightsInfo.ptr;
    float* dataPtr = (float*)dataInfo.ptr;
    float* outPtr = (float*)outInfo.ptr;

    // Release Python GIL after dealing with the input Python (NumPy)
    // arrays and getting the underlying data pointer.
    py::gil_scoped_release release;

    float* cumWeights = new float[N];

    // Calculate cumulative weights.
    for (int n = 0; n < N; n++) {
        float cumWeight = 0.0f;

        for (int m = 0; m < M; m++) {
            cumWeight += weightsPtr[m * N + n];
        }

        cumWeights[n] = cumWeight;
    }

    for (int li = 0; li < L; li++) {
        for (int n = 0; n < N; n++) {
            float cumVal = 0.0f;

            for (int m = 0; m < M; m++) {
                float weight = weightsPtr[m * N + n];

                if (weight < 1e-9) continue;

                float value = dataPtr[m * L + li];
                cumVal += weight * value;
            }

            outPtr[n * L + li] = cumVal / cumWeights[n];
        }
    }

    delete [] cumWeights;

    py::gil_scoped_acquire acquire;
}


PYBIND11_MODULE(py_gpu_inferno, m) {
    m.doc() = "GPU INFERNO";
    py::class_<GPUCompute>(m, "GPUCompute")
        .def(py::init<>())
        .def("set_data", &GPUCompute::set_data, "Set input data.",
            py::arg("_ignitionMethod"),
            py::arg("_flammabilityMethod"),
            py::arg("_drynessMethod"),
            py::arg("_fuelBuildUpMethod"),
            py::arg("_includeTemperature"),
            py::arg("_Nt"),
            py::arg("t1p5m_tile"),
            py::arg("q1p5m_tile"),
            py::arg("pstar"),
            py::arg("sthu_soilt_single"),
            py::arg("frac"),
            py::arg("c_soil_dpm_gb"),
            py::arg("c_soil_rpm_gb"),
            py::arg("canht"),
            py::arg("ls_rain"),
            py::arg("con_rain"),
            py::arg("pop_den"),
            py::arg("flash_rate"),
            py::arg("fuel_build_up"),
            py::arg("fapar_diag_pft"),
            py::arg("grouped_dry_bal"),
            py::arg("litter_pool"),
            py::arg("dry_days"),
            py::arg("checks_failed")
        )
        .def("set_params", &GPUCompute::set_params, "Set input parameters.",
            py::arg("fapar_factor"),
            py::arg("fapar_centre"),
            py::arg("fapar_shape"),
            py::arg("fuel_build_up_factor"),
            py::arg("fuel_build_up_centre"),
            py::arg("fuel_build_up_shape"),
            py::arg("temperature_factor"),
            py::arg("temperature_centre"),
            py::arg("temperature_shape"),
            py::arg("dry_day_factor"),
            py::arg("dry_day_centre"),
            py::arg("dry_day_shape"),
            py::arg("dry_bal_factor"),
            py::arg("dry_bal_centre"),
            py::arg("dry_bal_shape"),
            py::arg("litter_pool_factor"),
            py::arg("litter_pool_centre"),
            py::arg("litter_pool_shape"),
            py::arg("fapar_weight"),
            py::arg("dryness_weight"),
            py::arg("temperature_weight"),
            py::arg("fuel_weight")
        )
        .def("run", &GPUCompute::run, "Run kernel with defined data and parameters.",
            py::arg("out")
        )
        .def("release", &GPUCompute::release, "Release autorelease pool.")
        .def("get_checks_failed_mask", &GPUCompute::get_checks_failed_mask, "Get checks failed array")
        .def("get_diagnostics", &GPUCompute::get_diagnostics, "Get diagnostic arrays");

    m.def("calculate_phase", &calculate_phase, "Calculate phase", py::arg("x"));

    py::class_<GPUCalculatePhase>(m, "GPUCalculatePhase")
        .def(py::init<int>())
        .def("run", &GPUCalculatePhase::run, "Run calculate_phase.",
                py::arg("x")
            );

    py::class_<GPUCalculateMPD>(m, "GPUCalculateMPD")
        .def(py::init<int>())
        .def("run", &GPUCalculateMPD::run, "Run calculate_mpd.",
                py::arg("obs"),
                py::arg("pred")
            );

    py::class_<GPUConsAvg>(m, "GPUConsAvg")
        .def(py::init<int, pyArray>())
        .def("run", &GPUConsAvg::run, "Run conservative averaging.",
                py::arg("inData"),
                py::arg("inMask")
            );

    py::class_<GPUConsAvgNoMask>(m, "GPUConsAvgNoMask")
        .def(py::init<int, pyArray>())
        .def("run", &GPUConsAvgNoMask::run, "Run conservative averaging.",
                py::arg("inData")
            );

    m.def("nme", &nme, "Calculate NME error", py::arg("obs"), py::arg("pred"));

    m.def(
        "cons_avg_no_mask",
        &cons_avg_no_mask,
        "Calculate conservative monthly averages without masks.",
        py::arg("weights"),
        py::arg("data"),
        py::arg("out")
    );

    py::class_<GPUInfernoAvg>(m, "GPUInfernoAvg")
        .def(py::init<int, pyArray>())
        .def("run", &GPUInfernoAvg::run, "Run INFERNO with conservative averaging.",
            py::arg("out")
        )
        .def("set_data", &GPUInfernoAvg::set_data, "Set input data.",
            py::arg("_ignitionMethod"),
            py::arg("_flammabilityMethod"),
            py::arg("_drynessMethod"),
            py::arg("_fuelBuildUpMethod"),
            py::arg("_includeTemperature"),
            py::arg("_Nt"),
            py::arg("t1p5m_tile"),
            py::arg("q1p5m_tile"),
            py::arg("pstar"),
            py::arg("sthu_soilt_single"),
            py::arg("frac"),
            py::arg("c_soil_dpm_gb"),
            py::arg("c_soil_rpm_gb"),
            py::arg("canht"),
            py::arg("ls_rain"),
            py::arg("con_rain"),
            py::arg("pop_den"),
            py::arg("flash_rate"),
            py::arg("fuel_build_up"),
            py::arg("fapar_diag_pft"),
            py::arg("grouped_dry_bal"),
            py::arg("litter_pool"),
            py::arg("dry_days"),
            py::arg("checks_failed")
        )
        .def("set_params", &GPUInfernoAvg::set_params, "Set input parameters.",
            py::arg("fapar_factor"),
            py::arg("fapar_centre"),
            py::arg("fapar_shape"),
            py::arg("fuel_build_up_factor"),
            py::arg("fuel_build_up_centre"),
            py::arg("fuel_build_up_shape"),
            py::arg("temperature_factor"),
            py::arg("temperature_centre"),
            py::arg("temperature_shape"),
            py::arg("dry_day_factor"),
            py::arg("dry_day_centre"),
            py::arg("dry_day_shape"),
            py::arg("dry_bal_factor"),
            py::arg("dry_bal_centre"),
            py::arg("dry_bal_shape"),
            py::arg("litter_pool_factor"),
            py::arg("litter_pool_centre"),
            py::arg("litter_pool_shape"),
            py::arg("fapar_weight"),
            py::arg("dryness_weight"),
            py::arg("temperature_weight"),
            py::arg("fuel_weight")
        )
        .def("release", &GPUInfernoAvg::release, "Release autorelease pool.");

    py::class_<GPUFlam2>(m, "GPUFlam2")
        .def(py::init<>())
        .def("run", &GPUFlam2::run, "Run flammability (only flam 2)",
            py::arg("temp_l"),
            py::arg("fuel_build_up"),
            py::arg("fapar"),
            py::arg("dry_days"),
            py::arg("dryness_method"),
            py::arg("fuel_build_up_method"),
            py::arg("fapar_factor"),
            py::arg("fapar_centre"),
            py::arg("fapar_shape"),
            py::arg("fuel_build_up_factor"),
            py::arg("fuel_build_up_centre"),
            py::arg("fuel_build_up_shape"),
            py::arg("temperature_factor"),
            py::arg("temperature_centre"),
            py::arg("temperature_shape"),
            py::arg("dry_day_factor"),
            py::arg("dry_day_centre"),
            py::arg("dry_day_shape"),
            py::arg("dry_bal"),
            py::arg("dry_bal_factor"),
            py::arg("dry_bal_centre"),
            py::arg("dry_bal_shape"),
            py::arg("litter_pool"),
            py::arg("litter_pool_factor"),
            py::arg("litter_pool_centre"),
            py::arg("litter_pool_shape"),
            py::arg("include_temperature"),
            py::arg("fapar_weight"),
            py::arg("dryness_weight"),
            py::arg("temperature_weight"),
            py::arg("fuel_weight")
        )
        .def("release", &GPUFlam2::release, "Release");
}
