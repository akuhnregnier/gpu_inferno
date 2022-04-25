#include "inferno.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


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
            py::arg("dry_days")
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
            py::arg("litter_pool_shape")
        )
        .def("run", &GPUCompute::run, "Run kernel with defined data and parameters.")
        .def("release", &GPUCompute::release, "Release autorelease pool.");
}
