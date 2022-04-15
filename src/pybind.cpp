#include "inferno.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(py_gpu_inferno, m) {
    m.doc() = "GPU INFERNO";
    m.def("test", &use_metal, "A simple test function that uses metal");
    py::class_<GPUCompute>(m, "GPUCompute")
        .def(py::init<>())
        .def("run", &GPUCompute::run, "Run a metal example", py::arg("x") = 0.0f);
}
