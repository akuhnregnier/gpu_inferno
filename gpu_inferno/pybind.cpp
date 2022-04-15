#include "inferno.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(py_gpu_inferno, m) {
    m.doc() = "GPU INFERNO";
    m.def("test", &use_metal, "A simple test function that uses metal");
}
