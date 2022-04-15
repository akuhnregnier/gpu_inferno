import inspect
import os
import sys
from pathlib import Path

import cmake_build_extension
import setuptools

setuptools.setup(
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="CMakePy",
            install_prefix="py_gpu_inferno",
            cmake_depends_on=["pybind11"],
            source_dir=str(Path(__file__).parent.absolute()),
            cmake_configure_options=[
                f"-DPython3_ROOT_DIR={Path(sys.prefix)}",
                "-DCALL_FROM_SETUP_PY:BOOL=ON",
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
            ]
        ),
    ],
    cmdclass=dict(
        build_ext=cmake_build_extension.BuildExtension,
    ),
)
