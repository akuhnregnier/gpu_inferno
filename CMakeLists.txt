cmake_minimum_required (VERSION 3.20)
project (py_gpu_inferno LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME}
    src/pybind.cpp
    src/common.hpp
    src/consAvg.hpp
    src/consAvgNoMask.hpp
    src/flam.hpp
    src/inferno.hpp
    src/infernoAvg.hpp
    src/infernoAvgScore.hpp
    src/loadConsAvgLibrary.hpp
    src/loadConsAvgNoMaskLibrary.hpp
    src/loadFlam2Library.hpp
    src/loadInfAvgLibrary.hpp
    src/loadInfAvgScoreLibrary.hpp
    src/loadLibrary.hpp
    src/loadLibraryCommon.hpp
    src/loadMPDLibrary.hpp
    src/loadPhaseLibrary.hpp
    src/loadSALibrary.hpp
    src/mpd.hpp
    src/phase.hpp
    src/sensitivity_analysis.hpp
)

set(PYTHON_EXECUTABLE "/Users/alexander/.pyenv-arm64/versions/miniconda3-latest/envs/python-inferno/bin/python3")

set(
    INC_DIRS

    SYSTEM
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/metal-cpp
    ${CMAKE_CURRENT_LIST_DIR}/pybind11/include
    "/Users/alexander/.pyenv-arm64/versions/miniconda3-latest/envs/python-inferno/include/python3.10"
)
set(
    LIBS

    ${APPLE_FWK_FOUNDATION}
    ${APPLE_FWK_QUARTZ_CORE}
    ${APPLE_FWK_METAL}
)

target_include_directories(${PROJECT_NAME} PRIVATE ${INC_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
