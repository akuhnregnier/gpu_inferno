cmake_minimum_required (VERSION 3.20)
project (py_gpu_inferno LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} src/pybind.cpp)

set(PYTHON_EXECUTABLE "/Users/alexander/.pyenv-arm64/versions/miniconda3-latest/envs/python-inferno/bin/python3")

set(
    INC_DIRS

    SYSTEM
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/metal-cpp
    ${CMAKE_SOURCE_DIR}/pybind11/include
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
