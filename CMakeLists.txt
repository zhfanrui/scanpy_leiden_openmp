cmake_minimum_required(VERSION 3.18)
project(scanpy_leiden_openmp LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(cpp)
