set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

pybind11_add_module(_core MODULE
  src/adapter.cpp
  src/bindings.cpp
)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/leiden_communities_openmp/upstream/inc
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_core PRIVATE SLO_HAS_OPENMP=1 OPENMP=1)
else()
  target_compile_definitions(_core PRIVATE SLO_HAS_OPENMP=0)
endif()

install(TARGETS _core LIBRARY DESTINATION scanpy_leiden_openmp)
