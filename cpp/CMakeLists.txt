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

if(APPLE)
  set(_slo_rpaths "")
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib")
    list(APPEND _slo_rpaths "$ENV{CONDA_PREFIX}/lib")
  endif()
  if(EXISTS "/opt/homebrew/opt/libomp/lib")
    list(APPEND _slo_rpaths "/opt/homebrew/opt/libomp/lib")
  endif()
  if(EXISTS "/usr/local/opt/libomp/lib")
    list(APPEND _slo_rpaths "/usr/local/opt/libomp/lib")
  endif()
  if(_slo_rpaths)
    list(REMOVE_DUPLICATES _slo_rpaths)
    set_target_properties(_core PROPERTIES
      BUILD_RPATH "${_slo_rpaths}"
      INSTALL_RPATH "${_slo_rpaths}"
    )
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION scanpy_leiden_openmp)
