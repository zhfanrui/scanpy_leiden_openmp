#include "adapter.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

py::tuple run_leiden_csr(py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indptr,
                         py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indices,
                         py::array_t<double, py::array::c_style | py::array::forcecast> data,
                         std::int64_t n_nodes,
                         double resolution,
                         std::int64_t seed,
                         std::int64_t n_iterations,
                         bool directed,
                         bool weighted,
                         std::int64_t n_threads) {
  const auto indptr_buf = indptr.request();
  const auto indices_buf = indices.request();
  const auto data_buf = data.request();

  if (indptr_buf.ndim != 1 || indices_buf.ndim != 1 || data_buf.ndim != 1) {
    throw std::invalid_argument("indptr/indices/data must be 1D arrays");
  }

  std::vector<std::int64_t> indptr_v(indptr_buf.size);
  std::vector<std::int64_t> indices_v(indices_buf.size);
  std::vector<double> data_v(data_buf.size);

  std::memcpy(indptr_v.data(), indptr_buf.ptr, static_cast<std::size_t>(indptr_buf.size) * sizeof(std::int64_t));
  std::memcpy(indices_v.data(), indices_buf.ptr, static_cast<std::size_t>(indices_buf.size) * sizeof(std::int64_t));
  std::memcpy(data_v.data(), data_buf.ptr, static_cast<std::size_t>(data_buf.size) * sizeof(double));

  slo::LeidenInput input{};
  input.n_nodes = n_nodes;
  input.indptr = std::move(indptr_v);
  input.indices = std::move(indices_v);
  input.data = std::move(data_v);
  input.resolution = resolution;
  input.seed = seed;
  input.n_iterations = n_iterations;
  input.directed = directed;
  input.weighted = weighted;
  input.n_threads = n_threads;

  const auto out = slo::run_leiden(input);

  py::array_t<std::int64_t> labels(out.labels.size());
  auto labels_buf = labels.request();
  std::memcpy(labels_buf.ptr, out.labels.data(), out.labels.size() * sizeof(std::int64_t));

  return py::make_tuple(labels, out.quality, out.iterations_done);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "OpenMP Leiden backend for Scanpy";

  m.def("run_leiden_csr",
        &run_leiden_csr,
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("data"),
        py::arg("n_nodes"),
        py::arg("resolution") = 1.0,
        py::arg("seed") = 0,
        py::arg("n_iterations") = -1,
        py::arg("directed") = false,
        py::arg("weighted") = true,
        py::arg("n_threads") = -1);
}
