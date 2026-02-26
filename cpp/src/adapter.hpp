#pragma once

#include <cstdint>
#include <vector>

namespace slo {

struct LeidenInput {
  std::int64_t n_nodes;
  std::vector<std::int64_t> indptr;
  std::vector<std::int64_t> indices;
  std::vector<double> data;
  double resolution;
  std::int64_t seed;
  std::int64_t n_iterations;
  bool directed;
  bool weighted;
  std::int64_t n_threads;
};

struct LeidenOutput {
  std::vector<std::int64_t> labels;
  double quality;
  std::int64_t iterations_done;
};

LeidenOutput run_leiden(const LeidenInput& input);

}  // namespace slo
