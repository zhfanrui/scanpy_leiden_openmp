#include "adapter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "Graph.hxx"
#include "leiden.hxx"
#include "properties.hxx"

#if SLO_HAS_OPENMP
#include <omp.h>
#endif

namespace slo {

namespace {

std::vector<std::int64_t> compact_labels(const std::vector<std::int64_t>& labels) {
  std::unordered_map<std::int64_t, std::int64_t> remap;
  remap.reserve(labels.size());
  std::vector<std::int64_t> out(labels.size());
  std::int64_t next_label = 0;

  for (std::size_t i = 0; i < labels.size(); ++i) {
    const auto key = labels[i];
    const auto it = remap.find(key);
    if (it == remap.end()) {
      remap[key] = next_label;
      out[i] = next_label;
      ++next_label;
    } else {
      out[i] = it->second;
    }
  }

  return out;
}

}  // namespace

LeidenOutput run_leiden(const LeidenInput& input) {
  using VertexId = std::uint32_t;
  using Weight = double;
  using Offset = std::size_t;

  if (input.n_nodes < 0) {
    throw std::invalid_argument("n_nodes must be non-negative");
  }
  if (static_cast<std::uint64_t>(input.n_nodes) > std::numeric_limits<VertexId>::max()) {
    throw std::invalid_argument("n_nodes exceeds uint32_t capacity of upstream implementation");
  }
  if (static_cast<std::size_t>(input.n_nodes) + 1 != input.indptr.size()) {
    throw std::invalid_argument("indptr must have length n_nodes + 1");
  }
  if (input.indptr.empty()) {
    throw std::invalid_argument("indptr must be non-empty");
  }
  if (input.data.size() != input.indices.size()) {
    throw std::invalid_argument("data/indices shape mismatch");
  }
  if (input.indptr.front() != 0) {
    throw std::invalid_argument("indptr must start at 0");
  }
  for (std::size_t i = 1; i < input.indptr.size(); ++i) {
    if (input.indptr[i] < input.indptr[i - 1]) {
      throw std::invalid_argument("indptr must be non-decreasing");
    }
  }
  if (static_cast<std::size_t>(input.indptr.back()) != input.indices.size()) {
    throw std::invalid_argument("indptr/indices shape mismatch");
  }
  for (const auto idx : input.indices) {
    if (idx < 0 || idx >= input.n_nodes) {
      throw std::invalid_argument("indices contains out-of-range node id");
    }
  }

#if SLO_HAS_OPENMP
  if (input.n_threads > 0) {
    omp_set_num_threads(static_cast<int>(input.n_threads));
  }
#endif

  const auto n_nodes = static_cast<std::size_t>(input.n_nodes);
  const auto n_edges = input.indices.size();

  DiGraphCsr<VertexId, None, Weight, Offset> graph(n_nodes, n_edges);

  for (std::size_t u = 0; u < n_nodes; ++u) {
    const auto begin = static_cast<std::size_t>(input.indptr[u]);
    const auto end = static_cast<std::size_t>(input.indptr[u + 1]);
    if (end - begin > static_cast<std::size_t>(std::numeric_limits<VertexId>::max())) {
      throw std::invalid_argument("vertex degree exceeds uint32_t capacity");
    }
    graph.offsets[u] = begin;
    graph.degrees[u] = static_cast<VertexId>(end - begin);
    graph.values[u] = None();
  }
  graph.offsets[n_nodes] = n_edges;

  for (std::size_t e = 0; e < n_edges; ++e) {
    graph.edgeKeys[e] = static_cast<VertexId>(input.indices[e]);
    graph.edgeValues[e] = input.weighted ? input.data[e] : 1.0;
  }

  LeidenOptions options;
  options.repeat = 1;
  options.resolution = input.resolution;
  if (input.n_iterations > 0) {
    options.maxIterations = static_cast<int>(input.n_iterations);
  }

#if SLO_HAS_OPENMP
  const auto result = leidenStaticOmp(graph, options);
#else
  const auto result = leidenStatic(graph, options);
#endif

  std::vector<std::int64_t> labels(result.membership.size());
  for (std::size_t i = 0; i < result.membership.size(); ++i) {
    labels[i] = static_cast<std::int64_t>(result.membership[i]);
  }
  auto compact = compact_labels(labels);

  const double total_edge_weight =
#if SLO_HAS_OPENMP
      edgeWeightOmp(graph);
#else
      edgeWeight(graph);
#endif
  const double M = total_edge_weight / 2.0;
  double modularity = 0.0;
  if (M > 0.0) {
    const auto fc = [&](auto u) { return result.membership[u]; };
    modularity =
#if SLO_HAS_OPENMP
        modularityByOmp(graph, fc, M, options.resolution);
#else
        modularityBy(graph, fc, M, options.resolution);
#endif
  }

  LeidenOutput out;
  out.labels = std::move(compact);
  out.quality = modularity;
  out.iterations_done = static_cast<std::int64_t>(result.iterations);
  return out;
}

}  // namespace slo
