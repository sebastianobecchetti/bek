#pragma once
#include <vector>
#include <cstdint>

struct VQResult {
    std::vector<float> codebook;
    std::vector<uint16_t> indices;
};

VQResult compute_vq(const std::vector<std::vector<float>>& features, int num_clusters, int max_iters = 15);
std::vector<float> train_vq(const std::vector<std::vector<float>>& features, int num_clusters, int max_iters = 15);
std::vector<uint16_t> assign_vq(const std::vector<std::vector<float>>& features, const std::vector<float>& codebook);

struct RVQResult {
    std::vector<std::vector<float>> codebooks; // Multiple stages
    std::vector<std::vector<uint16_t>> indices; // Indices for each stage
};

// Optional per-dimension weights apply a diagonal Lloyd distance metric
// to assignment. Centroids remain ordinary means in the original space.
// Empty weights select unweighted L2. cb_chain_order enables codebook
// relabelling before index assignment.
RVQResult compute_rvq(const std::vector<std::vector<float>>& features, int num_stages, int clusters_per_stage, int max_iters = 15, const std::vector<float>& weights = {}, bool cb_chain_order = true);

// Reorder a codebook along a greedy nearest-neighbour chain through its
// centroids. Call before assignment so indices use the resulting numbering.
// This aims to improve locality for index-plane entropy coding.
void reorder_codebook_chain(std::vector<float>& codebook, int dims, const float* weights);

struct QuantizedCodebook {
    std::vector<float> levels;    // 256 levels (Lloyd centers)
    std::vector<uint8_t> indices; // Quantized indices for the codebook entries
};

QuantizedCodebook quantize_scalar_lloyd(const std::vector<float>& data, int num_levels = 256, int max_iters = 20);

// An RVQ already paired with the scalar quantization of its codebook entries,
// i.e. exactly what the decoder will rebuild: codebooks[s][e] == levels[s][cb_indices[s][e]].
struct RVQQuantized {
    std::vector<std::vector<float>> codebooks;    // dequantized centroids (decoder view)
    std::vector<std::vector<uint16_t>> indices;   // per stage, per point
    std::vector<std::vector<float>> levels;       // per stage (pooled mode repeats the same 256)
    std::vector<std::vector<uint8_t>> cb_indices; // per stage, into levels[s]
};

// Pool stage codebooks and Lloyd-quantize entries to shared scalar levels.
// Indices remain those assigned against unquantized centroids.
RVQQuantized pack_rvq_pooled(const RVQResult& res, int dims);
