#include "sh_vq.h"
#include <cstdlib>
#include <random>
#include <limits>
#include <cmath>
#include <thread>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cstring>

// Optimized distance calculation for interleaved data.
// `w` is either null (plain L2) or a per-dim diagonal weight vector.
static inline float compute_dist_sq(const float* a, const float* b, int dims, const float* w) {
    float dist = 0.0f;
    if (w) {
        for (int d = 0; d < dims; ++d) {
            float diff = a[d] - b[d];
            dist += w[d] * diff * diff;
        }
    } else {
        for (int d = 0; d < dims; ++d) {
            float diff = a[d] - b[d];
            dist += diff * diff;
        }
    }
    return dist;
}

std::vector<float> train_vq_optimized(const std::vector<float>& interleaved_features, int num_points, int dims, int num_clusters, int max_iters, const float* weights = nullptr) {
    if (interleaved_features.empty() || num_points == 0) return {};

    num_clusters = std::min(num_clusters, num_points);
    std::vector<float> codebook(num_clusters * dims);

    // Cap the training subset at 35000 samples by default.
    // BEK_VQ_SAMPLE overrides this cap with a positive integer.
    int sample_cap = 35000;
    if (const char* e = std::getenv("BEK_VQ_SAMPLE")) { int v = std::atoi(e); if (v > 0) sample_cap = v; }
    int sample_size = std::min(sample_cap, std::max(5000, static_cast<int>(num_points * 0.10)));
    sample_size = std::min(sample_size, num_points);
    num_clusters = std::min(num_clusters, sample_size);

    std::vector<int> subset_indices(num_points);
    std::iota(subset_indices.begin(), subset_indices.end(), 0);
    std::mt19937 rng(0);
    for (int i = 0; i < sample_size; ++i) {
        std::uniform_int_distribution<int> dist(i, num_points - 1);
        int j = dist(rng);
        std::swap(subset_indices[i], subset_indices[j]);
    }

    std::vector<float> subset(sample_size * dims);
    for (int i = 0; i < sample_size; ++i) {
        std::memcpy(&subset[i * dims], &interleaved_features[subset_indices[i] * dims], dims * sizeof(float));
    }

    for (int i = 0; i < num_clusters; ++i) {
        std::memcpy(&codebook[i * dims], &subset[i * dims], dims * sizeof(float));
    }

    std::vector<int> subset_labels(sample_size, 0);
    std::vector<int> counts(num_clusters, 0);
    std::vector<float> new_codebook(num_clusters * dims, 0.0f);

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<std::thread> threads;
        int chunk = (sample_size + num_threads - 1) / num_threads;

        auto worker = [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                const float* pt = &subset[i * dims];
                for (int c = 0; c < num_clusters; ++c) {
                    float dist = compute_dist_sq(pt, &codebook[c * dims], dims, weights);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                subset_labels[i] = best_c;
            }
        };

        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk;
            int end = std::min(start + chunk, sample_size);
            if (start < end) threads.emplace_back(worker, start, end);
        }
        for (auto& th : threads) th.join();

        std::fill(counts.begin(), counts.end(), 0);
        std::fill(new_codebook.begin(), new_codebook.end(), 0.0f);

        for (int i = 0; i < sample_size; ++i) {
            int c = subset_labels[i];
            counts[c]++;
            for (int d = 0; d < dims; ++d) {
                new_codebook[c * dims + d] += subset[i * dims + d];
            }
        }

        for (int c = 0; c < num_clusters; ++c) {
            if (counts[c] > 0) {
                for (int d = 0; d < dims; ++d) {
                    codebook[c * dims + d] = new_codebook[c * dims + d] / counts[c];
                }
            }
        }
    }
    return codebook;
}

std::vector<uint16_t> assign_vq_optimized(const std::vector<float>& interleaved_features, int num_points, int dims, const std::vector<float>& codebook, const float* weights = nullptr) {
    if (interleaved_features.empty() || codebook.empty()) return {};
    int num_clusters = codebook.size() / dims;

    std::vector<uint16_t> indices(num_points);
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    std::vector<std::thread> threads;
    int chunk = (num_points + num_threads - 1) / num_threads;
    auto final_worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            const float* pt = &interleaved_features[i * dims];
            for (int c = 0; c < num_clusters; ++c) {
                float dist = compute_dist_sq(pt, &codebook[c * dims], dims, weights);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_c = c;
                }
            }
            indices[i] = static_cast<uint16_t>(best_c);
        }
    };

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, num_points);
        if (start < end) threads.emplace_back(final_worker, start, end);
    }
    for (auto& th : threads) th.join();

    return indices;
}

std::vector<uint16_t> assign_vq(const std::vector<std::vector<float>>& features, const std::vector<float>& codebook) {
    if (features.empty()) return {};
    int num_points = features[0].size();
    int dims = features.size();
    
    // Transpose to interleaved for optimized call
    std::vector<float> interleaved(num_points * dims);
    for (int d = 0; d < dims; ++d) {
        for (int i = 0; i < num_points; ++i) {
            interleaved[i * dims + d] = features[d][i];
        }
    }
    return assign_vq_optimized(interleaved, num_points, dims, codebook);
}

std::vector<float> train_vq(const std::vector<std::vector<float>>& features, int num_clusters, int max_iters) {
    if (features.empty()) return {};
    int num_points = features[0].size();
    int dims = features.size();
    
    std::vector<float> interleaved(num_points * dims);
    for (int d = 0; d < dims; ++d) {
        for (int i = 0; i < num_points; ++i) {
            interleaved[i * dims + d] = features[d][i];
        }
    }
    return train_vq_optimized(interleaved, num_points, dims, num_clusters, max_iters);
}

VQResult compute_vq(const std::vector<std::vector<float>>& features, int num_clusters, int max_iters) {
    if (features.empty()) return {};
    int num_points = features[0].size();
    int dims = features.size();

    // Transpose to interleaved for speed
    std::vector<float> interleaved(num_points * dims);
    for (int d = 0; d < dims; ++d) {
        for (int i = 0; i < num_points; ++i) {
            interleaved[i * dims + d] = features[d][i];
        }
    }

    VQResult result;
    result.codebook = train_vq_optimized(interleaved, num_points, dims, num_clusters, max_iters);
    result.indices = assign_vq_optimized(interleaved, num_points, dims, result.codebook);
    return result;
}

void reorder_codebook_chain(std::vector<float>& codebook, int dims, const float* weights) {
    int K = codebook.empty() ? 0 : static_cast<int>(codebook.size() / dims);
    if (K <= 2) return;

    // Start from the centroid nearest the codebook mean, then walk a greedy
    // nearest-neighbour chain. Codeword labels are arbitrary, so this costs
    // nothing: the centroids are unchanged, only their numbering, and the
    // assignment that follows hands out the new labels directly.
    std::vector<float> mean(dims, 0.0f);
    for (int c = 0; c < K; ++c)
        for (int d = 0; d < dims; ++d) mean[d] += codebook[static_cast<size_t>(c) * dims + d];
    for (int d = 0; d < dims; ++d) mean[d] /= K;

    std::vector<char> used(K, 0);
    std::vector<int> order;
    order.reserve(K);

    int cur = 0;
    float best = std::numeric_limits<float>::max();
    for (int c = 0; c < K; ++c) {
        float dist = compute_dist_sq(mean.data(), &codebook[static_cast<size_t>(c) * dims], dims, weights);
        if (dist < best) { best = dist; cur = c; }
    }

    for (int step = 0; step < K; ++step) {
        used[cur] = 1;
        order.push_back(cur);
        int next = -1;
        float best_d = std::numeric_limits<float>::max();
        const float* pt = &codebook[static_cast<size_t>(cur) * dims];
        for (int c = 0; c < K; ++c) {
            if (used[c]) continue;
            float dist = compute_dist_sq(pt, &codebook[static_cast<size_t>(c) * dims], dims, weights);
            if (dist < best_d) { best_d = dist; next = c; }
        }
        if (next < 0) break;
        cur = next;
    }

    std::vector<float> reordered(codebook.size());
    for (int slot = 0; slot < static_cast<int>(order.size()); ++slot)
        std::memcpy(&reordered[static_cast<size_t>(slot) * dims],
                    &codebook[static_cast<size_t>(order[slot]) * dims], dims * sizeof(float));
    codebook.swap(reordered);
}

RVQResult compute_rvq(const std::vector<std::vector<float>>& features, int num_stages, int clusters_per_stage, int max_iters, const std::vector<float>& weights, bool cb_chain_order) {
    RVQResult result;
    if (features.empty() || features[0].empty()) return result;

    int num_points = features[0].size();
    int dims = features.size();

    // Only honour weights if they match the dim count; otherwise fall back to plain L2.
    const float* w = (static_cast<int>(weights.size()) == dims) ? weights.data() : nullptr;

    // 1. Transpose to interleaved layout (Cache Locality)
    std::vector<float> current_interleaved(num_points * dims);
    for (int d = 0; d < dims; ++d) {
        for (int i = 0; i < num_points; ++i) {
            current_interleaved[i * dims + d] = features[d][i];
        }
    }

    for (int s = 0; s < num_stages; ++s) {
        std::cout << "Training RVQ Stage " << (s+1) << "/" << num_stages << " (Interleaved)..." << std::endl;

        std::vector<float> codebook = train_vq_optimized(current_interleaved, num_points, dims, clusters_per_stage, max_iters, w);
        if (cb_chain_order) reorder_codebook_chain(codebook, dims, w);
        std::vector<uint16_t> indices = assign_vq_optimized(current_interleaved, num_points, dims, codebook, w);

        result.codebooks.push_back(codebook);
        result.indices.push_back(indices);

        // 2. Update residuals in interleaved layout
        for (int i = 0; i < num_points; ++i) {
            int c = indices[i];
            float* pt = &current_interleaved[i * dims];
            const float* cluster = &codebook[c * dims];
            for (int d = 0; d < dims; ++d) {
                pt[d] -= cluster[d];
            }
        }
    }

    return result;
}


QuantizedCodebook quantize_scalar_lloyd(const std::vector<float>& data, int num_levels, int max_iters) {
    QuantizedCodebook res;
    if (data.empty()) return res;
    
    int n = data.size();
    num_levels = std::min(num_levels, n);
    
    res.levels.resize(num_levels);
    float dmin = data[0], dmax = data[0];
    for (float v : data) {
        dmin = std::min(dmin, v);
        dmax = std::max(dmax, v);
    }
    
    if (num_levels > 1) {
        for (int i = 0; i < num_levels; ++i) {
            res.levels[i] = dmin + (dmax - dmin) * (static_cast<float>(i) / (num_levels - 1));
        }
    } else {
        res.levels[0] = (dmin + dmax) / 2.0f;
    }
    
    std::vector<int> labels(n);
    std::vector<float> new_levels(num_levels);
    std::vector<int> counts(num_levels);
    
    for (int iter = 0; iter < max_iters; ++iter) {
        for (int i = 0; i < n; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_l = 0;
            for (int l = 0; l < num_levels; ++l) {
                float dist = std::abs(data[i] - res.levels[l]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_l = l;
                }
            }
            labels[i] = best_l;
        }
        
        std::fill(new_levels.begin(), new_levels.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < n; ++i) {
            new_levels[labels[i]] += data[i];
            counts[labels[i]]++;
        }
        
        bool changed = false;
        for (int l = 0; l < num_levels; ++l) {
            if (counts[l] > 0) {
                float nl = new_levels[l] / counts[l];
                if (std::abs(res.levels[l] - nl) > 1e-6) {
                    changed = true;
                    res.levels[l] = nl;
                }
            }
        }
        if (!changed) break;
    }
    
    res.indices.resize(n);
    for (int i = 0; i < n; ++i) res.indices[i] = static_cast<uint8_t>(labels[i]);

    return res;
}

RVQQuantized pack_rvq_pooled(const RVQResult& res, int dims) {
    RVQQuantized q;
    q.indices = res.indices;
    if (res.codebooks.empty()) return q;

    std::vector<float> pool;
    for (const auto& cb : res.codebooks) pool.insert(pool.end(), cb.begin(), cb.end());
    QuantizedCodebook qcb = quantize_scalar_lloyd(pool, 256, 20);

    size_t off = 0;
    for (const auto& cb : res.codebooks) {
        std::vector<uint8_t> cbi(qcb.indices.begin() + off, qcb.indices.begin() + off + cb.size());
        std::vector<float> deq(cb.size());
        for (size_t e = 0; e < cb.size(); ++e) deq[e] = qcb.levels[cbi[e]];
        q.codebooks.push_back(std::move(deq));
        q.cb_indices.push_back(std::move(cbi));
        q.levels.push_back(qcb.levels);
        off += cb.size();
    }
    (void)dims;
    return q;
}
