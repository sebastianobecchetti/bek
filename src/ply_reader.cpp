#include "ply_reader.h"
#include <tinyply.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>

using namespace tinyply;

std::unique_ptr<VertexData> read_ply(const std::string& filepath) {
    std::ifstream filepath_stream(filepath, std::ios::binary);
    if (filepath_stream.fail()) {
        throw std::runtime_error("Failed to open PLY file: " + filepath);
    }

    PlyFile file;
    file.parse_header(filepath_stream);

    std::shared_ptr<PlyData> x, y, z;
    std::shared_ptr<PlyData> opacity;
    std::shared_ptr<PlyData> scale_0, scale_1, scale_2;
    std::shared_ptr<PlyData> rot_0, rot_1, rot_2, rot_3;
    std::shared_ptr<PlyData> f_dc_0, f_dc_1, f_dc_2;
    std::map<std::string, std::shared_ptr<PlyData>> f_rest;

    try { x = file.request_properties_from_element("vertex", { "x" }); } catch (const std::exception & e) { }
    try { y = file.request_properties_from_element("vertex", { "y" }); } catch (const std::exception & e) { }
    try { z = file.request_properties_from_element("vertex", { "z" }); } catch (const std::exception & e) { }
    try { opacity = file.request_properties_from_element("vertex", { "opacity" }); } catch (const std::exception & e) { }
    
    try { scale_0 = file.request_properties_from_element("vertex", { "scale_0" }); } catch (const std::exception & e) { }
    try { scale_1 = file.request_properties_from_element("vertex", { "scale_1" }); } catch (const std::exception & e) { }
    try { scale_2 = file.request_properties_from_element("vertex", { "scale_2" }); } catch (const std::exception & e) { }
    
    try { rot_0 = file.request_properties_from_element("vertex", { "rot_0" }); } catch (const std::exception & e) { }
    try { rot_1 = file.request_properties_from_element("vertex", { "rot_1" }); } catch (const std::exception & e) { }
    try { rot_2 = file.request_properties_from_element("vertex", { "rot_2" }); } catch (const std::exception & e) { }
    try { rot_3 = file.request_properties_from_element("vertex", { "rot_3" }); } catch (const std::exception & e) { }

    try { f_dc_0 = file.request_properties_from_element("vertex", { "f_dc_0" }); } catch (const std::exception & e) { }
    try { f_dc_1 = file.request_properties_from_element("vertex", { "f_dc_1" }); } catch (const std::exception & e) { }
    try { f_dc_2 = file.request_properties_from_element("vertex", { "f_dc_2" }); } catch (const std::exception & e) { }

    for (int i = 0; i < 45; ++i) {
        std::string name = "f_rest_" + std::to_string(i);
        try {
            f_rest[name] = file.request_properties_from_element("vertex", { name });
        } catch (const std::exception & e) { }
    }

    file.read(filepath_stream);

    auto extract_float = [](std::shared_ptr<PlyData> data) -> std::vector<float> {
        std::vector<float> vec;
        if (!data) return vec;
        vec.resize(data->count);
        if (data->t == Type::FLOAT32) {
            std::memcpy(vec.data(), data->buffer.get(), data->buffer.size_bytes());
        }
        return vec;
    };

    auto data = std::make_unique<VertexData>();
    if (x) data->count = x->count;

    data->x = extract_float(x);
    data->y = extract_float(y);
    data->z = extract_float(z);
    
    data->opacity = extract_float(opacity);
    data->scale_0 = extract_float(scale_0);
    data->scale_1 = extract_float(scale_1);
    data->scale_2 = extract_float(scale_2);
    
    data->rot_0 = extract_float(rot_0);
    data->rot_1 = extract_float(rot_1);
    data->rot_2 = extract_float(rot_2);
    data->rot_3 = extract_float(rot_3);
    
    data->f_dc_0 = extract_float(f_dc_0);
    data->f_dc_1 = extract_float(f_dc_1);
    data->f_dc_2 = extract_float(f_dc_2);
    
    for (int i = 0; i < 45; ++i) {
        std::string name = "f_rest_" + std::to_string(i);
        if (f_rest[name]) {
            data->f_rest[name] = extract_float(f_rest[name]);
        }
    }

    return data;
}

static uint64_t part1by2(uint64_t x) {
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

static uint64_t encode_morton(uint32_t x, uint32_t y, uint32_t z) {
    return part1by2(x) | part1by2(y) << 1 | part1by2(z) << 2;
}

// Number of significant bits of v (0 -> 0). The position coder's per-axis cost
// grows ~linearly with this (length prefix + direct mantissa bits), i.e. ~log2(d),
// so summing it over axes is a faithful proxy for what posmag actually pays.
static inline int bitlen32(uint32_t v) {
    int n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}

// Windowed 2-opt refinement of an ordering. `order[k]` is the original index at
// position k. For each position i, within a window of W following positions, find
// the j whose 2-opt move (reverse order[i+1..j], reconnecting as (i,j) and
// (i+1,j+1)) most shrinks the local path under `cost`, and apply the best improving
// move. `cost(a,b)` is the per-edge objective: either L1 distance, or the summed
// per-axis bit-length (what the position coder pays). A few passes settle the
// order; cost is O(passes * N * W).
// Return the number of passes run, capped at kPasses=4. Each accepted move
// strictly shortens the path; stop early when a pass makes no improvement.
template <typename Cost>
static int refine_2opt(std::vector<uint32_t>& order, int W, Cost cost) {
    const uint32_t n = static_cast<uint32_t>(order.size());
    if (n < 4 || W < 2) return 0;
    const int kPasses = 4;
    int pass = 0;
    for (; pass < kPasses; ++pass) {
        bool improved = false;
        for (uint32_t i = 0; i + 3 < n; ++i) {
            const uint32_t a = order[i];
            const uint32_t b = order[i + 1];
            const float dab = cost(a, b);
            uint32_t jmax = i + 1 + static_cast<uint32_t>(W);
            if (jmax > n - 2) jmax = n - 2; // need order[j+1] to exist
            float best_delta = -1e-6f;       // require a real improvement
            uint32_t best_j = 0;
            for (uint32_t j = i + 2; j <= jmax; ++j) {
                const uint32_t c = order[j];
                const uint32_t d = order[j + 1];
                // delta = new edges - old edges
                const float delta = (cost(a, c) + cost(b, d)) - (dab + cost(c, d));
                if (delta < best_delta) { best_delta = delta; best_j = j; }
            }
            if (best_j != 0) {
                std::reverse(order.begin() + i + 1, order.begin() + best_j + 1);
                improved = true;
            }
        }
        if (!improved) { ++pass; break; }
    }
    return pass;
}

// Greedy nearest-neighbour ordering over the 21-bit grid coords. From a start
// point it always steps to the nearest unvisited point, found via an expanding
// shell search on a spatial hash grid (visited points are physically removed from
// their cell). Unlike a fixed space-filling curve it adapts to the actual point
// distribution — on non-uniform clouds (splats on surfaces, empty space between)
// it follows the manifold instead of paying deltas to cross voids. A linear scan
// over the remainder backstops the rare case where the local shells empty out.
static std::vector<uint32_t> greedy_nn_order(const std::vector<uint32_t>& gx,
                                             const std::vector<uint32_t>& gy,
                                             const std::vector<uint32_t>& gz,
                                             uint32_t count) {
    std::vector<uint32_t> order;
    order.reserve(count);
    if (count <= 1) { if (count == 1) order.push_back(0); return order; }

    uint32_t mnx = gx[0], mxx = gx[0], mny = gy[0], mxy = gy[0], mnz = gz[0], mxz = gz[0];
    for (uint32_t i = 1; i < count; ++i) {
        mnx = std::min(mnx, gx[i]); mxx = std::max(mxx, gx[i]);
        mny = std::min(mny, gy[i]); mxy = std::max(mxy, gy[i]);
        mnz = std::min(mnz, gz[i]); mxz = std::max(mxz, gz[i]);
    }
    uint32_t cellsAxis = std::max<uint32_t>(1, (uint32_t)std::cbrt((double)count));
    uint64_t ext = std::max<uint64_t>({(uint64_t)(mxx - mnx), (uint64_t)(mxy - mny),
                                       (uint64_t)(mxz - mnz), 1});
    uint32_t cellsize = std::max<uint32_t>(1, (uint32_t)(ext / cellsAxis));
    auto cax = [&](uint32_t v, uint32_t mn) { return (v - mn) / cellsize; };
    uint32_t gdx = (mxx - mnx) / cellsize + 1;
    uint32_t gdy = (mxy - mny) / cellsize + 1;
    uint32_t gdz = (mxz - mnz) / cellsize + 1;
    auto key = [&](uint32_t cx, uint32_t cy, uint32_t cz) -> uint64_t {
        return ((uint64_t)cx * gdy + cy) * gdz + cz;
    };

    std::unordered_map<uint64_t, std::vector<uint32_t>> grid;
    grid.reserve(count);
    std::vector<uint32_t> cellPos(count);
    std::vector<uint64_t> cellKey(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t k = key(cax(gx[i], mnx), cax(gy[i], mny), cax(gz[i], mnz));
        auto& vec = grid[k];
        cellPos[i] = (uint32_t)vec.size();
        cellKey[i] = k;
        vec.push_back(i);
    }
    auto remove_grid = [&](uint32_t i) {
        auto& vec = grid[cellKey[i]];
        uint32_t p = cellPos[i], last = vec.back();
        vec[p] = last; cellPos[last] = p; vec.pop_back();
    };
    auto d2 = [&](uint32_t a, uint32_t b) -> uint64_t {
        int64_t dx = (int64_t)gx[a] - gx[b], dy = (int64_t)gy[a] - gy[b], dz = (int64_t)gz[a] - gz[b];
        return (uint64_t)(dx * dx + dy * dy + dz * dz);
    };

    std::vector<uint32_t> remaining(count), remPos(count);
    std::iota(remaining.begin(), remaining.end(), 0);
    std::iota(remPos.begin(), remPos.end(), 0);
    auto remove_rem = [&](uint32_t i) {
        uint32_t p = remPos[i], last = remaining.back();
        remaining[p] = last; remPos[last] = p; remaining.pop_back();
    };

    const uint32_t CAP = 40;
    const size_t LINEAR_THRESH = 1024;
    uint32_t gdmax = std::max({gdx, gdy, gdz});

    uint32_t cur = 0;
    remove_grid(cur); remove_rem(cur); order.push_back(cur);

    for (uint32_t step = 1; step < count; ++step) {
        uint32_t best = UINT32_MAX;
        uint64_t bestd = UINT64_MAX;
        if (remaining.size() <= LINEAR_THRESH) {
            for (uint32_t r : remaining) { uint64_t dd = d2(cur, r); if (dd < bestd) { bestd = dd; best = r; } }
        } else {
            uint32_t cx = cax(gx[cur], mnx), cy = cax(gy[cur], mny), cz = cax(gz[cur], mnz);
            int found_r = -1;
            for (uint32_t r = 0; r <= gdmax && r <= CAP; ++r) {
                int lo = -(int)r, hi = (int)r;
                for (int ax = lo; ax <= hi; ++ax)
                  for (int ay = lo; ay <= hi; ++ay)
                    for (int az = lo; az <= hi; ++az) {
                        if (std::max({std::abs(ax), std::abs(ay), std::abs(az)}) != (int)r) continue;
                        int64_t nx = (int64_t)cx + ax, ny = (int64_t)cy + ay, nz = (int64_t)cz + az;
                        if (nx < 0 || ny < 0 || nz < 0 || nx >= gdx || ny >= gdy || nz >= gdz) continue;
                        auto it = grid.find(key((uint32_t)nx, (uint32_t)ny, (uint32_t)nz));
                        if (it == grid.end() || it->second.empty()) continue;
                        for (uint32_t cand : it->second) { uint64_t dd = d2(cur, cand); if (dd < bestd) { bestd = dd; best = cand; } }
                    }
                if (best != UINT32_MAX) { if (found_r < 0) found_r = (int)r; if ((int)r >= found_r + 1) break; }
            }
            if (best == UINT32_MAX)
                for (uint32_t r : remaining) { uint64_t dd = d2(cur, r); if (dd < bestd) { bestd = dd; best = r; } }
        }
        remove_grid(best); remove_rem(best); order.push_back(best); cur = best;
    }
    return order;
}

void VertexData::sort_morton(int refine_window, int metric, int sort_mode) {
    if (count == 0) return;

    float min_x = x[0], max_x = x[0];
    float min_y = y[0], max_y = y[0];
    float min_z = z[0], max_z = z[0];
    
    for (uint32_t i = 1; i < count; ++i) {
        min_x = std::min(min_x, x[i]); max_x = std::max(max_x, x[i]);
        min_y = std::min(min_y, y[i]); max_y = std::max(max_y, y[i]);
        min_z = std::min(min_z, z[i]); max_z = std::max(max_z, z[i]);
    }
    
    float range_x = std::max(max_x - min_x, 1e-6f);
    float range_y = std::max(max_y - min_y, 1e-6f);
    float range_z = std::max(max_z - min_z, 1e-6f);
    
    std::vector<uint64_t> codes(count);
    std::vector<uint32_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    
    // 21 bits/axis (full 63-bit Morton code that part1by2 supports). A 10-bit
    // grid lumped many splats into one cell, so their 24-bit quantized positions
    // jumped around and inflated the DPCM deltas; a 21-bit grid orders them
    // almost monotonically, shrinking the residual entropy. Computed in double
    // to keep precision at the finer grid.
    const double MORTON_MAX = 2097151.0; // 2^21 - 1
    // Keep the 21-bit grid coords so the bits-metric 2-opt can score moves by the
    // per-axis magnitude the position coder pays for (not raw float distance).
    std::vector<uint32_t> gx(count), gy(count), gz(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t ix = static_cast<uint32_t>((x[i] - min_x) / range_x * MORTON_MAX);
        uint32_t iy = static_cast<uint32_t>((y[i] - min_y) / range_y * MORTON_MAX);
        uint32_t iz = static_cast<uint32_t>((z[i] - min_z) / range_z * MORTON_MAX);
        gx[i] = ix; gy[i] = iy; gz[i] = iz;
        codes[i] = encode_morton(ix, iy, iz);
    }
    
    if (sort_mode == 1) {
        indices = greedy_nn_order(gx, gy, gz, count);
    } else {
        std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
            return codes[a] < codes[b];
        });
    }

    if (refine_window > 1) {
        const bool use_bits = (metric == 1);
        auto adiff = [](uint32_t p, uint32_t q) { return p > q ? p - q : q - p; };
        auto cost_l1 = [&](uint32_t a, uint32_t b) -> float {
            return std::fabs(x[a] - x[b]) + std::fabs(y[a] - y[b]) + std::fabs(z[a] - z[b]);
        };
        auto cost_bits = [&](uint32_t a, uint32_t b) -> float {
            return float(bitlen32(adiff(gx[a], gx[b])) + bitlen32(adiff(gy[a], gy[b])) +
                         bitlen32(adiff(gz[a], gz[b])));
        };
        auto mean = [&](const std::vector<uint32_t>& ord, auto cost) {
            double s = 0.0;
            for (size_t k = 1; k < ord.size(); ++k) s += cost(ord[k - 1], ord[k]);
            return ord.size() > 1 ? s / (ord.size() - 1) : 0.0;
        };
        std::vector<uint32_t> ident(count);
        std::iota(ident.begin(), ident.end(), 0);
        double d_orig, d_morton, d_refined;
        int passes;
        if (use_bits) {
            d_orig = mean(ident, cost_bits);
            d_morton = mean(indices, cost_bits);
            passes = refine_2opt(indices, refine_window, cost_bits);
            d_refined = mean(indices, cost_bits);
        } else {
            d_orig = mean(ident, cost_l1);
            d_morton = mean(indices, cost_l1);
            passes = refine_2opt(indices, refine_window, cost_l1);
            d_refined = mean(indices, cost_l1);
        }
        const char* base = (sort_mode == 1) ? "nn" : "morton";
        std::cout << "[order] metric=" << (use_bits ? "bits" : "L1")
                  << " mean consecutive: original=" << d_orig
                  << " " << base << "=" << d_morton
                  << " " << base << "+2opt(w=" << refine_window << ")=" << d_refined
                  << "  (2opt " << 100.0 * (1.0 - d_refined / d_morton) << "% vs " << base
                  << ", " << passes << " passes)"
                  << std::endl;
    }

    auto reorder = [&](auto& vec) {
        if (vec.empty()) return;
        auto tmp = vec;
        for (uint32_t i = 0; i < count; ++i) {
            vec[i] = tmp[indices[i]];
        }
    };
    
    reorder(x); reorder(y); reorder(z);
    reorder(opacity);
    reorder(scale_0); reorder(scale_1); reorder(scale_2);
    reorder(rot_0); reorder(rot_1); reorder(rot_2); reorder(rot_3);
    reorder(f_dc_0); reorder(f_dc_1); reorder(f_dc_2);
    
    for (auto& pair : f_rest) {
        reorder(pair.second);
    }
}
