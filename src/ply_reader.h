#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

struct VertexData {
    uint32_t count = 0;
    std::vector<float> x, y, z;
    std::vector<float> opacity;
    std::vector<float> scale_0, scale_1, scale_2;
    std::vector<float> rot_0, rot_1, rot_2, rot_3;
    std::vector<float> f_dc_0, f_dc_1, f_dc_2;
    
    // Interleaved for decode/write
    std::vector<float> rot;
    std::vector<float> scale;
    std::map<std::string, std::vector<float>> f_dc;
    
    // f_rest channels, up to 45
    std::map<std::string, std::vector<float>> f_rest;
    
    const std::vector<float>& get_f_rest(int idx) const {
        static const std::vector<float> empty;
        auto it = f_rest.find("f_rest_" + std::to_string(idx));
        if (it == f_rest.end()) return empty;
        return it->second;
    }

    const std::vector<float>& get_scale(int idx) const {
        if (idx == 0) return scale_0;
        if (idx == 1) return scale_1;
        if (idx == 2) return scale_2;
        static const std::vector<float> empty;
        return empty;
    }

    // Orders splats for DPCM. sort_mode: 0 = Morton (Z-order curve), 1 = greedy
    // nearest-neighbour path. refine_window > 1 then runs windowed 2-opt passes to
    // shorten the consecutive path (smaller DPCM deltas); 0 = base order only.
    // metric: 0 = L1 distance objective, 1 = per-axis bit-length (~position coder).
    void sort_morton(int refine_window = 0, int metric = 0, int sort_mode = 0);
};

std::unique_ptr<VertexData> read_ply(const std::string& filepath);
