#include "quantization.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace {
    template<typename T>
    std::vector<uint8_t> pack_integer_effective_bytes_impl(const std::vector<T>& values, bool is_signed) {
        if (values.empty()) return {};
        
        int item_bits = sizeof(T) * 8;
        int64_t min_v = values[0];
        int64_t max_v = values[0];
        for (T v : values) {
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        
        int64_t levels = max_v - min_v + 1;
        int bits = 0;
        if (levels > 1) {
            bits = static_cast<int>(std::ceil(std::log2(levels)));
        }
        
        if (bits >= item_bits) {
            std::vector<uint8_t> out(values.size() * sizeof(T));
            std::memcpy(out.data(), values.data(), out.size());
            return out;
        }
        
        std::vector<uint8_t> bitplanes;
        if (bits > 0) {
            bitplanes.reserve(values.size() * bits);
            for (T v : values) {
                int64_t shifted = v - min_v;
                for (int b = 0; b < bits; ++b) {
                    bitplanes.push_back((shifted >> b) & 1);
                }
            }
        }
        
        std::vector<uint8_t> packed_payload;
        if (bits > 0) {
            packed_payload.resize((bitplanes.size() + 7) / 8, 0);
            for (size_t i = 0; i < bitplanes.size(); ++i) {
                if (bitplanes[i]) {
                    packed_payload[i / 8] |= (1 << (i % 8)); 
                }
            }
        }
        
        std::vector<uint8_t> header(20, 0);
        std::memcpy(&header[0], "BPK1", 4);
        header[4] = 1; 
        header[5] = is_signed ? 1 : 0;
        header[6] = item_bits;
        header[7] = bits;
        uint32_t count = values.size();
        std::memcpy(&header[8], &count, 4);
        std::memcpy(&header[12], &min_v, 8);
        
        std::vector<uint8_t> result;
        result.reserve(header.size() + packed_payload.size());
        result.insert(result.end(), header.begin(), header.end());
        result.insert(result.end(), packed_payload.begin(), packed_payload.end());
        
        return result;
    }
}

std::vector<uint8_t> pack_integer_effective_bytes(const std::vector<uint16_t>& values) {
    return pack_integer_effective_bytes_impl(values, false);
}

std::vector<uint8_t> pack_integer_effective_bytes_i16(const std::vector<int16_t>& values) {
    return pack_integer_effective_bytes_impl(values, true);
}

std::vector<uint8_t> byte_shuffle_16(const std::vector<uint8_t>& data) {
    int element_size = 2;
    if (data.empty() || data.size() % element_size != 0) return data;
    int n = data.size() / element_size;
    std::vector<uint8_t> out(data.size());
    for (int i = 0; i < n; ++i) {
        out[i] = data[i * 2];
        out[n + i] = data[i * 2 + 1];
    }
    return out;
}

std::vector<uint8_t> quantize_position_spz(const std::vector<float>& data, int fractional_bits) {
    double scale = static_cast<double>(1 << fractional_bits);
    std::vector<uint8_t> packed(data.size() * 3);
    uint32_t prev_q = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        int64_t fixed32 = std::round(static_cast<double>(data[i]) * scale);
        uint32_t cur_q = static_cast<uint32_t>(fixed32) & 0xFFFFFF;
        
        // DPCM: store delta (modular arithmetic handles negative results correctly)
        uint32_t delta = (cur_q - prev_q) & 0xFFFFFF;
        
        packed[i * 3 + 0] = delta & 0xFF;
        packed[i * 3 + 1] = (delta >> 8) & 0xFF;
        packed[i * 3 + 2] = (delta >> 16) & 0xFF;
        
        prev_q = cur_q;
    }
    return packed;
}

std::vector<uint8_t> quantize_position_spz_zz(const std::vector<float>& data, int fractional_bits) {
    // Same fixed-point DPCM as quantize_position_spz, but the signed delta is
    // zigzag-mapped to unsigned (0,-1,1,-2,... -> 0,1,2,3,...). After Morton
    // sort deltas are small, so the mid/high byte planes become near-zero
    // instead of the 0x00/0xFF split modular wrap produces — the image codec
    // collapses them.
    double scale = static_cast<double>(1 << fractional_bits);
    std::vector<uint8_t> packed(data.size() * 3);
    uint32_t prev_q = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        int64_t fixed32 = std::round(static_cast<double>(data[i]) * scale);
        uint32_t cur_q = static_cast<uint32_t>(fixed32) & 0xFFFFFF;

        uint32_t delta24 = (cur_q - prev_q) & 0xFFFFFF;
        // Sign-extend the 24-bit modular delta to a signed value in [-2^23, 2^23).
        int32_t d = static_cast<int32_t>(delta24);
        if (d & 0x800000) d -= 0x1000000;
        // Zigzag: result fits back into 24 bits.
        uint32_t zz = (static_cast<uint32_t>(d) << 1) ^ static_cast<uint32_t>(d >> 31);
        zz &= 0xFFFFFF;

        packed[i * 3 + 0] = zz & 0xFF;
        packed[i * 3 + 1] = (zz >> 8) & 0xFF;
        packed[i * 3 + 2] = (zz >> 16) & 0xFF;

        prev_q = cur_q;
    }
    return packed;
}

std::vector<uint8_t> quantize_opacity_sigmoid(const std::vector<float>& data) {
    std::vector<uint8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double sigmoid = 1.0 / (1.0 + std::exp(-static_cast<double>(data[i])));
        int val = std::round(sigmoid * 255.0);
        out[i] = static_cast<uint8_t>(std::clamp(val, 0, 255));
    }
    return out;
}

std::vector<uint8_t> pack_quaternion_smallest_three(
    const std::vector<float>& rot_0,
    const std::vector<float>& rot_1,
    const std::vector<float>& rot_2,
    const std::vector<float>& rot_3) 
{
    size_t n = rot_0.size();
    std::vector<uint8_t> out(n * 4);
    const double SQRT1_2 = 0.7071067811865476;
    
    for (size_t i = 0; i < n; ++i) {
        double q[4] = { rot_0[i], rot_1[i], rot_2[i], rot_3[i] };
        double norm2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
        double norm = std::sqrt(norm2);
        
        if (norm <= 1e-20) {
            q[0] = 0.0; q[1] = 0.0; q[2] = 0.0; q[3] = 1.0;
        } else {
            q[0] /= norm; q[1] /= norm; q[2] /= norm; q[3] /= norm;
        }
        
        int i_largest = 0;
        double max_abs = std::abs(q[0]);
        for (int c = 1; c < 4; ++c) {
            if (std::abs(q[c]) > max_abs) {
                max_abs = std::abs(q[c]);
                i_largest = c;
            }
        }
        
        if (q[i_largest] < 0.0) {
            q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3];
        }

        uint32_t comp = static_cast<uint32_t>(i_largest);
        
        for (int c = 0; c < 4; ++c) {
            if (c == i_largest) continue;
            double clamped_abs = std::min(SQRT1_2, std::abs(q[c]));
            uint32_t negbit = (q[c] < 0.0) ? 1 : 0;
            int mag = std::floor(511.0 * (clamped_abs / SQRT1_2) + 0.5);
            if (mag < 0) mag = 0;
            if (mag > 511) mag = 511;
            comp = (comp << 10) | (negbit << 9) | mag;
        }
        
        out[i * 4 + 0] = comp & 0xFF;
        out[i * 4 + 1] = (comp >> 8) & 0xFF;
        out[i * 4 + 2] = (comp >> 16) & 0xFF;
        out[i * 4 + 3] = (comp >> 24) & 0xFF;
    }
    return out;
}

std::vector<uint8_t> pack_quaternion_smallest_three_24(
    const std::vector<float>& rot_0,
    const std::vector<float>& rot_1,
    const std::vector<float>& rot_2,
    const std::vector<float>& rot_3) 
{
    size_t n = rot_0.size();
    std::vector<uint8_t> out(n * 3);
    const double SQRT1_2 = 0.7071067811865476;
    
    for (size_t i = 0; i < n; ++i) {
        double q[4] = { rot_0[i], rot_1[i], rot_2[i], rot_3[i] };
        double norm2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
        double norm = std::sqrt(norm2);
        
        if (norm <= 1e-20) {
            q[0] = 0.0; q[1] = 0.0; q[2] = 0.0; q[3] = 1.0;
        } else {
            q[0] /= norm; q[1] /= norm; q[2] /= norm; q[3] /= norm;
        }
        
        int i_largest = 0;
        double max_abs = std::abs(q[0]);
        for (int c = 1; c < 4; ++c) {
            if (std::abs(q[c]) > max_abs) {
                max_abs = std::abs(q[c]);
                i_largest = c;
            }
        }
        
        if (q[i_largest] < 0.0) {
            q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3];
        }

        uint32_t comp = static_cast<uint32_t>(i_largest);
        for (int c = 0; c < 4; ++c) {
            if (c == i_largest) continue;
            double clamped_abs = std::min(SQRT1_2, std::abs(q[c]));
            uint32_t negbit = (q[c] < 0.0) ? 1 : 0;
            int mag = std::floor(63.0 * (clamped_abs / SQRT1_2) + 0.5); // 6 bits mag
            if (mag < 0) mag = 0;
            if (mag > 63) mag = 63;
            comp = (comp << 7) | (negbit << 6) | mag;
        }
        
        out[i * 3 + 0] = comp & 0xFF;
        out[i * 3 + 1] = (comp >> 8) & 0xFF;
        out[i * 3 + 2] = (comp >> 16) & 0xFF;
    }
    return out;
}

std::vector<uint8_t> pack_quaternion_smallest_three_var(
    const std::vector<float>& rot_0,
    const std::vector<float>& rot_1,
    const std::vector<float>& rot_2,
    const std::vector<float>& rot_3,
    int mag_bits)
{
    // Generalization of pack_quaternion_smallest_three_24: 2-bit largest index
    // + 3 x (1-bit sign + mag_bits magnitude), still 3 bytes per quaternion.
    // mag_bits=6 reproduces the legacy layout bit for bit. Lower mag_bits cuts
    // the entropy the image codec has to pay; the value travels in meta.
    size_t n = rot_0.size();
    std::vector<uint8_t> out(n * 3);
    const double SQRT1_2 = 0.7071067811865476;
    const int max_mag = (1 << mag_bits) - 1;

    for (size_t i = 0; i < n; ++i) {
        double q[4] = { rot_0[i], rot_1[i], rot_2[i], rot_3[i] };
        double norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (norm <= 1e-20) { q[0]=0.0; q[1]=0.0; q[2]=0.0; q[3]=1.0; }
        else { q[0]/=norm; q[1]/=norm; q[2]/=norm; q[3]/=norm; }

        int i_largest = 0;
        double max_abs = std::abs(q[0]);
        for (int c = 1; c < 4; ++c)
            if (std::abs(q[c]) > max_abs) { max_abs = std::abs(q[c]); i_largest = c; }

        if (q[i_largest] < 0.0) { q[0]=-q[0]; q[1]=-q[1]; q[2]=-q[2]; q[3]=-q[3]; }

        uint32_t comp = static_cast<uint32_t>(i_largest);
        for (int c = 0; c < 4; ++c) {
            if (c == i_largest) continue;
            double clamped_abs = std::min(SQRT1_2, std::abs(q[c]));
            uint32_t negbit = (q[c] < 0.0) ? 1 : 0;
            int mag = std::floor(max_mag * (clamped_abs / SQRT1_2) + 0.5);
            if (mag < 0) mag = 0;
            if (mag > max_mag) mag = max_mag;
            comp = (comp << (mag_bits + 1)) | (negbit << mag_bits) | mag;
        }

        out[i * 3 + 0] = comp & 0xFF;
        out[i * 3 + 1] = (comp >> 8) & 0xFF;
        out[i * 3 + 2] = (comp >> 16) & 0xFF;
    }
    return out;
}

void pack_quaternion_smallest_three_split(
    const std::vector<float>& rot_0,
    const std::vector<float>& rot_1,
    const std::vector<float>& rot_2,
    const std::vector<float>& rot_3,
    std::vector<uint8_t>& out_rgb,
    std::vector<uint8_t>& out_idx)
{
    size_t n = rot_0.size();
    out_rgb.assign(n * 3, 0);
    out_idx.assign(n, 0);
    const double SQRT1_2 = 0.7071067811865476;

    for (size_t i = 0; i < n; ++i) {
        double q[4] = { rot_0[i], rot_1[i], rot_2[i], rot_3[i] };
        double norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (norm <= 1e-20) { q[0]=0.0; q[1]=0.0; q[2]=0.0; q[3]=1.0; }
        else { q[0]/=norm; q[1]/=norm; q[2]/=norm; q[3]/=norm; }

        int i_largest = 0;
        double max_abs = std::abs(q[0]);
        for (int c = 1; c < 4; ++c)
            if (std::abs(q[c]) > max_abs) { max_abs = std::abs(q[c]); i_largest = c; }

        if (q[i_largest] < 0.0) { q[0]=-q[0]; q[1]=-q[1]; q[2]=-q[2]; q[3]=-q[3]; }

        out_idx[i] = static_cast<uint8_t>(i_largest);
        // Store the three non-largest components, in increasing index order, as
        // 8-bit values mapping [-1/sqrt2, +1/sqrt2] -> [0,255]. Byte-aligned, so
        // each channel is a smooth field the image codec can predict.
        int ch = 0;
        for (int c = 0; c < 4; ++c) {
            if (c == i_largest) continue;
            double v = std::max(-SQRT1_2, std::min(SQRT1_2, q[c]));
            int u = static_cast<int>(std::floor((v / SQRT1_2 * 0.5 + 0.5) * 255.0 + 0.5));
            if (u < 0) u = 0; if (u > 255) u = 255;
            out_rgb[i * 3 + ch] = static_cast<uint8_t>(u);
            ++ch;
        }
    }
}

void unpack_quaternion_smallest_three_split(
    const std::vector<uint8_t>& rgb,
    const std::vector<uint8_t>& idx,
    size_t count,
    std::vector<float>& rot_0,
    std::vector<float>& rot_1,
    std::vector<float>& rot_2,
    std::vector<float>& rot_3)
{
    rot_0.resize(count); rot_1.resize(count); rot_2.resize(count); rot_3.resize(count);
    const double SQRT1_2 = 0.7071067811865476;
    for (size_t i = 0; i < count; ++i) {
        int i_largest = idx[i] & 0x3;
        double q[4] = {0.0, 0.0, 0.0, 0.0};
        double sum_sq = 0.0;
        int ch = 0;
        for (int c = 0; c < 4; ++c) {
            if (c == i_largest) continue;
            double v = (static_cast<double>(rgb[i*3 + ch]) / 255.0 - 0.5) * 2.0 * SQRT1_2;
            q[c] = v;
            sum_sq += v * v;
            ++ch;
        }
        q[i_largest] = std::sqrt(std::max(0.0, 1.0 - sum_sq));
        rot_0[i] = q[0]; rot_1[i] = q[1]; rot_2[i] = q[2]; rot_3[i] = q[3];
    }
}

std::vector<uint8_t> quantize_to_u8_levels(const std::vector<float>& data, int bits, double& out_min, double& out_max) {
    if (data.empty()) {
        out_min = 0.0; out_max = 0.0;
        return {};
    }
    out_min = data[0];
    out_max = data[0];
    for (float v : data) {
        if (v < out_min) out_min = v;
        if (v > out_max) out_max = v;
    }
    double range_val = out_max - out_min;
    int levels = (1 << bits) - 1;
    std::vector<uint8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double normalized = 0.0;
        if (range_val > 0.0) normalized = (data[i] - out_min) / range_val;
        int q = std::round(normalized * levels);
        out[i] = static_cast<uint8_t>(std::clamp(q, 0, levels));
    }
    return out;
}

std::vector<uint16_t> quantize_to_u16_levels(const std::vector<float>& data, int bits, double& out_min, double& out_max) {
    if (data.empty()) {
        out_min = 0.0; out_max = 0.0;
        return {};
    }
    out_min = data[0];
    out_max = data[0];
    for (float v : data) {
        if (v < out_min) out_min = v;
        if (v > out_max) out_max = v;
    }
    double range_val = out_max - out_min;
    int levels = (1 << bits) - 1;
    std::vector<uint16_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double normalized = 0.0;
        if (range_val > 0.0) normalized = (data[i] - out_min) / range_val;
        int q = std::round(normalized * levels);
        out[i] = static_cast<uint16_t>(std::clamp(q, 0, levels));
    }
    return out;
}

std::vector<float> dequantize_position_spz(const std::vector<uint8_t>& data, int fractional_bits, size_t count) {
    std::vector<float> out(count);
    double scale = static_cast<double>(1 << fractional_bits);
    uint32_t prev_q = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t delta = data[i * 3 + 0] | (data[i * 3 + 1] << 8) | (data[i * 3 + 2] << 16);
        
        // Reconstruct current value from delta
        uint32_t cur_q = (prev_q + delta) & 0xFFFFFF;
        
        // Sign extension for 24-bit fixed point
        int32_t fixed32 = static_cast<int32_t>(cur_q);
        if (fixed32 & 0x800000) fixed32 |= ~0xFFFFFF;
        
        out[i] = static_cast<float>(static_cast<double>(fixed32) / scale);
        prev_q = cur_q;
    }
    return out;
}

std::vector<float> dequantize_position_spz_zz(const std::vector<uint8_t>& data, int fractional_bits, size_t count) {
    std::vector<float> out(count);
    double scale = static_cast<double>(1 << fractional_bits);
    uint32_t prev_q = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t zz = data[i * 3 + 0] | (data[i * 3 + 1] << 8) | (data[i * 3 + 2] << 16);
        // Un-zigzag back to the signed delta.
        int32_t d = static_cast<int32_t>(zz >> 1) ^ -static_cast<int32_t>(zz & 1);

        uint32_t cur_q = (prev_q + static_cast<uint32_t>(d)) & 0xFFFFFF;

        int32_t fixed32 = static_cast<int32_t>(cur_q);
        if (fixed32 & 0x800000) fixed32 |= ~0xFFFFFF;

        out[i] = static_cast<float>(static_cast<double>(fixed32) / scale);
        prev_q = cur_q;
    }
    return out;
}

std::vector<float> dequantize_opacity_sigmoid(const std::vector<uint8_t>& data, size_t count) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        double alpha = data[i] / 255.0;
        if (alpha < 1e-6) alpha = 1e-6;
        if (alpha > 1.0 - 1e-6) alpha = 1.0 - 1e-6;
        out[i] = static_cast<float>(std::log(alpha / (1.0 - alpha)));
    }
    return out;
}

void unpack_quaternion_smallest_three(const std::vector<uint8_t>& data, size_t count,
                                      std::vector<float>& rot_0,
                                      std::vector<float>& rot_1,
                                      std::vector<float>& rot_2,
                                      std::vector<float>& rot_3) 
{
    rot_0.resize(count); rot_1.resize(count); rot_2.resize(count); rot_3.resize(count);
    const double SQRT1_2 = 0.7071067811865476;
    for (size_t i = 0; i < count; ++i) {
        uint32_t c = data[i * 4 + 0] | (data[i * 4 + 1] << 8) | (data[i * 4 + 2] << 16) | (data[i * 4 + 3] << 24);
        int i_largest = c >> 30;
        double sum_squares = 0.0;
        double q[4] = {0.0, 0.0, 0.0, 0.0};
        for (int k = 3; k >= 0; --k) {
            if (k == i_largest) continue;
            uint32_t mag = c & 0x1FF;
            uint32_t negbit = (c >> 9) & 0x1;
            c >>= 10;
            double val = SQRT1_2 * (static_cast<double>(mag) / 511.0);
            if (negbit == 1) val = -val;
            q[k] = val;
            sum_squares += val * val;
        }
        double remaining = 1.0 - sum_squares;
        q[i_largest] = std::sqrt(std::max(0.0, remaining));
        rot_0[i] = q[0]; rot_1[i] = q[1]; rot_2[i] = q[2]; rot_3[i] = q[3];
    }
}

void unpack_quaternion_smallest_three_24(const std::vector<uint8_t>& data, size_t count,
                                         std::vector<float>& rot_0,
                                         std::vector<float>& rot_1,
                                         std::vector<float>& rot_2,
                                         std::vector<float>& rot_3) 
{
    rot_0.resize(count); rot_1.resize(count); rot_2.resize(count); rot_3.resize(count);
    const double SQRT1_2 = 0.7071067811865476;
    for (size_t i = 0; i < count; ++i) {
        uint32_t c = data[i * 3 + 0] | (data[i * 3 + 1] << 8) | (data[i * 3 + 2] << 16);
        int i_largest = c >> 21;
        double sum_squares = 0.0;
        double q[4] = {0.0, 0.0, 0.0, 0.0};
        for (int k = 3; k >= 0; --k) {
            if (k == i_largest) continue;
            uint32_t mag = c & 0x3F;
            uint32_t negbit = (c >> 6) & 0x1;
            c >>= 7;
            double val = SQRT1_2 * (static_cast<double>(mag) / 63.0);
            if (negbit == 1) val = -val;
            q[k] = val;
            sum_squares += val * val;
        }
        double remaining = 1.0 - sum_squares;
        q[i_largest] = std::sqrt(std::max(0.0, remaining));
        rot_0[i] = q[0]; rot_1[i] = q[1]; rot_2[i] = q[2]; rot_3[i] = q[3];
    }
}

void unpack_quaternion_smallest_three_var(const std::vector<uint8_t>& data, size_t count, int mag_bits,
                                          std::vector<float>& rot_0,
                                          std::vector<float>& rot_1,
                                          std::vector<float>& rot_2,
                                          std::vector<float>& rot_3)
{
    rot_0.resize(count); rot_1.resize(count); rot_2.resize(count); rot_3.resize(count);
    const double SQRT1_2 = 0.7071067811865476;
    const uint32_t mag_mask = (1u << mag_bits) - 1;
    const double max_mag = static_cast<double>(mag_mask);
    for (size_t i = 0; i < count; ++i) {
        uint32_t c = data[i * 3 + 0] | (data[i * 3 + 1] << 8) | (data[i * 3 + 2] << 16);
        int i_largest = c >> (3 * (mag_bits + 1));
        double sum_squares = 0.0;
        double q[4] = {0.0, 0.0, 0.0, 0.0};
        for (int k = 3; k >= 0; --k) {
            if (k == i_largest) continue;
            uint32_t mag = c & mag_mask;
            uint32_t negbit = (c >> mag_bits) & 0x1;
            c >>= (mag_bits + 1);
            double val = SQRT1_2 * (static_cast<double>(mag) / max_mag);
            if (negbit == 1) val = -val;
            q[k] = val;
            sum_squares += val * val;
        }
        q[i_largest] = std::sqrt(std::max(0.0, 1.0 - sum_squares));
        rot_0[i] = q[0]; rot_1[i] = q[1]; rot_2[i] = q[2]; rot_3[i] = q[3];
    }
}

std::vector<float> dequantize_from_u8_levels(const std::vector<uint8_t>& data, int bits, double min_val, double max_val, size_t count) {
    std::vector<float> out(count);
    int levels = (1 << bits) - 1;
    double range_val = max_val - min_val;
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>((static_cast<double>(data[i]) / levels) * range_val + min_val);
    }
    return out;
}

std::vector<float> dequantize_from_u16_levels(const std::vector<uint16_t>& data, int bits, double min_val, double max_val, size_t count) {
    std::vector<float> out(count);
    int levels = (1 << bits) - 1;
    double range_val = max_val - min_val;
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>((static_cast<double>(data[i]) / levels) * range_val + min_val);
    }
    return out;
}

std::vector<int16_t> unpack_integer_effective_bytes_i16(const std::vector<uint8_t>& data) {
    if (data.size() < 20) return {};
    if (std::memcmp(data.data(), "BPK1", 4) != 0) {
        std::vector<int16_t> out(data.size() / 2);
        std::memcpy(out.data(), data.data(), out.size() * 2);
        return out;
    }
    int bits = data[7];
    uint32_t count;
    std::memcpy(&count, &data[8], 4);
    int64_t min_v;
    std::memcpy(&min_v, &data[12], 8);
    std::vector<int16_t> out(count, static_cast<int16_t>(min_v));
    if (bits == 0) return out;
    const uint8_t* payload = data.data() + 20;
    for (size_t i = 0; i < count; ++i) {
        int16_t shifted = 0;
        for (int b = 0; b < bits; ++b) {
            size_t bit_idx = i * bits + b;
            if (payload[bit_idx / 8] & (1 << (bit_idx % 8)))
                shifted |= (1 << b);
        }
        out[i] = shifted + static_cast<int16_t>(min_v);
    }
    return out;
}
