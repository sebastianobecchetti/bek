#pragma once
#include <vector>
#include <cstdint>

std::vector<uint8_t> pack_integer_effective_bytes(const std::vector<uint16_t>& values);
std::vector<uint8_t> pack_integer_effective_bytes_i16(const std::vector<int16_t>& values);
std::vector<int16_t> unpack_integer_effective_bytes_i16(const std::vector<uint8_t>& data);

std::vector<uint8_t> byte_shuffle_16(const std::vector<uint8_t>& data);

std::vector<uint8_t> quantize_position_spz(const std::vector<float>& data, int fractional_bits);
std::vector<uint8_t> quantize_position_spz_zz(const std::vector<float>& data, int fractional_bits);
std::vector<uint8_t> quantize_opacity_sigmoid(const std::vector<float>& data);
std::vector<uint8_t> pack_quaternion_smallest_three(
    const std::vector<float>& rot_0,
    const std::vector<float>& rot_1,
    const std::vector<float>& rot_2,
    const std::vector<float>& rot_3);
std::vector<uint8_t> pack_quaternion_smallest_three_24(
    const std::vector<float>& rot_0,
    const std::vector<float>& rot_1,
    const std::vector<float>& rot_2,
    const std::vector<float>& rot_3);
// Generalized smallest-three: 2-bit index + 3 x (sign + mag_bits) in 3 bytes.
// mag_bits=6 reproduces the _24 layout bit for bit; mag_bits travels in meta.
std::vector<uint8_t> pack_quaternion_smallest_three_var(
    const std::vector<float>& rot_0,
    const std::vector<float>& rot_1,
    const std::vector<float>& rot_2,
    const std::vector<float>& rot_3,
    int mag_bits);
// Smallest-three with byte-aligned channels: 3 components at 8 bits each in
// out_rgb (n*3) plus the largest-component index (0-3) in out_idx (n). Aligned
// fields compress better than the 24-bit-packed variant and gain a precision bit.
void pack_quaternion_smallest_three_split(
    const std::vector<float>& rot_0,
    const std::vector<float>& rot_1,
    const std::vector<float>& rot_2,
    const std::vector<float>& rot_3,
    std::vector<uint8_t>& out_rgb,
    std::vector<uint8_t>& out_idx);
void unpack_quaternion_smallest_three_split(
    const std::vector<uint8_t>& rgb,
    const std::vector<uint8_t>& idx,
    size_t count,
    std::vector<float>& rot_0,
    std::vector<float>& rot_1,
    std::vector<float>& rot_2,
    std::vector<float>& rot_3);

std::vector<uint8_t> quantize_to_u8_levels(const std::vector<float>& data, int bits, double& out_min, double& out_max);
std::vector<uint16_t> quantize_to_u16_levels(const std::vector<float>& data, int bits, double& out_min, double& out_max);

std::vector<float> dequantize_position_spz(const std::vector<uint8_t>& data, int fractional_bits, size_t count);
std::vector<float> dequantize_position_spz_zz(const std::vector<uint8_t>& data, int fractional_bits, size_t count);
std::vector<float> dequantize_opacity_sigmoid(const std::vector<uint8_t>& data, size_t count);
void unpack_quaternion_smallest_three(const std::vector<uint8_t>& data, size_t count,
                                      std::vector<float>& rot_0,
                                      std::vector<float>& rot_1,
                                      std::vector<float>& rot_2,
                                      std::vector<float>& rot_3);
void unpack_quaternion_smallest_three_24(const std::vector<uint8_t>& data, size_t count,
                                         std::vector<float>& rot_0,
                                         std::vector<float>& rot_1,
                                         std::vector<float>& rot_2,
                                         std::vector<float>& rot_3);
void unpack_quaternion_smallest_three_var(const std::vector<uint8_t>& data, size_t count, int mag_bits,
                                          std::vector<float>& rot_0,
                                          std::vector<float>& rot_1,
                                          std::vector<float>& rot_2,
                                          std::vector<float>& rot_3);
std::vector<float> dequantize_from_u8_levels(const std::vector<uint8_t>& data, int bits, double min_val, double max_val, size_t count);
std::vector<float> dequantize_from_u16_levels(const std::vector<uint16_t>& data, int bits, double min_val, double max_val, size_t count);
