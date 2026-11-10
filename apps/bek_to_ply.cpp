#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <fstream>
#include <webp/decode.h>
#include <nlohmann/json.hpp>
#include <zstd.h>
#include "config.h"
#include "ply_writer.h"
#include "quantization.h"
#include "range_coder.h"
#include "sh_vq.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "BEK " << BEK_VERSION << '\n';
        return 0;
    }
    if (argc < 3) {
        std::cerr << "Usage: bek_to_ply <input.bek> <output.ply>\n";
        return 1;
    }

    std::string input_bek = argv[1];
    std::string output_ply = argv[2];
    
    std::ifstream in(input_bek, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Could not open input file " << input_bek << std::endl;
        return 1;
    }

    // 1. Read Magic and Uncompressed Size
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, "BEK3", 4) != 0) {
        std::cerr << "Error: Not a BEK v3 file (Invalid Magic)" << std::endl;
        return 1;
    }

    uint64_t uncompressed_size = 0;
    in.read(reinterpret_cast<char*>(&uncompressed_size), 8);

    // 2. Read Compressed Payload
    auto compressed_start = in.tellg();
    in.seekg(0, std::ios::end);
    size_t compressed_size = static_cast<size_t>(in.tellg() - compressed_start);
    in.seekg(compressed_start, std::ios::beg);
    std::vector<uint8_t> compressed_buffer(compressed_size);
    in.read(reinterpret_cast<char*>(compressed_buffer.data()), compressed_size);
    in.close();

    // 3. Decompress Global Box (windowLogMax raised for the encoder's LDM window)
    std::vector<uint8_t> uncompressed_payload(uncompressed_size);
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    ZSTD_DCtx_setParameter(dctx, ZSTD_d_windowLogMax, 31);
    size_t const decompressed_size = ZSTD_decompressDCtx(dctx,
                                                    uncompressed_payload.data(), uncompressed_payload.size(),
                                                    compressed_buffer.data(), compressed_buffer.size());
    ZSTD_freeDCtx(dctx);
    if (ZSTD_isError(decompressed_size)) {
        std::cerr << "Error: Zstd decompression failed: " << ZSTD_getErrorName(decompressed_size) << std::endl;
        return 1;
    }

    // 4. Parse BEK v3 Structure from Uncompressed Buffer
    const uint8_t* ptr = uncompressed_payload.data();
    uint32_t header_size = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += 4;

    std::vector<uint8_t> header_bin(ptr, ptr + header_size);
    ptr += header_size;
    json header = json::from_cbor(header_bin);
    
    const uint8_t* payload_base = ptr; // Start of WebP blobs

    uint32_t N = header["count"];
    // Rotation split is delivered as two channels (rgb comps + index plane);
    // stash both and reconstruct after the channel loop.
    std::vector<uint8_t> rot_split_rgb, rot_split_idx;
    // Raw VQ index planes kept for cross-stage (rcx) contexts of later stages.
    std::map<std::string, std::vector<uint8_t>> vq_plane_cache;
    auto vertex_data = std::make_unique<VertexData>();
    vertex_data->count = N;
    vertex_data->x.resize(N); vertex_data->y.resize(N); vertex_data->z.resize(N);
    vertex_data->opacity.resize(N);
    vertex_data->scale.resize(N * 3);
    vertex_data->rot.resize(N * 4);

    for (auto& entry : header["channels"]) {
        std::string name = entry["name"];
        size_t offset = entry["offset"];
        size_t size = entry["size"];
        int channels = entry["channels"];
        auto m = entry["meta"];
        
        const uint8_t* blob_ptr = payload_base + offset;

        std::vector<uint8_t> raw;
        std::string codec = m.value("codec", "webp");
        if (codec == "raw") {
            // Raw bytes: the channel array verbatim (N*channels entries).
            raw.assign(blob_ptr, blob_ptr + size);
        } else if (codec == "rc2") {
            // Adaptive contextual range coder over the unpadded channel bytes.
            raw = rc2_decode(blob_ptr, size, static_cast<size_t>(N) * channels, channels);
        } else if (codec == "rc1") {
            raw = rc1_decode(blob_ptr, size, static_cast<size_t>(N) * channels, channels);
        } else if (codec == "rcx") {
            // Cross-stage coder: context plane is an earlier VQ stage, decoded
            // before this one (header preserves encoder channel order).
            auto it = vq_plane_cache.find(m.value("ctx_ch", ""));
            if (it == vq_plane_cache.end()) { std::cerr << "rcx: missing context channel\n"; continue; }
            raw = rcx_decode(blob_ptr, size, static_cast<size_t>(N) * channels, it->second.data());
        } else if (codec == "posmag") {
            // Dedicated position magnitude coder: rebuilds the 24-bit zigzag
            // DPCM byte triplets (3 bytes/splat) the encoder fed in.
            raw = posmag_decode(blob_ptr, size, static_cast<size_t>(N) * channels);
        } else if (codec == "posmag3") {
            // Joint xyz variant: 9 bytes/splat (x,y,z zigzag DPCM words).
            raw = posmag3_decode(blob_ptr, size, static_cast<size_t>(N) * channels);
        } else if (codec == "rotpack") {
            // Dedicated rotation coder: rebuilds the smallest3_24 packed word.
            raw = rotpack_decode(blob_ptr, size, static_cast<size_t>(N) * channels, m.value("mag_bits", 6));
        } else {
            int w, h;
            uint8_t* rgba = WebPDecodeRGBA(blob_ptr, size, &w, &h);
            if (!rgba) continue;
            if (channels == 4) {
                raw.assign(rgba, rgba + w * h * 4);
            } else if (channels == 3) {
                raw.resize(w * h * 3);
                for(int i=0; i<w*h; ++i) {
                    raw[i*3+0] = rgba[i*4+0]; raw[i*3+1] = rgba[i*4+1]; raw[i*3+2] = rgba[i*4+2];
                }
            } else if (channels == 1) {
                raw.resize(w * h);
                for(int i=0; i<w*h; ++i) raw[i] = rgba[i*4+0];
            }
            WebPFree(rgba);
        }

        // Keep VQ index planes around: later residual stages may use them as
        // rcx contexts.
        if (name.rfind("vq_", 0) == 0) vq_plane_cache[name] = raw;

        if (name == "pos_xyz") {
            // Deinterleave the 9-byte records back into the three per-axis
            // zigzag-DPCM planes, then reuse the standard dequantizer.
            const int frac = m.value("frac", 12);
            std::vector<uint8_t> px(N * 3), py(N * 3), pz(N * 3);
            for (uint32_t i = 0; i < N; ++i)
                for (int b = 0; b < 3; ++b) {
                    px[i * 3 + b] = raw[i * 9 + 0 + b];
                    py[i * 3 + b] = raw[i * 9 + 3 + b];
                    pz[i * 3 + b] = raw[i * 9 + 6 + b];
                }
            vertex_data->x = dequantize_position_spz_zz(px, frac, N);
            vertex_data->y = dequantize_position_spz_zz(py, frac, N);
            vertex_data->z = dequantize_position_spz_zz(pz, frac, N);
        } else if (name == "pos_x" || name == "pos_y" || name == "pos_z") {
            bool zz = m.value("type", "spz_dpcm") == "spz_dpcm_zz";
            auto dec = zz ? dequantize_position_spz_zz(raw, m.value("frac", 12), N)
                          : dequantize_position_spz(raw, m.value("frac", 12), N);
            if (name == "pos_x") vertex_data->x = std::move(dec);
            else if (name == "pos_y") vertex_data->y = std::move(dec);
            else vertex_data->z = std::move(dec);
        } else if (name == "opacity") {
            vertex_data->opacity = dequantize_opacity_sigmoid(raw, N);
        } else if (name == "rotation") {
            if (m.value("type", "smallest3_24") == "smallest3_split") {
                rot_split_rgb = std::move(raw); // reconstructed after the loop
            } else {
                std::vector<float> r0, r1, r2, r3;
                // mag_bits=6 is the legacy _24 layout; older files have no key.
                unpack_quaternion_smallest_three_var(raw, N, m.value("mag_bits", 6), r0, r1, r2, r3);
                for(uint32_t i=0; i<N; ++i) {
                    vertex_data->rot[i*4+0] = r0[i]; vertex_data->rot[i*4+1] = r1[i];
                    vertex_data->rot[i*4+2] = r2[i]; vertex_data->rot[i*4+3] = r3[i];
                }
            }
        } else if (name == "rotation_idx") {
            rot_split_idx = std::move(raw);
        } else if (name == "scales") {
            std::vector<double> min_v = m["min"];
            std::vector<double> max_v = m["max"];
            int bits = m.value("bits", 8); // absent in files written before --scale-bits existed
            for(int i=0; i<3; ++i) {
                std::vector<uint8_t> ch_data(N);
                for(uint32_t j=0; j<N; ++j) ch_data[j] = raw[j*3 + i];
                auto dec = dequantize_from_u8_levels(ch_data, bits, min_v[i], max_v[i], N);
                for(uint32_t j=0; j<N; ++j) vertex_data->scale[j*3+i] = dec[j];
            }
        } else if (name.find("vq_scale_s") == 0) {
            // Scale RVQ: accumulate each residual stage's 3D codebook entry.
            std::vector<float> codebook;
            if (m.contains("levels") && m.contains("cb_indices")) {
                std::vector<float> levels = m["levels"];
                std::vector<uint8_t> cb_indices = m["cb_indices"];
                for (uint8_t idx : cb_indices) codebook.push_back(levels[idx]);
            }
            for (uint32_t i = 0; i < N; ++i) {
                int idx = raw[i];
                for (int d = 0; d < 3; ++d) vertex_data->scale[i*3 + d] += codebook[idx*3 + d];
            }
        } else if (name == "km_dc") {
            // Flat k-means DC: single codebook, per-point index (1 or 2 bytes).
            int ibytes = m.value("index_bytes", 1);
            std::vector<float> codebook;
            if (m.contains("levels") && m.contains("cb_indices")) {
                std::vector<float> levels = m["levels"];
                std::vector<uint8_t> cb_indices = m["cb_indices"];
                for (uint8_t idx : cb_indices) codebook.push_back(levels[idx]);
            }
            if (vertex_data->f_dc["f_dc_0"].empty()) {
                vertex_data->f_dc["f_dc_0"].assign(N, 0.0f);
                vertex_data->f_dc["f_dc_1"].assign(N, 0.0f);
                vertex_data->f_dc["f_dc_2"].assign(N, 0.0f);
            }
            for (uint32_t i = 0; i < N; ++i) {
                int idx = (ibytes == 2) ? (raw[i*2] | (raw[i*2+1] << 8)) : raw[i];
                vertex_data->f_dc["f_dc_0"][i] = codebook[idx*3 + 0];
                vertex_data->f_dc["f_dc_1"][i] = codebook[idx*3 + 1];
                vertex_data->f_dc["f_dc_2"][i] = codebook[idx*3 + 2];
            }
        } else if (name == "km_sh_ac") {
            // Flat k-means joint SH AC: single codebook over all dims.
            int dims = m["dims"];
            int ibytes = m.value("index_bytes", 1);
            std::vector<float> codebook;
            if (m.contains("codebook")) {
                codebook = m["codebook"].get<std::vector<float>>();
            } else if (m.contains("levels") && m.contains("cb_indices")) {
                std::vector<float> levels = m["levels"];
                std::vector<uint8_t> cb_indices = m["cb_indices"];
                for (uint8_t idx : cb_indices) codebook.push_back(levels[idx]);
            }
            if (vertex_data->f_rest.empty()) {
                for (int i = 0; i < dims; ++i)
                    vertex_data->f_rest["f_rest_" + std::to_string(i)].assign(N, 0.0f);
            }
            for (uint32_t i = 0; i < N; ++i) {
                int idx = (ibytes == 2) ? (raw[i*2] | (raw[i*2+1] << 8)) : raw[i];
                for (int d = 0; d < dims; ++d)
                    vertex_data->f_rest["f_rest_" + std::to_string(d)][i] = codebook[idx*dims + d];
            }
        } else if (name.find("vq_dc_s") == 0) {
            std::vector<float> codebook;
            if (m.contains("codebook")) {
                codebook = m["codebook"].get<std::vector<float>>();
            } else if (m.contains("levels") && m.contains("cb_indices")) {
                std::vector<float> levels = m["levels"];
                std::vector<uint8_t> cb_indices = m["cb_indices"];
                for (uint8_t idx : cb_indices) codebook.push_back(levels[idx]);
            }

            if (vertex_data->f_dc["f_dc_0"].empty()) {
                vertex_data->f_dc["f_dc_0"].assign(N, 0.0f);
                vertex_data->f_dc["f_dc_1"].assign(N, 0.0f);
                vertex_data->f_dc["f_dc_2"].assign(N, 0.0f);
            }
            for(uint32_t i=0; i<N; ++i) {
                int idx = raw[i];
                vertex_data->f_dc["f_dc_0"][i] += codebook[idx*3 + 0];
                vertex_data->f_dc["f_dc_1"][i] += codebook[idx*3 + 1];
                vertex_data->f_dc["f_dc_2"][i] += codebook[idx*3 + 2];
            }
        } else if (name.find("vq_sh1_s") == 0 || name.find("vq_sh2_s") == 0 || name.find("vq_sh3_s") == 0) {
            // Banded SH AC: accumulate into f_rest starting at dim_offset.
            int dims = m["dims"];
            int dim_offset = m.value("dim_offset", 0);
            std::vector<float> codebook;
            if (m.contains("codebook")) {
                codebook = m["codebook"].get<std::vector<float>>();
            } else if (m.contains("levels") && m.contains("cb_indices")) {
                std::vector<float> levels = m["levels"];
                std::vector<uint8_t> cb_indices = m["cb_indices"];
                for (uint8_t idx : cb_indices) codebook.push_back(levels[idx]);
            }

            for(int d=0; d<dims; ++d) {
                std::string key = "f_rest_" + std::to_string(dim_offset + d);
                if (vertex_data->f_rest[key].empty()) vertex_data->f_rest[key].assign(N, 0.0f);
            }
            for(uint32_t i=0; i<N; ++i) {
                int idx = raw[i];
                for(int d=0; d<dims; ++d) {
                    vertex_data->f_rest["f_rest_" + std::to_string(dim_offset + d)][i] += codebook[idx*dims + d];
                }
            }
        } else if (name.find("vq_sh_ac_s") == 0) {
            int dims = m["dims"];
            std::vector<float> codebook;
            if (m.contains("codebook")) {
                codebook = m["codebook"].get<std::vector<float>>();
            } else if (m.contains("levels") && m.contains("cb_indices")) {
                std::vector<float> levels = m["levels"];
                std::vector<uint8_t> cb_indices = m["cb_indices"];
                for (uint8_t idx : cb_indices) codebook.push_back(levels[idx]);
            }

            if (vertex_data->f_rest.empty()) {
                for(int i=0; i<dims; ++i) {
                    vertex_data->f_rest["f_rest_" + std::to_string(i)].assign(N, 0.0f);
                }
            }
            for(uint32_t i=0; i<N; ++i) {
                int idx = raw[i];
                for(int d=0; d<dims; ++d) {
                    vertex_data->f_rest["f_rest_" + std::to_string(d)][i] += codebook[idx*dims + d];
                }
            }
        }
    }

    // Reconstruct rotations from the split planes (if present).
    if (!rot_split_rgb.empty() && !rot_split_idx.empty()) {
        std::vector<float> r0, r1, r2, r3;
        unpack_quaternion_smallest_three_split(rot_split_rgb, rot_split_idx, N, r0, r1, r2, r3);
        for (uint32_t i = 0; i < N; ++i) {
            vertex_data->rot[i*4+0] = r0[i]; vertex_data->rot[i*4+1] = r1[i];
            vertex_data->rot[i*4+2] = r2[i]; vertex_data->rot[i*4+3] = r3[i];
        }
    }

    // Defaults for omitted attributes.
    // Attribute arrays are initially zero-filled. Missing positions remain at
    // zero; opacity, rotation and scale receive the defaults below.
    bool had_opacity = false, had_rotation = false, had_scale = false;
    for (auto& entry : header["channels"]) {
        std::string name = entry["name"];
        if (name == "opacity") had_opacity = true;
        else if (name == "rotation" || name == "rotation_idx") had_rotation = true;
        else if (name == "scales" || name.rfind("vq_scale_s", 0) == 0) had_scale = true;
    }

    if (!had_opacity) {
        // Opaque default: the raw byte level that maps to alpha ~= 1.0 through
        // dequantize_opacity_sigmoid's inverse-sigmoid (raw=255 -> alpha
        // clamped to 1-1e-6 -> logit ~= 13.8155). Fully opaque splats instead
        // of invisible (raw=0/alpha~0) ones.
        float opaque_logit = dequantize_opacity_sigmoid(std::vector<uint8_t>(1, 255), 1)[0];
        std::fill(vertex_data->opacity.begin(), vertex_data->opacity.end(), opaque_logit);
    }
    if (!had_rotation) {
        // Identity quaternion: rot_0 is the real/w component in this codec's
        // convention (see pack_quaternion_smallest_three*), so w=1, x=y=z=0.
        for (uint32_t i = 0; i < N; ++i) {
            vertex_data->rot[i*4+0] = 1.0f;
            vertex_data->rot[i*4+1] = 0.0f;
            vertex_data->rot[i*4+2] = 0.0f;
            vertex_data->rot[i*4+3] = 0.0f;
        }
    }
    if (!had_scale) {
        // Fallback log-scale: exp(-5) gives small, nonzero splats.
        // A zero log-scale would correspond to unit size.
        std::fill(vertex_data->scale.begin(), vertex_data->scale.end(), -5.0f);
    }

    write_ply_file(output_ply, *vertex_data);
    return 0;
}
