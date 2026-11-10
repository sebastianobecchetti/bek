#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <filesystem>
#include <future>
#include <thread>
#include <mutex>
#include <map>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <webp/encode.h>
#include <nlohmann/json.hpp>
#include <zstd.h>
#include "config.h"
#include "ply_reader.h"
#include "quantization.h"
#include "range_coder.h"
#include "sh_vq.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

struct ChannelData {
    std::string name;
    int channels; // 1, 3, or 4
    std::vector<uint8_t> data;
    json metadata;
};

// Lossless WebP encode via the advanced API: method 6 + exact=1 squeezes the
// blob harder than the WebPEncodeLossless* wrappers (method 4) while producing
// bit-identical decoded pixels. Returns empty on failure.
static std::vector<uint8_t> webp_encode_lossless(const uint8_t* pixels, int W, int H, int channels) {
    WebPConfig config;
    WebPConfigInit(&config);
    WebPConfigLosslessPreset(&config, 6); // max effort, lossless
    config.lossless = 1;
    config.quality = 100.f;
    config.exact = 1; // keep exact RGB under fully-transparent regions

    WebPPicture pic;
    WebPPictureInit(&pic);
    pic.use_argb = 1;
    pic.width = W;
    pic.height = H;

    int imported = (channels == 4)
        ? WebPPictureImportRGBA(&pic, pixels, W * 4)
        : WebPPictureImportRGB(&pic, pixels, W * 3);

    std::vector<uint8_t> out;
    if (imported) {
        WebPMemoryWriter wrt;
        WebPMemoryWriterInit(&wrt);
        pic.writer = WebPMemoryWrite;
        pic.custom_ptr = &wrt;
        if (WebPEncode(&config, &pic)) {
            out.assign(wrt.mem, wrt.mem + wrt.size);
        }
        WebPMemoryWriterClear(&wrt);
    }
    WebPPictureFree(&pic);
    return out;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "BEK " << BEK_VERSION << '\n';
        return 0;
    }
    if (argc < 3) {
        std::cerr << "Usage: ply_to_bek <input.ply> <output_file.bek> [options]\n";
        return 1;
    }

    std::string input_ply = argv[1];
    std::string output_dir = argv[2];
    // Default encoder parameters; override individual attributes with CLI options.
    int dc_stages = 2;
    // SH AC is quantized jointly: one RVQ codebook spanning all 45 AC dims,
    // sh_stages residual stages of 256 clusters each. Joint RVQ exploits the
    // cross-band correlation a banded split throws away.
    int sh_stages = 3;
    // Scale RVQ uses a shared codebook over 3D log-scales when scale_stages > 0.
    // Otherwise scales use independent scalar min-max quantization.
    int scale_stages = 0;
    int scale_bits = 8;
    int pos_fractional_bits = 9;
    // Rotation magnitude bits for the smallest-three pack (<=6 keeps 3 bytes).
    int rot_mag_bits = 6;
    // Byte-aligned smallest-three split (3x8-bit planes + idx plane): more raw
    // bits than the 24-bit pack, but clean per-plane statistics for rc1.
    bool rot_split = false;
    int vq_iters = 30;
    std::string load_vq_path;
    std::string save_vq_path;
    // Disable specialized position and rotation coders; retain WebP, rc1, rc2,
    // rcx and raw candidates where applicable.
    bool legacy_tail = false;
    // Windowed 2-opt refinement shortens consecutive position jumps.
    // Pass --sort-refine 0 to disable refinement.
    int sort_refine = 128;
    // 2-opt objective: 0 = L1 distance, 1 = per-axis bit-length (~posmag cost).
    int sort_metric = 0;
    // Base ordering: 0 = Morton (Z-order), 1 = greedy nearest-neighbour path.
    int sort_mode = 0;
    // SH AC whitening applies a per-dimension weight var_d^(-P) to the
    // Lloyd distance metric. P=0 uses plain L2; P=1 gives full whitening.
    double sh_whiten = 0.0;
    // Attribute omission flags exclude both channel metadata and payload.
    // The decoder supplies defaults for omitted attributes.
    bool no_positions = false;
    bool no_opacity = false;
    bool no_rotation = false;
    bool no_scale = false;
    // Flat k-means uses one codebook instead of residual stages. Indices use
    // one byte for <=256 clusters and two bytes otherwise. The corresponding
    // dc_stages or sh_stages setting is ignored when flat k-means is enabled.
    int dc_kmeans = 0;
    int sh_kmeans = 0;
    // Store SH codebook entries as raw float32 instead of scalar-quantizing
    // them to shared levels.
    bool cb_float = false;
    // Positions as ONE interleaved xyz channel (9 bytes/splat) instead of three
    // independent pos_x/pos_y/pos_z planes, coded by posmag3 (per-axis models +
    // cross-axis magnitude context). Lossless, same quantization; default OFF.
    bool pos_joint = false;
    // Relabel each RVQ codebook along a greedy nearest-neighbour chain before
    // index assignment to improve locality in the entropy-coded index planes.
    // --no-rvq-cb-order disables relabelling.
    bool cb_chain_order = true;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dc-stages" && i + 1 < argc) dc_stages = std::stoi(argv[++i]);
        else if (arg == "--sh-stages" && i + 1 < argc) sh_stages = std::stoi(argv[++i]);
        // Compatibility aliases: --sh1-stages sets the joint stage count;
        // --sh2-stages and --sh3-stages consume a value but do not affect encoding.
        else if (arg == "--sh1-stages" && i + 1 < argc) sh_stages = std::stoi(argv[++i]);
        else if (arg == "--sh2-stages" && i + 1 < argc) ++i;
        else if (arg == "--sh3-stages" && i + 1 < argc) ++i;
        else if (arg == "--pos-bits" && i + 1 < argc) pos_fractional_bits = std::stoi(argv[++i]);
        else if (arg == "--rot-mag-bits" && i + 1 < argc) rot_mag_bits = std::clamp(std::stoi(argv[++i]), 2, 6);
        else if (arg == "--rot-split") rot_split = true;
        else if (arg == "--vq-iters" && i + 1 < argc) vq_iters = std::stoi(argv[++i]);
        else if (arg == "--scale-stages" && i + 1 < argc) scale_stages = std::stoi(argv[++i]);
        else if (arg == "--scale-bits" && i + 1 < argc) scale_bits = std::clamp(std::stoi(argv[++i]), 1, 8);
        else if (arg == "--load-vq" && i + 1 < argc) load_vq_path = argv[++i];
        else if (arg == "--save-vq" && i + 1 < argc) save_vq_path = argv[++i];
        else if (arg == "--legacy-tail") legacy_tail = true;
        else if (arg == "--sort-refine" && i + 1 < argc) sort_refine = std::stoi(argv[++i]);
        else if (arg == "--sort-metric" && i + 1 < argc) {
            std::string m = argv[++i];
            sort_metric = (m == "bits" || m == "1") ? 1 : 0;
        }
        else if (arg == "--sort-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            sort_mode = (m == "nn" || m == "1") ? 1 : 0;
        }
        else if (arg == "--sh-whiten" && i + 1 < argc) sh_whiten = std::stod(argv[++i]);
        else if (arg == "--no-positions") no_positions = true;
        else if (arg == "--no-opacity") no_opacity = true;
        else if (arg == "--no-rotation") no_rotation = true;
        else if (arg == "--no-scale") no_scale = true;
        else if (arg == "--dc-kmeans" && i + 1 < argc) dc_kmeans = std::stoi(argv[++i]);
        else if (arg == "--sh-kmeans" && i + 1 < argc) sh_kmeans = std::stoi(argv[++i]);
        else if (arg == "--cb-float") cb_float = true;
        else if (arg == "--pos-joint") pos_joint = true;
        else if (arg == "--rvq-cb-order") cb_chain_order = true;   // default; kept so old scripts still parse
        else if (arg == "--no-rvq-cb-order") cb_chain_order = false;
    }

    std::cout << "[BEK] DC Stages: " << dc_stages
              << ", SH Stages (joint): " << sh_stages
              << ", VQ Iters: " << vq_iters << ", Pos Bits: " << pos_fractional_bits << std::endl;
    if (no_positions || no_opacity || no_rotation || no_scale) {
        std::cout << "[BEK] Excluded channels:"
                  << (no_positions ? " positions" : "")
                  << (no_opacity ? " opacity" : "")
                  << (no_rotation ? " rotation" : "")
                  << (no_scale ? " scale" : "") << std::endl;
    }

    auto vertex_data = read_ply(input_ply);
    if (!vertex_data) return 1;

    std::cout << "[BEK] Points: " << vertex_data->count << std::endl;

    {
        const char* base = (sort_mode == 1) ? "NN" : "Morton";
        if (sort_refine > 1)
            std::cout << "Sorting (" << base << " + 2-opt refine, window=" << sort_refine
                      << ", metric=" << (sort_metric == 1 ? "bits" : "L1") << ")..." << std::endl;
        else
            std::cout << "Sorting (" << base << ")..." << std::endl;
    }
    // BEK_NO_SORT preserves input row order in the reconstructed PLY.
    if (!std::getenv("BEK_NO_SORT"))
        vertex_data->sort_morton(sort_refine, sort_metric, sort_mode);

    uint32_t N = vertex_data->count;
    int W = static_cast<int>(std::ceil(std::sqrt(N)));
    int H = static_cast<int>(std::ceil(static_cast<double>(N) / W));
    
    std::mutex channels_mutex;
    std::vector<ChannelData> channels;
    std::vector<std::future<void>> futures;

    // 1. Positions (with DPCM)
    futures.push_back(std::async(std::launch::async, [&, no_positions, pos_joint]() {
        if (!vertex_data->x.empty() && !no_positions) {
            // Zigzag DPCM: signed deltas map to small unsigned values, so the
            // mid/high byte planes are near-zero and compress away.
            auto pack_x = quantize_position_spz_zz(vertex_data->x, pos_fractional_bits);
            auto pack_y = quantize_position_spz_zz(vertex_data->y, pos_fractional_bits);
            auto pack_z = quantize_position_spz_zz(vertex_data->z, pos_fractional_bits);

            std::lock_guard<std::mutex> lock(channels_mutex);
            if (pos_joint) {
                // Same bytes, one stream: [x0 y0 z0][x1 y1 z1]... (3 bytes each).
                std::vector<uint8_t> pack(pack_x.size() * 3);
                for (size_t i = 0; i < pack_x.size() / 3; ++i)
                    for (int b = 0; b < 3; ++b) {
                        pack[i * 9 + 0 + b] = pack_x[i * 3 + b];
                        pack[i * 9 + 3 + b] = pack_y[i * 3 + b];
                        pack[i * 9 + 6 + b] = pack_z[i * 3 + b];
                    }
                channels.push_back(ChannelData{"pos_xyz", 9, pack, {{"type", "spz_dpcm_zz"}, {"bits", 24}, {"frac", pos_fractional_bits}}});
            } else {
            channels.push_back(ChannelData{"pos_x", 3, pack_x, {{"type", "spz_dpcm_zz"}, {"bits", 24}, {"frac", pos_fractional_bits}}});
            channels.push_back(ChannelData{"pos_y", 3, pack_y, {{"type", "spz_dpcm_zz"}, {"bits", 24}, {"frac", pos_fractional_bits}}});
            channels.push_back(ChannelData{"pos_z", 3, pack_z, {{"type", "spz_dpcm_zz"}, {"bits", 24}, {"frac", pos_fractional_bits}}});
            }
        }
    }));

    // 2. Opacity
    futures.push_back(std::async(std::launch::async, [&, no_opacity]() {
        if (!vertex_data->opacity.empty() && !no_opacity) {
            auto raw = quantize_opacity_sigmoid(vertex_data->opacity);
            std::lock_guard<std::mutex> lock(channels_mutex);
            channels.push_back(ChannelData{"opacity", 1, raw, {{"type", "sigmoid"}, {"bits", 8}}});
        }
    }));

    // 3. Rotations (smallest-three: packed 24-bit, or byte-aligned split)
    futures.push_back(std::async(std::launch::async, [&, rot_mag_bits, rot_split, no_rotation]() {
        if (!vertex_data->rot_0.empty() && !no_rotation) {
            if (rot_split) {
                std::vector<uint8_t> pack_rgb, pack_idx;
                pack_quaternion_smallest_three_split(vertex_data->rot_0, vertex_data->rot_1, vertex_data->rot_2, vertex_data->rot_3, pack_rgb, pack_idx);
                std::lock_guard<std::mutex> lock(channels_mutex);
                channels.push_back(ChannelData{"rotation", 3, pack_rgb, {{"type", "smallest3_split"}, {"bits", 24}}});
                channels.push_back(ChannelData{"rotation_idx", 1, pack_idx, {{"type", "rot_idx"}, {"bits", 8}}});
            } else {
                auto pack = pack_quaternion_smallest_three_var(vertex_data->rot_0, vertex_data->rot_1, vertex_data->rot_2, vertex_data->rot_3, rot_mag_bits);
                std::lock_guard<std::mutex> lock(channels_mutex);
                channels.push_back(ChannelData{"rotation", 3, pack, {{"type", "smallest3_24"}, {"bits", 24}, {"mag_bits", rot_mag_bits}}});
            }
        }
    }));

    // 4. Scales
    futures.push_back(std::async(std::launch::async, [&, scale_stages, scale_bits, vq_iters, no_scale]() {
        if (no_scale) return;
        // Scale RVQ: shared 3D codebook, scale_stages residual stages, emitted as
        // vq_scale_s{n} index planes (same scheme as DC). Much cheaper than 3
        // independent bytes since log-scales cluster tightly.
        if (scale_stages > 0) {
            if (scale_bits != 8) {
                std::cerr << "[BEK] Warning: --scale-bits " << scale_bits
                          << " ignored (--scale-stages " << scale_stages
                          << " > 0 selects the RVQ path, not the uniform quantizer)."
                          << std::endl;
            }
            std::vector<std::vector<float>> features = {
                vertex_data->get_scale(0), vertex_data->get_scale(1), vertex_data->get_scale(2) };
            RVQQuantized q = pack_rvq_pooled(
                compute_rvq(features, scale_stages, 256, vq_iters, {}, cb_chain_order), 3);

            std::lock_guard<std::mutex> lock(channels_mutex);
            for (int s = 0; s < scale_stages; ++s) {
                std::vector<uint8_t> idx8(N);
                for (uint32_t i = 0; i < N; ++i) idx8[i] = static_cast<uint8_t>(q.indices[s][i]);
                json m;
                m["type"] = "rvq_stage_scale_quantized";
                m["stage"] = s;
                m["levels"] = q.levels[s];
                m["cb_indices"] = q.cb_indices[s];
                channels.push_back(ChannelData{"vq_scale_s" + std::to_string(s), 1, idx8, m});
            }
            return;
        }
        std::vector<uint8_t> all_scales(N * 3);
        json m;
        double min_v[3], max_v[3];
        for (int i = 0; i < 3; ++i) {
            auto raw = quantize_to_u8_levels(vertex_data->get_scale(i), scale_bits, min_v[i], max_v[i]);
            for(uint32_t j=0; j<N; ++j) all_scales[j*3 + i] = raw[j];
        }
        m["type"] = "u8_levels";
        m["bits"] = scale_bits;
        m["min"] = {min_v[0], min_v[1], min_v[2]};
        m["max"] = {max_v[0], max_v[1], max_v[2]};
        std::lock_guard<std::mutex> lock(channels_mutex);
        channels.push_back(ChannelData{"scales", 3, all_scales, m});
    }));

    // 5. VQ Groups (DC and joint SH AC)
    futures.push_back(std::async(std::launch::async, [&, dc_stages, sh_stages, vq_iters, load_vq_path, save_vq_path, dc_kmeans, sh_kmeans, cb_chain_order]() {
        // DC flat k-means uses a single codebook.
        if (!vertex_data->f_dc_0.empty() && dc_kmeans > 0) {
            std::vector<std::vector<float>> features = { vertex_data->f_dc_0, vertex_data->f_dc_1, vertex_data->f_dc_2 };
            std::cout << "Quantizing DC with flat k-means (" << dc_kmeans << " clusters, " << vq_iters << " iters)..." << std::endl;
            VQResult vq = compute_vq(features, dc_kmeans, vq_iters);
            int K = vq.codebook.empty() ? 0 : static_cast<int>(vq.codebook.size() / 3);
            // Lloyd-quantize the codebook floats to 256 shared levels (JSON space).
            auto qcb = quantize_scalar_lloyd(vq.codebook, 256, 20);
            int ibytes = (K <= 256) ? 1 : 2;
            std::vector<uint8_t> idxb(static_cast<size_t>(N) * ibytes);
            for (uint32_t i = 0; i < N; ++i) {
                uint16_t idx = vq.indices[i];
                idxb[i * ibytes] = static_cast<uint8_t>(idx & 0xFF);
                if (ibytes == 2) idxb[i * ibytes + 1] = static_cast<uint8_t>((idx >> 8) & 0xFF);
            }
            json m;
            m["type"] = "kmeans_dc";
            m["clusters"] = K;
            m["index_bytes"] = ibytes;
            m["levels"] = qcb.levels;
            m["cb_indices"] = qcb.indices;
            std::lock_guard<std::mutex> lock(channels_mutex);
            channels.push_back(ChannelData{"km_dc", ibytes, idxb, m});
        }
        // DC RVQ (default)
        else if (!vertex_data->f_dc_0.empty() && dc_stages > 0) {
            std::vector<std::vector<float>> features = { vertex_data->f_dc_0, vertex_data->f_dc_1, vertex_data->f_dc_2 };
            RVQResult res;
            
            bool loaded = false;
            if (!load_vq_path.empty()) {
                std::string path = load_vq_path + "_rvq_dc.json";
                if (fs::exists(path)) {
                    try {
                        std::ifstream in(path);
                        json j; in >> j;
                        if (j["stages"] == dc_stages) {
                            res.codebooks = j["codebooks"].get<std::vector<std::vector<float>>>();
                            // Re-assign indices
                            std::vector<std::vector<float>> cur = features;
                            for(int s=0; s<dc_stages; ++s) {
                                auto idx = assign_vq(cur, res.codebooks[s]);
                                res.indices.push_back(idx);
                                for(uint32_t i=0; i<N; ++i) {
                                    for(int d=0; d<3; ++d) cur[d][i] -= res.codebooks[s][idx[i]*3+d];
                                }
                            }
                            loaded = true;
                            std::cout << "Loaded DC RVQ codebooks from " << path << std::endl;
                        }
                    } catch(...) {}
                }
            }

            if (!loaded) {
                std::cout << "Quantizing DC with RVQ (" << dc_stages << " stages, " << vq_iters << " iters)..." << std::endl;
                res = compute_rvq(features, dc_stages, 256, vq_iters, {}, cb_chain_order);
                if (!save_vq_path.empty()) {
                    json j; j["stages"] = dc_stages; j["codebooks"] = res.codebooks;
                    std::ofstream out(save_vq_path + "_rvq_dc.json");
                    out << j.dump();
                }
            }
            // Lloyd Quantization for the codebook entries themselves (to save JSON space)
            RVQQuantized q = pack_rvq_pooled(res, 3);

            std::lock_guard<std::mutex> lock(channels_mutex);
            for(int s=0; s<dc_stages; ++s) {
                std::vector<uint8_t> idx8(N);
                for(uint32_t i=0; i<N; ++i) idx8[i] = static_cast<uint8_t>(q.indices[s][i]);

                json m;
                m["type"] = "rvq_stage_quantized";
                m["stage"] = s;
                m["levels"] = q.levels[s];
                m["cb_indices"] = q.cb_indices[s]; // Much smaller in JSON than float list
                channels.push_back(ChannelData{"vq_dc_s" + std::to_string(s), 1, idx8, m});
            }
        }

        // Joint SH AC flat k-means uses one codebook over all AC dimensions.
        if (!vertex_data->get_f_rest(0).empty() && sh_kmeans > 0) {
            std::vector<std::vector<float>> features;
            for (int i = 0; i < 45; ++i) {
                const auto& vec = vertex_data->get_f_rest(i);
                if (!vec.empty()) features.push_back(vec);
            }
            if (!features.empty()) {
                int dims = features.size();
                std::cout << "Quantizing ALL SH AC (" << dims << " dims) with flat k-means ("
                          << sh_kmeans << " clusters, " << vq_iters << " iters)..." << std::endl;
                VQResult vq = compute_vq(features, sh_kmeans, vq_iters);
                int K = vq.codebook.empty() ? 0 : static_cast<int>(vq.codebook.size() / dims);
                int ibytes = (K <= 256) ? 1 : 2;
                std::vector<uint8_t> idxb(static_cast<size_t>(N) * ibytes);
                for (uint32_t i = 0; i < N; ++i) {
                    uint16_t idx = vq.indices[i];
                    idxb[i * ibytes] = static_cast<uint8_t>(idx & 0xFF);
                    if (ibytes == 2) idxb[i * ibytes + 1] = static_cast<uint8_t>((idx >> 8) & 0xFF);
                }
                json m;
                m["type"] = "kmeans_sh_ac";
                m["dims"] = dims;
                m["clusters"] = K;
                m["index_bytes"] = ibytes;
                if (cb_float) {
                    m["codebook"] = vq.codebook;  // raw float centroids (no scalar requant)
                } else {
                    auto qcb = quantize_scalar_lloyd(vq.codebook, 256, 20);
                    m["levels"] = qcb.levels;
                    m["cb_indices"] = qcb.indices;
                }
                std::lock_guard<std::mutex> lock(channels_mutex);
                channels.push_back(ChannelData{"km_sh_ac", ibytes, idxb, m});
            }
        }
        // Joint SH AC: a single RVQ codebook spanning all 45 AC dims, sh_stages
        // residual stages. Joint quantization captures the cross-band correlation
        // a banded split discards.
        else if (!vertex_data->get_f_rest(0).empty() && sh_stages > 0) {
            std::vector<std::vector<float>> features;
            for (int i = 0; i < 45; ++i) {
                const auto& vec = vertex_data->get_f_rest(i);
                if (!vec.empty()) features.push_back(vec);
            }
            if (!features.empty()) {
                int dims = features.size();

                // Per-dim whitening weights for the joint RVQ metric. var_d^(-P):
                // P=0 leaves weights empty (plain L2). Normalised to geometric mean
                // 1 so the absolute distortion scale (and thus codebook span) is
                // preserved — only the relative per-dim emphasis changes.
                std::vector<float> sh_weights;
                if (sh_whiten > 0.0) {
                    std::vector<double> var(dims, 0.0);
                    for (int d = 0; d < dims; ++d) {
                        const auto& col = features[d];
                        double mean = 0.0; for (float v : col) mean += v; mean /= col.size();
                        double acc = 0.0; for (float v : col) { double e = v - mean; acc += e * e; }
                        var[d] = acc / std::max<size_t>(1, col.size());
                    }
                    std::vector<double> w(dims);
                    double log_sum = 0.0;
                    for (int d = 0; d < dims; ++d) {
                        w[d] = std::pow(var[d] + 1e-12, -sh_whiten);
                        log_sum += std::log(w[d]);
                    }
                    double gmean = std::exp(log_sum / dims);
                    sh_weights.resize(dims);
                    for (int d = 0; d < dims; ++d) sh_weights[d] = static_cast<float>(w[d] / gmean);
                    std::cout << "SH whitening P=" << sh_whiten << " (per-dim var-weighted Lloyd metric)" << std::endl;
                }

                RVQResult res;
                bool loaded = false;
                if (!load_vq_path.empty()) {
                    std::string path = load_vq_path + "_rvq_sh.json";
                    if (fs::exists(path)) {
                        try {
                            std::ifstream in(path);
                            json j; in >> j;
                            if (j["stages"] == sh_stages && j["dims"] == dims) {
                                res.codebooks = j["codebooks"].get<std::vector<std::vector<float>>>();
                                std::vector<std::vector<float>> cur = features;
                                for (int s = 0; s < sh_stages; ++s) {
                                    auto idx = assign_vq(cur, res.codebooks[s]);
                                    res.indices.push_back(idx);
                                    for (uint32_t i = 0; i < N; ++i)
                                        for (int d = 0; d < dims; ++d) cur[d][i] -= res.codebooks[s][idx[i]*dims+d];
                                }
                                loaded = true;
                                std::cout << "Loaded SH RVQ codebooks from " << path << std::endl;
                            }
                        } catch(...) {}
                    }
                }

                if (!loaded) {
                    std::cout << "Quantizing ALL SH AC (" << dims << " dims) with RVQ ("
                              << sh_stages << " stages, " << vq_iters << " iters)..." << std::endl;
                    res = compute_rvq(features, sh_stages, 256, vq_iters, sh_weights, cb_chain_order);
                    if (!save_vq_path.empty()) {
                        json j; j["stages"] = sh_stages; j["dims"] = dims; j["codebooks"] = res.codebooks;
                        std::ofstream out(save_vq_path + "_rvq_sh.json");
                        out << j.dump();
                    }
                }

                // Lloyd-quantize the codebook entries (saves JSON/header space).
                RVQQuantized q = pack_rvq_pooled(res, dims);

                std::lock_guard<std::mutex> lock(channels_mutex);
                for (int s = 0; s < sh_stages; ++s) {
                    std::vector<uint8_t> idx8(N);
                    for (uint32_t i = 0; i < N; ++i) idx8[i] = static_cast<uint8_t>(q.indices[s][i]);

                    json m;
                    m["type"] = "rvq_stage_sh_ac_quantized";
                    m["stage"] = s;
                    m["dims"] = dims;
                    if (cb_float) {
                        m["codebook"] = res.codebooks[s];  // raw float centroids (no requant)
                    } else {
                        m["levels"] = q.levels[s];
                        m["cb_indices"] = q.cb_indices[s];
                    }
                    channels.push_back(ChannelData{"vq_sh_ac_s" + std::to_string(s), 1, idx8, m});
                }
            }
        }
    }));

    for (auto& f : futures) f.get();

    // The attribute futures push into `channels` in completion order, which is
    // timing-dependent — so the payload byte layout (and thus the global-Zstd
    // output) was not reproducible run-to-run. Sort by channel name to pin a
    // canonical order: deterministic, byte-identical output for identical input.
    std::sort(channels.begin(), channels.end(),
              [](const ChannelData& a, const ChannelData& b) { return a.name < b.name; });

    // --- Final Binary Packing (BEK v3) with Global Zstd ---
    fs::path final_output = fs::absolute(output_dir);
    if (final_output.extension() != ".bek") final_output += ".bek";

    // 1. Assemble the uncompressed container in memory
    std::vector<uint8_t> uncompressed_buffer;
    
    // Placeholder for header size
    uint32_t header_size_placeholder = 0;
    uncompressed_buffer.insert(uncompressed_buffer.end(), reinterpret_cast<uint8_t*>(&header_size_placeholder), reinterpret_cast<uint8_t*>(&header_size_placeholder) + 4);

    json header_json;
    header_json["count"] = N;
    header_json["width"] = W;
    header_json["height"] = H;

    // 2. Prepare Data Blobs and Header Entries
    // WebP encoding is independent per-channel and deterministic; encode all
    // channels in parallel, then assemble payload/header in original channel
    // order so the produced bytes are bit-identical to the serial version.
    struct EncodedChannel {
        std::vector<uint8_t> blob;
        json meta;
        bool ok = false;
    };
    std::vector<EncodedChannel> encoded(channels.size());
    std::vector<std::future<void>> webp_futures;
    for (size_t ci = 0; ci < channels.size(); ++ci) {
        webp_futures.push_back(std::async(std::launch::async, [&, ci]() {
            const ChannelData& ch = channels[ci];
            std::vector<uint8_t> padded = ch.data;
            padded.resize(W * H * ch.channels, 0);

            // Skip WebP candidates for the channels listed below to reduce encoding
            // work. Other eligible channels retain WebP as a candidate.
            const bool skip_webp =
                ch.name == "rotation" || ch.name == "scales" ||
                ch.name == "opacity" || ch.name == "vq_dc_s0";

            std::vector<uint8_t> blob;
            if (!skip_webp) {
                if (ch.channels == 3) {
                    blob = webp_encode_lossless(padded.data(), W, H, 3);
                } else if (ch.channels == 4) {
                    blob = webp_encode_lossless(padded.data(), W, H, 4);
                } else if (ch.channels == 1) {
                    std::vector<uint8_t> rgb(W * H * 3, 0);
                    for (int i = 0; i < W * H; ++i) {
                        rgb[i*3] = padded[i]; rgb[i*3+1] = padded[i]; rgb[i*3+2] = padded[i];
                    }
                    blob = webp_encode_lossless(rgb.data(), W, H, 3);
                }
            }

            // Compare adaptive range-coder candidates against WebP and raw storage.
            auto rc2 = rc2_encode(ch.data, ch.channels);
            auto rc1 = rc1_encode(ch.data, ch.channels);

            // Residual VQ index planes: also try the cross-stage coder, with
            // the previous stage's index plane as context (residual stage
            // distributions depend strongly on the coarse stage's cell).
            std::vector<uint8_t> rcx;
            std::string ctx_name;
            const size_t spos = ch.name.find_last_of('s');
            if (ch.name.rfind("vq_", 0) == 0 && spos != std::string::npos) {
                const int stage = std::atoi(ch.name.c_str() + spos + 1);
                if (stage > 0) {
                    ctx_name = ch.name.substr(0, spos + 1) + std::to_string(stage - 1);
                    for (const auto& other : channels) {
                        if (other.name == ctx_name && other.data.size() == ch.data.size()) {
                            rcx = rcx_encode(ch.data, other.data.data());
                            break;
                        }
                    }
                }
            }

            // Position channels can use a dedicated length-prefix and mantissa coder
            // for 24-bit zigzag DPCM values. It participates in the payload-size comparison.
            std::vector<uint8_t> pmag;
            if (!legacy_tail && ch.name == "pos_xyz") pmag = posmag3_encode(ch.data);
            else if (!legacy_tail && ch.name.rfind("pos_", 0) == 0) pmag = posmag_encode(ch.data);

            // Rotation: dedicated field-decomposition coder for the smallest3_24
            // packed word (index + 3 sign/magnitude fields). Byte trees lose the
            // structure across byte boundaries. Only valid for mag_bits<=6 (the
            // 23-bit layout must fit 24 bits); split-mode rotation is byte-aligned
            // already and stays on webp/rc1.
            std::vector<uint8_t> rpk;
            if (!legacy_tail && ch.name == "rotation" && ch.metadata.value("type", "") == "smallest3_24") {
                const int mb = ch.metadata.value("mag_bits", 6);
                if (mb >= 1 && mb <= 6) rpk = rotpack_encode(ch.data, mb);
            }

            encoded[ci].meta = ch.metadata;
            const size_t webp_sz = blob.empty() ? SIZE_MAX : blob.size();
            const size_t rc2_sz = rc2.empty() ? SIZE_MAX : rc2.size();
            const size_t rc1_sz = rc1.empty() ? SIZE_MAX : rc1.size();
            const size_t rcx_sz = rcx.empty() ? SIZE_MAX : rcx.size();
            const size_t pmag_sz = pmag.empty() ? SIZE_MAX : pmag.size();
            const size_t rpk_sz = rpk.empty() ? SIZE_MAX : rpk.size();
            const size_t raw_sz = ch.data.size();
            const size_t best = std::min({pmag_sz, rpk_sz, rcx_sz, rc2_sz, rc1_sz, webp_sz, raw_sz});
            if (best == pmag_sz) {
                encoded[ci].blob = std::move(pmag);
                encoded[ci].meta["codec"] = ch.name == "pos_xyz" ? "posmag3" : "posmag";
            } else if (best == rpk_sz) {
                encoded[ci].blob = std::move(rpk);
                encoded[ci].meta["codec"] = "rotpack";
            } else if (best == rcx_sz) {
                encoded[ci].blob = std::move(rcx);
                encoded[ci].meta["codec"] = "rcx";
                encoded[ci].meta["ctx_ch"] = ctx_name;
            } else if (best == rc2_sz) {
                encoded[ci].blob = std::move(rc2);
                encoded[ci].meta["codec"] = "rc2";
            } else if (best == rc1_sz) {
                encoded[ci].blob = std::move(rc1);
                encoded[ci].meta["codec"] = "rc1";
            } else if (best == webp_sz) {
                encoded[ci].blob = std::move(blob);
                encoded[ci].meta["codec"] = "webp";
            } else {
                encoded[ci].blob = ch.data;
                encoded[ci].meta["codec"] = "raw";
            }
            encoded[ci].ok = true;
        }));
    }
    for (auto& f : webp_futures) f.get();

    std::vector<uint8_t> data_payload;
    for (size_t ci = 0; ci < channels.size(); ++ci) {
        if (!encoded[ci].ok) continue;
        size_t offset = data_payload.size();
        data_payload.insert(data_payload.end(), encoded[ci].blob.begin(), encoded[ci].blob.end());

        json entry;
        entry["name"] = channels[ci].name;
        entry["offset"] = offset;
        entry["size"] = encoded[ci].blob.size();
        entry["channels"] = channels[ci].channels;
        entry["meta"] = encoded[ci].meta;
        header_json["channels"].push_back(entry);
    }

    // Write CBOR header to buffer
    std::vector<uint8_t> header_bin = json::to_cbor(header_json);
    uint32_t actual_header_size = static_cast<uint32_t>(header_bin.size());
    std::memcpy(uncompressed_buffer.data(), &actual_header_size, 4);
    uncompressed_buffer.insert(uncompressed_buffer.end(), header_bin.begin(), header_bin.end());
    
    // Append payload to buffer
    uncompressed_buffer.insert(uncompressed_buffer.end(), data_payload.begin(), data_payload.end());

    // 3. Global Zstd Compression: max level + long-distance matching over a
    // 27-bit (128MB) window, so cross-channel redundancy at large offsets is
    // also removed. windowLog 27 is stored in the frame; the decoder raises its
    // d_windowLogMax to match.
    size_t const max_compressed_size = ZSTD_compressBound(uncompressed_buffer.size());
    std::vector<uint8_t> compressed_buffer(max_compressed_size);

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, 22);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_enableLongDistanceMatching, 1);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_windowLog, 27);
    // Multi-threaded final compress: split into jobs across cores. Produces a
    // valid frame the decoder reads unchanged; compressed bytes may differ
    // marginally from the single-thread output (job boundaries). No-op (param
    // returns an ignored error) if libzstd was built without ZSTD_MULTITHREAD.
    {
        unsigned hw = std::thread::hardware_concurrency();
        int workers = hw > 1 ? static_cast<int>(hw) : 1;
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers);
    }
    size_t const compressed_size = ZSTD_compress2(cctx,
                                                 compressed_buffer.data(), compressed_buffer.size(),
                                                 uncompressed_buffer.data(), uncompressed_buffer.size());
    ZSTD_freeCCtx(cctx);

    if (ZSTD_isError(compressed_size)) {
        std::cerr << "Error: Zstd compression failed: " << ZSTD_getErrorName(compressed_size) << std::endl;
        return 1;
    }

    // 4. Final Write to Disk
    std::ofstream out(final_output, std::ios::binary);
    out.write("BEK3", 4);
    uint64_t uncompressed_size_64 = static_cast<uint64_t>(uncompressed_buffer.size());
    out.write(reinterpret_cast<const char*>(&uncompressed_size_64), 8);
    out.write(reinterpret_cast<const char*>(compressed_buffer.data()), compressed_size);

    size_t input_ply_size = fs::file_size(input_ply);
    size_t final_bek_size = out.tellp();

    std::cout << "Success! Compressed Binary BEK v67 saved at: " << final_output << std::endl;
    std::cout << "Input PLY Size:  " << (float)input_ply_size/1024/1024 << " MB" << std::endl;
    std::cout << "Internal Buffer: " << (float)uncompressed_buffer.size()/1024/1024 << " MB" << std::endl;
    std::cout << "Final BEK Size:  " << (float)final_bek_size/1024/1024 << " MB" << std::endl;
    std::cout << "Comp. Ratio:     " << (float)input_ply_size/final_bek_size << "x" << std::endl;
    std::cout << "Zstd Efficiency: " << (float)uncompressed_buffer.size()/compressed_size << "x" << std::endl;

    return 0;
}
