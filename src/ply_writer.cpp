#include "ply_writer.h"
#include "tinyply.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

void write_ply_file(const std::string& path, const VertexData& vertex_data) {
    std::filebuf fb;
    if (fb.open(path, std::ios::out | std::ios::binary) == nullptr) {
        throw std::runtime_error("Failed to open PLY output file: " + path);
    }
    std::ostream out_stream(&fb);
    
    tinyply::PlyFile file;
    
    // 1. Core Geometry (Matches original order)
    file.add_properties_to_element("vertex", {"x"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(const_cast<float*>(vertex_data.x.data())), tinyply::Type::INVALID, 0);
    file.add_properties_to_element("vertex", {"y"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(const_cast<float*>(vertex_data.y.data())), tinyply::Type::INVALID, 0);
    file.add_properties_to_element("vertex", {"z"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(const_cast<float*>(vertex_data.z.data())), tinyply::Type::INVALID, 0);

    std::vector<float> zeros(vertex_data.count, 0.0f);
    file.add_properties_to_element("vertex", {"nx"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(zeros.data()), tinyply::Type::INVALID, 0);
    file.add_properties_to_element("vertex", {"ny"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(zeros.data()), tinyply::Type::INVALID, 0);
    file.add_properties_to_element("vertex", {"nz"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(zeros.data()), tinyply::Type::INVALID, 0);

    // 2. Base Color (f_dc) - Crucial for "darkness" issue
    for (int i = 0; i < 3; ++i) {
        std::string name = "f_dc_" + std::to_string(i);
        auto it = vertex_data.f_dc.find(name);
        if (it != vertex_data.f_dc.end() && !it->second.empty()) {
            file.add_properties_to_element("vertex", {name}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(const_cast<float*>(it->second.data())), tinyply::Type::INVALID, 0);
        } else {
            file.add_properties_to_element("vertex", {name}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(zeros.data()), tinyply::Type::INVALID, 0);
        }
    }

    // 3. SH Harmonics (f_rest)
    for (int i = 0; i < 45; ++i) {
        std::string name = "f_rest_" + std::to_string(i);
        auto it = vertex_data.f_rest.find(name);
        if (it != vertex_data.f_rest.end() && !it->second.empty()) {
            file.add_properties_to_element("vertex", {name}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(const_cast<float*>(it->second.data())), tinyply::Type::INVALID, 0);
        } else {
            file.add_properties_to_element("vertex", {name}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(zeros.data()), tinyply::Type::INVALID, 0);
        }
    }

    // 4. Appearance attributes (opacity, scale, rot)
    if (!vertex_data.opacity.empty()) {
        file.add_properties_to_element("vertex", {"opacity"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(const_cast<float*>(vertex_data.opacity.data())), tinyply::Type::INVALID, 0);
    }

    if (!vertex_data.scale.empty()) {
        file.add_properties_to_element("vertex", {"scale_0", "scale_1", "scale_2"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(const_cast<float*>(vertex_data.scale.data())), tinyply::Type::INVALID, 0);
    }

    if (!vertex_data.rot.empty()) {
        file.add_properties_to_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}, tinyply::Type::FLOAT32, vertex_data.count, reinterpret_cast<uint8_t*>(const_cast<float*>(vertex_data.rot.data())), tinyply::Type::INVALID, 0);
    }

    file.write(out_stream, true); // binary
}
