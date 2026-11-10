#pragma once

#include "ply_reader.h"
#include <string>
#include <vector>

void write_ply_file(const std::string& path, const VertexData& vertex_data);
