#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Adaptive contextual byte coder built on an LZMA-style binary range coder.
// Each byte is coded through a 255-node bit tree. Context selection (rc2):
//  - plane 0 (or stride 1): previous sample of the same plane (full byte)
//    plus the one before it quantized to 4 bits  -> 4096 contexts
//  - planes k>0 of interleaved data: the same sample's previous plane byte
//    (full) plus the previous same-plane byte quantized to 5 bits -> 8192
// Both sides derive contexts from already-decoded bytes, so the model adapts
// in lockstep and no tables travel in the container — decode just needs the
// exact original size and the same stride.
std::vector<uint8_t> rc2_encode(const std::vector<uint8_t>& data, int stride);
std::vector<uint8_t> rc2_decode(const uint8_t* blob, size_t blob_size, size_t out_size, int stride);

// Plain order-1 variant (context = previous same-plane byte, 256 contexts).
// Better than rc2 on channels whose values are arbitrary labels (VQ index
// planes): fewer contexts adapt faster. The encoder tries both and keeps the
// smaller; the codec name in meta says which one to use on decode.
std::vector<uint8_t> rc1_encode(const std::vector<uint8_t>& data, int stride);
std::vector<uint8_t> rc1_decode(const uint8_t* blob, size_t blob_size, size_t out_size, int stride);

// Cross-channel variant for residual VQ index planes: the context for byte i
// is the same point's index in a previously decoded plane (full byte) plus the
// previous byte of this plane quantized to 4 bits -> 4096 dense contexts.
// `ctx_plane` must hold at least data.size()/out_size bytes and be identical
// on both sides (the decoder passes the already-decoded earlier stage).
std::vector<uint8_t> rcx_encode(const std::vector<uint8_t>& data, const uint8_t* ctx_plane);
std::vector<uint8_t> rcx_decode(const uint8_t* blob, size_t blob_size, size_t out_size, const uint8_t* ctx_plane);

// Dedicated coder for position channels (pos_x/y/z): each splat is one 24-bit
// little-endian zigzag DPCM delta (3 bytes, `data.size() == 3*N`). Instead of a
// 256-ary byte tree per plane, each delta's magnitude is coded as a length
// prefix (number of significant bits, unary-continuation, context = bucket of
// the previous delta's length) + mantissa: the top two mantissa bits are
// modelled, the remaining low bits are coded as equiprobable direct bits (they
// are essentially random). This matches the Laplacian/geometric distribution of
// Morton-ordered DPCM residuals far better than the byte trees. Lossless: the
// decoder reproduces the exact 3-byte-per-splat buffer. No tables travel — both
// sides derive context only from already-coded values plus `out_size` (= 3*N).
std::vector<uint8_t> posmag_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> posmag_decode(const uint8_t* blob, size_t blob_size, size_t out_size);

// Joint xyz variant of posmag for the interleaved `pos_xyz` channel (9 bytes per
// splat: x,y,z zigzag-DPCM 24-bit words back to back). Same length-prefix +
// mantissa model, but each axis keeps its own model set and the length prefix is
// contexted on BOTH the previous value of the same axis and the previous axis of
// the same splat — a large Morton/NN step jumps all three axes at once, so the
// magnitude regime is correlated across the triplet. out_size = 9*N.
std::vector<uint8_t> posmag3_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> posmag3_decode(const uint8_t* blob, size_t blob_size, size_t out_size);

// Dedicated coder for the `smallest3_24` rotation channel (3 bytes/splat, one
// 24-bit packed word per quaternion: 2-bit largest-component index, then 3 x
// (1 sign bit + `mag_bits` magnitude), LSB-first). rc1/rc2 byte trees cut across
// these fields and lose the structure; this coder decomposes the word and codes
// each field with its own adaptive model — index ctx = previous splat's index
// (Morton-correlated), per-slot sign, magnitude bit-tree ctx = (index, slot).
// Lossless: reproduces the exact 3-byte word. Only valid for mag_bits in [1,6]
// (2 + 3*(mag_bits+1) must fit in 24 bits); caller must pass the channel's
// mag_bits (from meta) identically on both sides.
std::vector<uint8_t> rotpack_encode(const std::vector<uint8_t>& data, int mag_bits);
std::vector<uint8_t> rotpack_decode(const uint8_t* blob, size_t blob_size, size_t out_size, int mag_bits);
