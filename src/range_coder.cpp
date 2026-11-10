#include "range_coder.h"

namespace {

constexpr int kProbBits = 11;
constexpr uint16_t kProbInit = (1u << kProbBits) / 2; // 1024
constexpr int kMoveBits = 5;
constexpr uint32_t kTopValue = 1u << 24;

// LZMA-style carry-aware range encoder.
struct RangeEncoder {
    std::vector<uint8_t> out;
    uint64_t low = 0;
    uint32_t range = 0xFFFFFFFFu;
    uint8_t cache = 0;
    uint64_t cache_size = 1;

    void shift_low() {
        if (static_cast<uint32_t>(low) < 0xFF000000u || (low >> 32) != 0) {
            uint8_t carry = static_cast<uint8_t>(low >> 32);
            uint8_t temp = cache;
            do {
                out.push_back(static_cast<uint8_t>(temp + carry));
                temp = 0xFF;
            } while (--cache_size != 0);
            cache = static_cast<uint8_t>(low >> 24);
        }
        ++cache_size;
        low = (low << 8) & 0xFFFFFFFFull;
    }

    void encode_bit(uint16_t& prob, int bit) {
        uint32_t bound = (range >> kProbBits) * prob;
        if (bit == 0) {
            range = bound;
            prob += ((1u << kProbBits) - prob) >> kMoveBits;
        } else {
            low += bound;
            range -= bound;
            prob -= prob >> kMoveBits;
        }
        while (range < kTopValue) {
            range <<= 8;
            shift_low();
        }
    }

    // Equiprobable bit: split the range in half, no adaptive model. Exact 1 bit.
    void encode_bit_direct(int bit) {
        range >>= 1;
        if (bit) low += range;
        while (range < kTopValue) {
            range <<= 8;
            shift_low();
        }
    }

    void flush() {
        for (int i = 0; i < 5; ++i) shift_low();
    }
};

struct RangeDecoder {
    const uint8_t* buf;
    size_t size;
    size_t pos = 0;
    uint32_t range = 0xFFFFFFFFu;
    uint32_t code = 0;

    uint8_t read_byte() { return pos < size ? buf[pos++] : 0; }

    void init() {
        read_byte(); // encoder's initial cache byte (always 0)
        for (int i = 0; i < 4; ++i) code = (code << 8) | read_byte();
    }

    int decode_bit(uint16_t& prob) {
        uint32_t bound = (range >> kProbBits) * prob;
        int bit;
        if (code < bound) {
            range = bound;
            prob += ((1u << kProbBits) - prob) >> kMoveBits;
            bit = 0;
        } else {
            code -= bound;
            range -= bound;
            prob -= prob >> kMoveBits;
            bit = 1;
        }
        while (range < kTopValue) {
            range <<= 8;
            code = (code << 8) | read_byte();
        }
        return bit;
    }

    int decode_bit_direct() {
        range >>= 1;
        int bit;
        if (code >= range) { code -= range; bit = 1; }
        else { bit = 0; }
        while (range < kTopValue) {
            range <<= 8;
            code = (code << 8) | read_byte();
        }
        return bit;
    }
};

} // namespace

namespace {

constexpr size_t kNumContexts = 8192;

// Context id for byte i of an interleaved buffer (see header). Reads only
// bytes < i, so encoder and decoder stay in lockstep.
inline size_t rc2_context(const uint8_t* data, size_t i, size_t stride) {
    const size_t plane = stride > 1 ? i % stride : 0;
    if (plane == 0) {
        const unsigned prev = i >= stride ? data[i - stride] : 0;
        const unsigned prev2 = i >= 2 * stride ? data[i - 2 * stride] : 0;
        return (static_cast<size_t>(prev) << 4) | (prev2 >> 4); // [0, 4096)
    }
    const unsigned intra = data[i - 1];           // same sample, previous plane
    const unsigned prev = i >= stride ? data[i - stride] : 0;
    return (static_cast<size_t>(intra) << 5) | (prev >> 3);     // [0, 8192)
}

} // namespace

std::vector<uint8_t> rc2_encode(const std::vector<uint8_t>& data, int stride) {
    std::vector<uint16_t> probs(kNumContexts * 256, kProbInit);
    RangeEncoder enc;
    enc.out.reserve(data.size() / 2);
    const size_t n = data.size();
    const size_t st = static_cast<size_t>(stride);
    for (size_t i = 0; i < n; ++i) {
        uint16_t* tree = &probs[rc2_context(data.data(), i, st) << 8];
        unsigned m = 1;
        uint8_t b = data[i];
        for (int k = 7; k >= 0; --k) {
            int bit = (b >> k) & 1;
            enc.encode_bit(tree[m], bit);
            m = (m << 1) | static_cast<unsigned>(bit);
        }
    }
    enc.flush();
    return std::move(enc.out);
}

std::vector<uint8_t> rc2_decode(const uint8_t* blob, size_t blob_size, size_t out_size, int stride) {
    std::vector<uint16_t> probs(kNumContexts * 256, kProbInit);
    RangeDecoder dec{blob, blob_size};
    dec.init();
    std::vector<uint8_t> out(out_size);
    const size_t st = static_cast<size_t>(stride);
    for (size_t i = 0; i < out_size; ++i) {
        uint16_t* tree = &probs[rc2_context(out.data(), i, st) << 8];
        unsigned m = 1;
        for (int k = 0; k < 8; ++k) m = (m << 1) | static_cast<unsigned>(dec.decode_bit(tree[m]));
        out[i] = static_cast<uint8_t>(m & 0xFF);
    }
    return out;
}

std::vector<uint8_t> rc1_encode(const std::vector<uint8_t>& data, int stride) {
    std::vector<uint16_t> probs(256 * 256, kProbInit);
    RangeEncoder enc;
    enc.out.reserve(data.size() / 2);
    const size_t n = data.size();
    const size_t st = static_cast<size_t>(stride);
    for (size_t i = 0; i < n; ++i) {
        uint16_t* tree = &probs[static_cast<size_t>(i >= st ? data[i - st] : 0) << 8];
        unsigned m = 1;
        uint8_t b = data[i];
        for (int k = 7; k >= 0; --k) {
            int bit = (b >> k) & 1;
            enc.encode_bit(tree[m], bit);
            m = (m << 1) | static_cast<unsigned>(bit);
        }
    }
    enc.flush();
    return std::move(enc.out);
}

std::vector<uint8_t> rc1_decode(const uint8_t* blob, size_t blob_size, size_t out_size, int stride) {
    std::vector<uint16_t> probs(256 * 256, kProbInit);
    RangeDecoder dec{blob, blob_size};
    dec.init();
    std::vector<uint8_t> out(out_size);
    const size_t st = static_cast<size_t>(stride);
    for (size_t i = 0; i < out_size; ++i) {
        uint16_t* tree = &probs[static_cast<size_t>(i >= st ? out[i - st] : 0) << 8];
        unsigned m = 1;
        for (int k = 0; k < 8; ++k) m = (m << 1) | static_cast<unsigned>(dec.decode_bit(tree[m]));
        out[i] = static_cast<uint8_t>(m & 0xFF);
    }
    return out;
}

namespace {
inline size_t rcx_context(const uint8_t* ctx_plane, const uint8_t* data, size_t i) {
    const unsigned cross = ctx_plane[i];
    const unsigned prev = i > 0 ? data[i - 1] : 0;
    return (static_cast<size_t>(cross) << 4) | (prev >> 4); // [0, 4096)
}
} // namespace

std::vector<uint8_t> rcx_encode(const std::vector<uint8_t>& data, const uint8_t* ctx_plane) {
    std::vector<uint16_t> probs(4096 * 256, kProbInit);
    RangeEncoder enc;
    enc.out.reserve(data.size() / 2);
    for (size_t i = 0; i < data.size(); ++i) {
        uint16_t* tree = &probs[rcx_context(ctx_plane, data.data(), i) << 8];
        unsigned m = 1;
        uint8_t b = data[i];
        for (int k = 7; k >= 0; --k) {
            int bit = (b >> k) & 1;
            enc.encode_bit(tree[m], bit);
            m = (m << 1) | static_cast<unsigned>(bit);
        }
    }
    enc.flush();
    return std::move(enc.out);
}

std::vector<uint8_t> rcx_decode(const uint8_t* blob, size_t blob_size, size_t out_size, const uint8_t* ctx_plane) {
    std::vector<uint16_t> probs(4096 * 256, kProbInit);
    RangeDecoder dec{blob, blob_size};
    dec.init();
    std::vector<uint8_t> out(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        uint16_t* tree = &probs[rcx_context(ctx_plane, out.data(), i) << 8];
        unsigned m = 1;
        for (int k = 0; k < 8; ++k) m = (m << 1) | static_cast<unsigned>(dec.decode_bit(tree[m]));
        out[i] = static_cast<uint8_t>(m & 0xFF);
    }
    return out;
}

namespace {

// 24-bit delta magnitudes have 0..24 significant bits. Bucket the previous
// delta's length into 8 classes so the length prefix is contextualised by the
// local magnitude regime (Morton order makes magnitudes autocorrelated).
inline int posmag_bucket(int nbits) {
    if (nbits <= 3) return nbits;     // 0,1,2,3
    if (nbits <= 6) return 4;
    if (nbits <= 9) return 5;
    if (nbits <= 13) return 6;
    return 7;
}

constexpr int kPosBuckets = 8;
constexpr int kPosLenBits = 24;       // max significant bits of a 24-bit value

} // namespace

std::vector<uint8_t> posmag_encode(const std::vector<uint8_t>& data) {
    const size_t N = data.size() / 3;
    // Length-prefix models: [prev-bucket][bit-position]. Mantissa models: the
    // top two mantissa bits per length [nbits][0..1]. Fresh per channel (only 3
    // position channels), tiny, warm up almost instantly.
    std::vector<uint16_t> cont(kPosBuckets * kPosLenBits, kProbInit);
    std::vector<uint16_t> mant((kPosLenBits + 1) * 2, kProbInit);
    RangeEncoder enc;
    enc.out.reserve(data.size() / 2);
    int prevBucket = 0;
    for (size_t i = 0; i < N; ++i) {
        const uint32_t zz = static_cast<uint32_t>(data[i * 3]) |
                            (static_cast<uint32_t>(data[i * 3 + 1]) << 8) |
                            (static_cast<uint32_t>(data[i * 3 + 2]) << 16);
        const int nbits = zz ? (32 - __builtin_clz(zz)) : 0;
        // Length prefix: unary continuation, stop at the first 0. nbits==24 emits
        // 24 ones with no terminator (decoder caps at kPosLenBits).
        uint16_t* c = &cont[prevBucket * kPosLenBits];
        for (int k = 0; k < kPosLenBits; ++k) {
            const int bit = nbits > k ? 1 : 0;
            enc.encode_bit(c[k], bit);
            if (!bit) break;
        }
        // Mantissa: implicit top 1-bit; emit bits below it. Top two modelled,
        // the rest as equiprobable direct bits.
        for (int j = nbits - 2; j >= 0; --j) {
            const int bit = (zz >> j) & 1;
            const int pos = nbits - 1 - j;     // 1 = highest mantissa bit
            if (pos <= 2) enc.encode_bit(mant[nbits * 2 + (pos - 1)], bit);
            else enc.encode_bit_direct(bit);
        }
        prevBucket = posmag_bucket(nbits);
    }
    enc.flush();
    return std::move(enc.out);
}

std::vector<uint8_t> posmag_decode(const uint8_t* blob, size_t blob_size, size_t out_size) {
    const size_t N = out_size / 3;
    std::vector<uint16_t> cont(kPosBuckets * kPosLenBits, kProbInit);
    std::vector<uint16_t> mant((kPosLenBits + 1) * 2, kProbInit);
    RangeDecoder dec{blob, blob_size};
    dec.init();
    std::vector<uint8_t> out(out_size);
    int prevBucket = 0;
    for (size_t i = 0; i < N; ++i) {
        uint16_t* c = &cont[prevBucket * kPosLenBits];
        int nbits = kPosLenBits;               // no terminator seen -> max length
        for (int k = 0; k < kPosLenBits; ++k) {
            if (dec.decode_bit(c[k]) == 0) { nbits = k; break; }
        }
        uint32_t zz = 0;
        if (nbits >= 1) {
            zz = 1u << (nbits - 1);
            for (int j = nbits - 2; j >= 0; --j) {
                const int pos = nbits - 1 - j;
                const int bit = (pos <= 2) ? dec.decode_bit(mant[nbits * 2 + (pos - 1)])
                                           : dec.decode_bit_direct();
                zz |= static_cast<uint32_t>(bit) << j;
            }
        }
        out[i * 3 + 0] = zz & 0xFF;
        out[i * 3 + 1] = (zz >> 8) & 0xFF;
        out[i * 3 + 2] = (zz >> 16) & 0xFF;
        prevBucket = posmag_bucket(nbits);
    }
    return out;
}

std::vector<uint8_t> posmag3_encode(const std::vector<uint8_t>& data) {
    const size_t N = data.size() / 9;
    // [axis][prevSameAxisBucket * kPosBuckets + prevAxisBucket][bit-position]
    std::vector<uint16_t> cont(3 * kPosBuckets * kPosBuckets * kPosLenBits, kProbInit);
    std::vector<uint16_t> mant(3 * (kPosLenBits + 1) * 2, kProbInit);
    RangeEncoder enc;
    enc.out.reserve(data.size() / 2);
    int prevAxisBucket[3] = {0, 0, 0};   // same axis, previous splat
    int prevBucket = 0;                  // previously coded value (any axis)
    for (size_t i = 0; i < N; ++i) {
        for (int a = 0; a < 3; ++a) {
            const size_t o = i * 9 + a * 3;
            const uint32_t zz = static_cast<uint32_t>(data[o]) |
                                (static_cast<uint32_t>(data[o + 1]) << 8) |
                                (static_cast<uint32_t>(data[o + 2]) << 16);
            const int nbits = zz ? (32 - __builtin_clz(zz)) : 0;
            uint16_t* c = &cont[((a * kPosBuckets + prevAxisBucket[a]) * kPosBuckets + prevBucket) * kPosLenBits];
            for (int k = 0; k < kPosLenBits; ++k) {
                const int bit = nbits > k ? 1 : 0;
                enc.encode_bit(c[k], bit);
                if (!bit) break;
            }
            uint16_t* mt = &mant[a * (kPosLenBits + 1) * 2];
            for (int j = nbits - 2; j >= 0; --j) {
                const int bit = (zz >> j) & 1;
                const int pos = nbits - 1 - j;
                if (pos <= 2) enc.encode_bit(mt[nbits * 2 + (pos - 1)], bit);
                else enc.encode_bit_direct(bit);
            }
            prevAxisBucket[a] = prevBucket = posmag_bucket(nbits);
        }
    }
    enc.flush();
    return std::move(enc.out);
}

std::vector<uint8_t> posmag3_decode(const uint8_t* blob, size_t blob_size, size_t out_size) {
    const size_t N = out_size / 9;
    std::vector<uint16_t> cont(3 * kPosBuckets * kPosBuckets * kPosLenBits, kProbInit);
    std::vector<uint16_t> mant(3 * (kPosLenBits + 1) * 2, kProbInit);
    RangeDecoder dec{blob, blob_size};
    dec.init();
    std::vector<uint8_t> out(out_size);
    int prevAxisBucket[3] = {0, 0, 0};
    int prevBucket = 0;
    for (size_t i = 0; i < N; ++i) {
        for (int a = 0; a < 3; ++a) {
            uint16_t* c = &cont[((a * kPosBuckets + prevAxisBucket[a]) * kPosBuckets + prevBucket) * kPosLenBits];
            int nbits = kPosLenBits;
            for (int k = 0; k < kPosLenBits; ++k) {
                if (dec.decode_bit(c[k]) == 0) { nbits = k; break; }
            }
            uint16_t* mt = &mant[a * (kPosLenBits + 1) * 2];
            uint32_t zz = 0;
            if (nbits >= 1) {
                zz = 1u << (nbits - 1);
                for (int j = nbits - 2; j >= 0; --j) {
                    const int pos = nbits - 1 - j;
                    const int bit = (pos <= 2) ? dec.decode_bit(mt[nbits * 2 + (pos - 1)])
                                               : dec.decode_bit_direct();
                    zz |= static_cast<uint32_t>(bit) << j;
                }
            }
            const size_t o = i * 9 + a * 3;
            out[o + 0] = zz & 0xFF;
            out[o + 1] = (zz >> 8) & 0xFF;
            out[o + 2] = (zz >> 16) & 0xFF;
            prevAxisBucket[a] = prevBucket = posmag_bucket(nbits);
        }
    }
    return out;
}

std::vector<uint8_t> rotpack_encode(const std::vector<uint8_t>& data, int mag_bits) {
    const int m = mag_bits;
    const int fw = m + 1;                       // field width: sign + magnitude
    const uint32_t fmask = (1u << fw) - 1;
    const uint32_t magmask = (1u << m) - 1;
    const size_t N = data.size() / 3;
    std::vector<uint16_t> idxTree(4 * 4, kProbInit);                  // [prevIdx][node]
    std::vector<uint16_t> sign(4 * 3, kProbInit);                    // [idx][slot]
    std::vector<uint16_t> magTree(4 * 3 * (1u << m), kProbInit);     // [idx*3+slot][node]
    RangeEncoder enc;
    enc.out.reserve(data.size() / 2);
    unsigned prevIdx = 0;
    for (size_t i = 0; i < N; ++i) {
        const uint32_t word = static_cast<uint32_t>(data[i * 3]) |
                              (static_cast<uint32_t>(data[i * 3 + 1]) << 8) |
                              (static_cast<uint32_t>(data[i * 3 + 2]) << 16);
        const unsigned idx = (word >> (3 * fw)) & 0x3u;
        uint16_t* it = &idxTree[prevIdx * 4];
        unsigned mt = 1;
        for (int k = 1; k >= 0; --k) { int bit = (idx >> k) & 1; enc.encode_bit(it[mt], bit); mt = (mt << 1) | bit; }
        for (int s = 0; s < 3; ++s) {
            const uint32_t comp = (word >> ((2 - s) * fw)) & fmask;   // s=0 = high component
            const int sgn = (comp >> m) & 1;
            const uint32_t mag = comp & magmask;
            enc.encode_bit(sign[idx * 3 + s], sgn);
            uint16_t* mtree = &magTree[static_cast<size_t>(idx * 3 + s) << m];
            unsigned mm = 1;
            for (int k = m - 1; k >= 0; --k) { int bit = (mag >> k) & 1; enc.encode_bit(mtree[mm], bit); mm = (mm << 1) | bit; }
        }
        prevIdx = idx;
    }
    enc.flush();
    return std::move(enc.out);
}

std::vector<uint8_t> rotpack_decode(const uint8_t* blob, size_t blob_size, size_t out_size, int mag_bits) {
    const int m = mag_bits;
    const int fw = m + 1;
    const uint32_t magmask = (1u << m) - 1;
    const size_t N = out_size / 3;
    std::vector<uint16_t> idxTree(4 * 4, kProbInit);
    std::vector<uint16_t> sign(4 * 3, kProbInit);
    std::vector<uint16_t> magTree(4 * 3 * (1u << m), kProbInit);
    RangeDecoder dec{blob, blob_size};
    dec.init();
    std::vector<uint8_t> out(out_size);
    unsigned prevIdx = 0;
    for (size_t i = 0; i < N; ++i) {
        uint16_t* it = &idxTree[prevIdx * 4];
        unsigned mt = 1;
        for (int k = 0; k < 2; ++k) mt = (mt << 1) | static_cast<unsigned>(dec.decode_bit(it[mt]));
        const unsigned idx = mt & 0x3u;
        uint32_t word = idx << (3 * fw);
        for (int s = 0; s < 3; ++s) {
            const int sgn = dec.decode_bit(sign[idx * 3 + s]);
            uint16_t* mtree = &magTree[static_cast<size_t>(idx * 3 + s) << m];
            unsigned mm = 1;
            for (int k = 0; k < m; ++k) mm = (mm << 1) | static_cast<unsigned>(dec.decode_bit(mtree[mm]));
            const uint32_t comp = (static_cast<uint32_t>(sgn) << m) | (mm & magmask);
            word |= comp << ((2 - s) * fw);
        }
        out[i * 3 + 0] = word & 0xFF;
        out[i * 3 + 1] = (word >> 8) & 0xFF;
        out[i * 3 + 2] = (word >> 16) & 0xFF;
        prevIdx = idx;
    }
    return out;
}
