# BEK 3D Gaussian Splatting Compression

Software version **3.67**; container identifier **`BEK3`**.
This document describes the current default pipeline and selected optional channel
representations. Encoder settings are stored in channel metadata for decoding.

## Overview

The BEK pipeline compresses 3D Gaussian splat attributes through a six-stage process:

1. **Load PLY** → Read vertex positions, opacity, rotation quaternions, scales, DC and AC spherical harmonics (SH).
2. **Spatial ordering** → Reorder all attributes via 21-bit Z-order (Morton) **plus a windowed 2-opt refinement** (default `--sort-refine 128`) that trims the residual local jumps the fixed curve leaves — improving DPCM and spatial smoothness for entropy coding, at zero bit cost (the splat set is unordered).
3. **Per-attribute quantization** → Convert continuous floats and quaternions into fixed-width byte arrays. Position uses fixed-point DPCM with zigzag; opacity uses sigmoid-to-8-bit; rotations use quaternion smallest-three 24-bit pack; scales use u8 min-max; SH uses joint Residual Vector Quantization (RVQ).
4. **Channel layout** → Arrange each attribute into logical channels (separate x/y/z planes for positions, etc.).
5. **Per-channel entropy coding** → For each channel, try the smallest of: lossless WebP, adaptive range coders (rc1/rc2/rcx), specialized position or rotation coders where applicable, or raw bytes. Record the winning codec in metadata.
6. **Container wrapping** → Assemble CBOR header (count, grid width/height, channel offsets/sizes/metadata) + concatenated channel blobs, then compress the entire buffer with Zstd level 22 (LDM, 27-bit window). Write as: `"BEK3"` magic + uint64 uncompressed size + Zstd frame.

**Decoder** reverses the process: read magic+size → Zstd inflate → parse CBOR → decode channels by codec type → dequantize by meta type → accumulate RVQ residuals → write PLY.

---

## Container Format

On disk:

```
"BEK3"                     (4 bytes, ASCII magic)
uint64                     uncompressed buffer size (little-endian)
Zstd frame                 level 22, LDM, windowLog 27
  ↓ inflates to:
  uint32                   header_size
  uint8[header_size]       CBOR header
  uint8[]                  concatenated channel blobs at offsets
```

**Header structure** (CBOR JSON):
```json
{
  "count": N,              // total splat count
  "width": W,              // W = ceil(sqrt(N)), grid width for WebP
  "height": H,             // H = ceil(N/W), grid height for WebP
  "channels": [
    {
      "name": "pos_x",
      "offset": 0,         // byte offset in payload
      "size": 42,          // blob size in bytes
      "channels": 3,       // bytes per splat in the decoded channel
      "meta": {
        "type": "spz_dpcm_zz",
        "bits": 24,
        "frac": 9,
        "codec": "rc1"     // selected entropy codec
      }
    },
    ...
  ]
}
```

Decoder sets `ZSTD_d_windowLogMax=31` to accept the encoder's 27-bit window.

---

## Per-Attribute Encoding

### Positions (pos_x, pos_y, pos_z)

**Format:** Three separate channels, 3 bytes each per splat.

**Scheme:** Fixed-point position with `--pos-bits` fractional bits (default 9), stored as 24-bit signed deltas via Differential Pulse Code Modulation (DPCM) after Morton sort, with zigzag mapping of the signed delta.

**Steps:**
1. Convert float to fixed-point: `fixed32 = round(position * 2^pos_bits)`, mask to 24 bits.
2. Compute delta: `delta24 = (current - previous) & 0xFFFFFF` (modular wrap handles negative).
3. Sign-extend the 24-bit modular value to a signed 32-bit integer and zigzag-encode:
   - `d = delta24` (sign-extended to 32-bit)
   - `zigzag = (d << 1) ^ (d >> 31)`
4. Pack into three bytes (little-endian).

**Why zigzag?** Morton sort orders splats by Z-order, so neighboring splats in 3D space are adjacent in the stream. Their positions differ by small amounts. A signed delta with zigzag maps small values (near zero) to small unsigned values (0, 1, 2, 3, ...), so high bytes are near-zero and compress away; plain modular wrap produces 0x00/0xFF splits.

**Metadata:**
```json
{
  "type": "spz_dpcm_zz",
  "bits": 24,
  "frac": pos_bits
}
```

**Joint positions (`--pos-joint`):** The optional `pos_xyz` channel interleaves
three 24-bit words per splat (x, y, z), for nine bytes per splat before entropy
coding. Quantization and `spz_dpcm_zz` metadata are unchanged. The decoder
separates the axes before applying the same reconstruction.

**Legacy support:** Decoder checks for `"type": "spz_dpcm"` (old zigzag-less format) and uses `dequantize_position_spz()` instead.

---

### Opacity

**Format:** Single channel, 1 byte per splat.

**Scheme:** Sigmoid function applied to the float opacity, scaled to [0, 255].

**Steps:**
1. Compute sigmoid: `s = 1.0 / (1.0 + exp(-opacity_float))`.
2. Scale and round: `uint8 = round(s * 255.0)`, clamped to [0, 255].

**Metadata:**
```json
{
  "type": "sigmoid",
  "bits": 8
}
```

---

### Rotation (Quaternion Smallest-Three)

**Format:** Default packs into 3 bytes per splat (24-bit). Optional byte-aligned variant uses 4 bytes (3 component planes + 1 index plane).

**Scheme — Compact (default, `--rot-mag-bits` default 6, typical 24-bit):**
- Quaternions are unit-norm; only three components are transmitted; the fourth is derived at decode.
- Find the largest component by absolute value (index 0–3, 2 bits).
- Quantize the three smallest components to `mag_bits` bits each + 1 sign bit each.
- Pack: `[2-bit index] [sign+mag_bits] [sign+mag_bits] [sign+mag_bits]` into 3 bytes.

**Scheme — Byte-aligned (via `--rot-split`):**
- Three channels of 8 bits each (unpacked smallest components, full-byte precision).
- One channel of 8 bits for the largest-component index.
- Stored as a three-byte component channel and a separate one-byte index channel before entropy coding.

**Default metadata** (compact):
```json
{
  "type": "smallest3_24",
  "bits": 24,
  "mag_bits": 6
}
```

**Split metadata:**
```json
{
  "type": "smallest3_split",
  "bits": 24
}
```
(plus a separate `rotation_idx` channel with `"type": "rot_idx"`)

---

### Scales (3D log-scale per splat)

**Format:** Two modes selectable via `--scale-stages`.

**Mode 1: scalar min-max (default, `--scale-stages 0`, `--scale-bits 8`)** — Per-axis independent quantization:
- Quantize each log-scale axis using min/max normalization to `2^bits - 1` levels, where `bits` is `--scale-bits` (1–8, default 8). For a nonconstant axis: `u8 = round((log_scale - min) / (max - min) * (2^bits - 1))`.
- Store 3 bytes per splat (one per axis), inline in the scales channel.

**Metadata:**
```json
{
  "type": "u8_levels",
  "bits": 8,
  "min": [min_0, min_1, min_2],
  "max": [max_0, max_1, max_2]
}
```

**Mode 2: Residual Vector Quantization (RVQ, `--scale-stages > 0`):**
- Train a joint RVQ codebook over the 3D log-scale vectors (using the same codebook training as SH AC).
- Emit `scale_stages` index planes (`vq_scale_s0`, `vq_scale_s1`, ...), each 1 byte per splat.
- Each stage uses up to 256 codewords, each a 3D vector (fewer for small inputs).
- At decode, accumulate: `scale += codebooks[stage][index[i]]` for each stage.
- Codebook entries from all stages in the group are pooled and Lloyd-quantized to shared scalar levels, stored in the CBOR header.

**Metadata (per stage):**
```json
{
  "type": "rvq_stage_scale_quantized",
  "stage": 0,
  "levels": [float, ...],        // 256 scalar levels (Lloyd-quantized)
  "cb_indices": [uint8, ...]     // indices into levels, 256*3 per stage
}
```

`--scale-bits` applies only to scalar quantization and is ignored when scale RVQ is enabled.

---

### Spherical Harmonics (DC and AC)

**DC (f_dc_0, f_dc_1, f_dc_2)** — 3-component color at each splat.

**Scheme:** RVQ with `--dc-stages` stages (default 2), each stage using up to 256 codewords.
- Train a 3D codebook via Lloyd k-means on the DC vectors.
- Emit `--dc-stages` index planes (`vq_dc_s0`, `vq_dc_s1`, ...), each 1 byte per splat.
- Codebook entries from all stages in the group are pooled and Lloyd-quantized to shared scalar levels for header compaction.

**Metadata (per stage):**
```json
{
  "type": "rvq_stage_quantized",
  "stage": 0,
  "levels": [float, ...],        // 256 scalar levels (Lloyd-quantized)
  "cb_indices": [uint8, ...]     // indices into levels, 256*3 per stage
}
```

**SH AC (Spherical Harmonics, bands 1–3, 45 total dimensions)** — Residual coefficients per splat.

**Scheme:** Joint RVQ with `--sh-stages` stages (default 3), single codebook spanning all 45 AC dimensions.
- Train a 45D codebook via Lloyd k-means on the concatenated AC feature vectors.
- Emit `--sh-stages` index planes (`vq_sh_ac_s0`, `vq_sh_ac_s1`, ...), each 1 byte per splat.
- **Key difference:** This is **joint** RVQ, not banded (split). A single stage quantizes all 45 dimensions together, capturing cross-band correlations. The old banded split (vq_sh1_s*, vq_sh2_s*, vq_sh3_s*) is decoded for backward compatibility but no longer emitted.
- Codebook entries from all stages in the group are pooled and Lloyd-quantized to shared scalar levels.

**Metadata (per stage):**
```json
{
  "type": "rvq_stage_sh_ac_quantized",
  "stage": 0,
  "dims": 45,                    // all 45 AC dimensions
  "levels": [float, ...],        // 256 scalar levels (Lloyd-quantized)
  "cb_indices": [uint8, ...]     // indices into levels, 256*45 per stage
}
```

**Codeword relabelling (enabled by default):**
Each RVQ stage's codebook is reordered along a greedy nearest-neighbour chain
before index assignment. This changes the numbering used by the entropy coder.
`--no-rvq-cb-order` disables this step.

**RVQ training:**
Lloyd k-means trains each stage on the residual left by previous stages, with
`--vq-iters` controlling the maximum training iterations (default 30).
Codebooks can be saved and loaded using `--save-vq` and `--load-vq`.

**Codebook reconstruction:**
For each stage, the decoder reconstructs the centroid values as
`codebook[j] = levels[cb_indices[j]]` and adds the selected centroid to the
attribute vector. During training, residuals and indices are computed using
unquantized centroids; the decoder uses their scalar-quantized counterparts.
Thus codebook quantization introduces additional reconstruction error.

---

## Per-Channel Entropy Coding

After quantization, the encoder selects the smallest payload among the candidates
applicable to each channel and records the result in `meta.codec`.

| Codec | Representation and context | Applicable channels |
|---|---|---|
| `webp` | Lossless image coding on a padded `W × H` grid. Single-byte channels are replicated into RGB. | Channels with 1, 3 or 4 bytes per splat; the encoder skips WebP for `rotation`, `scales`, `opacity` and `vq_dc_s0`. |
| `rc1` | Adaptive binary range coder; previous same-plane byte context. | Channel byte streams. |
| `rc2` | Adaptive binary range coder with previous-sample and inter-plane contexts. | Channel byte streams. |
| `rcx` | Adaptive binary range coder conditioned on an earlier RVQ stage. | Residual RVQ index planes with an available preceding stage. |
| `posmag` | Length-prefix and mantissa coding of 24-bit zigzag DPCM values. | Separate position channels. |
| `posmag3` | Joint xyz position coding with per-axis and cross-axis contexts. | Optional interleaved `pos_xyz` channel. |
| `rotpack` | Separate adaptive models for index, sign and magnitude fields. | Compact `smallest3_24` rotation channel. |
| `raw` | Uncompressed channel bytes. | Fallback for every channel. |

`--legacy-tail` excludes `posmag`, `posmag3` and `rotpack` from the candidate set.
The other candidates remain subject to their normal eligibility rules.

Range coders and raw storage use the unpadded channel bytes. WebP uses zero padding
to fill the image grid; only the first `count` splats are reconstructed.
For `rcx`, metadata also records `ctx_ch`, the name of the context channel.
That channel must be decoded before the dependent stage. The decoder processes
channels in header order and retains decoded RVQ index planes for this purpose.
When `meta.codec` is absent, the decoder defaults to WebP for compatibility.

---

## Spatial Ordering (Morton Sort + 2-opt refinement)

**Purpose:** Group nearby splats in 3D space so consecutive splats differ by little, shrinking DPCM deltas and entropy.

**Implementation:**
1. Find bounding box of all splats (min/max across x, y, z).
2. Map each splat's position to a 21-bit grid per axis (2^21 − 1 ≈ 2M cells per axis).
3. Encode the (21-bit x, 21-bit y, 21-bit z) tuple as a 63-bit Morton code (Z-order curve).
4. Sort splats by Morton code.
5. **2-opt refinement (`--sort-refine 128` by default):** within a sliding window, reverse segments when doing so shortens the consecutive L1 path. Run up to four passes, stopping early if no improvement is found.
6. Reorder all attributes in the final order.

The spatial order is not transmitted: the decoder reconstructs splats in stream
order, so the original PLY row order is not restored.

Implementation: [`VertexData::sort_morton()`](../src/ply_reader.cpp).
`--sort-refine 0` disables refinement. Optional alternatives are
`--sort-metric bits` for the bit-length objective and `--sort-mode nn` for a
greedy nearest-neighbour base order.

---

## CLI Reference

Encoder command-line flags for `ply_to_bek`:

```
ply_to_bek <input.ply> <output.bek> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--dc-stages N` | 2 | RVQ stages for DC coefficients; 0 omits DC RVQ channels. |
| `--sh-stages N` | 3 | Joint RVQ stages for the 45 SH AC coefficients. |
| `--pos-bits N` | 9 | Fixed-point fractional bits for positions. |
| `--vq-iters N` | 30 | Maximum Lloyd k-means iterations per RVQ stage. |
| `--rot-mag-bits N` | 6 | Quaternion magnitude bits, clamped to 2–6. |
| `--rot-split` | Off | Separate rotation component and index channels. |
| `--scale-bits N` | 8 | Scalar scale precision, clamped to 1–8; ignored for scale RVQ. |
| `--scale-stages N` | 0 | Scale RVQ stages; 0 selects scalar min–max quantization. |
| `--sort-refine N` | 128 | 2-opt window size; 0 disables refinement. |
| `--sort-metric M` | l1 | Refinement objective: `l1` or `bits`. |
| `--sort-mode M` | morton | Base ordering: `morton` or `nn`. |
| `--sh-whiten P` | 0 | Per-dimension weighting `variance^-P` in SH AC training; 0 uses unweighted L2. |
| `--load-vq PATH` | Unset | Load codebooks from `PATH_rvq_dc.json` and `PATH_rvq_sh.json`. |
| `--save-vq PATH` | Unset | Save trained codebooks using those suffixes. |
| `--no-rvq-cb-order` | Off | Disable codebook relabelling. |
| `--pos-joint` | Off | Interleave xyz position values into one nine-byte channel. |
| `--legacy-tail` | Off | Disable the specialized position and rotation entropy coders. |

Both executables accept `--version` as a standalone argument. The decoder is used as
`bek_to_ply <input.bek> <output.ply>` and reads reconstruction settings from the file.

Additional encoder options include flat k-means (`--dc-kmeans`, `--sh-kmeans`),
float SH codebooks (`--cb-float`), and attribute omission (`--no-positions`,
`--no-opacity`, `--no-rotation`, `--no-scale`). These select alternative
representations or omit attributes from the stream; they are disabled by default.
See the encoder source for their exact handling.

## Source reference

- [Encoder](../apps/ply_to_bek.cpp) and [decoder](../apps/bek_to_ply.cpp).
- [Quantization](../src/quantization.cpp) and [entropy coding](../src/range_coder.cpp).
- [RVQ training and codebook packing](../src/sh_vq.cpp).
- [PLY input and spatial ordering](../src/ply_reader.cpp).
