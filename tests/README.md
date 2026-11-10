# Codec round-trip tests

These tests run the `ply_to_bek` and `bek_to_ply` executables on small synthetic
Gaussian Splatting scenes. They require Python 3 and its standard library, with
no renderer, downloaded scenes or additional Python packages.

## Run

From the repository root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run one test with, for example:

```bash
ctest --test-dir build -R '^roundtrip_binary$' --output-on-failure
```

Tests are disabled by default in CMake. Enabling them registers ten CTest cases;
it does not change the codec implementation or installation.

## Fixtures and cases

The ASCII PLY files in `data/` contain float32 properties for positions, SH DC and
45 SH AC coefficients, opacity, log-scales and quaternion rotations.

| Test | Fixture | Purpose |
|---|---|---|
| `roundtrip_single` | `single.ply`: 1 Gaussian | Smallest nonempty input; constant attribute ranges. |
| `roundtrip_coincident` | `coincident.ply`: 4 identical Gaussians | Coincident positions, degenerate bounding box and preservation of duplicate splats. |
| `roundtrip_cube` | `cube.ply`: 8 Gaussians | Cube corners with positive and negative coordinates and varying attributes. |
| `roundtrip_cloud` | `cloud.ply`: 32 Gaussians | Spatial grid with varying color, SH, opacity, scale and rotation. |
| `roundtrip_binary` | Binary conversion of `cloud.ply` | The same scene read as binary little-endian PLY. |
| `roundtrip_cloud_binary_normals` | `cloud_binary.ply`: 32 Gaussians | Stored binary little-endian PLY with zero-valued `nx`, `ny`, `nz`, following the GraphDeco-INRIA property layout. Checks that reconstructed normals remain zero. |
| `roundtrip_precision` | `cloud.ply` | Explicit `--pos-bits 12 --scale-bits 4`. |
| `roundtrip_joint` | `cloud.ply` | Interleaved positions using `--pos-joint`. |
| `roundtrip_rotation_split` | `cloud.ply` | Separate rotation component and index channels using `--rot-split`. |
| `roundtrip_legacy` | `cloud.ply` | Encoding with specialized position and rotation entropy coders disabled by `--legacy-tail`. |

`cloud_binary.ply` is a binary counterpart of `cloud.ply` with the three normal
properties added after the positions. The ASCII fixtures are unchanged.

The first four cases and both binary cases use default encoder settings. Optional
coder candidates are tested through the CLI options; a test does not require a
particular entropy coder to win the per-channel payload-size comparison.

## Checks

For each case, `roundtrip.py` encodes the input, decodes the resulting BEK file,
and independently reads the reconstructed PLY. It checks:

- Both applications exit successfully and the encoded file starts with `BEK3`.
- The Gaussian count is unchanged, including duplicates.
- All input properties are present in the reconstruction and output values are finite.
- Position error is at most half a fixed-point quantization step, plus `1e-6`.
- Opacity error, measured after sigmoid conversion to alpha, is at most half an
  8-bit step, plus `1e-6`.
- Scalar scale error is at most half a quantization step for that axis and the
  selected scale precision, plus `1e-6`.
- Reconstructed quaternions have unit norm within `1e-5`. Their absolute normalized
  dot product with the input exceeds `0.999`, or `0.9999` for split rotations.
  Absolute dot products account for the equivalence of `q` and `-q`.
- Each SH DC and AC coefficient differs from its input by less than `0.025`.
  This is a fixture-specific tolerance: these scenes have at most 32 distinct
  vectors, fewer than the 256 available RVQ codewords. Scalar quantization of
  codebook entries still introduces error.

Rows are associated by nearest position rather than file order, because spatial
ordering can permute the Gaussians. Coincident fixture rows have identical
attributes, so their pairing is interchangeable.

Each test uses a temporary directory, removed on completion. The binary fixture
is generated there from the ASCII source. `BEK_*` environment overrides are
removed from child-process environments so local tuning does not alter test
conditions. Each codec invocation has a 60-second timeout; CTest allows 150 seconds
per case.

## Scope

These are functional regression checks on small valid scenes. They do not measure
rendered quality, compression efficiency or performance on large scenes, and do
not test malformed files or every experimental option. They allow lossy
reconstruction and do not require byte-identical compressed files.
