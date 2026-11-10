"""Exercise the codec CLIs against small synthetic PLY scenes (stdlib only)."""
import argparse
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def read_ply(path):
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError(f'{path}: missing PLY signature')
        properties = []
        count = None
        vertex = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f'{path}: truncated header')
            fields = line.decode('ascii').split()
            if fields[:1] == ['format']:
                encoding = fields[1]
            elif fields[:1] == ['element']:
                vertex = fields[1] == 'vertex'
                if vertex:
                    count = int(fields[2])
            elif fields[:1] == ['property'] and vertex:
                if fields[1] != 'float':
                    raise ValueError('Test reader expects float32 properties')
                properties.append(fields[2])
            elif fields == ['end_header']:
                break
        if count is None:
            raise ValueError('Missing vertex count')
        if encoding == 'ascii':
            values = [float(x) for x in stream.read().split()]
        elif encoding == 'binary_little_endian':
            data = stream.read()
            values = list(struct.unpack('<' + 'f' * (count * len(properties)), data))
        else:
            raise ValueError(f'Unsupported test PLY format: {encoding}')
        if len(values) != count * len(properties):
            raise ValueError('Incorrect PLY payload length')
        width = len(properties)
        return properties, [dict(zip(properties, values[i:i + width]))
                            for i in range(0, len(values), width)]


def write_binary(path, properties, rows):
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {len(rows)}']
    header += [f'property float {p}' for p in properties]
    header += ['end_header']
    with path.open('wb') as stream:
        stream.write(('\n'.join(header) + '\n').encode('ascii'))
        for row in rows:
            stream.write(struct.pack('<' + 'f' * len(properties), *(row[p] for p in properties)))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_rows(original, decoded, pos_bits, scale_bits, split_rotation):
    require(len(original) == len(decoded), 'Gaussian count changed')
    require(bool(decoded), 'Empty reconstruction')
    properties = list(original[0])
    for row in decoded:
        require(all(p in row for p in properties), 'Missing reconstructed attributes')
        require(all(math.isfinite(v) for v in row.values()), 'Non-finite reconstructed attribute')
        if 'nx' in properties:
            require(all(row[p] == 0.0 for p in ('nx', 'ny', 'nz')),
                    'Reconstructed normals must be zero')
    # Associate rows through position, independent of the encoder's permutation.
    # Coincident fixture rows have identical attributes, so their pairing is arbitrary.
    remaining = list(decoded)
    for source in original:
        index = min(range(len(remaining)), key=lambda i: sum(
            (source[p] - remaining[i][p]) ** 2 for p in ('x', 'y', 'z')))
        target = remaining.pop(index)
        for p in ('x', 'y', 'z'):
            require(abs(source[p] - target[p]) <= 0.5 / (2 ** pos_bits) + 1e-6,
                    f'{p}: position quantization bound exceeded')
        alpha = lambda v: 1 / (1 + math.exp(-v))
        require(abs(alpha(source['opacity']) - alpha(target['opacity'])) <= 0.5 / 255 + 1e-6,
                'Opacity quantization bound exceeded')
        for p in ('scale_0', 'scale_1', 'scale_2'):
            span = max(r[p] for r in original) - min(r[p] for r in original)
            require(abs(source[p] - target[p]) <= span / (2 * (2 ** scale_bits - 1)) + 1e-6,
                    f'{p}: scale quantization bound exceeded')
        q = [source[f'rot_{i}'] for i in range(4)]
        r = [target[f'rot_{i}'] for i in range(4)]
        norm = math.sqrt(sum(v * v for v in r))
        require(abs(norm - 1) < 1e-5, 'Reconstructed quaternion is not normalized')
        # q and -q describe the same rotation.
        dot = abs(sum(a * b for a, b in zip(q, r))) / math.sqrt(sum(v * v for v in q))
        require(dot > (0.9999 if split_rotation else 0.999), 'Rotation error exceeded fixture tolerance')
        # These fixtures have <=32 distinct vectors, fewer than the 256 RVQ
        # codewords. Allow a small absolute error for codebook scalar quantization.
        for p in properties:
            if p.startswith(('f_dc_', 'f_rest_')):
                require(abs(source[p] - target[p]) < 0.025,
                        f'{p}: SH reconstruction error exceeded fixture tolerance')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True, type=Path)
    parser.add_argument('--decoder', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--pos-bits', type=int, default=9)
    parser.add_argument('--scale-bits', type=int, default=8)
    parser.add_argument('--variant', choices=['default', 'joint', 'split', 'legacy'], default='default')
    args = parser.parse_args()
    properties, original = read_ply(args.fixture)
    env = dict(os.environ)
    # Developer overrides must not change the conditions of the tests.
    for key in list(env):
        if key.startswith('BEK_'):
            del env[key]
    with tempfile.TemporaryDirectory(prefix='bek-roundtrip-') as tmp:
        directory = Path(tmp)
        input_path = args.fixture
        if args.binary:
            input_path = directory / 'input.ply'
            write_binary(input_path, properties, original)
            _, original = read_ply(input_path)
        encoded = directory / 'output.bek'
        decoded = directory / 'decoded.ply'
        flags = []
        if args.pos_bits != 9:
            flags += ['--pos-bits', str(args.pos_bits)]
        if args.scale_bits != 8:
            flags += ['--scale-bits', str(args.scale_bits)]
        flags += {'default': [], 'joint': ['--pos-joint'],
                  'split': ['--rot-split'], 'legacy': ['--legacy-tail']}[args.variant]
        for command in ([str(args.encoder), str(input_path), str(encoded)] + flags,
                        [str(args.decoder), str(encoded), str(decoded)]):
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            require(result.returncode == 0,
                    f'Command failed: {command}\n{result.stdout}\n{result.stderr}')
        require(encoded.read_bytes()[:4] == b'BEK3', 'Invalid BEK signature')
        _, reconstruction = read_ply(decoded)
        check_rows(original, reconstruction, args.pos_bits, args.scale_bits, args.variant == 'split')
        print(f'{args.fixture.name}: {len(original)} Gaussians, round-trip verified ({args.variant})')


if __name__ == '__main__':
    main()
