# Hoosat H100 GPU Miner

High-performance **Hoosat (HTN)** cryptocurrency miner built with CUDA, specifically optimized for the **NVIDIA H100 SXM5** and its FP64 Tensor Cores.

## Features

- **FP64 Tensor Core Optimized** — Leverages H100's 67 TFLOPS FP64 Tensor Core performance for HooHash matrix operations
- **Full HooHash Implementation** — Complete GPU implementation of the HooHash proof-of-work algorithm (BLAKE3 + 64×64 FP64 matrix multiply + nonlinear transforms)
- **Stratum Pool Mining** — Compatible with HTN Stratum Bridge (Kaspa-style stratum protocol)
- **Benchmark Mode** — Test hashrate without connecting to a pool
- **Cross-Platform** — Builds on Linux and Windows

## Requirements

- NVIDIA GPU with CUDA support (optimized for H100/sm_90, works on any CUDA GPU)
- CUDA Toolkit 12.0 or later
- CMake 3.24 or later
- C++17 compiler

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

On Windows (Visual Studio):
```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

## Usage

### Pool Mining

```bash
./hoosat_h100_miner \
  --stratum stratum+tcp://pool.example.com:40012 \
  --user hoosat:YOUR_WALLET_ADDRESS \
  --worker h100_rig1 \
  --intensity 22
```

### Benchmark

```bash
./hoosat_h100_miner --benchmark --intensity 22
```

### All Options

| Option | Description | Default |
|--------|-------------|---------|
| `--stratum <url>` | Stratum server URL | (required) |
| `--user <address>` | Hoosat wallet address | (required) |
| `--worker <name>` | Worker name | `h100` |
| `--password <pass>` | Pool password | `x` |
| `--device <id>` | CUDA device ID | `0` |
| `--intensity <n>` | Batch size = 2^n nonces | `22` |
| `--benchmark` | Run benchmark mode | off |

## Testing

```bash
cd build
cmake --build . --target test_hoohash
./test_hoohash
```

Verifies:
- BLAKE3 hash correctness against known test vectors
- HooHash matrix generation
- Full PoW hash against reference test vectors from [HoosatNetwork/hoohash](https://github.com/HoosatNetwork/hoohash)
- Mining kernel batch processing

## Architecture

```
main.cu          → CLI, mining loop, benchmark
hoohash.cu/cuh   → CUDA HooHash kernels (matrix gen, nonlinear transforms, PoW)
blake3_cuda.cu/h → BLAKE3 device implementation
stratum.cpp/h    → Kaspa-style stratum TCP/JSON-RPC client
utils.h          → Common utilities, CUDA macros, hex encoding
```

### HooHash Algorithm Pipeline (per nonce)

```
1. BLAKE3(PrevHeader || Timestamp || zeros[32] || Nonce) → firstPass[32]
2. HoohashMatrixMultiplication(mat[64×64], firstPass, nonce) → lastPass[32]
   ├─ Split firstPass into 64 nibbles → vector[64]
   ├─ For each (i,j): conditional path based on TransformFactor
   │   ├─ sw ≤ 0.02: ComplexNonLinear(mat[i][j] * hashMod * vector[j] + nonceMod)
   │   └─ sw > 0.02: mat[i][j] * 0.0001 * vector[j]  (linear path)
   ├─ Combine product pairs → scaledValues[32]
   ├─ XOR firstPass ⊕ scaledValues → intermediate[32]
   └─ BLAKE3(intermediate) → lastPass[32]
3. Compare lastPass against target difficulty
```

### H100 Optimization Strategy

The 64×64 FP64 matrix is constant per block — uploaded once to GPU memory. Each CUDA thread tests a unique nonce. The H100's FP64 Tensor Cores (67 TFLOPS) accelerate the matrix multiply operations that dominate the HooHash computation.

## License

GPL-3.0 (derivative of [HoosatNetwork/hoohash](https://github.com/HoosatNetwork/hoohash))

## Credits

- HooHash algorithm by [Hoosat Oy](https://hoosat.fi) / Toni Lukkaroinen
- [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) hash function
