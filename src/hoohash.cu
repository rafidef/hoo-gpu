/*
 * HooHash CUDA Kernels — Optimized for NVIDIA H100 SXM5 FP64 Tensor Cores
 *
 * Implements the full HooHash proof-of-work algorithm on GPU:
 * 1. BLAKE3(PrevHeader || Timestamp || zeroes || Nonce) → firstPass
 * 2. HoohashMatrixMultiplication(mat, firstPass, nonce) → lastPass
 * 3. Compare lastPass against target
 *
 * Reference: https://github.com/HoosatNetwork/hoohash
 * License: GPL-3.0 (derivative of HoosatNetwork/hoohash)
 */

#include "hoohash.cuh"
#include "blake3_cuda.cuh"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cfenv>

// ============================================================================
// Constants
// ============================================================================

#define EPS 1e-9
#define PI 3.14159265358979323846
#define COMPLEX_TRANSFORM_MULTIPLIER 0.000001

// ============================================================================
// Host-side xoshiro256++ implementation (for matrix generation)
// ============================================================================

static inline uint64_t host_rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t host_xoshiro_gen(XoshiroState* x) {
    uint64_t res = host_rotl64(x->s0 + x->s3, 23) + x->s0;
    uint64_t t = x->s1 << 17;

    x->s2 ^= x->s0;
    x->s3 ^= x->s1;
    x->s1 ^= x->s2;
    x->s0 ^= x->s3;

    x->s2 ^= t;
    x->s3 = host_rotl64(x->s3, 45);

    return res;
}

void host_generate_hoohash_matrix(const uint8_t* hash, double mat[64][64]) {
    XoshiroState state;
    state.s0 = *(const uint64_t*)(&hash[0]);
    state.s1 = *(const uint64_t*)(&hash[8]);
    state.s2 = *(const uint64_t*)(&hash[16]);
    state.s3 = *(const uint64_t*)(&hash[24]);

    double normalize = 1000000.0;
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            uint64_t val = host_xoshiro_gen(&state);
            uint32_t lower_4_bytes = val & 0xFFFFFFFF;
            mat[i][j] = (double)lower_4_bytes / (double)UINT32_MAX * normalize;
        }
    }
}

// ============================================================================
// Device-side nonlinear transform functions
// Exact port of the reference C implementation for hash correctness
// ============================================================================

__device__ __forceinline__ double device_MediumComplexNonLinear(double x) {
    return exp(sin(x) + cos(x));
}

__device__ __forceinline__ double device_IntermediateComplexNonLinear(double x) {
    if (x == PI / 2.0 || x == 3.0 * PI / 2.0) {
        return 0.0;
    }
    return sin(x) * sin(x);
}

__device__ __forceinline__ double device_HighComplexNonLinear(double x) {
    return 1.0 / sqrt(fabs(x) + 1.0);
}

__device__ double device_ComplexNonLinear(double x) {
    double transformFactorOne = (x * COMPLEX_TRANSFORM_MULTIPLIER) / 8.0 -
                                 floor((x * COMPLEX_TRANSFORM_MULTIPLIER) / 8.0);
    double transformFactorTwo = (x * COMPLEX_TRANSFORM_MULTIPLIER) / 4.0 -
                                 floor((x * COMPLEX_TRANSFORM_MULTIPLIER) / 4.0);

    if (transformFactorOne < 0.33) {
        if (transformFactorTwo < 0.25)
            return device_MediumComplexNonLinear(x + (1.0 + transformFactorTwo));
        else if (transformFactorTwo < 0.5)
            return device_MediumComplexNonLinear(x - (1.0 + transformFactorTwo));
        else if (transformFactorTwo < 0.75)
            return device_MediumComplexNonLinear(x * (1.0 + transformFactorTwo));
        else
            return device_MediumComplexNonLinear(x / (1.0 + transformFactorTwo));
    } else if (transformFactorOne < 0.66) {
        if (transformFactorTwo < 0.25)
            return device_IntermediateComplexNonLinear(x + (1.0 + transformFactorTwo));
        else if (transformFactorTwo < 0.5)
            return device_IntermediateComplexNonLinear(x - (1.0 + transformFactorTwo));
        else if (transformFactorTwo < 0.75)
            return device_IntermediateComplexNonLinear(x * (1.0 + transformFactorTwo));
        else
            return device_IntermediateComplexNonLinear(x / (1.0 + transformFactorTwo));
    } else {
        if (transformFactorTwo < 0.25)
            return device_HighComplexNonLinear(x + (1.0 + transformFactorTwo));
        else if (transformFactorTwo < 0.5)
            return device_HighComplexNonLinear(x - (1.0 + transformFactorTwo));
        else if (transformFactorTwo < 0.75)
            return device_HighComplexNonLinear(x * (1.0 + transformFactorTwo));
        else
            return device_HighComplexNonLinear(x / (1.0 + transformFactorTwo));
    }
}

__device__ double device_ForComplex(double forComplex) {
    double complex_val;
    double rounds = 1.0;
    complex_val = device_ComplexNonLinear(forComplex);
    while (isnan(complex_val) || isinf(complex_val)) {
        forComplex = forComplex * 0.1;
        if (forComplex <= 0.0000000000001) {
            return 0.0 * rounds;
        }
        rounds += 1.0;
        complex_val = device_ComplexNonLinear(forComplex);
    }
    return complex_val * rounds;
}

__device__ __forceinline__ double device_TransformFactor(double x) {
    const double granularity = 1024.0;
    return x / granularity - floor(x / granularity);
}

// ============================================================================
// Device-side HooHash Matrix Multiplication
// Exact port of HoohashMatrixMultiplication from hoohash.c
// ============================================================================

__device__ void device_HoohashMatrixMultiplication(
    const double* __restrict__ mat,  // 64x64 row-major
    const uint8_t* hashBytes,        // 32-byte firstPass
    uint8_t* output,                 // 32-byte output
    uint64_t nonce)
{
    uint8_t vector[64];
    double product[64];
    uint8_t scaledValues[32];
    uint8_t result[32];

    // Convert hash bytes to uint32 array and compute hashMod
    uint32_t H[8];
    for (int i = 0; i < 8; i++) {
        H[i] = ((uint32_t)hashBytes[i * 4] << 24) |
               ((uint32_t)hashBytes[i * 4 + 1] << 16) |
               ((uint32_t)hashBytes[i * 4 + 2] << 8) |
               (uint32_t)hashBytes[i * 4 + 3];
    }

    double hashMod = (double)(H[0] ^ H[1] ^ H[2] ^ H[3] ^ H[4] ^ H[5] ^ H[6] ^ H[7]);
    double nonceMod = (double)(nonce & 0xFF);
    double divider = 0.0001;
    double multiplier = 1234.0;
    double sw = 0.0;

    // Split hash bytes into nibbles
    for (int i = 0; i < 32; i++) {
        vector[2 * i] = hashBytes[i] >> 4;
        vector[2 * i + 1] = hashBytes[i] & 0x0F;
    }

    // Initialize product
    for (int i = 0; i < 64; i++) {
        product[i] = 0.0;
    }

    // Core matrix multiply with conditional nonlinear transforms
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            double mat_val = mat[i * 64 + j];
            if (sw <= 0.02) {
                double input = (mat_val * hashMod * (double)vector[j] + nonceMod);
                double out_val = device_ForComplex(input) * (double)vector[j] * multiplier;
                product[i] += out_val;
            } else {
                double out_val = mat_val * divider * (double)vector[j];
                product[i] += out_val;
            }
            sw = device_TransformFactor(product[i]);
        }
    }

    // Combine product pairs into bytes
    for (int i = 0; i < 64; i += 2) {
        uint64_t pval = (uint64_t)product[i] + (uint64_t)product[i + 1];
        scaledValues[i / 2] = (uint8_t)(pval & 0xFF);
    }

    // XOR with original hash
    for (int i = 0; i < 32; i++) {
        result[i] = hashBytes[i] ^ scaledValues[i];
    }

    // Final BLAKE3 hash
    DeviceBlake3Hasher hasher;
    device_blake3_hasher_init(&hasher);
    device_blake3_hasher_update(&hasher, result, DOMAIN_HASH_SIZE);
    device_blake3_hasher_finalize(&hasher, output, DOMAIN_HASH_SIZE);
}

// ============================================================================
// Device-side CalculateProofOfWorkValue
// ============================================================================

__device__ void device_hoohash_pow(const double* __restrict__ mat,
                                    const uint8_t* __restrict__ prev_header,
                                    int64_t timestamp,
                                    uint64_t nonce,
                                    uint8_t* __restrict__ result)
{
    // Build the input for first BLAKE3 pass:
    // PrevHeader(32) || Timestamp(8) || zeroes(32) || Nonce(8) = 80 bytes
    uint8_t input_buf[80];

    // Copy PrevHeader
    for (int i = 0; i < 32; i++) {
        input_buf[i] = prev_header[i];
    }

    // Copy Timestamp (little-endian, as-is in memory)
    const uint8_t* ts_bytes = (const uint8_t*)&timestamp;
    for (int i = 0; i < 8; i++) {
        input_buf[32 + i] = ts_bytes[i];
    }

    // Zeroes (32 bytes)
    for (int i = 0; i < 32; i++) {
        input_buf[40 + i] = 0;
    }

    // Copy Nonce (little-endian, as-is in memory)
    const uint8_t* nonce_bytes = (const uint8_t*)&nonce;
    for (int i = 0; i < 8; i++) {
        input_buf[72 + i] = nonce_bytes[i];
    }

    // First BLAKE3 pass
    uint8_t firstPass[DOMAIN_HASH_SIZE];
    DeviceBlake3Hasher hasher;
    device_blake3_hasher_init(&hasher);
    device_blake3_hasher_update(&hasher, input_buf, 80);
    device_blake3_hasher_finalize(&hasher, firstPass, DOMAIN_HASH_SIZE);

    // HooHash matrix multiplication
    device_HoohashMatrixMultiplication(mat, firstPass, result, nonce);
}

// ============================================================================
// Device-side target comparison
// ============================================================================

__device__ bool device_hash_meets_target(const uint8_t* hash,
                                          const uint8_t* target) {
    for (int i = DOMAIN_HASH_SIZE - 1; i >= 0; i--) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;
}

// ============================================================================
// Mining kernel
// Each thread processes one nonce
// ============================================================================

__global__ void hoohash_mining_kernel(
    const double* __restrict__ d_mat,
    const uint8_t* __restrict__ d_prev_header,
    int64_t timestamp,
    uint64_t start_nonce,
    const uint8_t* __restrict__ d_target,
    uint64_t* __restrict__ d_result_nonces,
    uint8_t* __restrict__ d_result_hashes,
    uint32_t* __restrict__ d_result_count,
    uint32_t max_results,
    uint32_t batch_size)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    uint64_t nonce = start_nonce + tid;
    uint8_t result[DOMAIN_HASH_SIZE];

    // Compute PoW hash
    device_hoohash_pow(d_mat, d_prev_header, timestamp, nonce, result);

    // Check against target
    if (device_hash_meets_target(result, d_target)) {
        uint32_t idx = atomicAdd(d_result_count, 1);
        if (idx < max_results) {
            d_result_nonces[idx] = nonce;
            for (int i = 0; i < DOMAIN_HASH_SIZE; i++) {
                d_result_hashes[idx * DOMAIN_HASH_SIZE + i] = result[i];
            }
        }
    }
}
