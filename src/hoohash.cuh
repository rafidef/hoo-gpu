#ifndef HOOHASH_CUH
#define HOOHASH_CUH

#include <cstdint>
#include <cuda_runtime.h>

#define DOMAIN_HASH_SIZE 32
#define HOOHASH_MATRIX_SIZE 64

// HooHash state for GPU mining
struct HoohashGPUState {
    double mat[HOOHASH_MATRIX_SIZE][HOOHASH_MATRIX_SIZE]; // Pre-generated matrix
    uint8_t prev_header[DOMAIN_HASH_SIZE];
    uint8_t target[DOMAIN_HASH_SIZE];
    int64_t timestamp;
};

// xoshiro256++ PRNG state
struct XoshiroState {
    uint64_t s0, s1, s2, s3;
};

// ============================================================================
// Host-side functions
// ============================================================================

// Generate the HooHash 64x64 FP64 matrix from a block header hash (host-side)
void host_generate_hoohash_matrix(const uint8_t* hash, double mat[64][64]);

// ============================================================================
// Device-side functions
// ============================================================================

// Compute HooHash PoW for a single nonce (device function)
__device__ void device_hoohash_pow(const double* __restrict__ mat,
                                    const uint8_t* __restrict__ prev_header,
                                    int64_t timestamp,
                                    uint64_t nonce,
                                    uint8_t* __restrict__ result);

// ============================================================================
// Kernel launches
// ============================================================================

// Mining kernel: tests a batch of nonces, writes valid results
// d_mat: device pointer to 64x64 FP64 matrix (row-major, 64*64 doubles)
// d_prev_header: device pointer to 32-byte prev_header
// timestamp: block timestamp
// start_nonce: first nonce in batch
// d_target: device pointer to 32-byte target
// d_results: device pointer to results buffer (nonce + hash for each valid share)
// d_result_count: device pointer to atomic counter for valid results
// batch_size: number of nonces to test
__global__ void hoohash_mining_kernel(const double* __restrict__ d_mat,
                                       const uint8_t* __restrict__ d_prev_header,
                                       int64_t timestamp,
                                       uint64_t start_nonce,
                                       const uint8_t* __restrict__ d_target,
                                       uint64_t* __restrict__ d_result_nonces,
                                       uint8_t* __restrict__ d_result_hashes,
                                       uint32_t* __restrict__ d_result_count,
                                       uint32_t max_results,
                                       uint32_t batch_size);

#endif // HOOHASH_CUH
