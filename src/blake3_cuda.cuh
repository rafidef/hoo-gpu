#ifndef BLAKE3_CUDA_CUH
#define BLAKE3_CUDA_CUH

#include <cstdint>

// BLAKE3 constants
#define BLAKE3_OUT_LEN 32
#define BLAKE3_KEY_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024

// BLAKE3 IV
__constant__ static const uint32_t BLAKE3_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// BLAKE3 message permutation
__constant__ static const uint8_t BLAKE3_MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

// BLAKE3 flags
#define BLAKE3_CHUNK_START  (1 << 0)
#define BLAKE3_CHUNK_END    (1 << 1)
#define BLAKE3_ROOT         (1 << 3)

// Device BLAKE3 hasher state
struct DeviceBlake3Hasher {
    uint8_t buf[BLAKE3_CHUNK_LEN];
    uint32_t buf_len;
};

// Initialize a device BLAKE3 hasher
__device__ void device_blake3_hasher_init(DeviceBlake3Hasher* hasher);

// Update hasher with data
__device__ void device_blake3_hasher_update(DeviceBlake3Hasher* hasher,
                                             const void* data, uint32_t len);

// Finalize and output hash
__device__ void device_blake3_hasher_finalize(const DeviceBlake3Hasher* hasher,
                                               uint8_t* out, uint32_t out_len);

// Convenience: hash data in one call
__device__ void device_blake3_hash(const void* data, uint32_t len,
                                    uint8_t* out);

#endif // BLAKE3_CUDA_CUH
