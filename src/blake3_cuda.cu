/*
 * BLAKE3 Device Implementation for CUDA
 *
 * Minimal single-chunk BLAKE3 implementation as __device__ functions.
 * Sufficient for HooHash which only hashes small inputs (< 1024 bytes).
 *
 * Reference: https://github.com/BLAKE3-team/BLAKE3-spec/blob/master/blake3.pdf
 */

#include "blake3_cuda.cuh"

// Rotate right 32-bit
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// BLAKE3 G mixing function
__device__ __forceinline__ void blake3_g(uint32_t* state, int a, int b, int c,
                                          int d, uint32_t mx, uint32_t my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

// BLAKE3 round function
__device__ void blake3_round(uint32_t* state, const uint32_t* msg,
                              const uint8_t* schedule) {
    // Column step
    blake3_g(state, 0, 4,  8, 12, msg[schedule[0]],  msg[schedule[1]]);
    blake3_g(state, 1, 5,  9, 13, msg[schedule[2]],  msg[schedule[3]]);
    blake3_g(state, 2, 6, 10, 14, msg[schedule[4]],  msg[schedule[5]]);
    blake3_g(state, 3, 7, 11, 15, msg[schedule[6]],  msg[schedule[7]]);
    // Diagonal step
    blake3_g(state, 0, 5, 10, 15, msg[schedule[8]],  msg[schedule[9]]);
    blake3_g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
    blake3_g(state, 2, 7,  8, 13, msg[schedule[12]], msg[schedule[13]]);
    blake3_g(state, 3, 4,  9, 14, msg[schedule[14]], msg[schedule[15]]);
}

// BLAKE3 compression function
// cv: 8-word chaining value
// block: 16-word message block
// counter: block counter
// block_len: bytes in this block
// flags: BLAKE3 flags
__device__ void blake3_compress(const uint32_t cv[8],
                                 const uint32_t block[16],
                                 uint64_t counter,
                                 uint32_t block_len,
                                 uint32_t flags,
                                 uint32_t out[16]) {
    uint32_t state[16];

    // Initialize state
    state[0]  = cv[0];
    state[1]  = cv[1];
    state[2]  = cv[2];
    state[3]  = cv[3];
    state[4]  = cv[4];
    state[5]  = cv[5];
    state[6]  = cv[6];
    state[7]  = cv[7];
    state[8]  = BLAKE3_IV[0];
    state[9]  = BLAKE3_IV[1];
    state[10] = BLAKE3_IV[2];
    state[11] = BLAKE3_IV[3];
    state[12] = (uint32_t)(counter);
    state[13] = (uint32_t)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    // 7 rounds
    for (int r = 0; r < 7; r++) {
        blake3_round(state, block, BLAKE3_MSG_SCHEDULE[r]);
    }

    // Finalize: XOR first 8 words with last 8, and last 8 with cv
    for (int i = 0; i < 8; i++) {
        out[i] = state[i] ^ state[i + 8];
        out[i + 8] = state[i + 8] ^ cv[i];
    }
}

// Load a 32-bit little-endian word from bytes
__device__ __forceinline__ uint32_t load32_le(const uint8_t* p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

// Store a 32-bit little-endian word to bytes
__device__ __forceinline__ void store32_le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

// Initialize the hasher
__device__ void device_blake3_hasher_init(DeviceBlake3Hasher* hasher) {
    hasher->buf_len = 0;
}

// Update the hasher with data
__device__ void device_blake3_hasher_update(DeviceBlake3Hasher* hasher,
                                             const void* data, uint32_t len) {
    const uint8_t* src = (const uint8_t*)data;
    uint32_t remaining = len;

    while (remaining > 0) {
        uint32_t space = BLAKE3_CHUNK_LEN - hasher->buf_len;
        uint32_t take = remaining < space ? remaining : space;
        for (uint32_t i = 0; i < take; i++) {
            hasher->buf[hasher->buf_len + i] = src[i];
        }
        hasher->buf_len += take;
        src += take;
        remaining -= take;
    }
}

// Finalize and produce the hash output
// For single-chunk inputs (≤ 1024 bytes), this is straightforward:
// Process each 64-byte block of the chunk, the last block gets ROOT flag
__device__ void device_blake3_hasher_finalize(const DeviceBlake3Hasher* hasher,
                                               uint8_t* out,
                                               uint32_t out_len) {
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = BLAKE3_IV[i];
    }

    uint32_t input_len = hasher->buf_len;
    uint32_t num_blocks = (input_len + BLAKE3_BLOCK_LEN - 1) / BLAKE3_BLOCK_LEN;
    if (num_blocks == 0) num_blocks = 1;

    for (uint32_t b = 0; b < num_blocks; b++) {
        // Load message block (pad with zeros if needed)
        uint32_t msg[16];
        uint32_t block_start = b * BLAKE3_BLOCK_LEN;
        uint32_t block_bytes = input_len - block_start;
        if (block_bytes > BLAKE3_BLOCK_LEN) block_bytes = BLAKE3_BLOCK_LEN;

        // Zero-pad the block
        uint8_t block_buf[BLAKE3_BLOCK_LEN];
        for (int i = 0; i < BLAKE3_BLOCK_LEN; i++) {
            if ((uint32_t)i < block_bytes) {
                block_buf[i] = hasher->buf[block_start + i];
            } else {
                block_buf[i] = 0;
            }
        }

        // Parse block into 32-bit words
        for (int i = 0; i < 16; i++) {
            msg[i] = load32_le(&block_buf[i * 4]);
        }

        // Determine flags
        uint32_t flags = 0;
        if (b == 0) flags |= BLAKE3_CHUNK_START;
        if (b == num_blocks - 1) flags |= BLAKE3_CHUNK_END;
        // For single-chunk, the last block also gets ROOT
        if (b == num_blocks - 1) flags |= BLAKE3_ROOT;

        if (b < num_blocks - 1) {
            // Non-last block: compress and take first 8 words as new CV
            uint32_t full_out[16];
            blake3_compress(cv, msg, 0, block_bytes, flags, full_out);
            for (int i = 0; i < 8; i++) {
                cv[i] = full_out[i];
            }
        } else {
            // Last block (ROOT): produce output
            // For out_len <= 32, one compression is enough
            uint32_t full_out[16];
            blake3_compress(cv, msg, 0, block_bytes, flags, full_out);

            // Extract output bytes (little-endian)
            uint32_t to_copy = out_len < 64 ? out_len : 64;
            for (uint32_t i = 0; i < to_copy; i++) {
                out[i] = (uint8_t)(full_out[i / 4] >> (8 * (i % 4)));
            }
        }
    }
}

// Convenience: hash in one call
__device__ void device_blake3_hash(const void* data, uint32_t len,
                                    uint8_t* out) {
    DeviceBlake3Hasher hasher;
    device_blake3_hasher_init(&hasher);
    device_blake3_hasher_update(&hasher, data, len);
    device_blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}
