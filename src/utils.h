#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>

#define DOMAIN_HASH_SIZE 32
#define HOOHASH_MATRIX_SIZE 64

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,          \
                    __LINE__, (int)status);                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Mining job from stratum
struct MiningJob {
    std::string job_id;
    uint8_t prev_header[DOMAIN_HASH_SIZE];
    uint8_t target[DOMAIN_HASH_SIZE];
    int64_t timestamp;
    bool valid;
};

// Miner configuration
struct MinerConfig {
    std::string stratum_url;
    std::string user;
    std::string worker;
    std::string password;
    int device_id;
    int intensity; // batch size = 2^intensity
};

// Mining statistics
struct MinerStats {
    std::atomic<uint64_t> total_hashes{0};
    std::atomic<uint64_t> accepted_shares{0};
    std::atomic<uint64_t> rejected_shares{0};
    std::chrono::steady_clock::time_point start_time;
};

// Hex encoding/decoding utilities
inline std::string encode_hex(const uint8_t* data, size_t len) {
    std::string hex;
    hex.reserve(len * 2);
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        hex.push_back(digits[data[i] >> 4]);
        hex.push_back(digits[data[i] & 0x0F]);
    }
    return hex;
}

inline bool decode_hex(const std::string& hex, uint8_t* out, size_t max_len) {
    if (hex.length() % 2 != 0 || hex.length() / 2 > max_len) return false;
    for (size_t i = 0; i < hex.length() / 2; i++) {
        char high = hex[2 * i];
        char low = hex[2 * i + 1];
        auto hexval = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        int h = hexval(high), l = hexval(low);
        if (h < 0 || l < 0) return false;
        out[i] = (uint8_t)((h << 4) | l);
    }
    return true;
}

// Compare hash against target (little-endian, lower = better)
inline bool hash_meets_target(const uint8_t* hash, const uint8_t* target) {
    // Compare from most significant byte (index 31) to least
    for (int i = DOMAIN_HASH_SIZE - 1; i >= 0; i--) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true; // equal
}

#endif // UTILS_H
