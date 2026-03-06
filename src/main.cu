/*
 * Hoosat H100 Miner — Main Entry Point
 *
 * High-performance HooHash miner optimized for NVIDIA H100 SXM5 FP64 Tensor Cores.
 *
 * Usage:
 *   hoosat_h100_miner --stratum stratum+tcp://host:port --user <wallet> [options]
 *
 * Options:
 *   --stratum <url>     Stratum server URL
 *   --user <address>    Hoosat wallet address
 *   --worker <name>     Worker name (default: h100)
 *   --password <pass>   Password (default: x)
 *   --device <id>       CUDA device ID (default: 0)
 *   --intensity <n>     Batch size = 2^n nonces (default: 22)
 *   --benchmark         Run benchmark mode (no pool connection)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <csignal>

#include "hoohash.cuh"
#include "blake3_cuda.cuh"
#include "stratum.h"
#include "utils.h"

// ============================================================================
// Globals
// ============================================================================

static std::atomic<bool> g_running(true);
static std::atomic<bool> g_new_job(false);
static std::mutex g_job_mutex;
static MiningJob g_current_job;

void signal_handler(int sig) {
    (void)sig;
    printf("\n[Miner] Shutting down...\n");
    g_running = false;
}

// ============================================================================
// Parse stratum URL: stratum+tcp://host:port → (host, port)
// ============================================================================

static bool parse_stratum_url(const std::string& url, std::string& host, int& port) {
    std::string s = url;

    // Strip protocol prefix
    auto pos = s.find("://");
    if (pos != std::string::npos) {
        s = s.substr(pos + 3);
    }

    // Split host:port
    pos = s.find(':');
    if (pos == std::string::npos) {
        host = s;
        port = 5555; // Default port
        return true;
    }

    host = s.substr(0, pos);
    try {
        port = std::stoi(s.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// ============================================================================
// Print CUDA device info
// ============================================================================

static void print_device_info(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           HOOSAT H100 GPU MINER v1.0.0                     ║\n");
    printf("║           Optimized for NVIDIA H100 SXM5                   ║\n");
    printf("║           FP64 Tensor Core Accelerated                     ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  GPU: %-53s ║\n", prop.name);
    printf("║  Compute: %d.%d    SMs: %-3d    Clock: %d MHz              ║\n",
           prop.major, prop.minor, prop.multiProcessorCount,
           prop.clockRate / 1000);
    printf("║  Memory: %zu MB    Bandwidth: %d GB/s                      ║\n",
           prop.totalGlobalMem / (1024 * 1024),
           (int)(2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6));
    if (prop.major >= 9) {
        printf("║  ✓ Hopper Architecture — FP64 Tensor Cores ACTIVE         ║\n");
    } else {
        printf("║  ⚠ Non-Hopper GPU — FP64 Tensor Cores not available       ║\n");
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

// ============================================================================
// Benchmark mode
// ============================================================================

static void run_benchmark(int device_id, int intensity) {
    CUDA_CHECK(cudaSetDevice(device_id));
    print_device_info(device_id);

    uint32_t batch_size = 1u << intensity;
    printf("[Benchmark] Batch size: %u nonces (2^%d)\n", batch_size, intensity);

    // Generate test data
    uint8_t prev_header[DOMAIN_HASH_SIZE] = {
        0xa4, 0x9d, 0xbc, 0x7d, 0x44, 0xae, 0x83, 0x25,
        0x38, 0x23, 0x59, 0x2f, 0xd3, 0x88, 0xf2, 0x19,
        0xf3, 0xcb, 0x83, 0x63, 0x9d, 0x54, 0xc9, 0xe4,
        0xc3, 0x15, 0x4d, 0xb3, 0x6f, 0x2b, 0x51, 0x57
    };
    int64_t timestamp = 1725374568455LL;
    uint8_t target[DOMAIN_HASH_SIZE];
    memset(target, 0xFF, DOMAIN_HASH_SIZE); // Easy target for benchmark

    // Generate matrix on host
    double h_mat[64][64];
    host_generate_hoohash_matrix(prev_header, h_mat);

    // Allocate device memory
    double* d_mat;
    uint8_t* d_prev_header;
    uint8_t* d_target;
    uint64_t* d_result_nonces;
    uint8_t* d_result_hashes;
    uint32_t* d_result_count;

    CUDA_CHECK(cudaMalloc(&d_mat, 64 * 64 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prev_header, DOMAIN_HASH_SIZE));
    CUDA_CHECK(cudaMalloc(&d_target, DOMAIN_HASH_SIZE));
    CUDA_CHECK(cudaMalloc(&d_result_nonces, 32 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_result_hashes, 32 * DOMAIN_HASH_SIZE));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, 64 * 64 * sizeof(double),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev_header, prev_header, DOMAIN_HASH_SIZE,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, DOMAIN_HASH_SIZE,
                           cudaMemcpyHostToDevice));

    // Warm up
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_result_count, &zero, sizeof(uint32_t),
                           cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    hoohash_mining_kernel<<<blocks, threads_per_block>>>(
        d_mat, d_prev_header, timestamp, 0, d_target,
        d_result_nonces, d_result_hashes, d_result_count, 32, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[Benchmark] Warmup complete. Starting measurement...\n\n");

    // Benchmark loop
    uint64_t total_hashes = 0;
    auto start = std::chrono::steady_clock::now();

    for (int round = 0; round < 10 && g_running; round++) {
        CUDA_CHECK(cudaMemcpy(d_result_count, &zero, sizeof(uint32_t),
                               cudaMemcpyHostToDevice));

        uint64_t start_nonce = (uint64_t)round * batch_size;

        hoohash_mining_kernel<<<blocks, threads_per_block>>>(
            d_mat, d_prev_header, timestamp, start_nonce, d_target,
            d_result_nonces, d_result_hashes, d_result_count, 32, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        total_hashes += batch_size;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        double hashrate = total_hashes / elapsed;

        const char* unit = "H/s";
        double display_rate = hashrate;
        if (display_rate >= 1e9) { display_rate /= 1e9; unit = "GH/s"; }
        else if (display_rate >= 1e6) { display_rate /= 1e6; unit = "MH/s"; }
        else if (display_rate >= 1e3) { display_rate /= 1e3; unit = "KH/s"; }

        printf("[Benchmark] Round %d/10 | Hashrate: %.2f %s | Total: %llu hashes\n",
               round + 1, display_rate, unit,
               (unsigned long long)total_hashes);
    }

    auto end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end - start).count();
    double final_hashrate = total_hashes / total_time;

    const char* unit = "H/s";
    double display_rate = final_hashrate;
    if (display_rate >= 1e9) { display_rate /= 1e9; unit = "GH/s"; }
    else if (display_rate >= 1e6) { display_rate /= 1e6; unit = "MH/s"; }
    else if (display_rate >= 1e3) { display_rate /= 1e3; unit = "KH/s"; }

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  BENCHMARK RESULTS                                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Average Hashrate: %.2f %-6s                              ║\n",
           display_rate, unit);
    printf("║  Total Hashes: %llu                                  ║\n",
           (unsigned long long)total_hashes);
    printf("║  Total Time: %.2f seconds                                  ║\n",
           total_time);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_prev_header));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_result_nonces));
    CUDA_CHECK(cudaFree(d_result_hashes));
    CUDA_CHECK(cudaFree(d_result_count));
}

// ============================================================================
// Pool mining mode
// ============================================================================

static void run_mining(const MinerConfig& config) {
    CUDA_CHECK(cudaSetDevice(config.device_id));
    print_device_info(config.device_id);

    // Parse stratum URL
    std::string host;
    int port;
    if (!parse_stratum_url(config.stratum_url, host, port)) {
        fprintf(stderr, "[Miner] Invalid stratum URL: %s\n",
                config.stratum_url.c_str());
        return;
    }

    // Connect to pool
    StratumClient stratum;
    if (!stratum.connect(host, port)) return;
    if (!stratum.subscribe()) return;
    if (!stratum.authorize(config.user, config.password, config.worker)) return;

    // Set job callback
    stratum.set_job_callback([](const MiningJob& job) {
        std::lock_guard<std::mutex> lock(g_job_mutex);
        g_current_job = job;
        g_new_job = true;
    });
    stratum.start_receiving();

    // Wait for first job
    printf("[Miner] Waiting for first job...\n");
    while (g_running && !g_new_job) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!g_running) return;

    uint32_t batch_size = 1u << config.intensity;
    printf("[Miner] Batch size: %u nonces (2^%d)\n", batch_size, config.intensity);

    // Allocate device memory
    double* d_mat;
    uint8_t* d_prev_header;
    uint8_t* d_target;
    uint64_t* d_result_nonces;
    uint8_t* d_result_hashes;
    uint32_t* d_result_count;

    CUDA_CHECK(cudaMalloc(&d_mat, 64 * 64 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_prev_header, DOMAIN_HASH_SIZE));
    CUDA_CHECK(cudaMalloc(&d_target, DOMAIN_HASH_SIZE));
    CUDA_CHECK(cudaMalloc(&d_result_nonces, 32 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_result_hashes, 32 * DOMAIN_HASH_SIZE));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));

    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    double h_mat[64][64];
    uint64_t nonce_counter = 0;
    uint64_t total_hashes = 0;
    uint64_t accepted_shares = 0;
    uint64_t rejected_shares = 0;
    auto mining_start = std::chrono::steady_clock::now();
    auto last_report = mining_start;
    std::string current_job_id;

    printf("[Miner] Mining started!\n\n");

    while (g_running && stratum.is_connected()) {
        // Check for new job
        MiningJob job;
        bool new_job = false;
        {
            std::lock_guard<std::mutex> lock(g_job_mutex);
            if (g_new_job) {
                job = g_current_job;
                g_new_job = false;
                new_job = true;
            } else {
                job = g_current_job;
            }
        }

        if (!job.valid) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (new_job) {
            // Generate new matrix and upload to GPU
            host_generate_hoohash_matrix(job.prev_header, h_mat);
            CUDA_CHECK(cudaMemcpy(d_mat, h_mat, 64 * 64 * sizeof(double),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_prev_header, job.prev_header,
                                   DOMAIN_HASH_SIZE, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_target, job.target, DOMAIN_HASH_SIZE,
                                   cudaMemcpyHostToDevice));
            current_job_id = job.job_id;
            nonce_counter = 0;
        }

        // Reset result counter
        uint32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(d_result_count, &zero, sizeof(uint32_t),
                               cudaMemcpyHostToDevice));

        // Launch mining kernel
        hoohash_mining_kernel<<<blocks, threads_per_block>>>(
            d_mat, d_prev_header, job.timestamp, nonce_counter, d_target,
            d_result_nonces, d_result_hashes, d_result_count, 32, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        nonce_counter += batch_size;
        total_hashes += batch_size;

        // Check for valid shares
        uint32_t result_count;
        CUDA_CHECK(cudaMemcpy(&result_count, d_result_count, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

        if (result_count > 0) {
            if (result_count > 32) result_count = 32;

            uint64_t h_nonces[32];
            uint8_t h_hashes[32 * DOMAIN_HASH_SIZE];
            CUDA_CHECK(cudaMemcpy(h_nonces, d_result_nonces,
                                   result_count * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_hashes, d_result_hashes,
                                   result_count * DOMAIN_HASH_SIZE,
                                   cudaMemcpyDeviceToHost));

            for (uint32_t i = 0; i < result_count; i++) {
                printf("[Miner] ★ Share found! Nonce: %llu\n",
                       (unsigned long long)h_nonces[i]);
                if (stratum.submit_share(current_job_id, h_nonces[i],
                                          &h_hashes[i * DOMAIN_HASH_SIZE])) {
                    accepted_shares++;
                } else {
                    rejected_shares++;
                }
            }
        }

        // Periodic hashrate report (every 10 seconds)
        auto now = std::chrono::steady_clock::now();
        double since_report = std::chrono::duration<double>(now - last_report).count();
        if (since_report >= 10.0) {
            double elapsed = std::chrono::duration<double>(now - mining_start).count();
            double hashrate = total_hashes / elapsed;

            const char* unit = "H/s";
            double display_rate = hashrate;
            if (display_rate >= 1e9) { display_rate /= 1e9; unit = "GH/s"; }
            else if (display_rate >= 1e6) { display_rate /= 1e6; unit = "MH/s"; }
            else if (display_rate >= 1e3) { display_rate /= 1e3; unit = "KH/s"; }

            printf("[Miner] Hashrate: %.2f %s | Shares: %llu/%llu | "
                   "Nonces: %llu\n",
                   display_rate, unit,
                   (unsigned long long)accepted_shares,
                   (unsigned long long)(accepted_shares + rejected_shares),
                   (unsigned long long)total_hashes);

            last_report = now;
        }
    }

    // Cleanup
    stratum.stop_receiving();
    stratum.disconnect();
    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_prev_header));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_result_nonces));
    CUDA_CHECK(cudaFree(d_result_hashes));
    CUDA_CHECK(cudaFree(d_result_count));

    printf("[Miner] Stopped.\n");
}

// ============================================================================
// CLI parsing and main
// ============================================================================

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --stratum <url>     Stratum server URL (stratum+tcp://host:port)\n");
    printf("  --user <address>    Hoosat wallet address\n");
    printf("  --worker <name>     Worker name (default: h100)\n");
    printf("  --password <pass>   Password (default: x)\n");
    printf("  --device <id>       CUDA device ID (default: 0)\n");
    printf("  --intensity <n>     Batch size = 2^n nonces (default: 22)\n");
    printf("  --benchmark         Run benchmark mode (no pool connection)\n");
    printf("  --help              Show this help\n");
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
#ifndef _WIN32
    signal(SIGTERM, signal_handler);
#endif

    MinerConfig config;
    config.stratum_url = "";
    config.user = "";
    config.worker = "h100";
    config.password = "x";
    config.device_id = 0;
    config.intensity = 22;
    bool benchmark = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--stratum" && i + 1 < argc) {
            config.stratum_url = argv[++i];
        } else if (arg == "--user" && i + 1 < argc) {
            config.user = argv[++i];
        } else if (arg == "--worker" && i + 1 < argc) {
            config.worker = argv[++i];
        } else if (arg == "--password" && i + 1 < argc) {
            config.password = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            config.device_id = std::stoi(argv[++i]);
        } else if (arg == "--intensity" && i + 1 < argc) {
            config.intensity = std::stoi(argv[++i]);
            if (config.intensity < 10) config.intensity = 10;
            if (config.intensity > 30) config.intensity = 30;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Check CUDA device count
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "[Miner] No CUDA devices found!\n");
        return 1;
    }
    if (config.device_id >= device_count) {
        fprintf(stderr, "[Miner] Invalid device ID %d (found %d devices)\n",
                config.device_id, device_count);
        return 1;
    }

    if (benchmark) {
        run_benchmark(config.device_id, config.intensity);
        return 0;
    }

    if (config.stratum_url.empty() || config.user.empty()) {
        fprintf(stderr, "[Miner] --stratum and --user are required for pool mining\n\n");
        print_usage(argv[0]);
        return 1;
    }

    run_mining(config);
    return 0;
}
