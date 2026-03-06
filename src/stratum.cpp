/*
 * Stratum Client Implementation
 *
 * Kaspa-style stratum protocol (JSON-RPC over TCP).
 * Compatible with HTN Stratum Bridge (https://github.com/HoosatNetwork/htn-stratum-bridge)
 *
 * Protocol flow:
 *   1. mining.subscribe → receive extra_nonce
 *   2. mining.authorize → authenticate with wallet address
 *   3. mining.set_difficulty → receive share difficulty as float
 *   4. mining.notify → receive new jobs (JobID, HeaderHash as uint64[], Timestamp)
 *   5. mining.submit → submit [worker, job_id, nonce_hex, pow_hash_hex]
 */

#include "stratum.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <sstream>
#include <algorithm>

// ============================================================================
// Simple JSON helpers (no external deps)
// ============================================================================

static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return "";

    pos++;
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos >= json.length()) return "";

    if (json[pos] == '"') {
        pos++;
        auto end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }

    auto end = json.find_first_of(",}]", pos);
    if (end == std::string::npos) end = json.length();
    std::string val = json.substr(pos, end - pos);
    while (!val.empty() && val.back() == ' ') val.pop_back();
    return val;
}

static std::string json_get_method(const std::string& json) {
    return json_get_string(json, "method");
}

static int json_get_int(const std::string& json, const std::string& key) {
    std::string val = json_get_string(json, key);
    if (val.empty()) return -1;
    try { return std::stoi(val); } catch (...) { return -1; }
}

static std::string json_get_params(const std::string& json) {
    auto pos = json.find("\"params\"");
    if (pos == std::string::npos) return "";
    pos = json.find('[', pos);
    if (pos == std::string::npos) return "";
    int depth = 0;
    size_t start = pos;
    for (size_t i = pos; i < json.length(); i++) {
        if (json[i] == '[') depth++;
        else if (json[i] == ']') {
            depth--;
            if (depth == 0) return json.substr(start, i - start + 1);
        }
    }
    return "";
}

static std::vector<std::string> json_parse_params(const std::string& params) {
    std::vector<std::string> result;
    if (params.empty() || params[0] != '[') return result;

    size_t i = 1;
    while (i < params.length()) {
        while (i < params.length() && (params[i] == ' ' || params[i] == ',' || params[i] == '\t')) i++;
        if (i >= params.length() || params[i] == ']') break;

        if (params[i] == '"') {
            i++;
            size_t start = i;
            while (i < params.length() && params[i] != '"') i++;
            result.push_back(params.substr(start, i - start));
            i++;
        } else if (params[i] == '[') {
            int depth = 0;
            size_t start = i;
            while (i < params.length()) {
                if (params[i] == '[') depth++;
                else if (params[i] == ']') {
                    depth--;
                    if (depth == 0) { i++; break; }
                }
                i++;
            }
            result.push_back(params.substr(start, i - start));
        } else {
            size_t start = i;
            while (i < params.length() && params[i] != ',' && params[i] != ']') i++;
            std::string val = params.substr(start, i - start);
            while (!val.empty() && val.back() == ' ') val.pop_back();
            result.push_back(val);
        }
    }
    return result;
}

// ============================================================================
// Difficulty → Target conversion
// target = maxTarget / difficulty
// maxTarget = 0xFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF (28 bytes of 0xFF)
// Since we work with 256-bit values as byte arrays, we do big integer division.
// ============================================================================

// Simple 256-bit division for difficulty conversion
// We compute: target[32] = (2^224 - 1) / difficulty, stored as little-endian bytes
static void difficulty_to_target(double difficulty, uint8_t target[32]) {
    // maxTarget = 2^224 - 1 (28 bytes of 0xFF, 4 bytes of 0x00 at the top)
    // As a double: roughly 2^224
    // target = maxTarget / difficulty
    //
    // We'll compute this as: target_val = 2^224 / difficulty
    // Then convert to 32-byte little-endian

    if (difficulty <= 0.0) {
        memset(target, 0xFF, 32);
        return;
    }

    // 2^224 as long double for more precision
    long double max_target_f = powl(2.0L, 224.0L);
    long double target_f = max_target_f / (long double)difficulty;

    // Convert to bytes (big-endian first, then reverse to little-endian)
    // We'll extract byte-by-byte by dividing by 256
    uint8_t be_bytes[32];
    memset(be_bytes, 0, 32);

    long double remaining = target_f;
    for (int i = 31; i >= 0; i--) {
        long double byte_val = fmodl(remaining, 256.0L);
        be_bytes[i] = (uint8_t)byte_val;
        remaining = floorl(remaining / 256.0L);
    }

    // The bytes are in big-endian, copy to target (which we store as-is for comparison)
    // Our hash comparison does big-endian comparison from byte[31] down to byte[0]
    // So store as big-endian directly
    memcpy(target, be_bytes, 32);
}

// ============================================================================
// StratumClient implementation
// ============================================================================

StratumClient::StratumClient()
    : sock_(INVALID_SOCK), connected_(false), running_(false),
      msg_id_(1) {
    current_job_.valid = false;
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

StratumClient::~StratumClient() {
    stop_receiving();
    disconnect();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool StratumClient::connect(const std::string& host, int port) {
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0) {
        fprintf(stderr, "[Stratum] Failed to resolve host: %s\n", host.c_str());
        return false;
    }

    sock_ = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock_ == INVALID_SOCK) {
        fprintf(stderr, "[Stratum] Failed to create socket\n");
        freeaddrinfo(res);
        return false;
    }

    if (::connect(sock_, res->ai_addr, (int)res->ai_addrlen) != 0) {
        fprintf(stderr, "[Stratum] Failed to connect to %s:%d\n",
                host.c_str(), port);
        CLOSE_SOCKET(sock_);
        sock_ = INVALID_SOCK;
        freeaddrinfo(res);
        return false;
    }

    freeaddrinfo(res);
    connected_ = true;
    printf("[Stratum] Connected to %s:%d\n", host.c_str(), port);
    return true;
}

void StratumClient::disconnect() {
    if (sock_ != INVALID_SOCK) {
        CLOSE_SOCKET(sock_);
        sock_ = INVALID_SOCK;
    }
    connected_ = false;
}

bool StratumClient::send_json(const std::string& json) {
    std::string msg = json + "\n";
    int total = (int)msg.length();
    int sent = 0;
    while (sent < total) {
        int n = send(sock_, msg.c_str() + sent, total - sent, 0);
        if (n <= 0) {
            fprintf(stderr, "[Stratum] Send failed\n");
            connected_ = false;
            return false;
        }
        sent += n;
    }
    return true;
}

std::string StratumClient::recv_line() {
    std::string line;
    char ch;
    while (true) {
        int n = recv(sock_, &ch, 1, 0);
        if (n <= 0) {
            connected_ = false;
            return "";
        }
        if (ch == '\n') return line;
        if (ch != '\r') line += ch;
    }
}

bool StratumClient::subscribe() {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"id\":%d,\"method\":\"mining.subscribe\",\"params\":[\"hoosat_h100_miner/1.0.0\"]}",
             msg_id_++);
    if (!send_json(buf)) return false;

    std::string response = recv_line();
    if (response.empty()) return false;

    printf("[Stratum] Subscribe response: %s\n", response.c_str());
    return true;
}

bool StratumClient::authorize(const std::string& user,
                               const std::string& password,
                               const std::string& worker) {
    std::string full_user = user;
    if (!worker.empty()) {
        full_user += "." + worker;
    }
    worker_name_ = full_user;

    char buf[1024];
    snprintf(buf, sizeof(buf),
             "{\"id\":%d,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}",
             msg_id_++, full_user.c_str(), password.c_str());
    if (!send_json(buf)) return false;

    std::string response = recv_line();
    if (response.empty()) return false;

    printf("[Stratum] Authorize response: %s\n", response.c_str());
    if (response.find("true") != std::string::npos) {
        printf("[Stratum] Authorization successful\n");
        return true;
    }

    fprintf(stderr, "[Stratum] Authorization failed\n");
    return false;
}

bool StratumClient::submit_share(const std::string& job_id, uint64_t nonce,
                                  const uint8_t* pow_hash) {
    // Format nonce as hex (no 0x prefix)
    char nonce_hex[17];
    snprintf(nonce_hex, sizeof(nonce_hex), "%016llx",
             (unsigned long long)nonce);

    // Format pow_hash as hex
    std::string hash_hex = encode_hex(pow_hash, DOMAIN_HASH_SIZE);

    // HTN stratum bridge expects 4 params: [worker, job_id, nonce, pow_hash]
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"id\":%d,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\"]}",
             msg_id_++, worker_name_.c_str(), job_id.c_str(),
             nonce_hex, hash_hex.c_str());

    return send_json(buf);
}

void StratumClient::set_job_callback(JobCallback cb) {
    job_callback_ = cb;
}

MiningJob StratumClient::get_current_job() {
    std::lock_guard<std::mutex> lock(job_mutex_);
    return current_job_;
}

void StratumClient::start_receiving() {
    running_ = true;
    recv_thread_ = std::thread(&StratumClient::receive_loop, this);
}

void StratumClient::stop_receiving() {
    running_ = false;
    if (recv_thread_.joinable()) {
        recv_thread_.join();
    }
}

void StratumClient::receive_loop() {
    while (running_ && connected_) {
        std::string line = recv_line();
        if (line.empty()) {
            if (!connected_) break;
            continue;
        }
        handle_message(line);
    }
}

void StratumClient::handle_message(const std::string& msg) {
    std::string method = json_get_method(msg);

    if (method == "mining.notify") {
        parse_notify(msg);
    } else if (method == "mining.set_difficulty") {
        auto params = json_get_params(msg);
        auto plist = json_parse_params(params);
        if (!plist.empty()) {
            double diff = 0.0;
            try { diff = std::stod(plist[0]); } catch (...) {}
            if (diff > 0) {
                printf("[Stratum] New difficulty: %.10f\n", diff);
                std::lock_guard<std::mutex> lock(job_mutex_);
                difficulty_to_target(diff, current_job_.target);
                current_difficulty_ = diff;
            }
        }
    } else {
        // Response to our requests (share accept/reject)
        int id = json_get_int(msg, "id");
        if (id > 0) {
            if (msg.find("\"result\"") != std::string::npos) {
                if (msg.find("true") != std::string::npos) {
                    printf("[Stratum] Share ACCEPTED!\n");
                } else if (msg.find("null") != std::string::npos &&
                           msg.find("\"error\"") != std::string::npos) {
                    // Only print first few rejections to avoid spam
                    fprintf(stderr, "[Stratum] Share REJECTED: %s\n", msg.c_str());
                }
            }
        }
    }
}

void StratumClient::parse_notify(const std::string& msg) {
    auto params = json_get_params(msg);
    auto plist = json_parse_params(params);

    // HTN stratum bridge sends:
    //   mining.notify params: [job_id, [uint64_0, uint64_1, uint64_2, uint64_3], timestamp]
    //   OR for big jobs: [job_id, "hex_string_80chars"]
    //
    // The 4 uint64s are the serialized block header hash (32 bytes, 4x uint64 LE)
    // This is NOT the PrevHeader — it's a BLAKE3 hash of the full block header
    // which serves as the pre-image input for HooHash

    if (plist.size() < 2) {
        fprintf(stderr, "[Stratum] Invalid notify params (got %zu)\n", plist.size());
        return;
    }

    MiningJob job;
    job.job_id = plist[0];
    job.valid = true;

    // Parse the header hash — could be array of uint64 or a hex string
    memset(job.prev_header, 0, DOMAIN_HASH_SIZE);

    if (plist.size() >= 3) {
        // Standard format: [job_id, [u64,u64,u64,u64], timestamp]
        // Parse the array of uint64s
        std::string arr = plist[1];
        if (arr.size() > 2 && arr[0] == '[') {
            // Parse uint64 array
            auto elements = json_parse_params(arr);
            if (elements.size() >= 4) {
                for (int i = 0; i < 4; i++) {
                    uint64_t val = 0;
                    try { val = std::stoull(elements[i]); } catch (...) {}
                    // Store as little-endian bytes into prev_header
                    memcpy(&job.prev_header[i * 8], &val, 8);
                }
            }
        } else if (arr.length() == 80) {
            // Big job format: 80-char hex string (40 bytes: 32 header + 8 timestamp)
            decode_hex(arr.substr(0, 64), job.prev_header, DOMAIN_HASH_SIZE);
            // Timestamp is in the last 16 hex chars
            std::string ts_hex = arr.substr(64, 16);
            uint64_t ts_val = 0;
            try { ts_val = std::stoull(ts_hex, nullptr, 16); } catch (...) {}
            job.timestamp = (int64_t)ts_val;
        } else if (arr.length() == 64) {
            // Plain hex header
            decode_hex(arr, job.prev_header, DOMAIN_HASH_SIZE);
        }

        // Parse timestamp (3rd param)
        if (plist.size() > 2) {
            try { job.timestamp = std::stoll(plist[2]); } catch (...) { job.timestamp = 0; }
        }
    } else {
        // Only 2 params — big job format
        std::string data = plist[1];
        if (data.length() == 80) {
            decode_hex(data.substr(0, 64), job.prev_header, DOMAIN_HASH_SIZE);
            std::string ts_hex = data.substr(64, 16);
            uint64_t ts_val = 0;
            try { ts_val = std::stoull(ts_hex, nullptr, 16); } catch (...) {}
            job.timestamp = (int64_t)ts_val;
        }
    }

    // Copy current target (set by mining.set_difficulty)
    {
        std::lock_guard<std::mutex> lock(job_mutex_);
        // Preserve the target from set_difficulty
        memcpy(job.target, current_job_.target, DOMAIN_HASH_SIZE);
        current_job_ = job;
    }

    printf("[Stratum] New job: %s (ts=%lld, header=%s)\n",
           job.job_id.c_str(), (long long)job.timestamp,
           encode_hex(job.prev_header, 8).c_str());

    if (job_callback_) {
        job_callback_(job);
    }
}
