#ifndef STRATUM_H
#define STRATUM_H

#include "utils.h"
#include <string>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define INVALID_SOCK INVALID_SOCKET
#define CLOSE_SOCKET closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCK (-1)
#define CLOSE_SOCKET close
#endif

class StratumClient {
public:
    using JobCallback = std::function<void(const MiningJob&)>;

    StratumClient();
    ~StratumClient();

    // Connect to stratum server
    bool connect(const std::string& host, int port);

    // Disconnect
    void disconnect();

    // Subscribe and authorize
    bool subscribe();
    bool authorize(const std::string& user, const std::string& password,
                   const std::string& worker);

    // Submit a share
    bool submit_share(const std::string& job_id, uint64_t nonce,
                      const uint8_t* hash);

    // Set callback for new jobs
    void set_job_callback(JobCallback cb);

    // Start/stop receiving thread
    void start_receiving();
    void stop_receiving();

    // Check if connected
    bool is_connected() const { return connected_; }

    // Get current job
    MiningJob get_current_job();

private:
    socket_t sock_;
    bool connected_;
    std::atomic<bool> running_;
    std::thread recv_thread_;
    std::mutex job_mutex_;
    MiningJob current_job_;
    JobCallback job_callback_;
    int msg_id_;
    std::string extra_nonce_;
    std::string worker_name_;
    double current_difficulty_;

    bool send_json(const std::string& json);
    std::string recv_line();
    void receive_loop();
    void handle_message(const std::string& msg);
    void parse_notify(const std::string& msg);
};

#endif // STRATUM_H
