#pragma once
/**
 * bridge-client.h
 * Simple TCP client for Python co-sim. Length-prefixed JSON strings (portable).
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include <string>
#include <queue>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

// POSIX sockets (available in WSL/Linux builds)
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

namespace ns3 {

static inline uint32_t HostToNet32(uint32_t x) {
  return ((x & 0x000000FFu) << 24) | ((x & 0x0000FF00u) << 8) |
         ((x & 0x00FF0000u) >> 8)  | ((x & 0xFF000000u) >> 24);
}
static inline uint32_t NetToHost32(uint32_t x) { return HostToNet32(x); }

class BridgeClient : public Object {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("ns3::BridgeClient")
      .SetParent<Object>()
      .SetGroupName("Applications");
    return tid;
  }

  BridgeClient() = default;
  ~BridgeClient() override {
    Stop();
  }

  void SetNode(Ptr<Node> node) { m_node = node; }
  void Configure(const std::string& host, uint16_t port) { m_host = host; m_port = port; }

  void Connect() {
    StartThreadIfNeeded();
  }

  void SendJson(const std::string& json) {
    if (!m_threadStarted.load()) StartThreadIfNeeded();
    // If not yet connected, wait briefly to establish the connection so the first
    // request isn't lost; fallback to drop if still not connected.
    if (!m_connected.load()) {
      for (int i = 0; i < 50 && !m_connected.load(); ++i) { SleepTiny(); }
    }
    uint32_t len = static_cast<uint32_t>(json.size());
    uint32_t nlen = HostToNet32(len);
    std::string pkt;
    pkt.resize(4);
    std::memcpy(pkt.data(), &nlen, 4);
    pkt += json;
    if (m_fd < 0) return; // still not connected
    const char* buf = pkt.data();
    size_t total = 0, toSend = pkt.size();
    while (total < toSend && m_fd >= 0) {
      ssize_t n = ::send(m_fd, buf + total, toSend - total, 0);
      if (n < 0) {
        if (errno == EINTR) continue;
        // error: will reconnect
        break;
      }
      total += static_cast<size_t>(n);
    }
  }

  bool TryPop(std::string& out) {
    std::lock_guard<std::mutex> lock(m_mu);
    if (m_incoming.empty()) return false;
    out = std::move(m_incoming.front());
    m_incoming.pop();
    return true;
  }

private:
  void StartThreadIfNeeded() {
    if (m_threadStarted.exchange(true)) return;
    m_stop.store(false);
    m_thread = std::thread([this]() { this->ThreadMain(); });
  }

  void Stop() {
    m_stop.store(true);
    if (m_fd >= 0) { ::shutdown(m_fd, SHUT_RDWR); ::close(m_fd); m_fd = -1; }
    if (m_thread.joinable()) m_thread.join();
    m_connected.store(false);
    m_threadStarted.store(false);
  }

  void ThreadMain() {
    while (!m_stop.load()) {
      if (m_fd < 0) {
        // Connect
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) { SleepShort(); continue; }
        struct sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(m_port);
        if (::inet_pton(AF_INET, m_host.c_str(), &sa.sin_addr) != 1) {
          ::close(fd); SleepShort(); continue;
        }
        if (::connect(fd, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) < 0) {
          std::cout << "[BridgeClient] connect failed to " << m_host << ":" << m_port << ", retrying in 2s\n";
          ::close(fd);
          SleepRetry();
          continue;
        }
        m_fd = fd;
        m_connected.store(true);
        std::cout << "[BridgeClient] connected to " << m_host << ":" << m_port << "\n";
        m_buffer.clear();
      }

      // Read length-prefixed frames
      uint32_t nlen = 0;
      if (!ReadFull(reinterpret_cast<char*>(&nlen), 4)) { HandleDisconnect(); continue; }
      uint32_t len = NetToHost32(nlen);
      if (len == 0 || len > (32u * 1024u * 1024u)) { // sanity
        HandleDisconnect(); continue;
      }
      std::string payload; payload.resize(len);
      if (!ReadFull(payload.data(), len)) { HandleDisconnect(); continue; }
      {
        std::lock_guard<std::mutex> lock(m_mu);
        m_incoming.push(std::move(payload));
      }
    }
  }

  bool ReadFull(char* buf, size_t len) {
    size_t got = 0;
    while (got < len && !m_stop.load()) {
      ssize_t n = ::recv(m_fd, buf + got, len - got, 0);
      if (n == 0) return false; // closed
      if (n < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) { SleepTiny(); continue; }
        return false;
      }
      got += static_cast<size_t>(n);
    }
    return got == len;
  }

  void HandleDisconnect() {
    if (m_fd >= 0) { ::close(m_fd); m_fd = -1; }
    if (m_connected.exchange(false)) {
      std::cout << "[BridgeClient] disconnected, will retry in 2s\n";
    }
    SleepRetry();
  }

  static void SleepTiny() { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
  static void SleepShort() { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
  static void SleepRetry() { std::this_thread::sleep_for(std::chrono::seconds(2)); }

  Ptr<Node> m_node;
  std::string m_host{"127.0.0.1"};
  uint16_t m_port{50051};
  std::string m_buffer;
  std::queue<std::string> m_incoming;
  std::mutex m_mu;
  std::thread m_thread;
  std::atomic<bool> m_threadStarted{false};
  std::atomic<bool> m_connected{false};
  std::atomic<bool> m_stop{false};
  int m_fd{-1};
};

} // namespace ns3
