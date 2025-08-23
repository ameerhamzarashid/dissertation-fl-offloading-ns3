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
  ~BridgeClient() override = default;

  void SetNode(Ptr<Node> node) { m_node = node; }
  void Configure(const std::string& host, uint16_t port) { m_host = host; m_port = port; }

  void Connect() {
    if (m_socket) return;
    m_socket = Socket::CreateSocket(m_node, TcpSocketFactory::GetTypeId());
    m_socket->SetConnectCallback(MakeCallback(&BridgeClient::OnConnect, this),
                                 MakeCallback(&BridgeClient::OnConnectFail, this));
    m_socket->SetRecvCallback(MakeCallback(&BridgeClient::OnRecv, this));

    InetSocketAddress addr(Ipv4Address(m_host.c_str()), m_port);
    m_socket->Connect(addr);
  }

  void SendJson(const std::string& json) {
    uint32_t len = static_cast<uint32_t>(json.size());
    uint32_t nlen = HostToNet32(len);
    std::string pkt;
    pkt.resize(4);
    std::memcpy(pkt.data(), &nlen, 4);
    pkt += json;
    if (!m_socket) Connect();
    if (m_socket) {
      m_socket->Send(reinterpret_cast<const uint8_t*>(pkt.data()), pkt.size(), 0);
    }
  }

  bool TryPop(std::string& out) {
    if (m_incoming.empty()) return false;
    out = std::move(m_incoming.front());
    m_incoming.pop();
    return true;
  }

private:
  void OnConnect(Ptr<Socket> s) { /* connected */ }
  void OnConnectFail(Ptr<Socket> s) {
    Simulator::Schedule(Seconds(2.0), &BridgeClient::Connect, this);
  }
  void OnRecv(Ptr<Socket> s) {
    Ptr<Packet> pkt;
    while ((pkt = s->Recv())) {
      uint32_t size = pkt->GetSize();
      std::string data; data.resize(size);
      pkt->CopyData(reinterpret_cast<uint8_t*>(data.data()), size);
      m_buffer += data;

      while (m_buffer.size() >= 4) {
        uint32_t nlen; std::memcpy(&nlen, m_buffer.data(), 4);
        uint32_t len = NetToHost32(nlen);
        if (m_buffer.size() < 4 + len) break;
        std::string payload = m_buffer.substr(4, len);
        m_incoming.push(payload);
        m_buffer.erase(0, 4 + len);
      }
    }
  }

  Ptr<Node> m_node;
  Ptr<Socket> m_socket;
  std::string m_host{"127.0.0.1"};
  uint16_t m_port{50051};
  std::string m_buffer;
  std::queue<std::string> m_incoming;
};

} // namespace ns3
