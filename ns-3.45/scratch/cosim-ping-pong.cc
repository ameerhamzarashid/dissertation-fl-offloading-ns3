#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include <arpa/inet.h>  // for htonl, ntohl

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("CosimPingPongApp");

class CosimPingPongApp : public Application {
public:
  CosimPingPongApp() : socket(nullptr), count(0) {}
  virtual ~CosimPingPongApp() {}

  void Setup(Ptr<Socket> sock, Ipv4Address peer, uint16_t port) {
    socket   = sock;
    peerAddr = peer;
    peerPort = port;
  }

private:
  virtual void StartApplication() override {
    socket->SetConnectCallback(
      MakeCallback(&CosimPingPongApp::OnConnectSuccess, this),
      MakeCallback(&CosimPingPongApp::OnConnectFail, this)
    );
    socket->Connect(InetSocketAddress(peerAddr, peerPort));
  }

  virtual void StopApplication() override {
    if (socket) socket->Close();
  }

  void OnConnectSuccess(Ptr<Socket> sock) {
    NS_LOG_INFO("TCP connect succeeded");
    sock->SetRecvCallback(MakeCallback(&CosimPingPongApp::HandleRead, this));
    SendOne();
  }

  void OnConnectFail(Ptr<Socket> sock) {
    NS_LOG_ERROR("TCP connect failed");
  }

  void SendOne() {
    ++count;
    NS_LOG_INFO("C++ sending: " << count);
    uint32_t netCount = htonl(count);
    socket->Send(Create<Packet>((uint8_t*)&netCount, sizeof(netCount)));
  }

  void HandleRead(Ptr<Socket> s) {
    Ptr<Packet> p = s->Recv();
    uint32_t netCount;
    p->CopyData((uint8_t*)&netCount, sizeof(netCount));
    uint32_t recv = ntohl(netCount);
    NS_LOG_INFO("C++ received: " << recv);
    Simulator::Schedule(Seconds(1.0), &CosimPingPongApp::SendOne, this);
  }

  Ptr<Socket>   socket;
  Ipv4Address   peerAddr;
  uint16_t      peerPort;
  uint32_t      count;
};

int main(int argc, char *argv[]) {
  CommandLine cmd;
  std::string host = "127.0.0.1";
  uint16_t port    = 12345;
  cmd.AddValue("host", "Server IP",   host);
  cmd.AddValue("port", "Server port", port);
  cmd.Parse(argc, argv);

  Ptr<Node> node = CreateObject<Node>();
  InternetStackHelper().Install(node);

  Ptr<Socket> sock = Socket::CreateSocket(node, TcpSocketFactory::GetTypeId());

  LogComponentEnable("CosimPingPongApp", LOG_LEVEL_INFO);

  Ptr<CosimPingPongApp> app = CreateObject<CosimPingPongApp>();
  app->Setup(sock, Ipv4Address(host.c_str()), port);
  node->AddApplication(app);
  app->SetStartTime(Seconds(0.1));
  app->SetStopTime(Seconds(20.0));

  Simulator::Run();
  Simulator::Destroy();
  return 0;
}