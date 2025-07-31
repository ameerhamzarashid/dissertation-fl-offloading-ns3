#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("FederatedClientApp");

class FederatedClientApp : public Application {
public:
  FederatedClientApp() : sockFd(-1) {}
  void Setup(Ipv4Address serverIp, uint16_t serverPort) {
    peerIp   = serverIp;
    peerPort = serverPort;
  }

private:
  virtual void StartApplication() override {
    Simulator::Schedule(Seconds(1.0),
        &FederatedClientApp::SendState, this);
  }
  virtual void StopApplication() override {
    if (sockFd >= 0) {
      ::close(sockFd);
      sockFd = -1;
    }
  }

  void SendState() {
    // 1) (Re)create & connect socket
    if (sockFd < 0) {
      sockFd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (sockFd < 0) {
        NS_FATAL_ERROR("socket() failed: " << strerror(errno));
      }
      struct sockaddr_in addr = {};
      addr.sin_family      = AF_INET;
      addr.sin_port        = htons(peerPort);
      addr.sin_addr.s_addr = peerIp.Get();
      if (::connect(sockFd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        NS_FATAL_ERROR("connect() failed: " << strerror(errno));
      }
      NS_LOG_INFO("Connected to " << peerIp << ":" << peerPort);
    }

    // 2) Pack & send 40‐byte state
    uint32_t state[10] = {0}; 
    uint32_t size     = sizeof(state);
    uint32_t netSize  = htonl(size);
    ::send(sockFd, &netSize, sizeof(netSize), 0);
    ::send(sockFd, state, size, 0);
    NS_LOG_INFO("Sent " << size << " bytes of state");

    // 3) Receive 4‐byte action
    uint32_t netAction = 0;
    ssize_t r = ::recv(sockFd, &netAction, sizeof(netAction), 0);
    if (r != sizeof(netAction)) {
      NS_FATAL_ERROR("recv() failed");
    }
    uint32_t action = ntohl(netAction);
    NS_LOG_INFO("Received action: " << action);

    // 4) Close socket & reschedule
    ::close(sockFd);
    sockFd = -1;
    Simulator::Schedule(Seconds(1.0),
        &FederatedClientApp::SendState, this);
  }

  int sockFd;
  Ipv4Address peerIp;
  uint16_t    peerPort;
};

int main(int argc, char *argv[]) {
  CommandLine cmd;
  std::string host = "127.0.0.1";
  uint16_t    port = 12345;
  uint32_t    N    = 10;
  cmd.AddValue("host","FL server IP",      host);
  cmd.AddValue("port","FL server port",    port);
  cmd.AddValue("N",   "Number of clients", N);
  cmd.Parse(argc, argv);

  GlobalValue::Bind("SimulatorImplementationType",
      StringValue("ns3::RealtimeSimulatorImpl"));

  NodeContainer clients;
  clients.Create(N);
  InternetStackHelper().Install(clients);

  for (uint32_t i = 0; i < N; ++i) {
    Ptr<FederatedClientApp> app = CreateObject<FederatedClientApp>();
    app->Setup(Ipv4Address(host.c_str()), port);
    clients.Get(i)->AddApplication(app);
    app->SetStartTime(Seconds(0.5));
    app->SetStopTime(Seconds(100.0));
  }

  LogComponentEnable("FederatedClientApp", LOG_LEVEL_INFO);
  Simulator::Run();
  Simulator::Destroy();
  return 0;
}
