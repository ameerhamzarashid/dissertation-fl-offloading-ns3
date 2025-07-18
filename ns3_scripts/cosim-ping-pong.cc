// cosim-ping-pong.cc

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

// --- CRITICAL CHANGE: Platform-specific headers for socket communication ---
#ifdef _WIN32
    // For Windows Sockets (Winsock2)
    #include <winsock2.h>
    #include <ws2tcpip.h>
    // Link with Ws2_32.lib
    #pragma comment(lib, "Ws2_32.lib")
#else
    // For Linux/macOS (POSIX sockets)
    #include <arpa/inet.h>
    #include <sys/socket.h>
    #include <unistd.h> // For close()
    #define SOCKET int
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR -1
    #define closesocket close
#endif

#include <cstring>
#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("CoSimPingPongApp");

class CoSimApp : public Application
{
public:
    CoSimApp();
    virtual ~CoSimApp();
    void Setup(Address address);

private:
    virtual void StartApplication(void);
    virtual void StopApplication(void);

    void ConnectionSucceeded(Ptr<Socket> socket);
    void ConnectionFailed(Ptr<Socket> socket);
    void HandleRead(Ptr<Socket> socket);
    void SendData(void);

    Ptr<Socket> m_socket;
    Address m_peerAddress;
    uint32_t m_valueToSend;
};

CoSimApp::CoSimApp()
    : m_socket(nullptr),
      m_peerAddress(),
      m_valueToSend(10)
{
#ifdef _WIN32
    // --- CRITICAL: Initialize Winsock on Windows ---
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        NS_LOG_ERROR("WSAStartup failed: " << result);
    }
#endif
}

CoSimApp::~CoSimApp()
{
    m_socket = nullptr;
#ifdef _WIN32
    // --- CRITICAL: Clean up Winsock on Windows ---
    WSACleanup();
#endif
}

void CoSimApp::Setup(Address address)
{
    m_peerAddress = address;
}

void CoSimApp::StartApplication(void)
{
    NS_LOG_INFO("[ns-3] Starting CoSimApp");
    m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());

    m_socket->SetConnectCallback(
        MakeCallback(&CoSimApp::ConnectionSucceeded, this),
        MakeCallback(&CoSimApp::ConnectionFailed, this));
    
    m_socket->SetRecvCallback(MakeCallback(&CoSimApp::HandleRead, this));
    
    m_socket->Connect(m_peerAddress);
    NS_LOG_INFO("[ns-3] Attempting to connect to Python server...");
}

void CoSimApp::StopApplication(void)
{
    NS_LOG_INFO("[ns-3] Stopping CoSimApp");
    if (m_socket)
    {
        m_socket->Close();
    }
}

void CoSimApp::ConnectionSucceeded(Ptr<Socket> socket)
{
    NS_LOG_INFO("[ns-3] --- Connection to Python server SUCCEEDED! ---");
    Simulator::Schedule(Seconds(1.0), &CoSimApp::SendData, this);
}

void CoSimApp::ConnectionFailed(Ptr<Socket> socket)
{
    NS_LOG_ERROR("[ns-3] --- Connection to Python server FAILED! ---");
}

void CoSimApp::SendData(void)
{
    NS_LOG_INFO("[ns-3] Sending value to Python: " << m_valueToSend);
    uint32_t val_net = htonl(m_valueToSend);
    Ptr<Packet> packet = Create<Packet>((uint8_t*)&val_net, sizeof(val_net));
    m_socket->Send(packet);
}

void CoSimApp::HandleRead(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    while ((packet = socket->Recv()))
    {
        if (packet->GetSize() == 0)
        {
            break;
        }
        uint32_t received_val_net;
        packet->CopyData((uint8_t*)&received_val_net, sizeof(received_val_net));
        m_valueToSend = ntohl(received_val_net);
        NS_LOG_INFO("[ns-3] Received response from Python: " << m_valueToSend);
        
        Simulator::Schedule(Seconds(2.0), &CoSimApp::SendData, this);
    }
}

int main(int argc, char* argv[])
{
    LogComponentEnable("CoSimPingPongApp", LOG_LEVEL_INFO);

    NodeContainer nodes;
    nodes.Create(1);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ptr<CoSimApp> app = CreateObject<CoSimApp>();
    Address remoteAddress(InetSocketAddress(Ipv4Address("127.0.0.1"), 12345));
    app->Setup(remoteAddress);
    nodes.Get(0)->AddApplication(app);
    
    app->SetStartTime(Seconds(0.5));
    app->SetStopTime(Seconds(20.0));

    NS_LOG_INFO("[ns-3] Running simulation...");
    Simulator::Run();
    Simulator::Destroy();
    NS_LOG_INFO("[ns-3] Simulation finished.");

    return 0;
}