/*
 * Baseline NS-3 Federated Learning Edge Client
 * Simple implementation for edge computing offloading decisions
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FederatedClient");

class FederatedClient : public Application {
public:
    FederatedClient() : m_socket(-1), m_running(false) {}
    
    void Setup(Ipv4Address serverIp, uint16_t serverPort) {
        m_serverIp = serverIp;
        m_serverPort = serverPort;
    }

private:
    virtual void StartApplication() override {
        m_running = true;
        NS_LOG_INFO("Starting Federated Client");
        
        // Create socket and connect to FL server
        m_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (m_socket < 0) {
            NS_FATAL_ERROR("Failed to create socket");
        }
        
        struct sockaddr_in serverAddr = {};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(m_serverPort);
        serverAddr.sin_addr.s_addr = m_serverIp.Get();
        
        if (connect(m_socket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            NS_FATAL_ERROR("Failed to connect to FL server: " << strerror(errno));
        }
        
        NS_LOG_INFO("Connected to FL server at " << m_serverIp << ":" << m_serverPort);
        
        // Start sending federated learning requests
        Simulator::Schedule(Seconds(1.0), &FederatedClient::SendFLRequest, this);
    }
    
    virtual void StopApplication() override {
        m_running = false;
        if (m_socket >= 0) {
            close(m_socket);
            m_socket = -1;
        }
        NS_LOG_INFO("Federated Client stopped");
    }
    
    void SendFLRequest() {
        if (!m_running) return;
        
        // Simulate edge device state (CPU utilization, network load, battery, etc.)
        double currentTime = Simulator::Now().GetSeconds();
        float state = static_cast<float>(fmod(currentTime * 10.0, 100.0)); // 0-100 range
        
        // Send state as 4-byte float
        uint32_t networkState = htonl(*reinterpret_cast<uint32_t*>(&state));
        ssize_t sent = send(m_socket, &networkState, sizeof(networkState), 0);
        
        if (sent != sizeof(networkState)) {
            NS_LOG_ERROR("Failed to send state");
            return;
        }
        
        NS_LOG_INFO("Sent state: " << state);
        
        // Receive offloading decision
        uint32_t networkAction;
        ssize_t received = recv(m_socket, &networkAction, sizeof(networkAction), 0);
        
        if (received == sizeof(networkAction)) {
            uint32_t hostAction = ntohl(networkAction);
            int action = *reinterpret_cast<int*>(&hostAction);
            
            NS_LOG_INFO("Received offloading decision: " << action << 
                       " (0=local, 1=edge, 2=cloud)");
            
            // Apply the offloading decision
            ApplyOffloadingDecision(action, state);
        }
        
        // Schedule next FL request
        Simulator::Schedule(Seconds(2.0), &FederatedClient::SendFLRequest, this);
    }
    
    void ApplyOffloadingDecision(int action, float state) {
        // Simple simulation of offloading decision impact
        switch(action) {
            case 0: // Local processing
                NS_LOG_INFO("Processing locally - Low latency, high energy");
                break;
            case 1: // Edge processing  
                NS_LOG_INFO("Offloading to edge - Medium latency, medium energy");
                break;
            case 2: // Cloud processing
                NS_LOG_INFO("Offloading to cloud - High latency, low energy");
                break;
            default:
                NS_LOG_WARN("Unknown action: " << action);
        }
    }
    
    int m_socket;
    bool m_running;
    Ipv4Address m_serverIp;
    uint16_t m_serverPort;
};

int main(int argc, char *argv[]) {
    // Enable logging
    LogComponentEnable("FederatedClient", LOG_LEVEL_INFO);
    
    // Parse command line arguments
    CommandLine cmd;
    std::string serverHost = "127.0.0.1";
    uint16_t serverPort = 12345;
    uint32_t numNodes = 3;
    
    cmd.AddValue("host", "FL server IP address", serverHost);
    cmd.AddValue("port", "FL server port", serverPort);
    cmd.AddValue("nodes", "Number of edge nodes", numNodes);
    cmd.Parse(argc, argv);
    
    // Create nodes
    NodeContainer nodes;
    nodes.Create(numNodes);
    
    // Install Internet stack
    InternetStackHelper internet;
    internet.Install(nodes);
    
    // Create applications
    Ipv4Address serverIp = Ipv4Address(serverHost.c_str());
    
    for (uint32_t i = 0; i < numNodes; ++i) {
        Ptr<FederatedClient> app = CreateObject<FederatedClient>();
        app->Setup(serverIp, serverPort);
        nodes.Get(i)->AddApplication(app);
        app->SetStartTime(Seconds(1.0 + i * 0.5)); // Stagger start times
        app->SetStopTime(Seconds(30.0));
    }
    
    NS_LOG_INFO("Starting simulation with " << numNodes << " edge nodes");
    NS_LOG_INFO("FL Server: " << serverHost << ":" << serverPort);
    
    Simulator::Stop(Seconds(35.0));
    Simulator::Run();
    Simulator::Destroy();
    
    return 0;
}
