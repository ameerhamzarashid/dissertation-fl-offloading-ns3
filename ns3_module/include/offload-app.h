#pragma once
/**
 * offload-app.h
 * Application that receives Task objects and either executes locally or offloads.
 * This is a simplified scaffold: by default it offloads; later the action comes from Python RL agent.
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "bridge-client.h"
#include "energy-tracker.h"
#include "logging-helper.h"
#include "task-generator.h"
#include <string>

namespace ns3 {

class OffloadApp : public Application {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("ns3::OffloadApp")
      .SetParent<Application>()
      .SetGroupName("Applications");
    return tid;
  }

  OffloadApp() = default;
  ~OffloadApp() override = default;

  void SetBridge(Ptr<BridgeClient> b) { m_bridge = b; }
  void SetEnergy(Ptr<EnergyTracker> e) { m_energy = e; }
  void SetLogger(Ptr<LoggingHelper> l) { m_logger = l; }
  void SetComputeFreqHz(double f) { m_cpuHz = f; }

  void OnTask(const Task& t) {
    // For now, simple policy: offload if size <= threshold else local
    bool offload = (t.sizeBytes <= m_offloadSizeThreshold);
    if (offload) {
      DoOffload(t);
    } else {
      DoLocal(t);
    }
  }

protected:
  void StartApplication() override {
    if (m_bridge) m_bridge->Connect();
  }
  void StopApplication() override {}

private:
  void DoLocal(const Task& t) {
    // Compute time (s) = cycles / freq
    double exec = t.cycles / m_cpuHz;
    // CPU energy
    if (m_energy) m_energy->AddCpuCycles(t.cycles);
    // Finish after exec time
    Simulator::Schedule(Seconds(exec), &OffloadApp::OnComplete, this, t, false, exec, 0.0);
  }

  void DoOffload(const Task& t) {
    // Send bytes upstream (simplified): serialize a small JSON string to Python
    std::string json = std::string("{\"type\":\"state\",\"ue\":") + std::to_string(GetNode()->GetId())
                     + ",\"size\":" + std::to_string(t.sizeBytes)
                     + ",\"cycles\":" + std::to_string((uint64_t)t.cycles) + "}";

    if (m_bridge) m_bridge->SendJson(json);

    // Radio time estimate (rough): assume uplink rate R = 10 Mbps
    double ratebps = 10e6;
    double txTime = (double)t.sizeBytes * 8.0 / ratebps;
    if (m_energy) m_energy->AddRadioTx(txTime);

    // Assume edge compute 10x faster
    double edgeHz = m_cpuHz * 10.0;
    double edgeExec = t.cycles / edgeHz;

    double rtt = 0.02; // 20 ms RTT placeholder
    double total = txTime + rtt + edgeExec;
    Simulator::Schedule(Seconds(total), &OffloadApp::OnComplete, this, t, true, total, txTime);
  }

  void OnComplete(const Task& t, bool offloaded, double latency, double txTime) {
    m_txBytes += offloaded ? t.sizeBytes : 0;
    if (m_logger) {
      m_logger->LogBytes(GetNode()->GetId(), m_txBytes, m_rxBytes);
      double cpuJ = m_energy ? m_energy->GetCpuEnergyJ() : 0.0;
      double radioJ = m_energy ? m_energy->GetRadioEnergyJ() : 0.0;
      m_logger->LogEnergy(GetNode()->GetId(), cpuJ, radioJ);
    }
    // TODO: Pull action from Python and adapt threshold dynamically.
    // Placeholder: adjust threshold slightly to simulate learning
    if (offloaded) m_offloadSizeThreshold = std::max<uint64_t>(512*1024, m_offloadSizeThreshold - 4096);
    else           m_offloadSizeThreshold = std::min<uint64_t>(64*1024*1024, m_offloadSizeThreshold + 4096);
  }

  Ptr<BridgeClient> m_bridge;
  Ptr<EnergyTracker> m_energy;
  Ptr<LoggingHelper> m_logger;
  double m_cpuHz{1.5e9};
  uint64_t m_txBytes{0}, m_rxBytes{0};
  uint64_t m_offloadSizeThreshold{10 * 1024 * 1024}; // 10 MB default
};

} // namespace ns3
