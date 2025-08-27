#pragma once
#include "ns3/core-module.h"
#include "ns3/log.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include <string>
#include <cstring>
#include <iostream>

#include "task-generator.h"
#include "bridge-client.h"
#include "energy-tracker.h"
#include "logging-helper.h"

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
  void SetUlRateBps(double bps) { m_ulRateBps = bps; }

  // Task arrival from generator
  void OnTask(const Task& t) {
    DecideAndProcess(t);
  }

  // Completion hook (latency logging + bytes)
  void OnComplete(const Task& t, bool offloaded, double latency, double txTime) {
    if (m_logger) {
      m_logger->LogBytes(GetNode()->GetId(), m_txBytes, m_rxBytes);
      m_logger->LogLatency(t.id, GetNode()->GetId(), latency * 1000.0, offloaded);
      if (m_energy) {
        m_logger->LogEnergy(GetNode()->GetId(), m_energy->GetCpuEnergyJ(), m_energy->GetRadioEnergyJ());
      }
    }
  }

private:
  struct ActionResp { int action{0}; uint64_t updateBytes{0}; };
  bool ParseActionPayload(const std::string& payload, ActionResp& out) {
    auto findInt = [&](const std::string& key)->long long {
      size_t pos = payload.find(key); if (pos == std::string::npos) return -1;
      pos = payload.find(':', pos);   if (pos == std::string::npos) return -1;
      ++pos; while (pos < payload.size() && (payload[pos]==' '||payload[pos]=='\"')) ++pos;
      long long val=0; bool any=false;
      while (pos < payload.size() && isdigit(payload[pos])) { any=true; val = val*10 + (payload[pos]-'0'); ++pos; }
      return any ? val : -1;
    };
    long long a = findInt("\"action\"");
    long long u = findInt("\"update_bytes\"");
    if (a >= 0) out.action = (int)a;
    if (u >= 0) out.updateBytes = (uint64_t)u;
    return a >= 0;
  }

  void DecideAndProcess(const Task& t) {
    std::string json = std::string("{\"type\":\"state\",\"ue\":") + std::to_string(GetNode()->GetId())
                     + ",\"size\":" + std::to_string(t.sizeBytes)
                     + ",\"cycles\":" + std::to_string((uint64_t)t.cycles) + "}";
    if (m_bridge) m_bridge->SendJson(json);
  m_waitTask = t; m_waiting = true; m_waitDeadline = Simulator::Now() + MilliSeconds(1000);
    Simulator::Schedule(MilliSeconds(5), &OffloadApp::PollAction, this);
  }

  void PollAction() {
    if (!m_waiting) return;
    std::string payload;
    if (m_bridge && m_bridge->TryPop(payload)) {
      std::cout << "[OffloadApp] Bridge reply: " << payload << std::endl;
      ActionResp ar; bool ok = ParseActionPayload(payload, ar);
      bool offload = ok ? (ar.action == 1) : (m_waitTask.sizeBytes <= m_offloadSizeThreshold);
      if (offload) DoOffload(m_waitTask); else DoLocal(m_waitTask);
      if (ar.updateBytes > 0) {
        double txu = (double)ar.updateBytes * 8.0 / m_ulRateBps;
        if (m_energy) m_energy->AddRadioTx(txu);
        m_txBytes += ar.updateBytes;
      }
      m_waiting = false; return;
    }
    if (Simulator::Now() < m_waitDeadline) {
      Simulator::Schedule(MilliSeconds(5), &OffloadApp::PollAction, this);
    } else {
      // No reply from bridge: decide + apply environment-driven update_bytes fallback.
      bool offload = (m_waitTask.sizeBytes <= m_offloadSizeThreshold);
      if (offload) DoOffload(m_waitTask); else DoLocal(m_waitTask);

      // Fallback: emulate FL update bytes based on BRIDGE_VARIANT/NUM_PARAMS/K_PERCENT
      const char* var = std::getenv("BRIDGE_VARIANT");
      std::string variant = var ? std::string(var) : std::string("baseline");
      for (auto &c : variant) c = std::tolower(c);
      uint64_t numParams = 1000000; // default 1M
      if (const char* np = std::getenv("NUM_PARAMS")) { try { numParams = std::stoull(np); } catch (...) {} }
      double kPercent = 10.0;
      if (const char* kp = std::getenv("K_PERCENT")) { try { kPercent = std::stod(kp); } catch (...) {} }
      uint64_t upd = 0;
      if (variant == "sfea") {
        uint64_t kept = std::max<uint64_t>(1, (uint64_t)std::llround((double)numParams * (kPercent / 100.0)));
        upd = kept * 8ULL; // 4B value + 4B index per kept param
      } else {
        upd = numParams * 4ULL; // dense 4B values
      }
      if (upd > 0) {
        double txu = (double)upd * 8.0 / m_ulRateBps;
        if (m_energy) m_energy->AddRadioTx(txu);
        m_txBytes += upd;
        std::cout << "[OffloadApp] No bridge reply; applied env fallback update_bytes=" << upd
                  << " variant=" << variant << "\n";
      }
      m_waiting = false;
    }
  }

  void DoLocal(const Task& t) {
    double computeSec = (double)t.cycles / std::max(1.0, m_cpuHz);
    if (m_energy) { m_energy->AddCpuCycles((double)t.cycles); }
    Simulator::Schedule(Seconds(computeSec), &OffloadApp::OnComplete, this, t, false, computeSec, 0.0);
  }

  void DoOffload(const Task& t) {
    double txSec = (double)t.sizeBytes * 8.0 / std::max(1.0, m_ulRateBps);
    if (m_energy) m_energy->AddRadioTx(txSec);
    m_txBytes += t.sizeBytes;
    double edgeSec = (double)t.cycles / (10.0 * std::max(1.0, m_cpuHz)); // edge ~10× faster
    Simulator::Schedule(Seconds(txSec + edgeSec), &OffloadApp::OnComplete, this, t, true, txSec + edgeSec, txSec);
  }

private:
  Ptr<BridgeClient>  m_bridge;
  Ptr<EnergyTracker> m_energy;
  Ptr<LoggingHelper> m_logger;

  double   m_cpuHz{1.5e9};
  double   m_ulRateBps{10e6};
  uint64_t m_txBytes{0}, m_rxBytes{0};

  uint64_t m_offloadSizeThreshold{20 * 1024 * 1024};
  Task     m_waitTask{}; bool m_waiting{false}; Time m_waitDeadline;
};

} // namespace ns3
