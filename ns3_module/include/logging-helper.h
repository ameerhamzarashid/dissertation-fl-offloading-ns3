#pragma once
#include "ns3/core-module.h"
#include "ns3/system-path.h"
#include <fstream>
#include <string>
#include <cstdint>

namespace ns3 {

class LoggingHelper : public Object {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("ns3::LoggingHelper")
      .SetParent<Object>()
      .SetGroupName("Applications");
    return tid;
  }

  LoggingHelper() = default;
  ~LoggingHelper() override { Close(); }

  void Open(const std::string& dir, const std::string& bytesCsv, const std::string& energyCsv) {
    SystemPath::MakeDirectories(dir);
    m_bytes.open((dir + "/" + bytesCsv).c_str(), std::ios::out);
    m_energy.open((dir + "/" + energyCsv).c_str(), std::ios::out);
    m_latency.open((dir + "/task_latency.csv").c_str(), std::ios::out);
    if (m_bytes && m_energy && m_latency) {
      m_bytes   << "time_s,ue_id,tx_bytes,rx_bytes\n";
      m_energy  << "time_s,ue_id,cpu_j,radio_j,total_j\n";
      m_latency << "time_s,task_id,ue_id,latency_ms,offloaded\n";
    }
  }

  void LogBytes(uint32_t ueId, uint64_t tx, uint64_t rx) {
    if (m_bytes) {
      m_bytes << Simulator::Now().GetSeconds() << "," << ueId << "," << tx << "," << rx << "\n";
    }
  }

  void LogEnergy(uint32_t ueId, double cpuJ, double radioJ) {
    if (m_energy) {
      m_energy << Simulator::Now().GetSeconds() << "," << ueId << "," << cpuJ << "," << radioJ
               << "," << (cpuJ + radioJ) << "\n";
    }
  }

  void LogLatency(uint64_t taskId, uint32_t ueId, double latencyMs, bool offloaded) {
    if (m_latency) {
      m_latency << Simulator::Now().GetSeconds() << "," << taskId << "," << ueId
                << "," << latencyMs << "," << (offloaded ? 1 : 0) << "\n";
    }
  }

  void Close() {
    if (m_bytes)   { m_bytes.flush();   m_bytes.close(); }
    if (m_energy)  { m_energy.flush();  m_energy.close(); }
    if (m_latency) { m_latency.flush(); m_latency.close(); }
  }

private:
  std::ofstream m_bytes;
  std::ofstream m_energy;
  std::ofstream m_latency;
};

} // namespace ns3
