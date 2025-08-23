#pragma once
/**
 * energy-tracker.h
 * Lightweight energy accounting helper.
 * - CPU energy: cycles * e_cycle (J) (approx)
 * - Radio energy: P_tx * t_tx + P_rx * t_rx
 * Later, hook to ns-3 energy models; this keeps a simple first-order estimate.
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"

namespace ns3 {

class EnergyTracker : public Object {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("ns3::EnergyTracker")
      .SetParent<Object>()
      .SetGroupName("Applications");
    return tid;
  }

  EnergyTracker() = default;

  void SetNode(Ptr<Node> n) { m_node = n; }
  void SetCpuParams(double freqHz, double joulePerCycle) {
    m_cpuFreqHz = freqHz; m_eCycle = joulePerCycle;
  }
  void SetRadioParams(double pTxW, double pRxW) { m_pTx = pTxW; m_pRx = pRxW; }

  void AddCpuCycles(double cycles) {
    m_cpuCycles += cycles;
    m_eCpuJ += cycles * m_eCycle;
  }
  void AddRadioTx(double seconds) {
    m_tTx += seconds;
    m_eRadioJ += seconds * m_pTx;
  }
  void AddRadioRx(double seconds) {
    m_tRx += seconds;
    m_eRadioJ += seconds * m_pRx;
  }

  double GetCpuEnergyJ() const { return m_eCpuJ; }
  double GetRadioEnergyJ() const { return m_eRadioJ; }
  double GetTotalEnergyJ() const { return m_eCpuJ + m_eRadioJ; }

private:
  Ptr<Node> m_node;
  // CPU
  double m_cpuFreqHz{1.5e9};
  double m_eCycle{1.0e-9};
  double m_cpuCycles{0.0};
  double m_eCpuJ{0.0};
  // Radio
  double m_pTx{0.2}; // Watts
  double m_pRx{0.1}; // Watts
  double m_tTx{0.0}, m_tRx{0.0};
  double m_eRadioJ{0.0};
};

} // namespace ns3
