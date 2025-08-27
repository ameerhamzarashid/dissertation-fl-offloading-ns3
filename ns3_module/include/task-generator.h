#pragma once
/**
 * task-generator.h
 * Poisson task arrivals with size/complexity distributions.
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include <functional>

namespace ns3 {

struct Task {
  uint64_t id = 0;
  uint32_t ownerUeIndex = 0;
  uint64_t sizeBytes = 0;   // input size to send if offloaded
  double   cycles = 0.0;    // required CPU cycles if executed
  double   createdAt = 0.0; // sim time
};

// Signature: void (const Task&)
using TaskCallback = Callback<void, const Task&>;

class TaskGenerator : public Object {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("ns3::TaskGenerator")
      .SetParent<Object>()
      .SetGroupName("Applications");
    return tid;
  }

  TaskGenerator() {
    m_rng = CreateObject<ExponentialRandomVariable>();
    m_rng->SetAttribute("Mean", DoubleValue(2.0)); // default 0.5 tasks/s
    m_sizeUniform = CreateObject<UniformRandomVariable>();
    m_cyclesUniform = CreateObject<UniformRandomVariable>();
  }

  void SetNode(Ptr<Node> n) { m_node = n; }
  void SetCallback(TaskCallback cb) { m_cb = cb; }
  void SetLambda(double lambdaPerSec) {
    double meanInter = 1.0 / std::max(1e-9, lambdaPerSec);
    m_rng->SetAttribute("Mean", DoubleValue(meanInter));
  }
  void SetSizeBytesRange(uint64_t low, uint64_t high) {
    m_sizeLow = low; m_sizeHigh = high;
  }
  void SetCyclesRange(double low, double high) {
    m_cyclesLow = low; m_cyclesHigh = high;
  }

  void Start(Time t) {
    m_running = true;
    m_next = Simulator::Schedule(t, &TaskGenerator::Generate, this);
  }
  void Stop(Time t) {
    Simulator::Schedule(t, &TaskGenerator::DoStop, this);
  }

private:
  void DoStop() { m_running = false; if (m_next.IsPending()) m_next.Cancel(); }

  void Generate() {
    if (!m_running) return;
    Task task;
    task.id = ++m_id;
    task.ownerUeIndex = m_node ? m_node->GetId() : 0;
    task.sizeBytes = (uint64_t) m_sizeUniform->GetInteger(m_sizeLow, m_sizeHigh);
    task.cycles = m_cyclesUniform->GetValue(m_cyclesLow, m_cyclesHigh);
    task.createdAt = Simulator::Now().GetSeconds();

    if (!m_cb.IsNull()) m_cb(task);

    // Schedule next arrival
    Time inter = Seconds(m_rng->GetValue());
    m_next = Simulator::Schedule(inter, &TaskGenerator::Generate, this);
  }

  Ptr<Node> m_node;
  EventId m_next;
  bool m_running{false};
  uint64_t m_id{0};

  // RNGs
  Ptr<ExponentialRandomVariable> m_rng;
  Ptr<UniformRandomVariable> m_sizeUniform;
  Ptr<UniformRandomVariable> m_cyclesUniform;
  uint64_t m_sizeLow{1*1024*1024}, m_sizeHigh{2*1024*1024};
  double m_cyclesLow{9e8}, m_cyclesHigh{1.1e9};

  TaskCallback m_cb;
};

} // namespace ns3
