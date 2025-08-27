#pragma once
/**
 * mec-sim.h
 * MEC + FL simulation plumbing in ns-3 (energy-first).
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/lte-module.h"
#include "ns3/point-to-point-epc-helper.h"
#include "ns3/config-store-module.h"
#include <string>
#include <vector>
#include <memory>

#include "task-generator.h"
#include "offload-app.h"
#include "bridge-client.h"
#include "energy-tracker.h"
#include "logging-helper.h"

namespace ns3 {

struct MecSimParams {
  uint32_t seed = 42;
  double   duration = 600.0;                     // seconds
  uint32_t nUe = 20;
  uint32_t nEnb = 5;
  double   areaX = 1000.0, areaY = 1000.0;       // meters
  // LTE / PHY settings
  double   bandwidthMhz = 20.0;
  double   ueTxPowerDbm = 23.0;
  // Mobility
  double   meanSpeed = 1.0;
  double   speedStd = 0.5;
  // Bridge (Python co-sim)
  std::string bridgeHost = "127.0.0.1";
  uint16_t    bridgePort = 50051;
};

class MecSim : public Object {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("ns3::MecSim")
      .SetParent<Object>()
      .SetGroupName("Applications");
    return tid;
  }

  MecSim() = default;
  ~MecSim() override = default;

  void Configure(const MecSimParams& p) { m_params = p; }

  void Build() {
    RngSeedManager::SetSeed(m_params.seed);
    // Nodes
    m_enbs.Create(m_params.nEnb);
    m_ues.Create(m_params.nUe);

    // Mobility
    MobilityHelper mobEnb;
    mobEnb.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobEnb.Install(m_enbs);

  MobilityHelper mobUe;
  // Use explicit ns-3 RandomVariable syntax to avoid LookupByName asserts
  mobUe.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                 "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(m_params.areaX) + "]"),
                 "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(m_params.areaY) + "]"));
  mobUe.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
               "Mode", StringValue("Time"),
               "Time", StringValue("2s"),
               "Speed", StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(m_params.meanSpeed) + "]"),
               "Bounds", RectangleValue(Rectangle(0, m_params.areaX, 0, m_params.areaY)));
    mobUe.Install(m_ues);

    // LTE/EPC
    m_epcHelper = CreateObject<PointToPointEpcHelper>();
    m_lteHelper = CreateObject<LteHelper>();
    m_lteHelper->SetEpcHelper(m_epcHelper);

    // Internet stack
    InternetStackHelper internet;
    internet.Install(m_ues);

    // LTE devices
    NetDeviceContainer enbDevs = m_lteHelper->InstallEnbDevice(m_enbs);
    NetDeviceContainer ueDevs  = m_lteHelper->InstallUeDevice(m_ues);

    // IP assign + attach
    Ipv4InterfaceContainer ueIfaces = m_epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
    for (uint32_t i = 0; i < m_ues.GetN(); ++i) {
      m_lteHelper->Attach(ueDevs.Get(i), enbDevs.Get(i % enbDevs.GetN())); // round-robin
      Ipv4StaticRoutingHelper ipv4RoutingHelper;
      Ptr<Ipv4StaticRouting> srt = ipv4RoutingHelper.GetStaticRouting(m_ues.Get(i)->GetObject<Ipv4>());
      srt->SetDefaultRoute(m_epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // Logging helper (CSV)
    m_logger = CreateObject<LoggingHelper>();
    m_logger->Open("data/raw_logs", "link_bytes.csv", "radio_energy.csv");

    // Per-UE components
    for (uint32_t i = 0; i < m_ues.GetN(); ++i) {
      Ptr<Node> ue = m_ues.Get(i);

      Ptr<EnergyTracker> energy = CreateObject<EnergyTracker>();
      energy->SetNode(ue);
      energy->SetCpuParams(1.5e9 /*Hz*/, 1.0e-9 /*J per cycle*/);
      energy->SetRadioParams(0.2 /*W tx*/, 0.1 /*W rx*/);

      Ptr<BridgeClient> bridge = CreateObject<BridgeClient>();
      bridge->Configure(m_params.bridgeHost, m_params.bridgePort);
      bridge->SetNode(ue);
  bridge->Connect(); // start background thread and connect early

      Ptr<OffloadApp> app = CreateObject<OffloadApp>();
      app->SetNode(ue);
      app->SetBridge(bridge);
      app->SetEnergy(energy);
      app->SetLogger(m_logger);
      app->SetComputeFreqHz(1.5e9);
      ue->AddApplication(app);
      app->SetStartTime(Seconds(0.1));
      app->SetStopTime(Seconds(m_params.duration));

      Ptr<TaskGenerator> gen = CreateObject<TaskGenerator>();
      gen->SetNode(ue);
      gen->SetLambda(0.5);
      gen->SetSizeBytesRange(10 * 1024 * 1024, 20 * 1024 * 1024);
      gen->SetCyclesRange(900e6, 1100e6);
      gen->SetCallback(MakeCallback(&OffloadApp::OnTask, app));
      gen->Start(Seconds(0.2));
      gen->Stop(Seconds(m_params.duration));

      m_apps.push_back(app);
      m_gens.push_back(gen);
      m_energies.push_back(energy);
      m_bridges.push_back(bridge);
    }
  }

  void Run() {
    Simulator::Stop(Seconds(m_params.duration + 1.0));
    Simulator::Run();
    if (m_logger) m_logger->Close();
    Simulator::Destroy();
  }

  NodeContainer GetUes() const { return m_ues; }
  NodeContainer GetEnbs() const { return m_enbs; }

private:
  MecSimParams m_params;
  NodeContainer m_enbs, m_ues;
  Ptr<PointToPointEpcHelper> m_epcHelper;
  Ptr<LteHelper> m_lteHelper;
  Ptr<LoggingHelper> m_logger;

  std::vector< Ptr<OffloadApp> >    m_apps;
  std::vector< Ptr<TaskGenerator> > m_gens;
  std::vector< Ptr<EnergyTracker> > m_energies;
  std::vector< Ptr<BridgeClient> >  m_bridges;
};

} // namespace ns3
