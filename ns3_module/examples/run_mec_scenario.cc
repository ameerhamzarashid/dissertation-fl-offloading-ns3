#include "mec-sim.h"
#include "ns3/core-module.h"

using namespace ns3;

int main (int argc, char *argv[]) {
  CommandLine cmd;
  uint32_t seed = 42;
  uint32_t nUe = 20;
  uint32_t nEnb = 5;
  double   duration = 600.0;
  double   areaX = 1000.0, areaY = 1000.0;
  std::string host = "127.0.0.1";
  uint16_t port = 50051;

  cmd.AddValue("seed", "RNG seed", seed);
  cmd.AddValue("nUe", "Number of mobile users", nUe);
  cmd.AddValue("nEnb", "Number of eNBs (edge servers in LTE cells)", nEnb);
  cmd.AddValue("duration", "Simulation duration (s)", duration);
  cmd.AddValue("areaX", "Area X (m)", areaX);
  cmd.AddValue("areaY", "Area Y (m)", areaY);
  cmd.AddValue("bridgeHost", "Python bridge host", host);
  cmd.AddValue("bridgePort", "Python bridge port", port);
  cmd.Parse(argc, argv);

  MecSimParams p;
  p.seed = seed; p.nUe = nUe; p.nEnb = nEnb; p.duration = duration; p.areaX = areaX; p.areaY = areaY;
  p.bridgeHost = host; p.bridgePort = port;

  Ptr<MecSim> sim = CreateObject<MecSim>();
  sim->Configure(p);
  sim->Build();
  sim->Run();

  return 0;
}
