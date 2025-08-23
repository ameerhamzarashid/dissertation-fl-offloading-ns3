cmake_minimum_required(VERSION 3.16)
project(mec_ns3_module LANGUAGES CXX)

# Enforce C++20 to match modern ns-3 headers
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ns3_DIR must point to a folder containing ns3Config.cmake + ns3Targets.cmake
find_package(ns3 CONFIG REQUIRED)

# ---- Library
add_library(mec_sim
  src/mec-sim.cc
  src/task-generator.cc
  src/offload-app.cc
  src/bridge-client.cc
  src/energy-tracker.cc
  src/logging-helper.cc
)

target_include_directories(mec_sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(mec_sim
  PUBLIC
    ns3::core
    ns3::network
    ns3::internet
    ns3::applications
    ns3::mobility
    ns3::lte
    ns3::energy
    ns3::stats
)

target_compile_features(mec_sim PUBLIC cxx_std_20)

# ---- Example entry
add_executable(run_mec_scenario
  examples/run_mec_scenario.cc
)
target_link_libraries(run_mec_scenario PRIVATE mec_sim)
target_compile_features(run_mec_scenario PRIVATE cxx_std_20)
