param(
  [string]$BuildType = "Release",
  [string]$Ns3Dir = ""
)
$ErrorActionPreference = "Stop"
$buildDir = "build"
if (!(Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }

cmake -S . -B $buildDir -G "NMake Makefiles" `
  -DCMAKE_BUILD_TYPE=$BuildType `
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON `
  -Dns3_DIR="$Ns3Dir"

cmake --build $buildDir --config $BuildType
