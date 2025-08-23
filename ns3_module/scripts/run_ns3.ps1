param(
  [string]$BuildType = "Release",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)
$bin = "build\run_mec_scenario.exe"
if (!(Test-Path $bin)) { Write-Error "Binary not found at $bin. Build first with scripts\build_ns3_module.ps1"; exit 1 }
& $bin @Args
