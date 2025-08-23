Param(
  [string]$VenvDir = ".venv"
)
# Sets up a Python venv and installs required packages.
python -m venv $VenvDir
& "$VenvDir\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install numpy pyyaml matplotlib
# Optional (CPU PyTorch):
# pip install torch --index-url https://download.pytorch.org/whl/cpu
Write-Host "✅ Environment ready. Activate with: `"$VenvDir\Scripts\Activate.ps1`""
