$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceDir = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$ToolDir = Join-Path $WorkspaceDir '.tools\llama-b10369'
$DownloadDir = Join-Path $WorkspaceDir '.tools\downloads'
$LlamaZip = Join-Path $DownloadDir 'llama-b10369-bin-win-cuda-12.4-x64.zip'
$CudaZip = Join-Path $DownloadDir 'cudart-llama-bin-win-cuda-12.4-x64.zip'

New-Item -ItemType Directory -Force -Path $ToolDir, $DownloadDir | Out-Null

function Get-PinnedAsset {
    param([string]$Path, [long]$ExpectedSize, [string]$Url)
    if ((Test-Path -LiteralPath $Path) -and ((Get-Item -LiteralPath $Path).Length -eq $ExpectedSize)) {
        return
    }
    if (Test-Path -LiteralPath $Path) {
        throw "Existing partial or mismatched archive was preserved: $Path. Move it aside and retry."
    }
    & curl.exe -L --fail --retry 3 --output $Path $Url
    if ((Get-Item -LiteralPath $Path).Length -ne $ExpectedSize) {
        throw "Downloaded archive size mismatch: $Path"
    }
}

Get-PinnedAsset -Path $LlamaZip -ExpectedSize 250748190 -Url 'https://github.com/ggml-org/llama.cpp/releases/download/b10369/llama-b10369-bin-win-cuda-12.4-x64.zip'
Get-PinnedAsset -Path $CudaZip -ExpectedSize 391443627 -Url 'https://github.com/ggml-org/llama.cpp/releases/download/b10369/cudart-llama-bin-win-cuda-12.4-x64.zip'

Expand-Archive -LiteralPath $LlamaZip -DestinationPath $ToolDir -Force
Expand-Archive -LiteralPath $CudaZip -DestinationPath $ToolDir -Force

$Server = Join-Path $ToolDir 'llama-server.exe'
if (-not (Test-Path -LiteralPath $Server)) {
    throw "llama-server.exe was not found after extraction."
}
& $Server --version
