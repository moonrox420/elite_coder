# Windows PowerShell Environment Initialization and Setup script for Elite Coding Assistant.
# Save this file as: init_workspace.ps1

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Logging Functions
# ---------------------------------------------------------------------------
function Write-Info ($Message) {
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Warn ($Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-ErrorMessage ($Message) {
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Success ($Message) {
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Pre-flight Tool Assertions
# ---------------------------------------------------------------------------
Write-Info "Starting structural system pre-flight checks on Windows..."

# Verify Python install
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-ErrorMessage "Python is not installed or not found in system environment variables (Path)."
    Exit 1
}

# Verify pip is present
if (-not (Get-Command "pip" -ErrorAction SilentlyContinue)) {
    Write-ErrorMessage "Pip is missing from current Path. Ensure Pip is installed and added to environment variables."
    Exit 1
}

# Ensure Ollama service is reachable on Windows
$OllamaReachable = $false
try {
    $Response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($null -ne $Response) {
        $OllamaReachable = $true
    }
}
catch {
    $OllamaReachable = $false
}

if (-not $OllamaReachable) {
    Write-ErrorMessage "Ollama service is unreachable on http://localhost:11434."
    Write-Info "Ensure the local Windows Ollama Desktop application is running in your system tray."
    Exit 1
}

# ---------------------------------------------------------------------------
# Workspace Structure Mapping
# ---------------------------------------------------------------------------
Write-Info "Constructing local Windows directory storage layers..."

$RequiredDirectories = @("multi-task-final", "rag_db")
foreach ($Dir in $RequiredDirectories) {
    if (-not (Test-Path -Path $Dir)) {
        try {
            New-Item -ItemType Directory -Path $Dir | Out-Null
            Write-Info "  → Created Windows directory: '$Dir'"
        }
        catch {
            Write-ErrorMessage "Failed to construct local folder path: '$Dir'"
            Exit 1
        }
    }
    else {
        Write-Info "  → Verified directory path: '$Dir'"
    }
}
Write-Success "Workspace folder paths mapped successfully."

# ---------------------------------------------------------------------------
# Ollama Models Procurement
# ---------------------------------------------------------------------------
Write-Info "Syncing LLMs and embedder weights from local Windows Ollama daemon..."

# Pull nomic-embed-text for vector search embeddings
Write-Info "Pulling Nomic Embed Text model..."
& ollama pull nomic-embed-text

# Pull Qwen 14B for generation and memory summarization
Write-Info "Pulling Qwen 14B LLM model..."
& ollama pull gemma4:cloud

# Pull Mistral for offline synthetic dataset generation routines
Write-Info "Pulling Mistral model..."
& ollama pull mistral

Write-Success "All required local Ollama models successfully synced."

# ---------------------------------------------------------------------------
# Setup Execution Call
# ---------------------------------------------------------------------------
Write-Info "Running dependency builder and virtual package setup inside PowerShell..."
& python run.py --setup

Write-Success "System is fully initialized and operational on Windows."
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  To train the classifier model:  python run.py --train" -ForegroundColor Cyan
Write-Host "  To start the coding assistant:  python run.py --start" -ForegroundColor Cyan
Write-Host "  To run benchmarks/evaluation:  python run.py --benchmark" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green