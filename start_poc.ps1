$instances = @(
    @{ Name="Admin"; BackendPort=8000; FrontendPort=3000; DecisionsDB="artifacts/actions/admin_decisions.db" },
    @{ Name="Bank_A"; BackendPort=8001; FrontendPort=3001; DecisionsDB="artifacts/actions/bank_a_decisions.db" },
    @{ Name="Bank_B"; BackendPort=8002; FrontendPort=3002; DecisionsDB="artifacts/actions/bank_b_decisions.db" },
    @{ Name="Bank_C"; BackendPort=8003; FrontendPort=3003; DecisionsDB="artifacts/actions/bank_c_decisions.db" }
)

# Build the frontend ONCE to prevent 4 concurrent Dev servers from crashing your laptop
Write-Host "Building frontend (this may take a minute but saves massive CPU/RAM)..."
cd frontend
npm run build
cd ..

# Start Backend and Frontend for each instance
foreach ($inst in $instances) {
    Write-Host "Starting $($inst.Name)..."
    
    # 1. Start Backend
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "& {
        `$env:DECISIONS_DB_PATH='$($inst.DecisionsDB)'
        # All share the same gateway DB to simulate a single federated network
        `$env:GATEWAY_DB_PATH='artifacts/gateway/fl_gateway.db'
        
        # Activate virtual environment if it exists
        if (Test-Path 'venv\Scripts\Activate.ps1') {
            . '.\venv\Scripts\Activate.ps1'
        }
        
        # Start uvicorn backend
        uvicorn backend.main:app --port $($inst.BackendPort)
    }"
    
    # 2. Start Frontend in Production mode
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "& {
        cd frontend
        `$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:$($inst.BackendPort)'
        `$env:PORT=$($inst.FrontendPort)
        
        # Start Next.js frontend
        npm run start -- -p $($inst.FrontendPort)
    }"
    
    # Stagger slightly just to be safe
    Write-Host "Waiting 3 seconds before starting the next node..."
    Start-Sleep -Seconds 3
}

Write-Host "All instances started successfully."
Write-Host "Admin:   http://localhost:3000 (Backend: 8000)"
Write-Host "Bank A:  http://localhost:3001 (Backend: 8001)"
Write-Host "Bank B:  http://localhost:3002 (Backend: 8002)"
Write-Host "Bank C:  http://localhost:3003 (Backend: 8003)"
Write-Host ""
Write-Host "To stop, you may need to close the terminal or kill the node/python processes manually."
