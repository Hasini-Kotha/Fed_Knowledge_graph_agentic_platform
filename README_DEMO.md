# TraceAI - Federated Knowledge Graph Platform (Demo Setup Guide)

This guide provides instructions for a new developer or presenter to set up and run the 4-node concurrent Proof of Concept (POC) demo locally.

## 1. System Requirements

Ensure your system has the following installed before proceeding:
- **Python 3.10+** (Required for PyTorch and backend components)
- **Node.js 18+** (Required for Next.js frontend)
- **Windows PowerShell** (Required to run the startup script)

## 2. First-Time Setup

1. **Install Python Dependencies:**
   Open a terminal in the root directory of the project and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node Dependencies:**
   Navigate into the `frontend` folder and install the web packages:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Initialize the Artifact Directories:**
   Run the setup script to generate the required folders and configuration files (this creates the `artifacts/` folder where databases and models are stored):
   ```bash
   python setup.py
   ```

## 3. How to Start the 4-Node Demo

We have created an automated PowerShell script that compiles the frontend for production (to save RAM) and automatically spins up 4 independent Python servers and 4 independent Node.js servers.

1. Open **Windows PowerShell** as Administrator (or standard user).
2. Navigate to the project root directory.
3. Execute the startup script:
   ```powershell
   .\start_poc.ps1
   ```

**What this script does:**
- It runs `npm run build` once to create optimized static pages.
- It opens 8 separate PowerShell pop-up windows.
- 4 windows run the Python backends (`port 8000` for Admin, `8001-8003` for Banks).
- 4 windows run the Node.js frontends (`port 3000` for Admin, `3001-3003` for Banks).

## 4. Navigating the Demo

Once all windows are open, use your browser to navigate to the nodes:

- **Admin Control Room:** [http://localhost:3000](http://localhost:3000) (Login as `admin` / `AdminSecure123!`)
- **Bank A:** [http://localhost:3001](http://localhost:3001) (Login as `bank_a` / `BankA_Secure1!`)
- **Bank B:** [http://localhost:3002](http://localhost:3002) (Login as `bank_b` / `BankB_Secure2@`)
- **Bank C:** [http://localhost:3003](http://localhost:3003) (Login as `bank_c` / `BankC_Secure3#`)

## 5. How to Stop the Demo

To cleanly shut down the environment, you can either:
1. Click the "X" on all 8 of the popup PowerShell windows.
2. OR, open your main terminal and run the following command to forcefully kill all node and python processes:
   ```powershell
   Stop-Process -Name node -Force -ErrorAction SilentlyContinue
   Stop-Process -Name python -Force -ErrorAction SilentlyContinue
   ```

## Troubleshooting
- **Database Locked / Desync:** If the rounds get stuck or you want to restart the demo from Round 1, simply delete the `artifacts/gateway/fl_gateway.db` file and restart `.\start_poc.ps1`.
- **Port in Use:** If a server fails to start because a port is blocked, run the `Stop-Process` commands above to ensure no ghost processes are hogging ports 3000-3003 or 8000-8003.
