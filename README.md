# TraceAI - Federated Knowledge Graph-Enhanced Agentic AI Platform

[![Unisys Innovation Program](https://img.shields.io/badge/Idea-Unisys_UIP-blue)](https://idea.unisys.com/D8927)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)

## Overview
Enterprises today face three fundamental barriers preventing effective AI adoption across organizational boundaries:
- **Data Privacy Constraints:** Organizations cannot share raw data due to strict regulations.
- **Lack of Explainability:** Black-box AI models limit trust and auditability.
- **Limited Autonomy:** Siloed systems prevent real-time, automated response to threats.

These challenges are particularly critical in high-stakes industries such as finance, healthcare, and cybersecurity. We propose a **Federated, Knowledge Graph-Enhanced Agentic AI Platform** that enables secure cross-enterprise learning, structured relational reasoning, transparent AI decisions, and autonomous response actions—all without sharing any raw data.

---

## 🏗 Architecture & Core Layers

The platform integrates five layers into a unified end-to-end pipeline:

### 1. Federated Learning Layer
- Built using the **Flower framework**.
- Each organization trains a local model on its own data, and only encrypted model weights are shared.
- Aggregation is performed using **Federated Averaging**, ensuring no raw data leaves the organization.

### 2. Prediction Layer
- Produces calibrated risk scores using the globally aggregated model.
- Serves as the baseline signal for downstream reasoning.

### 3. Knowledge Graph Layer
- Implemented using **NetworkX / Neo4j**.
- Models relationships between entities such as accounts, transactions, merchants, and risk clusters.
- Provides structured relational context beyond numerical features.

### 4. Explainability Layer
- Uses a locally hosted LLM (**Llama 3 via Ollama**).
- Combines model predictions and Knowledge Graph evidence to output clear, natural-language explanations and auditable decision reasoning.

### 5. Agentic Engine
- Built using **LangGraph** and the **ReAct framework**.
- Performs autonomous reasoning over explanations, selecting and executing actions like graph querying, alert escalation, threat remediation, and report generation.

---

## 🚀 Getting Started & Demo Setup

This guide provides instructions to set up and run the 4-node concurrent Proof of Concept (POC) demo locally.

### 1. System Requirements
- **Python 3.10+** (Required for PyTorch and backend components)
- **Node.js 18+** (Required for the Next.js frontend)
- **Windows PowerShell** (Required to run the automated startup scripts)

### 2. First-Time Setup

1. **Install Python Dependencies:**
   Open a terminal in the root directory and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node Dependencies:**
   Navigate to the frontend folder and install the packages:
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

### 3. Running the 4-Node POC Demo

We have included an automated PowerShell script that compiles the frontend for production (saving RAM) and automatically spins up 4 independent Python servers and 4 independent Node.js servers (simulating an Admin and 3 Banks).

1. Open **Windows PowerShell**.
2. Navigate to the project root directory.
3. Execute the startup script:
   ```powershell
   .\start_poc.ps1
   ```

**What this script does:**
- Runs `npm run build` once to create optimized static pages.
- Opens 8 separate PowerShell windows.
- 4 windows for the Python backends (`port 8000` for Admin, `8001-8003` for Banks).
- 4 windows for the Next.js frontends (`port 3000` for Admin, `3001-3003` for Banks).

### 4. Navigating the Demo

Once the servers are running, open your browser and navigate to the nodes:

| Node | URL | Login (Username) | Password |
|---|---|---|---|
| **Admin Control Room** | [http://localhost:3000](http://localhost:3000) | `admin` | `AdminSecure123!` |
| **Bank A** | [http://localhost:3001](http://localhost:3001) | `bank_a` | `BankA_Secure1!` |
| **Bank B** | [http://localhost:3002](http://localhost:3002) | `bank_b` | `BankB_Secure2@` |
| **Bank C** | [http://localhost:3003](http://localhost:3003) | `bank_c` | `BankC_Secure3#` |

### 5. Stopping the Demo & Troubleshooting

- **To Stop:** Click the "X" on all 8 PowerShell popup windows, OR open your main terminal and run:
  ```powershell
  Stop-Process -Name node -Force -ErrorAction SilentlyContinue
  Stop-Process -Name python -Force -ErrorAction SilentlyContinue
  ```
- **Database Locked / Desync:** If the FL rounds get stuck or you want to restart from Round 1, simply delete the `artifacts/gateway/fl_gateway.db` file and restart `.\start_poc.ps1`.
- **Port in Use:** If a server fails to start because a port is blocked, run the `Stop-Process` commands above to ensure no ghost processes are lingering.

---

## 🎯 Applications & Validation

This architecture is **domain-agnostic** and reusable across industries. 
The same pipeline has been successfully applied and validated on:
- **Financial Fraud Detection** (Dataset: IEEE-CIS | Setup: 3 Banks)
- **Cybersecurity Threat Detection** (Dataset: UNSW-NB15 | Setup: 3 Enterprise Networks)
- **Healthcare & Enterprise IT** (Theoretical expansions for patient privacy and infrastructure risk)

## Keywords
`Federated Learning`, `Knowledge Graph`, `Explainable AI`, `Agentic AI`, `LLM`, `Cross-Enterprise Intelligence`, `Privacy-Preserving ML`
