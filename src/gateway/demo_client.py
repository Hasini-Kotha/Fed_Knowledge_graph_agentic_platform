"""demo_client.py — Shows how a bank client interacts with the FL gateway.

Run AFTER starting the gateway:
    uvicorn src.gateway.gateway:app --port 8000

Then in a second terminal:
    python src/gateway/demo_client.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import requests

BASE = "http://localhost:8000"

 
# 1. Register the bank
 
print("\n=== Step 1: Register ===")
resp = requests.post(f"{BASE}/register-client", json={
    "client_id": "bank_alpha",
    "bank_name": "Alpha National Bank",
    "password": "SecurePass123!",
})
print(resp.status_code, resp.json())

 
# 2. Login and get JWT
 
print("\n=== Step 2: Login ===")
resp = requests.post(f"{BASE}/login", json={
    "client_id": "bank_alpha",
    "password": "SecurePass123!",
})
token_data = resp.json()
token = token_data["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("Token received:", token[:40], "...")

 
# 3. Check round status
 
print("\n=== Step 3: Round Status ===")
resp = requests.get(f"{BASE}/round-status", headers=headers)
print(resp.json())

 
# 4. Check global model info
 
print("\n=== Step 4: Global Model Info ===")
resp = requests.get(f"{BASE}/global-model", headers=headers)
print(resp.status_code, resp.json())

 
# 5. Submit a weight update (dummy weights for demo)
#    In real usage: load your locally-trained model parameters here.
 
print("\n=== Step 5: Submit Weight Update ===")

# Build dummy weights matching LiteFraudNet parameter shapes
from src.gateway.validator import EXPECTED_SHAPES
dummy_weights = [np.zeros(shape, dtype=np.float32).tolist() for shape in EXPECTED_SHAPES]

resp = requests.post(f"{BASE}/submit-update", headers=headers, json={
    "round_num": 1,
    "model_version": "LiteFraudNet-v1",
    "n_samples": 64556,
    "weights": dummy_weights,
})
print(resp.status_code, resp.json())

 
# 6. View submission logs
 
print("\n=== Step 6: Admin Logs ===")
resp = requests.get(f"{BASE}/admin/logs", headers=headers)
logs = resp.json()["logs"]
for log in logs[:5]:
    print(log)
