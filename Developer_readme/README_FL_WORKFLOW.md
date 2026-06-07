# Federated Learning End-to-End Workflow

This document outlines the exact command sequence required to run the Domain-Agnostic Federated Learning pipeline from scratch, including an explanation of what is happening under the hood at each step.

---

## 1. Data Preparation (The Split)

**Command:**
```bash
python src/main/run_data_pipeline.py
```

**What happens here?**
* **Ingestion:** The system loads your massive, single raw dataset (e.g., `creditcard.csv`).
* **Validation:** It verifies that the dataset has all the correct columns specified in your schema.
* **Splitting:** It mathematically splits the dataset into separate CSVs for each client (e.g., `client_a_train.csv`, `client_b_train.csv`) and creates a `global_test.csv` for final evaluation. These are saved in `data/splits/`.
* **Why it's needed:** Federated Learning requires decentralized data. This step simulates that decentralization by breaking a centralized dataset into isolated "silos".

---

## 2. Local Baselines & Preprocessor Initialization

**Commands:**
```bash
python src/main/run_single_baseline.py --client client_a
python src/main/run_single_baseline.py --client client_b
python src/main/run_single_baseline.py --client client_c
```

**What happens here?**
* **Dynamic Vectorization:** For each client, the system reads `configs/mapping.json`. It looks at the specific client's data, handles missing values, scales the numbers (Standard/Robust), encodes categorical text, and mathematically forces the dataset into a strict feature dimension (e.g., exactly 64 columns using PCA or zero-padding).
* **Model Training:** It trains a local Neural Network model (MLP) purely on that individual client's data for a set number of epochs. 
* **Artifact Generation (Crucial):** It saves the customized data pipeline logic as a pickle file (`artifacts/preprocessors/client_a_preprocessor.pkl`). 
* **Why it's needed:** The global Federated system needs to know how to transform the raw data into tensors. By running this, you generate the `preprocessor.pkl` files that the Flower simulation relies on. You also get a "baseline" F1-score to prove your Federated model is actually an improvement!

---

## 3. Global Federated Learning Simulation

**Command:**
```bash
python src/main/run_fl_simulation.py
```

*(Optional: Override the number of rounds by adding `--num_rounds 10`)*

**What happens here?**
* **Initialization:** The script starts a virtual Federated Server (via the Flower framework) and connects to 3 virtual Clients. It automatically loads the `preprocessor.pkl` files generated in Step 2.
* **The FL Loop (per round):**
  1. **Broadcast:** The Server sends the current Global Model weights to all clients.
  2. **Local Fit:** Each client trains the model on their own isolated data (`client_a_train.csv`, etc.) for a few epochs.
  3. **Upload:** Each client sends their updated weights back to the Server.
  4. **Aggregate:** The Server averages the weights together (FedAvg) to create a smarter Global Model.
* **Evaluation:** At the end of the round, the Server evaluates the new Global Model to see if the F1-score and PR-AUC improved.
* **Why it's needed:** This is the core magic! It allows multiple organizations to collaboratively build a highly accurate machine learning model without ever sharing their underlying raw data with each other.
