# Implementation Status — Full Platform Audit

> **Last updated**: 12 May 2026  
> **Scope reviewed**: Every `.py`, `.yaml`, `.json`, and artifact file in the repository

---

## The Full Platform Has 5 Layers

Your README defines 5 layers. Here is where each one stands:

| # | Layer | Status | What It Does |
|---|-------|--------|--------------|
| 1 | **Federated Learning** | ✅ Code complete, has been run | 3 banks train locally → aggregate → global model |
| 2 | **Prediction Layer** | ✅ Code complete | Use global model to produce fraud risk scores |
| 3 | **Knowledge Graph** | ✅ Code complete | Relational reasoning between entities (accounts, merchants, transactions) |
| 4 | **Explainability (LLM)** | ❌ Not started | Llama 3 generates human-readable explanations |
| 5 | **Agentic Engine** | ❌ Not started | LangGraph/ReAct autonomous reasoning + actions |

> [!IMPORTANT]
> **Your part (Knowledge Graph) is Layer 3.** It is now **COMPLETED**. It consumes the global model's predictions from Layer 2 and provides structured relational evidence to Layer 4 (Explainability).

---

## Layer 1: Federated Learning — ✅ Fully Implemented

#### 1A. Data Foundation
- **Schema contract & Validator**: Defines "what columns must any dataset have" in a domain-agnostic way.
- **Client splitter**: Splits one big dataset into 3 "bank" datasets + 1 global holdout test set.
- **Mapping engine**: Reads `mapping.json` — ensuring the platform stays domain-agnostic.

#### 1B. Local ML Pipeline
- **Preprocessor**: `ClientPreprocessor` — enforces fixed output dimension via zero-padding.
- **Models**: `TabularMLP` and `TabularTransformer`.
- **Training engine**: Forward pass, optional FedProx, local metrics evaluation.

#### 1C. Federated Orchestration
- **Flower client**: Receives global weights, applies secure update protection (norm clipping + DP noise).
- **Aggregation strategies**: `WeightedFedAvg`, `FedProxStrategy`, `TrimmedMeanStrategy`.
- **Manual FL loop**: Fallback for simulation.

*Note: Minor bugs regarding Python import paths and tensor/numpy conversion have been resolved.*

---

## Layer 2: Prediction Layer — ✅ Fully Implemented

The Prediction Layer (Layer 2) provides a clean **inference API** that bridges the Federated Learning models with downstream components.

| What | File | What It Actually Does |
|------|------|----------------------|
| **Predictor Wrapper** | `src/prediction/predictor.py` | `GlobalModelPredictor` loads the `FINAL_global_model.pt`, `model_card.json`, and preprocessor. It provides `predict()` to attach raw `fraud_risk_score` and `classify()` to apply the optimal threshold and assign a `predicted_label`. |
| **Runner Script** | `src/main/run_prediction.py` | Command-line tool to score an entire CSV, outputting a scored file and a summary JSON. |

---

## Layer 3: Knowledge Graph — ✅ Fully Implemented (YOUR PART)

The Knowledge Graph (Layer 3) is a domain-agnostic engine that builds relational structures from tabular data and risk scores.

### 1. The Architecture We Used
We used an **In-Memory, Schema-Driven Architecture** built on top of the **NetworkX** Python library. 
- **Why NetworkX?** It is incredibly fast, allows complex matrix math (for similarity edges) in memory, and doesn't require setting up a heavy external database like Neo4j right now. The graph is serialized and saved as a `.graphml` file.
- **Why Schema-Driven?** The Python code contains ZERO fraud-specific logic. It doesn't know what "Amount", "Time", or "Merchant" is. Everything is dynamically read from `kg_config.yaml`. This means you can swap this pipeline to a Hospital or Cybersecurity dataset with zero code changes, just config updates!

### 2. What Each Config Represents (`configs/kg_config.yaml`)
The config is the "brain" of the graph. It defines the rules for the architecture:

**A. Primary Entities (Transactions)**
- It defines the `transaction` as the core node. It tells the pipeline to extract `Amount` and `Time` columns as raw attributes.

**B. "Derived" Entities (Time Buckets & Amount Buckets)**
Because the MLG-ULB dataset is anonymized, we don't have columns for "Merchant Name" or "Account ID" to connect transactions together. To solve this, the config instructs the graph builder to dynamically create "Derived" hub nodes out of the raw numbers:
- **Amount Buckets (`amount_bucket`)**: The config defines exact threshold edges (`[0, 50, 200, 1000, 5000, 999999]`) and assigns them plain-English labels. 
  - `$0 to $50` = `micro`
  - `$50 to $200` = `small`
  - `$200 to $1000` = `medium`
  - `$1000 to $5000` = `large`
  - `$5000+` = `whale`
  - *Why?* If 50 different transactions are all between $200 and $1000, they will all connect to the single `medium` amount bucket node, forming a structural relationship.
- **Time Windows (`time_window`)**: The config tells the builder to take the `Time` column (which is raw seconds) and mathematically split the entire dataset into exactly `24` equal chronological buckets (like hours in a day). 
  - *Why?* This allows the graph to connect transactions that occurred at exactly the same time. If a fraudster runs a script that blasts 100 transactions in the same 5-minute window, they will all instantly connect to the same `time_window` node in the graph, making the anomaly highly visible to the analytics engine!

**C. Relationships (Edges)**
- Defines how to connect the nodes: `HAS_AMOUNT_BUCKET`, `IN_TIME_WINDOW`.
- Defines `SIMILAR_PATTERN`: This tells the graph to look at features `V1, V2, V3, V14, V17`. If two transactions have a cosine similarity above `0.85`, it draws a "similarity edge" between them (acting as a mathematical proxy for "they look exactly the same").

**D. Risk Thresholds & Analytics Methods**
- **Risk**: Defines that `risk_score >= 0.7` is high risk, and specifies that risk should mathematically propagate `2` hops outward through the graph.
- **Analytics**: Defines that we should use the `louvain` algorithm for community detection and `degree` for centrality.

### 3. What Input It Takes
The KG pipeline takes **three specific inputs**:
1. **Raw CSV Data**: (e.g. `global_test.csv`). It extracts the columns specified in the config (`Amount`, `Time`, and `V1-V28` features).
2. **Global Model Risk Scores**: The KG calls the Layer 2 Predictor to run the CSV through `FINAL_global_model.pt`, giving each transaction a `fraud_risk_score` from 0.0 to 1.0.
3. **The YAML Config**: To know how to glue the data and the scores together into a graph.

### 4. File-by-File Breakdown

| What | File | What It Actually Does |
|------|------|----------------------|
| **Schema** | `src/kg/kg_schema.py` | Parses `kg_config.yaml` into strict, typed Python dataclasses (`EntityType`, `RelationshipType`) to validate input datasets. |
| **Builder** | `src/kg/kg_builder.py` | Reads transaction records and constructs the NetworkX graph. It creates the structural edges and uses batched K-Nearest-Neighbor cosine similarity to safely compute `SIMILAR_PATTERN` edges between transactions without running out of memory. |
| **Enricher** | `src/kg/kg_enricher.py` | Takes the risk scores from Layer 2, attaches them to nodes, and propagates risk through multi-hop BFS (breadth-first search) to compute contextual `neighborhood_risk`. |
| **Analytics** | `src/kg/kg_analytics.py` | Performs Louvain community detection to find dense risk clusters, and degree centrality to find highly connected (suspicious) nodes. |
| **Query Engine**| `src/kg/kg_query.py` | Generates structured **evidence bundles** (JSON dictionaries). It compiles the node's risk, neighborhood stats, community size, and connected entities into one clean payload. Consumed by Layer 4. |
| **Runners** | `src/main/run_kg_pipeline.py` | The end-to-end pipeline script: load data → score (Layer 2) → build graph → enrich → analyze → save `.graphml` and evidence bundles. |

---

## What Remains to Be Built (Layers 4–5)

### Layer 4: Explainability (LLM) — ❌ Not Started
- Llama 3 via Ollama
- Takes: the JSON Evidence Bundle from Layer 3 → produces human-readable explanation
- Depends on Layer 3 (which is now done!)

### Layer 5: Agentic Engine — ❌ Not Started
- LangGraph + ReAct framework
- Takes: explanation + KG queries → autonomous actions (alerts, remediation)
