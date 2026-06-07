# 2026-Federated-Knowledge-Graph-Enhanced-Agentic-AI-Platform-for-Secure-Explainable-Cross-Enterprise
https://idea.unisys.com/D8927

## Project Overview

This project builds the **federated learning layer** of a privacy-preserving fraud detection platform designed for multi-organization collaboration. The system simulates three banks that each keep their customer transaction data locally, train a fraud model on-site, and contribute only model updates to a central aggregation server. The server combines these updates into a shared **global fraud detection model** without collecting raw transaction records from any bank.[1][2]

The work in this repository stops at the point where a reliable global model is produced and evaluated. That global model can later be connected to downstream layers such as risk scoring, knowledge graph context, explainability, and agentic response, which are part of the broader architecture described in the project documents.[1][2]

## Problem Statement

Banks face a common challenge in fraud detection: each institution sees only its own transaction history, while fraud patterns often spread across institutions. A single bank's local model may miss attack styles that are already visible at another bank. At the same time, banks cannot directly pool raw data because of privacy, compliance, competition, and governance constraints.[1][2]

This project addresses that problem by implementing **federated learning**, where multiple banks train collaboratively without exchanging raw customer-level records. Instead of centralizing data, the system centralizes learning through iterative model aggregation. This allows the final global model to benefit from cross-bank fraud patterns while preserving local data ownership.[1]

## What This Repository Does

This repository implements the complete path from standardized local datasets to a globally aggregated fraud detection model. It focuses on the following responsibilities:

- Loading and validating a fraud dataset.
- Standardizing feature expectations through a dataset schema contract.
- Splitting the dataset into three simulated bank clients.
- Preparing local train and validation datasets for each client.
- Training a local fraud classifier on each client.
- Running federated learning rounds using Flower.
- Sharing protected model updates instead of raw records.
- Aggregating local updates into a global model using weighted federated averaging.
- Saving and evaluating the final global model.

The repository deliberately stops before knowledge graph augmentation, LLM explanation, or autonomous response logic. Those later layers depend on the global model created here, but are outside the scope of this codebase.[1][2]

## Core Goal

The primary objective is to demonstrate that multiple banks can collaboratively train a fraud detector while keeping transaction rows private. The central technical outcome is a reproducible **global model development pipeline** that starts from local client data silos and ends with a shared fraud classification model ready for downstream use.[1]

the project is structured so student developers can clearly observe what happens at each stage of federated learning. The implementation is meant to make the separation between input standardization, local preprocessing, local training, protected update exchange, server-side aggregation, and global evaluation explicit and testable.[3]

## Scope Boundaries

The repository includes:

- Input validation and schema enforcement.
- Local preprocessing for fraud data.
- Three-client federated simulation.
- Local and global model training logic.
- Aggregation strategy and round logging.
- Global checkpoint saving and evaluation.

The repository does **not** include:

- Real bank deployment infrastructure.
- Production identity and access management.
- Full cryptographic homomorphic encryption pipeline.
- Knowledge graph construction.
- LLM-generated explanations.
- Agentic remediation workflows.

This boundary is intentional. The project documents define federated learning as the privacy-preserving training layer that produces the global model consumed by the later prediction and reasoning layers.[1][2]

## Why Federated Learning Is Used

Traditional centralized machine learning requires all transaction data to be collected in one place. That is difficult or impossible in regulated industries because raw financial data contains personally sensitive and institution-sensitive information. The project therefore uses federated learning so that each bank trains locally and only shares learned parameters or updates rather than underlying records.[1]

This design follows the architecture in the project documents, where local organizations train on-site, send protected model weights to a central aggregation server, and receive back an improved global model for the next round. The model becomes stronger over time because it learns from multiple data silos without dissolving those silos.[1][2]

## Why Fraud Detection Is a Good Use Case

Fraud detection is an especially strong fit for federated learning because fraud is distributed, adaptive, and sparse. A single bank usually sees only a partial picture of attacker behavior, while fraudulent patterns may emerge sooner when institutions learn together. At the same time, financial institutions cannot casually exchange raw transaction logs, which makes privacy-preserving collaborative learning highly relevant.[1][2]

The project is built around credit card fraud data to simulate this setting. Three demo clients represent three banks with separate local datasets, and a federated server coordinates the learning process across them.[1]

## Conceptual System Flow

The system works as a sequence of tightly separated layers:

1. **Input standardization** makes the dataset structurally consistent.
2. **Preprocessing** converts validated data into model-ready features.
3. **Local training** updates a client model using only local data.
4. **Protected update exchange** sends only model parameters or parameter changes.
5. **Federated aggregation** merges client learning into one global model.
6. **Global evaluation** measures whether the shared model actually improves.

This separation is critical. The input-layer document explicitly states that standardization must be distinct from preprocessing because heterogeneous datasets with inconsistent feature meaning will cause federated learning to fail.[3]

## Architectural Interpretation of the Demo

The broader project architecture contains multiple layers beyond federated learning, but this repository implements only the training core. In the larger design, the federated layer acts as the privacy guard and collaborative intelligence layer, producing the shared model that later generates risk scores and supports explainable reasoning.[1][2]

Within this repository, the architecture is intentionally simplified to one high-value path: **three local banks → local model training → aggregated global fraud model**. That keeps the implementation manageable while still aligning with the system described in the project documents.[1]

## Demo Scenario: Three Simulated Banks

The project simulates three financial institutions, each acting as an independent federated client:

- **Client A**: a bank with one local fraud distribution.
- **Client B**: a bank with a different transaction profile.
- **Client C**: a bank with a third local data pattern.

These clients do not exchange their CSV files with one another. Each client trains the same model architecture on its own local split and sends back only model updates for aggregation. This mirrors the decentralized silo setup emphasized in the architecture notes.[1]

The three-client demo is important because it allows the system to show collaborative learning behavior without requiring real multi-bank infrastructure. It is a controlled development environment for demonstrating how a global fraud model emerges from separate local learners.[1]

## Dataset Strategy

The first implementation uses a single fraud dataset and partitions it into three bank-like clients for simulation. This is a standard way to prototype federated learning when real institution-partitioned data is unavailable. The dataset is treated as if it came from multiple organizations, and the split logic creates three local data silos plus one untouched global holdout test set.

This design is especially useful in early development because it allows the project to validate the full federated pipeline before handling the complexity of real cross-institution ingestion. The documents already frame the goal as collaborative learning across siloed organizations while protecting raw records; a client-partitioned simulation is an acceptable first-stage implementation of that idea.[1][2]

## Input Layer Responsibility

A foundational idea in this project is that **federated learning must not begin on messy, inconsistent raw data**. Before preprocessing or model training, each client dataset must satisfy a common schema contract that defines expected columns, target labels, and feature meaning. The attached input-layer notes explicitly warn that feature mismatch across nodes makes weight averaging meaningless and breaks federated learning.[3]

For this reason, the repository starts with schema validation rather than with model code. Every later component depends on a shared understanding of what each feature represents and where it appears in the final model input vector.[3]

## Input Standardization vs Preprocessing

This project separates two responsibilities that are often mixed incorrectly:

- **Input standardization** checks whether the data is structurally valid.
- **Preprocessing** transforms valid data into model-ready form.

Input standardization includes things like required columns, label verification, feature typing, and sanity checks. Preprocessing includes operations like scaling, imputation, encoding, and tensor conversion. The project documents and the input-layer explanation strongly support keeping these concerns separate for reliability and debuggability.[3]

## Local Preprocessing Philosophy

Each client preprocesses its own data locally using the same code path and feature contract. This preserves privacy while ensuring that the model sees a consistent feature layout across clients. Even if preprocessing is fit locally, the expected final feature ordering must stay aligned so that model weights refer to the same learned meaning at every node.[3]

This is one of the most important engineering rules in the project. Federated averaging only makes sense if parameter positions correspond to the same features and model structure across all clients.[3]

## Local Model Training

Every client holds a copy of the same base model architecture. At the start of each round, the server provides the current global weights. The client loads those weights, trains locally for a small number of epochs using only its own data, evaluates on its local validation set, and returns updated parameters plus metrics.[1]

The local model is designed to be lightweight and stable rather than unnecessarily complex. In an educational federated pipeline, a small tabular neural network is usually the best starting point because it makes debugging convergence, client drift, and aggregation behavior easier.[1]

## Protected Model Update Exchange

A central rule of the project is that **raw transaction data never leaves the client**. What leaves the client is only a model update or model weights, ideally under additional protection. The project documents describe this as encrypted or protected weight sharing between local organizations and the aggregation server.[1][2]

In practice, this repository treats protected exchange as a staged privacy-preserving mechanism. The first version focuses on secure handling and validation of updates, then later extensions can strengthen privacy through clipping, masking, secure aggregation, or differential privacy. This is a realistic student implementation path that remains faithful to the privacy goals described in the architecture.[1]

## Federated Aggregation

The central server receives client updates from the three simulated banks and combines them into a new global model. The default aggregation strategy is weighted federated averaging, where larger client datasets contribute proportionally more to the updated model. This directly matches the project description of a global aggregation server that merges local learning into a smarter shared model.[1]

Aggregation is the point where decentralized learning becomes collaborative intelligence. No client sees another client's raw transactions, but the global model still accumulates the collective fraud knowledge represented in all local updates.[1][2]

## Global Model Development

The major output of the repository is the **global fraud detection model** produced after multiple federated rounds. This model is the shared artifact that has learned from all clients' local fraud patterns while preserving data locality. It is the key deliverable of the federated learning layer.[1]

Global model development includes more than aggregation alone. It also includes checkpointing, evaluation against a holdout test set, round-wise performance tracking, and selecting the final version of the model based on fraud-sensitive metrics such as PR-AUC, recall, and calibration quality. The project documents mention that the aggregated model feeds a later risk-scoring layer, so quality at this stage matters directly for downstream system behavior.[1][2]

## Why a Global Holdout Test Set Is Used

A separate global test set is maintained outside local client training. This allows the project to evaluate whether the aggregated model generalizes beyond the local training partitions. Without such a holdout set, it would be difficult to tell whether the global model is actually improving or merely memorizing client-specific patterns.

This centralized evaluation does not violate the project goal in a simulation setting because it is used only for research validation, not for cross-bank raw-data sharing in operational training. It provides one consistent reference point for comparing local and global performance.[1]

## Metrics That Matter

Fraud detection is usually highly imbalanced, which means accuracy alone is misleading. A model can appear accurate while still missing most fraud cases if it predicts the majority class too often. For that reason, the project emphasizes metrics such as precision, recall, F1, ROC-AUC, PR-AUC, and calibration behavior.[1]

These metrics help determine whether federated learning is actually improving fraud sensitivity. Since later layers are supposed to produce calibrated risk scores and decision support, the quality of the global model must be judged in a way that reflects rare-event detection rather than raw accuracy.[1]

## Why the Project Uses a Simulation First

The first implementation is a simulation rather than a production multi-bank deployment because the primary goal is to validate the learning pipeline itself. By controlling all three clients in one development environment, the repository can test schema handling, split logic, local training, communication flow, aggregation, and global checkpointing before worrying about real distributed infrastructure.

This simulation-first approach is consistent with the project documents, which describe a modular architecture that can be validated across domains and then extended. The three-client setup is therefore a development scaffold, not a claim of real-world deployment maturity.[1][2]

## Technology Choices

The project uses a stack that is realistic for professional student development:

- **Python** for the overall implementation.
- **Pandas** for data loading and validation.
- **scikit-learn** for preprocessing utilities.
- **PyTorch** for the tabular fraud classifier.
- **Flower** for federated learning orchestration and simulation.
- **YAML or Python config files** for experiment settings.
- **Artifact directories** for saved models, preprocessors, and metrics.

This stack is strongly aligned with the project material, which explicitly names Flower as the federated learning framework used for local training and global aggregation.[1][2]

## Repository Design Philosophy

The codebase is  modular so that each major stage of the pipeline can be understood, tested, and replaced independently. Data validation, preprocessing, local model training, federated orchestration, aggregation strategy, and evaluation are separated into dedicated components rather than merged into one script.

That modularity is important for both learning and reliability. The attached notes emphasize that if responsibilities are overloaded into one stage, debugging becomes difficult and failure sources become ambiguous.[3]

## Major Components

### 1. Data Schema Module

This component defines the expected dataset contract. It validates required columns, the label column, and feature expectations before any downstream processing starts. Its role is to prevent structural inconsistency from reaching preprocessing and federated training.[3]

### 2. Data Loading Module

This component reads the raw fraud dataset, checks shape and label distribution, and exposes a validated DataFrame to the rest of the pipeline. It provides the first sanity checkpoint for fraud imbalance and data completeness.

### 3. Client Split Module

This component partitions the validated dataset into three simulated bank clients and one holdout global test set. It is responsible for saving local train and validation splits in a way that preserves the federated-learning illusion of decentralized silos.

### 4. Preprocessing Module

This component transforms validated local data into model-ready tensors or arrays. It handles steps such as scaling, imputing, and preserving stable feature ordering while staying entirely local to each client.[3]

### 5. Local Model Module

This component defines the fraud classification network used by all clients. Every client must share the same architecture so that server-side aggregation is mathematically meaningful.

### 6. Federated Client Module

This component wraps local training and evaluation into the interface expected by Flower. It receives global weights, trains locally, and returns updated weights plus metrics to the server.[1]

### 7. Federated Strategy Module

This component defines how updates are aggregated, how many rounds are executed, and how round-level logging is handled. It governs the creation of the evolving global model.[1]

### 8. Global Evaluation Module

This component loads the final aggregated model and evaluates it on the global holdout set. It determines whether the federated pipeline produced a usable fraud detector.

### 9. Artifact Management

This component stores preprocessing artifacts, local checkpoints, global checkpoints, metrics, and experiment outputs. It supports reproducibility and inspection.

## End-to-End Learning Cycle

Each federated round follows the same pattern:

1. The server initializes or loads the current global model.
2. The server sends the global weights to each selected client.
3. Each client trains locally using its own local data only.[1]
4. Each client returns updated model parameters and metadata.[1]
5. The server aggregates those updates using weighted federated averaging.[1]
6. The updated global model is saved and optionally evaluated.[1]
7. The next round starts with the improved global weights.[1]

This repeated cycle is the core mechanism through which collaborative fraud intelligence emerges without centralized raw data sharing.[1][2]

## Privacy Positioning

This repository should be described carefully. It is a **privacy-preserving federated learning prototype**, not a fully audited enterprise security product. The project follows the architectural rule that raw data remains local and only model updates are exchanged, but advanced cryptographic guarantees may be implemented progressively rather than all at once.[1]

That distinction matters in academic presentations and professional documentation. It is more credible to claim a staged privacy pipeline than to overstate cryptographic completeness when the main engineering goal is federated model development.[1]

## Project learning objectives

The project is strong from a learning standpoint because it exposes several real machine learning engineering issues at once:

- class imbalance in fraud detection,
- non-IID distributions across clients,
- privacy constraints,
- aggregation strategy design,
- local versus global performance trade-offs,
- reproducible experiment management.

It also demonstrates how modular AI systems are built in layers rather than as one monolithic script. That matches the larger platform vision described in the project documents.[1][2]

## Expected Output of the Repository

When this repository is complete, it should produce the following core outputs:

- validated fraud dataset report,
- three client train/validation splits,
- local training logs,
- federated round metrics,
- saved global checkpoints,
- final global fraud model,
- evaluation summary on the holdout test set.

These outputs together define success for the current scope. They prove that the federated learning layer is functioning and that the downstream platform layers can now consume a trained shared model.[1][2]

## Intended Audience

This repository is designed for:

- student developers building a professional academic project,
- reviewers who need to understand the FL layer in isolation,
- teammates integrating later explainability or agentic layers,
- evaluators assessing privacy-preserving collaborative training design.

The README is therefore written to explain both the technical flow and the scope limits clearly.

## In conclusion

The project should be considered successful when the following conditions are met:

- all three clients can train locally without schema mismatch,
- no raw transaction records are exchanged across clients,
- the server successfully aggregates multiple rounds of updates,
- a final global model is saved and reloadable,
- the global model is evaluated on a holdout set,
- the results are documented clearly enough for downstream integration.

That outcome corresponds directly to the federated-learning layer described in the platform architecture: decentralized local learning, protected update sharing, centralized aggregation, and a globally improved model ready for the next stage of the system.[1][2]