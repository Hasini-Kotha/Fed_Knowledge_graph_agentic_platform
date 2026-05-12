# The Fraud Detection Platform: A Plain-English Guide




## SECTION 1 — OVERALL SYSTEM OVERVIEW

### What problem does this platform solve?
Fraudsters are smart. They don't just attack one bank; they attack multiple banks simultaneously using coordinated networks (fraud rings). If Bank A only looks at its own data, it will completely miss the wider pattern happening at Bank B and Bank C. 

To catch organized fraud, banks need a way to combine their intelligence and see the big picture. 

### Why can't banks just share their raw data?
Because of strict privacy laws (like GDPR and CCPA) and business competition, Bank A is legally forbidden from handing over its customers' private transaction records to Bank B. 

### Enter the Architecture: FL + KG + LLM
To solve this, our platform uses three major technologies:
1. **Federated Learning (FL):** Allows banks to train a shared AI together *without* ever moving or sharing the actual data.
2. **Knowledge Graphs (KG):** Maps out the relationships between transactions to catch organized fraud rings.
3. **LLM Explainability:** Translates complex AI math into plain-English reports for human investigators.

### The End-to-End Flow
Here is the high-level flow of our platform:
**Transactions** (Customers swiping cards at local banks) 
→ **Federated Learning** (Banks collaboratively training an AI) 
→ **Global Model** (The final shared AI brain) 
→ **Risk Prediction** (Scoring new transactions from 0 to 100%) 
→ **Knowledge Graph** (Connecting suspicious transactions together) 
→ **Fraud Communities** (Discovering organized rings) 
→ **Evidence JSON** (Generating hard proof) 
→ **LLM Report** (Creating a human-readable alert).

---

## SECTION 2 — FEDERATED LEARNING


* **Analogy:** Imagine three detectives in different cities trying to catch a serial bank robber. Instead of mailing their highly sensitive case files to a central police chief, the chief mails them a blank "suspect profile notebook" (the AI model). Each detective updates the notebook based on what they've learned locally, and then mails the *updated notebook* back to the chief. The chief averages the notes together and sends the improved notebook back out.

### Central Server vs. Local Bank
In our code, we have a **Central Server** and **Local Banks** (Clients A, B, and C). 
The raw data *never leaves the banks*. Instead, the local banks train their copy of the AI on their own data. During this training, the AI learns patterns (e.g., "transactions at 3 AM for $999 are usually fraud"). 

### What are "Weights"?
When the local bank finishes training, it sends its **weights** to the central server. "Weights" are just mathematical numbers—the internal "knobs and dials" of the AI's brain. They represent *patterns*, not *people*. Because they are just math, it is impossible to reverse-engineer them to see John Doe's credit card swipe.

### How Aggregation Works (FedAvg)
The central server receives the weights from Bank A, Bank B, and Bank C. It uses a mathematical formula called **Federated Averaging (FedAvg)** to literally average the numbers together. This creates a new, smarter "Global Model" that contains the combined intelligence of all three banks. This process repeats for multiple "rounds" until the AI is incredibly smart.

The final result is saved as **`FINAL_global_model.pt`**, which is the fully trained brain ready to catch fraud.

---

## SECTION 3 — DIFFERENTIAL PRIVACY

### Why Differential Privacy (DP) exists
Even though "weights" are just math, extremely advanced hackers can sometimes use "Membership Inference Attacks" to analyze the weights and guess if a specific person's data was used to train the model. 

### What is "Noise Injection"?
To prevent this, we use Differential Privacy. Before Bank A sends its weights to the central server, it injects "mathematical noise" (random static) into the numbers.

### Why adding noise still preserves learning
* **Analogy:** Imagine trying to listen to a symphony orchestra, but someone is rustling a candy wrapper nearby. The rustling is the "noise." Even with the noise, you can still perfectly hear the melody of the symphony. 

In DP, the "symphony" is the massive, undeniable pattern of fraud. The "candy wrapper" is the random noise we add to hide the specific details of individual people. The central server can still learn the general patterns of fraud while mathematically guaranteeing that no individual's privacy is compromised.

---

## SECTION 4 — GLOBAL MODEL PREDICTION FLOW

### How a new transaction enters the system
When a customer swipes their card, the transaction data is fed into the local bank's copy of `FINAL_global_model.pt`. 

### How risk scores are generated
The AI analyzes the transaction and spits out a **Risk Score**—a probability between 0.0 and 1.0. For example, a score of `0.85` means the AI is 85% confident this transaction is fraudulent. 

### The limitation of isolated prediction
The AI model is incredibly fast, but it has a major blind spot: **It only sees one transaction at a time.** 
If a fraudster tests a stolen card with a $1 purchase, the AI might give it a low risk score (e.g., `0.10`) because $1 purchases happen all the time. The AI doesn't realize that the *exact same $1 purchase pattern* just happened at 50 other banks in the last hour. 

To fix this blind spot, we need the Knowledge Graph.

---

## SECTION 5 — KNOWLEDGE GRAPH

### What is a Knowledge Graph (KG)?
A Knowledge Graph is the digital equivalent of a detective's investigation board—the kind with pictures pinned to a corkboard connected by red string. 

### Nodes and Edges
* **Nodes (The Pins):** These are the entities in our system. A node can be a Transaction, a Time Window (e.g., "3:00 AM to 4:00 AM"), or an Amount Bucket (e.g., "Micro-transactions under $5").
* **Edges (The Red String):** These are the relationships connecting the nodes. If Transaction A and Transaction B both occurred at 3:15 AM, we draw a string connecting them to the "3:00 AM Time Window" node.

### Why the KG is NOT an AI model
The KG does not "predict" anything. It does no math to guess if something is fraud. Its only job is to organize relationships. It receives the transactions *after* the AI has already assigned them Risk Scores, and simply maps out how they are connected.

---

## SECTION 6 — SIMILARITY EDGE CREATION

### How transactions become connected
We connect transactions using two types of edges:
1. **Structural Edges:** Hard facts. (e.g., Transaction A and B both share the "Large Amount" bucket).
2. **Similarity Edges:** Deep behavioral patterns. 

### What is Cosine Similarity?
The AI looks at the 28 hidden features of a transaction (V1 through V28). You can think of these features as an arrow pointing in a specific direction in a 28-dimensional space. 

**Cosine Similarity** is a mathematical way of asking, "How closely do these two arrows point in the exact same direction?" If two transactions have a similarity of 0.95, it means their behavioral DNA is 95% identical, even if they happened on different days. When we find high similarity, we draw a "SIMILAR_PATTERN" edge between them on the graph.

---

## SECTION 7 — COMMUNITY DETECTION

### What is the Louvain algorithm?
Once our graph is built and thousands of red strings are connecting our transactions, we run the **Louvain algorithm**. This is a mathematical tool that searches the graph for densely tangled webs of strings—called **Communities**. 

### How fraud rings are discovered
If a group of 50 transactions are all heavily connected to each other by "SIMILAR_PATTERN" edges, Louvain flags them as a Community. In the banking world, a highly-connected community of identical behaviors is usually an organized **Fraud Ring**.

### Risk Enrichment
This is where the magic happens. Remember that $1 test transaction that the AI gave a low risk score of `0.10`? 
The Knowledge Graph notices that this $1 transaction is connected to a community where 10 other transactions have a confirmed fraud score of `0.99`. 

Because of its bad neighborhood, the KG artificially inflates the $1 transaction's risk. It becomes "suspicious by association." This is called **Risk Enrichment**.

---

## SECTION 8 — WHY BOTH FL + KG ARE NEEDED

* **The AI (FL) is the Smoke Detector:** It is instantly reactive. It looks at a single puff of smoke (a single transaction) and sounds the alarm based on known patterns.
* **The Knowledge Graph (KG) is the Arson Investigation Team:** It is slower, but it looks at the whole building. It finds out *who* started the fire and *how* the fires are connected.

If you only have the AI, you catch isolated incidents but miss the organized gangs. 
If you only have the KG, you have a beautiful map of connections, but no way of knowing which connections are actually dangerous. You need both.

---

## SECTION 9 — MOCK DATA VS REAL KAGGLE DATA

### Why testing on small datasets is useful
The real Kaggle credit card dataset has 284,000 rows. Training an AI on that takes a long time. When we are writing code and testing for bugs, we don't want to wait 20 minutes to see if our code works. 

### Stratified Sampling
We use a tool (`run_split.py`) to chop the data down to just 5,000 rows. However, we use **Stratified Sampling**. 
Fraud is incredibly rare (only about 0.17% of the Kaggle dataset). If we just grabbed 5,000 random rows, we might accidentally grab zero fraud cases, and the AI wouldn't be able to learn anything! Stratified sampling ensures that our tiny 5,000-row test dataset has the *exact same 0.17% ratio* of fraud as the massive production dataset.

---

## SECTION 10 — COMPLETE FILE/FOLDER FLOW

Here is what our codebase does when you run the commands:

1. **`Sample_datasets/`**: The folder where you place the raw, downloaded Kaggle CSV file.
2. **`run_split.py`**: Reads the raw CSV, stratifies it, and chops it into private chunks for Bank A, Bank B, Bank C, and a Global Test set. It saves these into the `data/splits/` folder.
3. **`run_fl_simulation.py`**: Wakes up the simulated banks. They train their local AIs, add Differential Privacy noise, and send their weights to the server. The server averages them and saves the result as **`FINAL_global_model.pt`**.
4. **`run_kg_pipeline.py`**: Loads the Global Model, scores the test transactions, and builds the GraphML network. It finds the communities and writes the final evidence to **`top_risk_evidence.json`**.

---

## SECTION 11 — FINAL OUTPUT

### What does `top_risk_evidence.json` contain?
Instead of just handing a human investigator a spreadsheet of numbers, the KG pipeline generates plain-English JSON evidence. For example:
> *"This transaction has a risk score of 0.50. It shares identical feature patterns with 5 other transactions. It belongs to Community #13."*

### How an LLM uses it
A Large Language Model (like GPT-4) thrives on text. In the final layer of our platform, an LLM Agent reads that JSON file and writes a highly professional, contextual security report. 

It translates the cold math into a readable alert: *"Alert: We have detected a coordinated micro-transaction attack originating from Community 13. Transaction ID txn_96 is highly suspicious due to its behavioral similarity to known fraud patterns."*

---

## SECTION 12 — THE COMPLETE END-TO-END STORY

Let's put it all together:

1. **The Swipe:** A criminal swipes a stolen credit card for a tiny $5 purchase, trying to test if the card works. 
2. **The Prediction:** The local bank runs the transaction through the **Global AI Model** (which was trained collaboratively by multiple banks). Because $5 is a normal amount, the AI gives it a low baseline risk score of `0.20`.
3. **The Graph Builds:** Later that day, the transaction is loaded into the **Knowledge Graph**. The Graph draws red strings connecting this transaction to 50 other transactions across the network because their mathematical signatures (Cosine Similarities) match perfectly. 
4. **The Fraud Ring Discovered:** The **Louvain algorithm** scans the graph and realizes that out of those 50 connected transactions, 45 of them were already confirmed as massive $5,000 frauds at other banks. 
5. **Risk Enrichment:** The Graph realizes our $5 transaction is part of this gang. It artificially enriches the risk score from `0.20` up to `0.95`. 
6. **The Explanation:** The Graph packages this discovery into an **Evidence JSON** file.
7. **The Alert:** The **LLM Agent** reads the JSON, writes a clear, understandable report detailing the entire fraud ring, and emails it to the human fraud investigator. 

The investigator clicks a button, blocks the entire network of cards simultaneously, and the bank saves millions of dollars.

Welcome to the future of Agentic AI. Let me know if you have any questions!
