import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("=" * 60)
print("TESTING FEDERATED LEARNING SIMULATION")
print("=" * 60)

from src.core.metadata_engine import MetadataMapper
from src.core.vectorizer import DynamicVectorizer
from src.models import create_model
from src.models.train_engine import train_one_round, evaluate_client

mapper = MetadataMapper('configs/mapping.json')
vectorizer = DynamicVectorizer.load('artifacts/global_vectorizer_kaggle.pkl')

import pandas as pd

client_data = {}
client_ids = ['client_a', 'client_b', 'client_c']

for cid in client_ids:
    train_df = pd.read_csv(f'data/splits/{cid}_train.csv')
    val_df = pd.read_csv(f'data/splits/{cid}_val.csv')
    
    if cid == 'client_a':
        train_result = vectorizer.fit_transform(train_df, mapper)
    else:
        train_result = vectorizer.transform(train_df, mapper)
    
    X_train = train_result["data"] if isinstance(train_result, dict) else train_result
    y_train = torch.tensor(train_df[mapper.get_target_column()].values.astype(np.float32))
    
    val_result = vectorizer.transform(val_df, mapper)
    X_val = val_result["data"] if isinstance(val_result, dict) else val_result
    y_val = torch.tensor(val_df[mapper.get_target_column()].values.astype(np.float32))
    
    client_data[cid] = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}
    print(f"{cid}: train={X_train.shape}, val={X_val.shape}")

print("\n--- Manual FL Simulation (3 rounds) ---")

model_config = {"hidden_dims": [64, 32], "dropout": 0.2}
train_config = {"epochs": 2, "batch_size": 256, "lr": 0.001, "mu": 0.01}
device = torch.device("cpu")
input_dim = vectorizer.get_feature_dim()

global_model = create_model(input_dim, model_config, "mlp")
global_params = [p.clone().detach() for p in global_model.parameters()]

for round_num in range(1, 4):
    print(f"\n--- Round {round_num} ---")
    
    fit_results = []
    for cid in client_ids:
        local_model = create_model(input_dim, model_config, "mlp").to(device)
        local_model.set_parameters(global_params)
        
        tc = {**train_config, "round": round_num}
        updated, metrics = train_one_round(local_model, client_data[cid]['X_train'], client_data[cid]['y_train'], tc, device)
        
        n_samples = len(client_data[cid]['X_train'])
        fit_results.append((cid, updated, n_samples, metrics))
        print(f"  {cid}: loss={metrics['train_loss']:.4f}")
    
    total_samples = sum(r[2] for r in fit_results)
    aggregated = None
    for _, params, n_samples, _ in fit_results:
        weight = n_samples / total_samples
        if aggregated is None:
            aggregated = [p * weight for p in params]
        else:
            aggregated = [a + p * weight for a, p in zip(aggregated, params)]
    
    global_params = aggregated
    global_model.set_parameters(global_params)
    
    eval_results = []
    for cid in client_ids:
        eval_model = create_model(input_dim, model_config, "mlp").to(device)
        eval_model.set_parameters(global_params)
        
        ec = {"batch_size": 512}
        em = evaluate_client(eval_model, client_data[cid]['X_val'], client_data[cid]['y_val'], ec, device)
        eval_results.append((cid, em, len(client_data[cid]['X_val'])))
    
    round_metrics = {}
    total_eval = sum(r[2] for r in eval_results)
    for key in ["roc_auc", "pr_auc", "f1"]:
        ws = sum(r[1].get(key, 0) * r[2] for r in eval_results)
        round_metrics[key] = ws / total_eval
    
    print(f"  Global: roc_auc={round_metrics['roc_auc']:.4f}, pr_auc={round_metrics['pr_auc']:.4f}, f1={round_metrics['f1']:.4f}")

print("\n" + "=" * 60)
print("FL SIMULATION TEST PASSED")
print("=" * 60)
