import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

import torch
import numpy as np
from src.models import TabularMLP, TabularTransformer, create_model
from src.models.train_engine import train_one_round, evaluate_client, predict_proba

print("=" * 60)
print("TESTING MODEL + TRAINING ENGINE")
print("=" * 60)

print("\n--- Step 1: Create Models ---")
input_dim = 128

mlp = create_model(input_dim, {"hidden_dims": [64, 32], "dropout": 0.2}, "mlp")
print(f"MLP: {sum(p.numel() for p in mlp.parameters()):,} parameters")

transformer = create_model(input_dim, {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1}, "transformer")
print(f"Transformer: {sum(p.numel() for p in transformer.parameters()):,} parameters")

print("\n--- Step 2: Test Forward Pass ---")
x = torch.randn(32, 128)
mlp_out = mlp(x)
trans_out = transformer(x)
print(f"MLP output: {mlp_out.shape}")
print(f"Transformer output: {trans_out.shape}")

mlp_prob = mlp.predict_proba(x)
print(f"MLP probability: {mlp_prob.shape}, range=[{mlp_prob.min():.4f}, {mlp_prob.max():.4f}]")

print("\n--- Step 3: Test get/set parameters ---")
params = mlp.get_parameters()
print(f"Number of parameter tensors: {len(params)}")

mlp2 = create_model(input_dim, {"hidden_dims": [64, 32], "dropout": 0.2}, "mlp")
mlp2.set_parameters(params)
print("Parameters set successfully")

print("\n--- Step 4: Test Training ---")
X_train = torch.randn(500, 128)
y_train = torch.tensor(np.random.binomial(1, 0.1, 500).astype(np.float32))
X_val = torch.randn(100, 128)
y_val = torch.tensor(np.random.binomial(1, 0.1, 100).astype(np.float32))

device = torch.device("cpu")
train_config = {
    "epochs": 3,
    "batch_size": 64,
    "lr": 0.001,
    "mu": 0.01,
}

test_model = create_model(input_dim, {"hidden_dims": [64, 32], "dropout": 0.2}, "mlp")
updated_params, metrics = train_one_round(test_model, X_train, y_train, train_config, device)
print(f"Training metrics: {metrics}")

print("\n--- Step 5: Test Evaluation ---")
eval_config = {"batch_size": 128}
eval_metrics = evaluate_client(test_model, X_val, y_val, eval_config, device)
print(f"Evaluation metrics:")
for k, v in eval_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n--- Step 6: Test Predict Proba ---")
probs = predict_proba(test_model, X_val, device, batch_size=128)
print(f"Predictions: {probs.shape}, mean={probs.mean():.4f}")

print("\n" + "=" * 60)
print("ALL MODEL TESTS PASSED")
print("=" * 60)