import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path.cwd()))

import pandas as pd
import numpy as np
import torch

from src.prediction.predictor import GlobalModelPredictor

def test():
    predictor = GlobalModelPredictor.from_artifacts("artifacts")
    
    # Check predictions on demo_200.csv with Time = 0
    demo_df = pd.read_csv("Sample_datasets/demo_200.csv")
    
    # Actual Time
    res_actual = predictor.classify(demo_df)
    fraud_actual = res_actual[res_actual["Class"] == 1]
    
    # Time = 0
    demo_df_zero_time = demo_df.copy()
    demo_df_zero_time["Time"] = 0.0
    res_zero = predictor.classify(demo_df_zero_time)
    fraud_zero = res_zero[res_zero["Class"] == 1]
    
    print("Fraud risk scores (Actual Time):")
    print(fraud_actual["fraud_risk_score"].describe())
    
    print("\nFraud risk scores (Time = 0):")
    print(fraud_zero["fraud_risk_score"].describe())

if __name__ == "__main__":
    test()
