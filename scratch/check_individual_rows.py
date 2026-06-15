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
    demo_df = pd.read_csv("Sample_datasets/demo_200.csv")
    
    # Predict directly
    res_actual = predictor.classify(demo_df)
    
    # Predict with Time=0 and Class=0
    records = []
    _FEATURE_COLS = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                     "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                     "V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
    for idx, row in demo_df.iterrows():
        rec = {"Time": 0.0, "Class": 0}
        for col in _FEATURE_COLS:
            rec[col] = float(row[col])
        records.append(rec)
    df_zero = pd.DataFrame(records)
    res_zero = predictor.classify(df_zero)
    
    for i in range(10):
        print(f"Row {i+1} | Class: {demo_df['Class'].iloc[i]} | Amount: {demo_df['Amount'].iloc[i]:.2f}")
        print(f"  Actual prediction: {res_actual['fraud_risk_score'].iloc[i]*100:.2f}%")
        print(f"  Zero-time/Class prediction: {res_zero['fraud_risk_score'].iloc[i]*100:.2f}%")

if __name__ == "__main__":
    test()
