import sys
import sqlite3
from pathlib import Path

# Add current working directory to sys.path
sys.path.insert(0, str(Path.cwd()))

def read_db():
    conn = sqlite3.connect("artifacts/gateway/fl_gateway.db")
    cursor = conn.cursor()
    
    print("--- Rounds ---")
    cursor.execute("SELECT round_id, round_number, started_at, completed_at, participating_clients FROM rounds")
    for row in cursor.fetchall():
        print(row)
        
    print("\n--- Submissions ---")
    cursor.execute("SELECT submission_id, client_id, round_number, submitted_at, validation_status, rejection_reason FROM submissions")
    for row in cursor.fetchall():
        print(row)
        
    print("\n--- Model Registry ---")
    cursor.execute("SELECT round_number, model_version, global_pr_auc, optimal_threshold, encrypted_snapshot_path FROM model_registry")
    for row in cursor.fetchall():
        print(row)
        
    conn.close()

if __name__ == "__main__":
    read_db()
