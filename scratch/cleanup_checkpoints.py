import os
from pathlib import Path

def cleanup():
    dir_path = Path("artifacts/global_model")
    if not dir_path.exists():
        print("Directory does not exist")
        return
        
    for p in dir_path.glob("*"):
        if p.name in ["model_card.json", "training_history.json"]:
            # Keep model card for structure reference, but let's delete the checkpoints
            continue
        try:
            p.unlink()
            print(f"Deleted: {p.name}")
        except Exception as e:
            print(f"Failed to delete {p.name}: {e}")

if __name__ == "__main__":
    cleanup()
