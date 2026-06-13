import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gateway.database import init_db

if __name__ == "__main__":
    print("Starting database schema migrations ...")
    init_db()
    print("Database migrations applied successfully!")
