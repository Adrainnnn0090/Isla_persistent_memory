from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isla_memory.config import MemoryConfig
from isla_memory.memory_store import MemoryStore


def main() -> None:
    config = MemoryConfig.from_env()
    db_path = Path(config.db_path)
    if db_path.exists():
        db_path.unlink()
    MemoryStore(config.db_path)
    print(f"Initialized memory database at {config.db_path}")


if __name__ == "__main__":
    main()
