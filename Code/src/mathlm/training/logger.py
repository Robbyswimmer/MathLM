"""Simple JSONL logger for PPO training metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(record) + "\n")
