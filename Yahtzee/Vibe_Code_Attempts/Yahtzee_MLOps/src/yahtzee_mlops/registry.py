"""Tiny local model registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch


@dataclass(frozen=True)
class RegisteredModel:
    run_id: str
    checkpoint_path: Path
    metrics_path: Path


class LocalModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def save(self, checkpoint: dict[str, object], metrics: dict[str, object], run_id: str | None = None) -> RegisteredModel:
        run_id = run_id or self.create_run_id()
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = run_dir / "model.pt"
        metrics_path = run_dir / "metrics.json"
        torch.save(checkpoint, checkpoint_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        shutil.copy2(checkpoint_path, self.root / "latest.pt")
        shutil.copy2(metrics_path, self.root / "latest_metrics.json")
        return RegisteredModel(run_id=run_id, checkpoint_path=checkpoint_path, metrics_path=metrics_path)

    def latest_checkpoint(self) -> Path | None:
        path = self.root / "latest.pt"
        return path if path.exists() else None
