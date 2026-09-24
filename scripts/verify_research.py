"""Recompute published pooled metrics from archived predictions, without retraining."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    research = root / "research"
    source = research / "predictions.csv"
    manifest = json.loads((research / "source_manifest.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["archived_predictions"]["sha256"]:
        raise ValueError("Archived prediction checksum mismatch")
    frame = pd.read_csv(source)
    summary = json.loads((research / "summary.json").read_text())
    if len(frame) != 1140 or frame.match_id.duplicated().any():
        raise ValueError("Expected 1,140 unique historical predictions")
    if not frame.outcome.isin([0, 1, 2]).all():
        raise ValueError("Invalid outcome labels")
    y = frame.outcome.to_numpy(int)
    observed = np.eye(3)[y]
    report = {}
    for label, prefix in [("pooled_baseline", ""), ("pooled_xgboost", "xgb_")]:
        p = frame[[prefix + c for c in ["p_home", "p_draw", "p_away"]]].to_numpy(float)
        # Original CSV serializes XGBoost float32 probabilities. Preserve its
        # stored values and the original scoring convention, allowing rounding.
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Invalid archived probabilities")
        if not np.allclose(p.sum(axis=1), 1, atol=2e-7, rtol=0):
            raise ValueError("Archived probabilities do not sum to one")
        metrics = {
            "n": len(y),
            "rps": float(np.mean((p.cumsum(1)[:, :2] - observed.cumsum(1)[:, :2]) ** 2)),
            "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
            "accuracy": float((p.argmax(1) == y).mean()),
        }
        for key, value in metrics.items():
            if not np.isclose(value, summary[label][key], atol=1e-7, rtol=0):
                raise ValueError(f"Published metric mismatch: {label}.{key}")
        report[label] = metrics
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
