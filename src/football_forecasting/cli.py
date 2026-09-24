"""Offline demonstration and evaluation of a user-supplied match CSV."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .probabilities import run
from .synthetic import synthetic
from .validation import validate_competition


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Evaluate seeded artificial matches offline")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--seed", type=int, default=23)
    evaluate = commands.add_parser("evaluate", help="Chronologically replay a match CSV")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--min-history", type=int, default=80)
    evaluate.add_argument("--half-life", type=float, default=240)
    evaluate.add_argument("--shrink-k", type=float, default=2)
    evaluate.add_argument("--rho", type=float, default=-0.04)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path to preserve the run.")

    if args.command == "demo":
        matches, _, _ = synthetic(100, noise=True, seed=args.seed)
        settings = dict(min_history=80, half_life=120, shrink_k=8, rho=0)
    else:
        matches = pd.read_csv(args.input, dtype={"match_id": str})
        validate_competition(matches, required=True)
        if "match_id" not in matches or matches.match_id.duplicated().any():
            parser.error("Input must have unique match_id values.")
        settings = dict(min_history=args.min_history, half_life=args.half_life,
                        shrink_k=args.shrink_k, rho=args.rho)

    predictions = run(matches, **settings)
    if predictions.empty:
        parser.error("No predictions: input has insufficient history after the warmup.")
    metrics = {**predictions.attrs, "synthetic_only": args.command == "demo",
               "predictions": len(predictions), "settings": settings,
               "interpretation": "Synthetic mechanics only" if args.command == "demo"
               else "Retrospective replay; not an untouched prospective test"}
    source_bytes = (matches.to_csv(index=False).encode("utf-8")
                    if args.command == "demo" else args.input.read_bytes())
    code_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    manifest = {
        "command": args.command,
        "seed": args.seed if args.command == "demo" else None,
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_sha256": code_hashes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ["numpy", "pandas", "scipy"]},
        "settings": settings,
        "cutoff_rule": "Strictly before target UTC calendar day",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.command == "demo":
        matches.to_csv(args.output / "synthetic_matches.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
