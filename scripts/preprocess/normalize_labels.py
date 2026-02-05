#!/usr/bin/env python3
"""
Standardize accessibility labels (mean/std) for training stability.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main(args: argparse.Namespace):
    labels_path = Path(args.labels)
    output_path = Path(args.output) if args.output else labels_path.with_name("accessibility_scores.norm.npy")
    stats_path = Path(args.stats) if args.stats else labels_path.with_name("accessibility_scores.norm.stats.json")

    labels = np.load(labels_path)
    mean = float(labels.mean())
    std = float(labels.std())
    if std < args.eps:
        raise ValueError(f"Standard deviation too small ({std}); cannot normalize.")

    normalized = (labels - mean) / std
    np.save(output_path, normalized.astype(np.float32))

    stats = {"mean": mean, "std": std}
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Saved normalized labels -> {output_path}")
    print(f"[OK] Stats: mean={mean:.6f}, std={std:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize accessibility labels.")
    parser.add_argument("--labels", default="data/processed/labels/accessibility_scores.npy", help="Path to label numpy file.")
    parser.add_argument("--output", help="Output path for normalized labels.")
    parser.add_argument("--stats", help="Where to save normalization stats JSON.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Minimum std to avoid divide-by-zero.")
    main(parser.parse_args())

