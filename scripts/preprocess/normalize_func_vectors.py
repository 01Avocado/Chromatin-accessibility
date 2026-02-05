#!/usr/bin/env python3
"""
Normalize functional feature vectors per track/bin.

Steps:
  1. Load func_vectors.npy (N, n_features)
  2. Optionally load feature_map.json to group features by source
  3. Compute mean/std per feature (with epsilon for stability)
  4. Save normalized array as func_vectors.norm.npy (or overwrite when --inplace)
  5. Save stats to JSON for reproducibility
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_feature_groups(func_vectors: np.ndarray, feature_map_path: Path):
    """
    Returns list of (name, indices) for each feature group.
    If feature_map missing, treat each column separately.
    """
    n_features = func_vectors.shape[1]
    if not feature_map_path.exists():
        return [("feature_%d" % i, slice(i, i + 1)) for i in range(n_features)]

    with feature_map_path.open("r") as f:
        fmap = json.load(f)

    n_bins = fmap.get("n_bins", 1)
    groups = []
    for idx, track in enumerate(fmap.get("bigwigs", [])):
        start = idx * n_bins
        end = start + n_bins
        groups.append((track, slice(start, end)))
    return groups


def normalize(func_vectors: np.ndarray, groups, eps: float = 1e-6):
    normalized = func_vectors.copy().astype(np.float32)
    stats = {}

    for name, idx_slice in groups:
        segment = normalized[:, idx_slice]
        mean = segment.mean(axis=0)
        std = segment.std(axis=0)
        std = np.where(std < eps, eps, std)
        normalized[:, idx_slice] = (segment - mean) / std
        stats[name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return normalized, stats


def main(args: argparse.Namespace):
    func_path = Path(args.func_vectors)
    feature_map_path = Path(args.feature_map) if args.feature_map else func_path.with_name("feature_map.json")
    out_path = Path(args.output) if args.output else func_path.with_name("func_vectors.norm.npy")
    stats_path = Path(args.stats_path) if args.stats_path else func_path.with_name("func_vectors.norm.stats.json")

    func_vectors = np.load(func_path)
    groups = load_feature_groups(func_vectors, feature_map_path)
    normalized, stats = normalize(func_vectors, groups, eps=args.eps)

    np.save(out_path, normalized)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Saved normalized vectors to {out_path}")
    print(f"[OK] Saved stats to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize functional feature vectors")
    parser.add_argument("--func_vectors", default="data/processed/funcvecs/func_vectors.npy", help="Path to func_vectors.npy")
    parser.add_argument("--feature_map", default=None, help="Path to feature_map.json")
    parser.add_argument("--output", default=None, help="Output path for normalized numpy file")
    parser.add_argument("--stats_path", default=None, help="Path to save normalization stats JSON")
    parser.add_argument("--eps", type=float, default=1e-6, help="Minimum std deviation")
    args = parser.parse_args()
    main(args)

