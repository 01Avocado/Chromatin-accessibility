#!/usr/bin/env python3
"""
Encode K562 1kb ATAC windows from FASTA + BED into NPZ tensors.

Inputs (expected paths):
  - data/processed/k562_atac_peaks_1kb_windows.bed
      columns: chr  start  end
  - data/processed/k562_sequences_1kb.clean.fa
      1kb sequences in the same order as BED (or very close)
  - data/processed/k562_labels_norm.npy
      accessibility labels per window (length = number of BED rows)

Outputs:
  - data/processed/sequences/encoded_sequences.k562_1kb.npz
      X:    (N, 1000, 4) one‑hot sequences
      meta: (N, 4) [chr, start, end, id]
  - data/processed/labels/accessibility_scores.k562_1kb.npy
      (N,) labels, aligned to X/meta
  - data/processed/labels/splits_k562_1kb_90_5_5.npz
      train/val/test index arrays
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def read_fasta_sequences(fasta_path: Path) -> List[str]:
    sequences: List[str] = []
    cur: List[str] = []
    with fasta_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    sequences.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line)
        if cur:
            sequences.append("".join(cur).upper())
    return sequences


def one_hot_encode(seq: str, target_len: int = 1000) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    if len(seq) > target_len:
        seq = seq[:target_len]
    elif len(seq) < target_len:
        seq = seq + ("N" * (target_len - len(seq)))

    arr = np.zeros((target_len, 4), dtype=np.uint8)
    for i, base in enumerate(seq):
        idx = mapping.get(base)
        if idx is not None:
            arr[i, idx] = 1
    return arr


def create_splits(n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.arange(n, dtype=int)
    rng.shuffle(indices)
    n_train = int(0.90 * n)
    n_val = int(0.05 * n)
    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode K562 1kb windows from FASTA + BED")
    parser.add_argument(
        "--bed",
        type=str,
        default="data/processed/k562_atac_peaks_1kb_windows.bed",
        help="BED file of 1kb windows",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default="data/processed/k562_sequences_1kb.clean.fa",
        help="FASTA with 1kb sequences",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/processed/k562_labels_norm.npy",
        help="Numpy array of labels for each window",
    )
    parser.add_argument(
        "--output_seq_npz",
        type=str,
        default="data/processed/sequences/encoded_sequences.k562_1kb.npz",
        help="Output NPZ for encoded sequences",
    )
    parser.add_argument(
        "--output_labels_npy",
        type=str,
        default="data/processed/labels/accessibility_scores.k562_1kb.npy",
        help="Output NPY for labels aligned to sequences",
    )
    parser.add_argument(
        "--output_splits_npz",
        type=str,
        default="data/processed/labels/splits_k562_1kb_90_5_5.npz",
        help="Output NPZ with train/val/test indices",
    )
    args = parser.parse_args()

    bed_path = Path(args.bed)
    fasta_path = Path(args.fasta)
    labels_path = Path(args.labels)

    print("Loading BED:", bed_path)
    bed_df = pd.read_csv(bed_path, sep="\t", header=None, names=["chr", "start", "end"])
    n_bed = len(bed_df)
    print(f"  BED rows: {n_bed}")

    print("Loading FASTA:", fasta_path)
    seqs = read_fasta_sequences(fasta_path)
    n_fa = len(seqs)
    print(f"  FASTA sequences: {n_fa}")

    print("Loading labels:", labels_path)
    labels = np.load(labels_path)
    n_lab = labels.shape[0]
    print(f"  Labels: {n_lab}")

    # Align by truncating to common length (small mismatch is acceptable)
    n = min(n_bed, n_fa, n_lab)
    if n_bed != n or n_fa != n or n_lab != n:
        print(f"WARNING: length mismatch (bed={n_bed}, fasta={n_fa}, labels={n_lab}), truncating to {n}")

    bed_df = bed_df.iloc[:n].reset_index(drop=True)
    seqs = seqs[:n]
    labels = labels[:n]

    print("Encoding sequences to one-hot...")
    X = np.stack([one_hot_encode(s, 1000) for s in seqs], axis=0)
    print(f"  Encoded X shape: {X.shape}")

    # Build meta: chr, start, end, id (simple incremental id)
    peak_ids = [f"peak_{i+1}" for i in range(n)]
    meta = np.column_stack([bed_df["chr"].astype(object), bed_df["start"], bed_df["end"], np.array(peak_ids, dtype=object)])

    out_seq_npz = Path(args.output_seq_npz)
    out_seq_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_seq_npz, X=X, meta=meta)
    print(f"Saved sequences NPZ -> {out_seq_npz}")

    out_labels = Path(args.output_labels_npy)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_labels, labels.astype(np.float32))
    print(f"Saved labels NPY -> {out_labels} (shape={labels.shape})")

    # Create 90/5/5 splits
    train_idx, val_idx, test_idx = create_splits(n, seed=42)
    out_splits = Path(args.output_splits_npz)
    np.savez(out_splits, train=train_idx, val=val_idx, test=test_idx)
    print(f"Saved splits NPZ -> {out_splits}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    print("Done.")


if __name__ == "__main__":
    main()

