#!/usr/bin/env python3
"""
Create synthetic accessibility labels for testing model architecture.

This script generates temporary labels to allow Step 1 implementation.
For production, replace with real ATAC-seq data from ENCODE.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def create_synthetic_labels(n_sequences, method='gaussian', seed=42):
    """
    Generate synthetic accessibility scores for sequences.
    
    Args:
        n_sequences: Number of sequences to generate labels for
        method: 'gaussian', 'binary', or 'pattern'
        seed: Random seed for reproducibility
    
    Returns:
        labels: Array of accessibility scores (0-1 range)
    """
    np.random.seed(seed)
    
    if method == 'gaussian':
        # Generate values from normal distribution, clip to [0, 1]
        labels = np.random.normal(0.5, 0.2, n_sequences)
        labels = np.clip(labels, 0.0, 1.0)
        
    elif method == 'binary':
        # Binary classification: accessible (1) vs non-accessible (0)
        labels = np.random.choice([0, 1], size=n_sequences, p=[0.3, 0.7])
        
    elif method == 'pattern':
        # Create a pattern: higher values for some sequences
        labels = np.random.beta(2, 5, n_sequences)  # Skewed distribution
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels.astype(np.float32)


def create_splits(n_sequences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create train/validation/test splits.
    
    Args:
        n_sequences: Total number of sequences
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
    
    Returns:
        splits: Dictionary with 'train', 'val', 'test' indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    indices = np.random.permutation(n_sequences)
    
    n_train = int(n_sequences * train_ratio)
    n_val = int(n_sequences * val_ratio)
    
    splits = {
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:]
    }
    
    return splits


def main(args):
    print("Creating synthetic accessibility labels...")
    print(f"Method: {args.method}")
    print(f"Output directory: {args.output_dir}")
    
    # Load encoded sequences to get count
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    n_sequences = data['X'].shape[0]
    print(f"Found {n_sequences} sequences")
    
    # Create synthetic labels
    print(f"\nGenerating synthetic labels ({args.method} method)...")
    labels = create_synthetic_labels(n_sequences, method=args.method, seed=args.seed)
    print(f"Label statistics:")
    print(f"  Mean: {labels.mean():.4f}")
    print(f"  Std: {labels.std():.4f}")
    print(f"  Min: {labels.min():.4f}")
    print(f"  Max: {labels.max():.4f}")
    
    # Create train/val/test splits
    print(f"\nCreating train/val/test splits...")
    splits = create_splits(
        n_sequences,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"  Train: {len(splits['train'])} sequences ({len(splits['train'])/n_sequences*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} sequences ({len(splits['val'])/n_sequences*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} sequences ({len(splits['test'])/n_sequences*100:.1f}%)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save labels
    labels_path = output_dir / "synthetic_labels.npy"
    np.save(labels_path, labels)
    print(f"\nSaved labels to: {labels_path}")
    
    # Save splits
    splits_path = output_dir / "splits.npz"
    np.savez(splits_path, **splits)
    print(f"Saved splits to: {splits_path}")
    
    # Save split information as text files (for reference)
    for split_name, indices in splits.items():
        split_file = output_dir / f"{split_name}_indices.txt"
        np.savetxt(split_file, indices, fmt='%d')
        print(f"Saved {split_name} indices to: {split_file}")
    
    # Create summary
    summary = {
        'total_sequences': n_sequences,
        'label_method': args.method,
        'label_statistics': {
            'mean': float(labels.mean()),
            'std': float(labels.std()),
            'min': float(labels.min()),
            'max': float(labels.max())
        },
        'splits': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        }
    }
    
    print(f"\nSUCCESS: Synthetic labels created successfully!")
    print(f"\nNOTE: These are temporary labels for testing.")
    print(f"For production, replace with real ATAC-seq data from ENCODE.")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create synthetic accessibility labels for model testing"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/sequences/encoded_sequences.npz",
        help="Path to encoded sequences NPZ file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/labels",
        help="Output directory for labels and splits"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gaussian",
        choices=['gaussian', 'binary', 'pattern'],
        help="Method for generating synthetic labels"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction of data for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    main(args)

