#!/usr/bin/env python3
"""
Visualize Genomic Regions with Predictions
Creates genome browser-style visualizations showing:
- Genomic coordinates
- DNA sequence
- Predicted vs actual accessibility
- Functional signal tracks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def onehot_to_sequence(onehot_seq):
    """Convert one-hot encoded sequence to nucleotide string."""
    nucleotide_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq = []
    for pos in onehot_seq:
        nucleotide_idx = np.argmax(pos)
        if np.sum(pos) == 0:
            seq.append('N')
        else:
            seq.append(nucleotide_map[nucleotide_idx])
    return ''.join(seq)

def _read_fasta_sequences(fasta_path: Path) -> Tuple[list[str], list[str]]:
    """Read sequences from a FASTA file. Returns (headers, sequences)."""
    headers: list[str] = []
    seqs: list[str] = []
    cur_header: Optional[str] = None
    cur_seq_parts: list[str] = []

    with open(fasta_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_header is not None:
                    headers.append(cur_header)
                    seqs.append("".join(cur_seq_parts).upper())
                cur_header = line[1:]
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)

    if cur_header is not None:
        headers.append(cur_header)
        seqs.append("".join(cur_seq_parts).upper())

    return headers, seqs


def _parse_header_to_meta(header: str) -> Tuple[str, int, int, str]:
    """
    Parse header formats produced by our preprocessing, e.g.:
      peak_1_chr1:9750-11250
      peak_1_chr1:9750-11250 (anything after '>' is header)
    Returns (chr, start, end, id).
    """
    # Split at last "_" to separate peak_id and chrom/coords (works for peak_XXX)
    # Fallback: treat everything before first "_" as id.
    if "_" in header:
        peak_id, rest = header.split("_", 1)
        # In our data the id is like peak_1, peak_2,...
        # So if rest starts with digit, keep peak_id + "_" + rest-part?
        # Safer: id is everything up to first ":"
        before_colon = header.split(":", 1)[0]
        peak_id = before_colon.rsplit("_", 1)[0] if "_" in before_colon else before_colon
        chr_part = before_colon.split("_")[-1]
    else:
        peak_id = header.split(":", 1)[0]
        chr_part = peak_id

    coords_part = header.split(":", 1)[1] if ":" in header else ""
    if "-" in coords_part:
        start_s, end_s = coords_part.split("-", 1)
        try:
            start = int(start_s.replace(",", ""))
            end = int(end_s.replace(",", ""))
        except ValueError:
            start, end = 0, 0
    else:
        start, end = 0, 0

    return chr_part, start, end, peak_id


def _one_hot_encode_sequence(seq: str, target_length: int) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    if len(seq) > target_length:
        seq = seq[:target_length]
    elif len(seq) < target_length:
        seq = seq + ("N" * (target_length - len(seq)))

    arr = np.zeros((target_length, 4), dtype=np.uint8)
    for i, base in enumerate(seq):
        idx = mapping.get(base)
        if idx is not None:
            arr[i, idx] = 1
    return arr


def load_all_data(data_dir="data/processed", results_dir="results/evaluation", use_filtered: bool = False, split_file: str | None = None):
    """Load all necessary data for visualization."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)

    # Choose sequence source
    seq_npz = data_dir / "sequences" / ("encoded_sequences.filtered.npz" if use_filtered else "encoded_sequences.npz")
    sequences = None
    meta = None

    if seq_npz.exists():
        seq_data = np.load(seq_npz, allow_pickle=True)
        sequences = seq_data["X"]
        meta = seq_data["meta"]
    else:
        # Fallback: reconstruct from FASTA if present
        fasta_path = data_dir / "sequences" / "full_sequences.fa"
        if not fasta_path.exists():
            raise FileNotFoundError(f"Could not find sequences NPZ or FASTA. Missing: {seq_npz} and {fasta_path}")

        headers, seqs = _read_fasta_sequences(fasta_path)
        target_len = max((len(s) for s in seqs), default=1500)
        sequences = np.stack([_one_hot_encode_sequence(s, target_len) for s in seqs], axis=0)
        meta_rows = [_parse_header_to_meta(h) for h in headers]
        meta = np.array(meta_rows, dtype=object)

        if use_filtered:
            # If we have kept_indices, filter down to the 93-seq subset
            summary_path = data_dir / "sequences" / "filter_summary.json"
            if summary_path.exists():
                kept = json.load(open(summary_path, "r")).get("kept_indices")
                if kept:
                    kept_idx = np.array(kept, dtype=int)
                    sequences = sequences[kept_idx]
                    meta = meta[kept_idx]
    
    # Load predictions and labels
    predictions = None
    labels = None
    pred_npz = results_dir / "predictions_and_labels.npz"
    if pred_npz.exists():
        pred_data = np.load(pred_npz)
        predictions = pred_data["predictions"]
        labels = pred_data["labels"]
    else:
        # Fallback: if a predictions_table.csv exists, we can load from it
        pred_csv = results_dir / "predictions_table.csv"
        if pred_csv.exists():
            df = pd.read_csv(pred_csv)
            if {"predicted_accessibility", "actual_accessibility"}.issubset(df.columns):
                predictions = df["predicted_accessibility"].to_numpy()
                labels = df["actual_accessibility"].to_numpy()
    
    # Load test indices
    if split_file is None:
        split_file = "test_indices_filtered.txt" if use_filtered else "test_indices.txt"
    test_path = data_dir / "labels" / split_file
    if test_path.exists():
        test_indices = np.loadtxt(test_path, dtype=int)
        test_indices = np.atleast_1d(test_indices).astype(int)
    else:
        # Fallback to empty: script can still render sequence-only plots
        test_indices = np.array([], dtype=int)
    
    # Load functional vectors
    func_vectors = None
    func_dir = data_dir / "funcvecs"
    if func_dir.exists():
        # Prefer filtered if available
        candidates = [
            func_dir / "func_vectors.norm.filtered.npy",
            func_dir / "func_vectors.filtered.npy",
            func_dir / "func_vectors.norm.npy",
            func_dir / "func_vectors.npy",
        ]
        for p in candidates:
            if p.exists():
                func_vectors = np.load(p)
                break
    
    # Load feature map
    with open(data_dir / "funcvecs" / "feature_map.json", 'r') as f:
        feature_map = json.load(f)
    
    # Load BED file for coordinates
    bed_df = pd.read_csv(data_dir / "peaks" / "full_peaks.bed", sep='\t', 
                        header=None, names=['chr', 'start', 'end', 'id'], comment='#')
    
    return {
        'sequences': sequences,
        'meta': meta,
        'predictions': predictions,
        'labels': labels,
        'test_indices': test_indices,
        'func_vectors': func_vectors,
        'feature_map': feature_map,
        'bed_df': bed_df
    }

def plot_genomic_region(region_idx, data, output_dir, show_sequence=True, max_seq_display=200):
    """
    Create a comprehensive visualization for a single genomic region.
    
    Args:
        region_idx: Index of the region in the full dataset
        data: Dictionary containing all loaded data
        output_dir: Directory to save the plot
        show_sequence: Whether to show DNA sequence
        max_seq_display: Maximum sequence length to display (for readability)
    """
    sequences = data['sequences']
    meta = data['meta']
    predictions = data['predictions']
    labels = data['labels']
    func_vectors = data['func_vectors']
    feature_map = data['feature_map']
    test_indices = data['test_indices']
    
    # Get region data
    seq = sequences[region_idx]
    chr_name = str(meta[region_idx, 0])
    start = int(meta[region_idx, 1])
    end = int(meta[region_idx, 2])
    peak_id = str(meta[region_idx, 3])
    
    # Get predictions (if in test set)
    pred_score = None
    true_score = None
    if predictions is not None and labels is not None and region_idx in test_indices:
        test_pos = np.where(test_indices == region_idx)[0][0]
        pred_score = predictions[test_pos]
        true_score = labels[test_pos]
    
    # Get functional signals
    func_signals = None
    n_bins = feature_map.get('n_bins', 16)
    n_tracks = len(feature_map.get('bigwigs', []))
    if func_vectors is not None:
        func_vec = func_vectors[region_idx]  # (80,) -> reshape to (5, 16)
        try:
            func_signals = func_vec.reshape(n_tracks, n_bins)
        except Exception:
            func_signals = None
    
    # Convert sequence
    dna_seq = onehot_to_sequence(seq)
    seq_len = len(dna_seq)
    
    # Create figure with multiple subplots
    has_tracks = func_signals is not None
    if has_tracks:
        # 1 title + 1 sequence + 1 accessibility + 5 tracks = 8 rows
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(8, 1, height_ratios=[0.5, 1, 1, 1, 1, 1, 1, 1], hspace=0.4)
    else:
        # 1 title + 1 sequence + 1 accessibility = 3 rows
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(3, 1, height_ratios=[0.5, 1, 1], hspace=0.4)
    
    # 1. Title and coordinates
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis('off')
    title_text = f"Genomic Region: {chr_name}:{start:,}-{end:,} ({end-start:,} bp)\n"
    title_text += f"Peak ID: {peak_id}"
    if pred_score is not None:
        title_text += f" | Predicted: {pred_score:.4f} | Actual: {true_score:.4f} | Error: {abs(pred_score-true_score):.4f}"
    ax_title.text(0.5, 0.5, title_text, ha='center', va='center', 
                  fontsize=14, fontweight='bold', transform=ax_title.transAxes)
    
    # 2. DNA Sequence (colored by nucleotide)
    ax_seq = fig.add_subplot(gs[1, 0])
    if show_sequence and seq_len <= max_seq_display:
        # Display full sequence
        display_seq = dna_seq
        x_positions = np.arange(len(display_seq))
    else:
        # Show summary or truncated
        display_seq = dna_seq[:max_seq_display] + "..." if seq_len > max_seq_display else dna_seq
        x_positions = np.arange(len(display_seq))
    
    # Color mapping for nucleotides
    colors = {'A': '#FF6B6B', 'C': '#4ECDC4', 'G': '#45B7D1', 'T': '#FFA07A', 'N': '#CCCCCC'}
    
    for i, base in enumerate(display_seq):
        ax_seq.bar(i, 1, color=colors.get(base, '#CCCCCC'), width=0.8, edgecolor='black', linewidth=0.1)
    
    ax_seq.set_xlim(-0.5, len(display_seq) - 0.5)
    ax_seq.set_ylim(0, 1.2)
    ax_seq.set_ylabel('Sequence', fontweight='bold')
    ax_seq.set_title('DNA Sequence (A=Red, C=Teal, G=Blue, T=Orange, N=Gray)', fontsize=10)
    ax_seq.set_xticks([])
    ax_seq.set_yticks([])
    
    # Add position labels every 100 bp
    if len(display_seq) > 100:
        tick_positions = np.arange(0, len(display_seq), 100)
        tick_labels = [f"{start + pos}" for pos in tick_positions]
        ax_seq.set_xticks(tick_positions)
        ax_seq.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        ax_seq.set_xlabel('Genomic Position', fontsize=9)
    
    # 3. Accessibility Scores
    ax_acc = fig.add_subplot(gs[2, 0])
    if pred_score is not None:
        # Create a bar showing predicted vs actual
        ax_acc.barh(0, pred_score, height=0.3, color='blue', alpha=0.7, label=f'Predicted: {pred_score:.4f}')
        ax_acc.barh(0.5, true_score, height=0.3, color='red', alpha=0.7, label=f'Actual: {true_score:.4f}')
        ax_acc.set_xlim(0, max(pred_score, true_score) * 1.2 if max(pred_score, true_score) > 0 else 1.0)
        ax_acc.set_yticks([0, 0.5])
        ax_acc.set_yticklabels(['Predicted', 'Actual'])
        ax_acc.set_xlabel('Accessibility Score', fontweight='bold')
        ax_acc.legend(loc='upper right')
        ax_acc.set_title('Model Predictions', fontsize=10)
        ax_acc.grid(True, alpha=0.3, axis='x')
    else:
        ax_acc.text(0.5, 0.5, 'Not in test set', ha='center', va='center', 
                   transform=ax_acc.transAxes, fontsize=12, style='italic')
        ax_acc.set_xticks([])
        ax_acc.set_yticks([])
    
    if has_tracks:
        # 4-8. Functional Signal Tracks
        track_names = ['ATAC-seq', 'H3K27ac', 'H3K27me3', 'H3K4me3', 'CTCF']
        track_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#9B59B6']
        
        for i, (track_name, color) in enumerate(zip(track_names, track_colors)):
            ax_track = fig.add_subplot(gs[3 + i, 0])  # 3, 4, 5, 6, 7
            
            # Get signal for this track
            signal = func_signals[i, :]  # (16,)
            positions = np.linspace(start, end, n_bins)
            
            # Plot signal
            ax_track.fill_between(positions, 0, signal, color=color, alpha=0.6, step='mid')
            ax_track.plot(positions, signal, color=color, linewidth=2, marker='o', markersize=4)
            ax_track.axhline(0, color='black', linestyle='--', linewidth=0.5)
            
            ax_track.set_ylabel(track_name, fontweight='bold', fontsize=9)
            ax_track.set_xlim(start, end)
            ax_track.grid(True, alpha=0.3, axis='y')
            
            if i == len(track_names) - 1:
                ax_track.set_xlabel('Genomic Position', fontweight='bold')
            else:
                ax_track.set_xticklabels([])
    
    plt.suptitle(f'Genomic Region Visualization: {chr_name}:{start:,}-{end:,}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"region_{region_idx}_{chr_name}_{start}_{end}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / filename}")
    plt.close()

def create_summary_dashboard(data, output_dir, n_regions=10):
    """Create a summary dashboard showing multiple regions."""
    test_indices = data['test_indices']
    predictions = data['predictions']
    labels = data['labels']
    func_vectors = data.get("func_vectors")
    feature_map = data.get("feature_map", {})
    
    # Select top regions by prediction error (or other criteria)
    if predictions is None or labels is None or len(test_indices) == 0:
        print("No predictions/test indices available; skipping summary dashboard.")
        return

    errors = np.abs(predictions - labels)
    n_regions = min(int(n_regions), len(errors))
    top_error_indices = np.argsort(errors)[-n_regions:][::-1]  # Top N by error
    
    fig, axes = plt.subplots(n_regions, 1, figsize=(16, 2 * n_regions))
    if n_regions == 1:
        axes = [axes]
    
    for idx, (ax, test_pos) in enumerate(zip(axes, top_error_indices)):
        region_idx = test_indices[test_pos]
        meta = data['meta']
        chr_name = str(meta[region_idx, 0])
        start = int(meta[region_idx, 1])
        end = int(meta[region_idx, 2])
        
        # Plot ATAC signal if functional vectors available
        if func_vectors is not None:
            func_vec = func_vectors[region_idx]
            n_bins = int(feature_map.get("n_bins", 16))
            n_tracks = len(feature_map.get("bigwigs", []))
            try:
                func_signals = func_vec.reshape(n_tracks, n_bins)
                positions = np.linspace(start, end, n_bins)
                ax.fill_between(positions, 0, func_signals[0, :], alpha=0.6, color='red')
                ax.plot(positions, func_signals[0, :], color='darkred', linewidth=1.5)
            except Exception:
                pass
        
        # Add prediction info
        pred = predictions[test_pos]
        true = labels[test_pos]
        error = abs(pred - true)
        
        ax.text(0.02, 0.95, f"{chr_name}:{start:,}-{end:,} | Pred: {pred:.3f} | True: {true:.3f} | Err: {error:.3f}",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('ATAC Signal', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        if idx < n_regions - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Genomic Position', fontsize=9)
    
    plt.suptitle(f'Top {n_regions} Regions by Prediction Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "summary_dashboard.png", dpi=300, bbox_inches='tight')
    print(f"Saved summary dashboard: {output_dir / 'summary_dashboard.png'}")
    plt.close()

def create_predictions_table(data, output_dir):
    """Create a CSV table with all predictions and genomic coordinates."""
    test_indices = data['test_indices']
    predictions = data['predictions']
    labels = data['labels']
    meta = data['meta']

    if predictions is None or labels is None:
        print("No predictions available (missing predictions_and_labels.npz). Skipping predictions_table.csv.")
        return None
    
    results = []
    for test_pos, region_idx in enumerate(test_indices):
        chr_name = str(meta[region_idx, 0])
        start = int(meta[region_idx, 1])
        end = int(meta[region_idx, 2])
        peak_id = str(meta[region_idx, 3])
        
        results.append({
            'region_index': region_idx,
            'peak_id': peak_id,
            'chromosome': chr_name,
            'start': start,
            'end': end,
            'length': end - start,
            'predicted_accessibility': predictions[test_pos],
            'actual_accessibility': labels[test_pos],
            'error': abs(predictions[test_pos] - labels[test_pos]),
            'genomic_coordinate': f"{chr_name}:{start}-{end}"
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('error', ascending=False)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions table: {csv_path}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Visualize genomic regions with predictions")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--results_dir", type=str, default="results/evaluation",
                       help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="results/visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--use_filtered", action="store_true",
                       help="Use filtered dataset (93 sequences) if available")
    parser.add_argument("--split_file", type=str, default=None,
                       help="Override test indices filename (default: test_indices_filtered.txt when --use_filtered)")
    parser.add_argument("--region_idx", type=int, default=None,
                       help="Specific region index to visualize (if None, creates summary)")
    parser.add_argument("--n_regions", type=int, default=10,
                       help="Number of regions for summary dashboard")
    parser.add_argument("--all_test", action="store_true",
                       help="Create visualization for all test regions")
    parser.add_argument("--sample", type=int, default=5,
                       help="Number of sample regions to visualize")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Genomic Region Visualization")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data(args.data_dir, args.results_dir, use_filtered=args.use_filtered, split_file=args.split_file)
    print(f"  Loaded {len(data['sequences'])} regions")
    print(f"  Test set size: {len(data['test_indices'])}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create predictions table
    print("\nCreating predictions table...")
    df = create_predictions_table(data, output_dir)
    if df is not None:
        print(f"  Created table with {len(df)} regions")
    
    if args.region_idx is not None:
        # Visualize specific region
        print(f"\nVisualizing region {args.region_idx}...")
        plot_genomic_region(args.region_idx, data, output_dir)
    elif args.all_test:
        # Visualize all test regions
        print(f"\nVisualizing all {len(data['test_indices'])} test regions...")
        for i, region_idx in enumerate(data['test_indices']):
            print(f"  Processing {i+1}/{len(data['test_indices'])}: region {region_idx}")
            plot_genomic_region(region_idx, data, output_dir, show_sequence=False)
    else:
        # Create summary dashboard
        print(f"\nCreating summary dashboard with {args.n_regions} regions...")
        create_summary_dashboard(data, output_dir, args.n_regions)
        
        # Also create a few individual visualizations
        print(f"\nCreating {args.sample} sample individual visualizations...")
        sample_indices = data['test_indices'][:args.sample]  # First N test regions
        for region_idx in sample_indices:
            plot_genomic_region(region_idx, data, output_dir, show_sequence=True, max_seq_display=150)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
