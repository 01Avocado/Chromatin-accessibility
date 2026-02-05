#!/usr/bin/env python3
"""
Extract accessibility scores from ATAC-seq bigWig file for peak regions.

This script reads the ATAC-seq signal track (bigWig) and extracts accessibility
scores for each peak region in your BED file.
"""

import numpy as np
import pandas as pd
import pyBigWig
from pathlib import Path
import argparse

def extract_bigwig_scores(bed_file, bigwig_file, method='mean'):
    """
    Extract scores from bigWig file for regions in BED file.
    
    Args:
        bed_file: Path to BED file with peaks
        bigwig_file: Path to bigWig signal file
        method: 'mean', 'max', 'sum', or 'binned'
    
    Returns:
        scores: Array of accessibility scores per region
    """
    print(f"Loading BED file: {bed_file}")
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, 
                        names=['chr', 'start', 'end', 'id'],
                        comment='#')
    
    print(f"Opening bigWig file: {bigwig_file}")
    bw = pyBigWig.open(bigwig_file)
    
    if bw is None:
        raise FileNotFoundError(f"Could not open bigWig file: {bigwig_file}")
    
    scores = []
    total = len(bed_df)
    
    print(f"Extracting scores for {total} regions (method: {method})...")
    
    for idx, row in bed_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{total} regions...")
        
        try:
            # Get values for this region
            values = bw.values(row['chr'], row['start'], row['end'])
            
            # Handle None values (regions not in bigWig)
            values = [v if v is not None else 0.0 for v in values]
            
            if not values or all(v == 0 for v in values):
                # Region not covered, assign 0
                scores.append(0.0)
                continue
            
            # Calculate score based on method
            if method == 'mean':
                score = np.mean(values)
            elif method == 'max':
                score = np.max(values)
            elif method == 'sum':
                score = np.sum(values)
            elif method == 'binned':
                # Divide into bins and average
                n_bins = 16
                bin_size = len(values) // n_bins
                binned = [np.mean(values[i:i+bin_size]) 
                         for i in range(0, len(values), bin_size)]
                score = np.mean(binned)
            else:
                score = np.mean(values)
            
            scores.append(float(score))
            
        except Exception as e:
            print(f"  Warning: Could not process {row['chr']}:{row['start']}-{row['end']}: {e}")
            scores.append(0.0)
    
    bw.close()
    
    scores = np.array(scores, dtype=np.float32)
    print(f"\nExtracted {len(scores)} scores")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    
    return scores


def main(args):
    print("=" * 60)
    print("Extract Accessibility Scores from ATAC-seq bigWig")
    print("=" * 60)
    
    # Check files exist
    bed_path = Path(args.bed_file)
    bigwig_path = Path(args.bigwig_file)
    
    if not bed_path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_path}")
    if not bigwig_path.exists():
        raise FileNotFoundError(f"bigWig file not found: {bigwig_path}")
    
    # Extract scores
    scores = extract_bigwig_scores(bed_path, bigwig_path, method=args.method)
    
    # Save scores
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, scores)
    print(f"\n✅ Saved accessibility scores to: {output_path}")
    print(f"   Shape: {scores.shape}")
    print(f"   Format: numpy float32 array")
    
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract accessibility scores from ATAC-seq bigWig for peaks"
    )
    parser.add_argument(
        "--bed_file",
        type=str,
        default="data/processed/peaks/full_peaks.bed",
        help="Input BED file with peak coordinates"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="data/raw/encode/atac_seq/K562/atac_signal.bigWig",
        help="ATAC-seq bigWig signal file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/labels/accessibility_scores.npy",
        help="Output file for accessibility scores"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mean",
        choices=['mean', 'max', 'sum', 'binned'],
        help="Method to aggregate scores per region"
    )
    
    args = parser.parse_args()
    main(args)

