#!/usr/bin/env python3
"""
preprocess.py
--------------
Preprocessing pipeline for EpiBERT-style genomic modeling.
Steps:
  1. Load ATAC-seq peaks (BED file)
  2. Add flanks around peaks
  3. Extract DNA sequences from genome FASTA
  4. One-hot encode DNA sequences
  5. (Optional) Aggregate signal tracks (bigWig)
  6. Save encoded data
"""

import pandas as pd
import numpy as np
from pyfaidx import Fasta
import argparse
import os

# ======================
# Utility Functions
# ======================

def add_flanks(df, flank=250):
    """Add flanking sequence to peaks."""
    df["start_flank"] = df["start"] - flank
    df["end_flank"] = df["end"] + flank
    df["start_flank"] = df["start_flank"].clip(lower=0)
    return df[["chr", "start_flank", "end_flank", "id"]]


def extract_sequences(df, fasta_path, output_fa=None):
    """Extract genomic sequences from fasta file."""
    fasta = Fasta(fasta_path)
    sequences = []
    for _, row in df.iterrows():
        try:
            seq = fasta[row["chr"]][row["start_flank"]:row["end_flank"]].seq.upper()
        except KeyError:
            seq = "N" * (row["end_flank"] - row["start_flank"])
        sequences.append(seq)
    df["sequence"] = sequences

    if output_fa:
        with open(output_fa, "w") as f:
            for _, row in df.iterrows():
                f.write(f">{row['id']}_{row['chr']}:{row['start_flank']}-{row['end_flank']}\n{row['sequence']}\n")

    return df


def one_hot_encode(seq):
    """One-hot encode a DNA sequence."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((len(seq), 4), dtype=np.uint8)
    for i, base in enumerate(seq):
        if base in mapping:
            arr[i, mapping[base]] = 1
    return arr


def encode_all_sequences(df, target_length=None):
    """Convert all sequences to one-hot."""
    sequences = df["sequence"].tolist()
    
    # If no target length specified, use the most common length
    if target_length is None:
        lengths = [len(seq) for seq in sequences]
        target_length = max(set(lengths), key=lengths.count)
        print(f"Using target length: {target_length} (most common length)")
    
    # Pad or truncate sequences to target length
    padded_sequences = []
    for seq in sequences:
        if len(seq) > target_length:
            # Truncate
            seq = seq[:target_length]
        elif len(seq) < target_length:
            # Pad with N
            seq = seq + "N" * (target_length - len(seq))
        padded_sequences.append(seq)
    
    # Now encode
    encoded = [one_hot_encode(seq) for seq in padded_sequences]
    return np.array(encoded, dtype=np.uint8)


# ======================
# Main
# ======================

def main(args):
    print(f"Loading peaks from {args.bed_file} ...")
    cols = ["chr", "start", "end", "id"]
    # this would be the bed file
    df = pd.read_csv(args.bed_file, sep="\t", comment="#", header=None, names=cols) 

    print(f"Adding ±{args.flank} bp flanks ...") 
    df = add_flanks(df, flank=args.flank)

    print(f"Extracting sequences from {args.fasta} ...")
    # the args.fasta below would have the fasta path
    df = extract_sequences(df, args.fasta, output_fa=args.output_fa)

    if args.onehot:
        print("One-hot encoding sequences ...")
        # Calculate expected length (original peak + 2*flank)
        # We need to get this from the original dataframe before adding flanks
        original_df = pd.read_csv(args.bed_file, sep="\t", comment="#", header=None, names=["chr", "start", "end", "id"])
        expected_length = (original_df["end"] - original_df["start"]).iloc[0] + 2 * args.flank
        print(f"Expected sequence length: {expected_length}")
        encoded = encode_all_sequences(df, target_length=expected_length)
        np.savez_compressed(args.output_npz, X=encoded, meta=df[["chr", "start_flank", "end_flank", "id"]].values)
        print(f"Saved encoded data -> {args.output_npz}")
    else:
        print(f"Sequences extracted -> {args.output_fa or 'in-memory dataframe'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ATAC-seq peaks for EpiBERT")
    parser.add_argument("--bed_file", required=True, help="Input BED file with peaks")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA (e.g. hg38.fa)")
    parser.add_argument("--flank", type=int, default=250, help="Flank length to add on both sides")
    parser.add_argument("--output_fa", default="flanked_sequences.fa", help="Output FASTA file")
    parser.add_argument("--output_npz", default="encoded_sequences.npz", help="Output NPZ file for one-hot encoded data")
    parser.add_argument("--onehot", action="store_true", help="Enable one-hot encoding")

    args = parser.parse_args()
    main(args)
