#!/usr/bin/env python3
"""
Optimized preprocessing pipeline for large genomic datasets
"""
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import argparse
import os
from tqdm import tqdm

def process_large_bed(bed_file, fasta_path, output_dir, flank=250, batch_size=100):
    """
    Process large BED file in batches to handle memory constraints
    """
    print(f"Processing {bed_file} with {fasta_path}")
    print(f"Output directory: {output_dir}")
    print(f"Flank size: {flank} bp")
    print(f"Batch size: {batch_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load FASTA file
    print("Loading FASTA file...")
    fasta = Fasta(fasta_path)
    
    # Read BED file
    print("Reading BED file...")
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chr', 'start', 'end', 'id'])
    
    # Add flanks
    print("Adding flanks...")
    bed_df['start_flank'] = (bed_df['start'] - flank).clip(lower=0)
    bed_df['end_flank'] = bed_df['end'] + flank
    
    total_peaks = len(bed_df)
    print(f"Total peaks to process: {total_peaks}")
    
    # Process in batches
    all_sequences = []
    all_metadata = []
    
    for batch_start in tqdm(range(0, total_peaks, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_peaks)
        batch_df = bed_df.iloc[batch_start:batch_end]
        
        batch_sequences = []
        batch_metadata = []
        
        for _, row in batch_df.iterrows():
            try:
                # Extract sequence
                seq = fasta[row['chr']][row['start_flank']:row['end_flank']].seq.upper()
                batch_sequences.append(seq)
                batch_metadata.append([row['chr'], row['start_flank'], row['end_flank'], row['id']])
            except KeyError:
                # Handle missing chromosome
                seq = "N" * (row['end_flank'] - row['start_flank'])
                batch_sequences.append(seq)
                batch_metadata.append([row['chr'], row['start_flank'], row['end_flank'], row['id']])
        
        all_sequences.extend(batch_sequences)
        all_metadata.extend(batch_metadata)
    
    # Convert to one-hot encoding
    print("Converting to one-hot encoding...")
    encoded_sequences = []
    
    # Determine target length (most common length)
    lengths = [len(seq) for seq in all_sequences]
    target_length = max(set(lengths), key=lengths.count)
    print(f"Using target length: {target_length}")
    
    # Pad/truncate sequences to uniform length
    padded_sequences = []
    for seq in all_sequences:
        if len(seq) > target_length:
            seq = seq[:target_length]
        elif len(seq) < target_length:
            seq = seq + "N" * (target_length - len(seq))
        padded_sequences.append(seq)
    
    for seq in tqdm(padded_sequences, desc="One-hot encoding"):
        encoded_seq = one_hot_encode(seq)
        encoded_sequences.append(encoded_seq)
    
    # Save results
    print("Saving results...")
    
    # Save FASTA file
    fasta_output = os.path.join(output_dir, "flanked_sequences.fa")
    with open(fasta_output, 'w') as f:
        for i, (seq, meta) in enumerate(zip(padded_sequences, all_metadata)):
            f.write(f">{meta[3]}_{meta[0]}:{meta[1]}-{meta[2]}\n{seq}\n")
    
    # Save encoded data
    encoded_array = np.array(encoded_sequences, dtype=np.uint8)
    metadata_array = np.array(all_metadata)
    
    npz_output = os.path.join(output_dir, "encoded_sequences.npz")
    np.savez_compressed(npz_output, X=encoded_array, meta=metadata_array)
    
    print(f"Processing complete!")
    print(f"FASTA file: {fasta_output}")
    print(f"Encoded data: {npz_output}")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Sequence length: {len(all_sequences[0]) if all_sequences else 0}")

def one_hot_encode(seq):
    """One-hot encode a DNA sequence"""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((len(seq), 4), dtype=np.uint8)
    for i, base in enumerate(seq):
        if base in mapping:
            arr[i, mapping[base]] = 1
    return arr

def main():
    parser = argparse.ArgumentParser(description="Preprocess large genomic datasets")
    parser.add_argument("--bed_file", required=True, help="Input BED file")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA")
    parser.add_argument("--output_dir", default="preprocessed_output", help="Output directory")
    parser.add_argument("--flank", type=int, default=250, help="Flank size")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    
    args = parser.parse_args()
    process_large_bed(args.bed_file, args.fasta, args.output_dir, args.flank, args.batch_size)

if __name__ == "__main__":
    main()
