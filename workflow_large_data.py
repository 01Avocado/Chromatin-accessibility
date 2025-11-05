#!/usr/bin/env python3
"""
Complete workflow for preprocessing large genomic datasets
"""
import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("OUTPUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stderr:
            print("ERROR OUTPUT:", e.stderr)
        return False

def main():
    print("Large Genomic Data Preprocessing Workflow")
    print("=" * 50)
    
    # Check if files exist
    required_files = ["converted/hg38_converted.csv", "hg38.fa"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: Required file {file} not found!")
            return
    
    # Step 1: Convert CSV to BED (sample)
    print("\n1. Converting CSV to BED format (first 1000 rows)...")
    cmd1 = 'python csv_to_bed.py --csv_file "converted/hg38_converted.csv" --output_bed "peaks_sample.bed" --chunk_size 1000'
    if not run_command(cmd1, "Convert CSV to BED"):
        print("Failed to convert CSV to BED")
        return
    
    # Step 2: Process with original script (small sample)
    print("\n2. Processing sample with original script...")
    cmd2 = 'python preprocess.py --bed_file peaks_sample.bed --fasta hg38.fa --flank 250 --output_fa sample_sequences.fa --onehot'
    if not run_command(cmd2, "Process sample with original script"):
        print("Failed to process sample")
        return
    
    # Step 3: Process with optimized script (larger dataset)
    print("\n3. Processing with optimized script...")
    cmd3 = 'python preprocess_large.py --bed_file peaks_sample.bed --fasta hg38.fa --output_dir preprocessed_large --flank 250 --batch_size 50'
    if not run_command(cmd3, "Process with optimized script"):
        print("Failed to process with optimized script")
        return
    
    print("\n" + "="*50)
    print("WORKFLOW COMPLETE!")
    print("="*50)
    print("Generated files:")
    print("- peaks_sample.bed (BED file from CSV)")
    print("- sample_sequences.fa (FASTA sequences)")
    print("- encoded_sequences.npz (One-hot encoded)")
    print("- preprocessed_large/ (Directory with batch-processed results)")
    
    print("\nTo process your full dataset:")
    print("1. Create a proper BED file with your peak coordinates")
    print("2. Run: python preprocess_large.py --bed_file your_peaks.bed --fasta hg38.fa --output_dir results")

if __name__ == "__main__":
    main()

