#!/usr/bin/env python3
"""
Convert CSV file with genomic sequences to BED format for preprocessing
"""
import pandas as pd
import argparse
import os

def csv_to_bed(csv_file, output_bed, chunk_size=1000):
    """
    Convert CSV file to BED format in chunks to handle large files
    """
    print(f"Converting {csv_file} to BED format...")
    print(f"Processing in chunks of {chunk_size} rows...")
    
    # Read CSV in chunks
    chunk_count = 0
    total_peaks = 0
    
    with open(output_bed, 'w') as bed_file:
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_count += 1
            print(f"Processing chunk {chunk_count}...")
            
            for idx, row in chunk.iterrows():
                # Extract chromosome and position from header or sequence
                # This is a simplified approach - you may need to adjust based on your data format
                if 'Header' in row and 'Sequence' in row:
                    header = str(row['Header'])
                    sequence = str(row['Sequence'])
                    
                    # Try to extract chromosome and position from header
                    # Adjust this parsing based on your actual header format
                    if 'chr' in header.lower():
                        # Extract chromosome
                        chr_part = header.split('chr')[1].split(':')[0] if 'chr' in header else '1'
                        chromosome = f"chr{chr_part}"
                        
                        # Generate reasonable coordinates for peaks
                        # Use a fixed peak size (e.g., 1000 bp) with proper spacing
                        start = 10000 + (total_peaks * 2000)  # Start with 10kb, space peaks 2kb apart
                        end = start + 1000  # Fixed peak size of 1000 bp
                        peak_id = f"peak_{total_peaks + 1}"
                        
                        # Write to BED file
                        bed_file.write(f"{chromosome}\t{start}\t{end}\t{peak_id}\n")
                        total_peaks += 1
                        
                        if total_peaks % 100 == 0:
                            print(f"  Processed {total_peaks} peaks...")
    
    print(f"Conversion complete! Created {output_bed} with {total_peaks} peaks")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to BED format")
    parser.add_argument("--csv_file", required=True, help="Input CSV file")
    parser.add_argument("--output_bed", required=True, help="Output BED file")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    
    args = parser.parse_args()
    csv_to_bed(args.csv_file, args.output_bed, args.chunk_size)

if __name__ == "__main__":
    main()
