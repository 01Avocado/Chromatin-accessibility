# Folder Contents Summary

## 📁 Complete Data Inventory

### 🗂️ **RAW INPUT DATA** (Original Dataset)

#### 1. **Reference Genome Files** (~3.5 GB total)
- **`hg38.fa`** (3.27 GB) 
  - Complete human reference genome hg38
  - Used for extracting genomic sequences
  
- **`hg38.fa.fai`** (19.8 KB)
  - FASTA index file for random access
  
- **`chr1.fa`** (254 MB)
  - Chromosome 1 sequence only
  
- **`chr2.fa`** (247 MB)
  - Chromosome 2 sequence only

#### 2. **Original Data Files** (~7.5 GB total)
- **`hg38 text.csv.txt`** (4.23 GB)
  - **PROBABLY YOUR MAIN INPUT DATA**
  - Text/CSV format with genomic data
  
- **`converted/hg38_converted.csv`** (3.21 GB)
  - Converted version of the above
  - Contains "Header" and "Sequence" columns
  - 455 rows with very long sequences (millions of bp each)

---

### 🔧 **PROCESSING SCRIPTS**

#### Python Scripts Created:
1. **`preprocess.py`** (4.8 KB)
   - Main preprocessing pipeline
   - Extracts sequences, adds flanks, one-hot encodes
   - Handles variable-length sequences

2. **`preprocess_large.py`** (4.9 KB)
   - Optimized for large datasets
   - Batch processing with memory management
   - Progress bars with tqdm

3. **`csv_to_bed.py`** (2.8 KB)
   - Converts CSV format to BED format
   - Handles large files in chunks
   - Generates synthetic coordinates

4. **`workflow_large_data.py`** (2.9 KB)
   - Complete workflow automation
   - Runs all preprocessing steps sequentially

---

### 📊 **PROCESSED OUTPUT DATA**

#### Final Results (Use These!):
- **`full_peaks.bed`** (19.4 KB) ⭐
  - **FINAL BED FILE** - 455 genomic regions
  - Format: chr, start, end, peak_id
  
- **`full_sequences.fa`** (182 KB) ⭐
  - **FINAL FASTA FILE** - Extracted sequences with 250bp flanks
  - 455 sequences, 1500 bp each
  
- **`encoded_sequences.npz`** (71 KB) ⭐
  - **FINAL ENCODED DATA** - One-hot encoded for ML
  - Shape: (455, 1500, 4)
  - Format: uint8 arrays
  - Includes metadata

#### Batch Processing Results:
- **`full_results/encoded_sequences.npz`** (7.7 KB)
- **`full_results/flanked_sequences.fa`** (20.7 KB)
- **`preprocessed_large/encoded_sequences.npz`** (7.7 KB)
- **`preprocessed_large/flanked_sequences.fa`** (20.7 KB)

#### Test/Sample Files:
- `sample_peaks.bed` (119 bytes) - Test BED file
- `peaks_sample.bed` (19.5 KB) - Sample output
- `peaks_fixed.bed` (19.4 KB) - Fixed coordinate version
- `fixed_sequences.fa` (182 KB) - Fixed sequence file
- `sample_sequences.fa` (3.14 GB) - **ERROR FILE** (too large, contains artifacts)
- `flanked_sequences.fa` (7.6 KB) - Early test output

#### Visualization:
- **`sequence_analysis.png`** (120 KB) ⭐
  - Quality distribution plots
  - GC-content analysis
  - N-content visualization

---

### 📈 **Data Statistics**

#### Final Dataset (455 sequences):
- **High quality**: 92 sequences (20.2%)
  - < 25% N-content
  - Ready for analysis
  
- **Medium quality**: 1 sequence (0.2%)
  - 25-50% N-content
  
- **Low quality**: 362 sequences (79.6%)
  - > 50% N-content (mostly gaps)
  - Consider filtering out

#### Sequence Properties:
- **Length**: 1500 bp per sequence (1000 bp original + 250 bp flanks each side)
- **GC Content**: Mean 8.69%, Median 0%
- **Format**: One-hot encoded (A, C, G, T)
- **Memory**: 2.6 MB when loaded

---

### 🎯 **What Data Types Are Present?**

✅ **WHAT YOU HAVE:**
- DNA sequences from human genome (hg38)
- Genomic coordinates (BED format)
- One-hot encoded nucleotide data
- Chromosome-level annotations

❌ **WHAT YOU DON'T HAVE:**
- ATAC-seq signal intensities
- Histone modification data (H3K27ac, H3K4me3, etc.)
- Transcription factor binding data
- RNA-seq expression data
- Chromatin accessibility scores
- Functional annotations

---

### 🔍 **Recommendations**

#### For Analysis:
1. **Use `full_peaks.bed`, `full_sequences.fa`, `encoded_sequences.npz`**
2. **Filter to 92 high-quality sequences** for better results
3. **Consider getting actual ATAC-seq data** if studying chromatin accessibility

#### To Download Functional Data:
- **ENCODE Project**: https://www.encodeproject.org/
- **GEO Database**: https://www.ncbi.nlm.nih.gov/geo/
- **UCSC Genome Browser**: https://genome.ucsc.edu/

#### Next Steps:
```bash
# Filter high-quality sequences
python filter_high_quality.py --input encoded_sequences.npz --output high_quality_encoded.npz

# Download ATAC-seq data from ENCODE
# Process and merge with your sequences
```

---

### 📝 **File Size Summary**

| Category | Total Size |
|----------|-----------|
| Raw Genome Files | ~3.5 GB |
| Original CSV Data | ~7.5 GB |
| Processed Output | ~500 KB |
| Scripts & Docs | ~15 KB |
| **TOTAL** | **~11 GB** |

---

**Last Updated**: November 2025
**Dataset**: Human Genome hg38, 455 genomic regions
**Primary Format**: DNA sequences only (no epigenetic signals)

