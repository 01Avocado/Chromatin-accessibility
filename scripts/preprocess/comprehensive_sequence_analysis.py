#!/usr/bin/env python3
"""
Comprehensive Sequence Analysis for Chromatin Accessibility Dataset

Analyzes:
- GC content distribution
- Sequence composition (A, T, G, C frequencies)
- Sequence length statistics
- N-content (missing/ambiguous bases)
- Dinucleotide frequencies
- Sequence complexity
- Correlation with accessibility scores
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json
from scipy import stats
from scipy.stats import chi2_contingency

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_sequences():
    """Load encoded sequences and labels."""
    data_dir = Path("data/processed")
    
    # Load sequences
    sequences_data = np.load(data_dir / "sequences" / "encoded_sequences.npz", allow_pickle=True)
    sequences = sequences_data['X']  # (N, seq_len, 4)
    
    # Load labels
    labels = np.load(data_dir / "labels" / "accessibility_scores.npy")
    
    # Load BED file for coordinates
    bed_file = data_dir / "peaks" / "full_peaks.bed"
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, 
                        names=['chr', 'start', 'end', 'id'],
                        comment='#')
    
    return sequences, labels, bed_df

def onehot_to_sequence(onehot_seq):
    """Convert one-hot encoded sequence to nucleotide string."""
    nucleotide_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq = []
    for pos in onehot_seq:
        nucleotide_idx = np.argmax(pos)
        seq.append(nucleotide_map[nucleotide_idx])
    return ''.join(seq)

def calculate_gc_content(sequence):
    """Calculate GC content of a sequence."""
    gc_count = (sequence == 'G').sum() + (sequence == 'C').sum()
    total = len(sequence)
    return gc_count / total if total > 0 else 0.0

def calculate_n_content(sequence):
    """Calculate N content (missing/ambiguous bases)."""
    n_count = (sequence == 'N').sum()
    return n_count / len(sequence) if len(sequence) > 0 else 0.0

def calculate_dinucleotide_freq(sequence):
    """Calculate dinucleotide frequencies."""
    dinucleotides = [sequence[i:i+2] for i in range(len(sequence)-1)]
    freq = Counter(dinucleotides)
    total = len(dinucleotides)
    return {k: v/total for k, v in freq.items()}

def calculate_sequence_complexity(sequence):
    """Calculate sequence complexity (Shannon entropy)."""
    from collections import Counter
    counts = Counter(sequence)
    total = len(sequence)
    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def analyze_sequences(sequences, labels):
    """Perform comprehensive sequence analysis."""
    n_sequences = sequences.shape[0]
    seq_len = sequences.shape[1]
    
    print(f"Analyzing {n_sequences} sequences of length {seq_len} bp...")
    
    # Initialize arrays for statistics
    gc_contents = []
    n_contents = []
    complexities = []
    nucleotide_freqs = {'A': [], 'C': [], 'G': [], 'T': []}
    dinucleotide_freqs_all = []
    
    # Analyze each sequence
    for i in range(n_sequences):
        # Convert one-hot to sequence
        onehot_seq = sequences[i]
        seq_array = np.argmax(onehot_seq, axis=1)
        nucleotide_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        sequence = ''.join([nucleotide_map[idx] for idx in seq_array])
        
        # Calculate statistics
        gc_content = calculate_gc_content(np.array(list(sequence)))
        n_content = calculate_n_content(np.array(list(sequence)))
        complexity = calculate_sequence_complexity(sequence)
        
        gc_contents.append(gc_content)
        n_contents.append(n_content)
        complexities.append(complexity)
        
        # Nucleotide frequencies
        for nt in ['A', 'C', 'G', 'T']:
            freq = sequence.count(nt) / len(sequence)
            nucleotide_freqs[nt].append(freq)
        
        # Dinucleotide frequencies (sample every 10th sequence for speed)
        if i % 10 == 0:
            dinuc_freq = calculate_dinucleotide_freq(sequence)
            dinucleotide_freqs_all.append(dinuc_freq)
    
    # Convert to numpy arrays
    gc_contents = np.array(gc_contents)
    n_contents = np.array(n_contents)
    complexities = np.array(complexities)
    
    # Calculate correlations with labels
    gc_corr, gc_p = stats.pearsonr(gc_contents, labels)
    complexity_corr, complexity_p = stats.pearsonr(complexities, labels)
    
    # Create summary statistics
    summary = {
        'n_sequences': int(n_sequences),
        'sequence_length': int(seq_len),
        'gc_content': {
            'mean': float(gc_contents.mean()),
            'std': float(gc_contents.std()),
            'min': float(gc_contents.min()),
            'max': float(gc_contents.max()),
            'correlation_with_labels': float(gc_corr),
            'correlation_p_value': float(gc_p)
        },
        'n_content': {
            'mean': float(n_contents.mean()),
            'std': float(n_contents.std()),
            'min': float(n_contents.min()),
            'max': float(n_contents.max())
        },
        'complexity': {
            'mean': float(complexities.mean()),
            'std': float(complexities.std()),
            'min': float(complexities.min()),
            'max': float(complexities.max()),
            'correlation_with_labels': float(complexity_corr),
            'correlation_p_value': float(complexity_p)
        },
        'nucleotide_frequencies': {
            nt: {
                'mean': float(np.mean(nucleotide_freqs[nt])),
                'std': float(np.std(nucleotide_freqs[nt]))
            }
            for nt in ['A', 'C', 'G', 'T']
        }
    }
    
    return {
        'gc_contents': gc_contents,
        'n_contents': n_contents,
        'complexities': complexities,
        'nucleotide_freqs': nucleotide_freqs,
        'dinucleotide_freqs': dinucleotide_freqs_all,
        'summary': summary
    }

def create_comprehensive_plots(analysis_results, labels, output_dir):
    """Create comprehensive sequence analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gc_contents = analysis_results['gc_contents']
    n_contents = analysis_results['n_contents']
    complexities = analysis_results['complexities']
    nucleotide_freqs = analysis_results['nucleotide_freqs']
    summary = analysis_results['summary']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. GC Content Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gc_contents, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(gc_contents.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gc_contents.mean():.3f}')
    ax1.set_xlabel('GC Content')
    ax1.set_ylabel('Frequency')
    ax1.set_title('GC Content Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. GC Content vs Accessibility
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(gc_contents, labels, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    corr = summary['gc_content']['correlation_with_labels']
    p_val = summary['gc_content']['correlation_p_value']
    ax2.set_xlabel('GC Content')
    ax2.set_ylabel('Accessibility Score')
    ax2.set_title(f'GC Content vs Accessibility\nr={corr:.3f}, p={p_val:.3f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sequence Complexity Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(complexities, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(complexities.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {complexities.mean():.3f}')
    ax3.set_xlabel('Sequence Complexity (Entropy)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Sequence Complexity Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Complexity vs Accessibility
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(complexities, labels, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    corr = summary['complexity']['correlation_with_labels']
    p_val = summary['complexity']['correlation_p_value']
    ax4.set_xlabel('Sequence Complexity')
    ax4.set_ylabel('Accessibility Score')
    ax4.set_title(f'Complexity vs Accessibility\nr={corr:.3f}, p={p_val:.3f}')
    ax4.grid(True, alpha=0.3)
    
    # 5. Nucleotide Frequency Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    nts = ['A', 'C', 'G', 'T']
    means = [summary['nucleotide_frequencies'][nt]['mean'] for nt in nts]
    stds = [summary['nucleotide_frequencies'][nt]['std'] for nt in nts]
    x_pos = np.arange(len(nts))
    ax5.bar(x_pos, means, yerr=stds, alpha=0.7, color=['red', 'blue', 'green', 'orange'], 
            edgecolor='black', capsize=5)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(nts)
    ax5.set_ylabel('Frequency')
    ax5.set_title('Nucleotide Frequencies')
    ax5.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Expected (0.25)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. N-content Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(n_contents, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax6.axvline(n_contents.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {n_contents.mean():.4f}')
    ax6.set_xlabel('N Content (Missing Bases)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('N Content Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. GC Content by Accessibility (box plot)
    ax7 = fig.add_subplot(gs[2, 0])
    # Bin labels by accessibility
    high_acc = gc_contents[labels >= np.median(labels)]
    low_acc = gc_contents[labels < np.median(labels)]
    ax7.boxplot([low_acc, high_acc], labels=['Low\nAccessibility', 'High\nAccessibility'])
    ax7.set_ylabel('GC Content')
    ax7.set_title('GC Content by Accessibility Level')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Sequence Statistics Summary
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    # Create summary table
    stats_data = [
        ['Metric', 'Value'],
        ['Total Sequences', f"{summary['n_sequences']}"],
        ['Sequence Length', f"{summary['sequence_length']} bp"],
        ['GC Content Mean', f"{summary['gc_content']['mean']:.3f} ± {summary['gc_content']['std']:.3f}"],
        ['GC-Label Correlation', f"{summary['gc_content']['correlation_with_labels']:.3f} (p={summary['gc_content']['correlation_p_value']:.3f})"],
        ['Complexity Mean', f"{summary['complexity']['mean']:.3f} ± {summary['complexity']['std']:.3f}"],
        ['Complexity-Label Correlation', f"{summary['complexity']['correlation_with_labels']:.3f} (p={summary['complexity']['correlation_p_value']:.3f})"],
        ['N Content Mean', f"{summary['n_content']['mean']:.4f}"],
        ['A Frequency', f"{summary['nucleotide_frequencies']['A']['mean']:.3f}"],
        ['C Frequency', f"{summary['nucleotide_frequencies']['C']['mean']:.3f}"],
        ['G Frequency', f"{summary['nucleotide_frequencies']['G']['mean']:.3f}"],
        ['T Frequency', f"{summary['nucleotide_frequencies']['T']['mean']:.3f}"]
    ]
    
    table = ax8.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax8.set_title('Sequence Statistics Summary', fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Sequence Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_dir / "comprehensive_sequence_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  Saved comprehensive sequence analysis: {output_dir / 'comprehensive_sequence_analysis.png'}")
    plt.close()
    
    # Create additional plots
    # Dinucleotide frequency heatmap
    if analysis_results['dinucleotide_freqs']:
        create_dinucleotide_heatmap(analysis_results['dinucleotide_freqs'], output_dir)
    
    # Sequence quality by accessibility
    create_quality_analysis(analysis_results, labels, output_dir)

def create_dinucleotide_heatmap(dinucleotide_freqs, output_dir):
    """Create dinucleotide frequency heatmap."""
    # Aggregate dinucleotide frequencies
    all_dinucs = set()
    for freq_dict in dinucleotide_freqs:
        all_dinucs.update(freq_dict.keys())
    
    # Create matrix
    nts = ['A', 'C', 'G', 'T']
    matrix = np.zeros((4, 4))
    
    for i, nt1 in enumerate(nts):
        for j, nt2 in enumerate(nts):
            dinuc = nt1 + nt2
            if dinuc in all_dinucs:
                # Average frequency across sequences
                freqs = [f.get(dinuc, 0) for f in dinucleotide_freqs]
                matrix[i, j] = np.mean(freqs)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=nts, yticklabels=nts, cbar_kws={'label': 'Frequency'})
    plt.xlabel('Second Nucleotide')
    plt.ylabel('First Nucleotide')
    plt.title('Dinucleotide Frequency Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / "dinucleotide_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  Saved dinucleotide heatmap: {output_dir / 'dinucleotide_heatmap.png'}")
    plt.close()

def create_quality_analysis(analysis_results, labels, output_dir):
    """Create sequence quality analysis by accessibility."""
    gc_contents = analysis_results['gc_contents']
    complexities = analysis_results['complexities']
    n_contents = analysis_results['n_contents']
    
    # Bin by accessibility
    high_acc_idx = labels >= np.median(labels)
    low_acc_idx = labels < np.median(labels)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # GC content comparison
    axes[0].boxplot([gc_contents[low_acc_idx], gc_contents[high_acc_idx]], 
                    labels=['Low\nAccessibility', 'High\nAccessibility'])
    axes[0].set_ylabel('GC Content')
    axes[0].set_title('GC Content by Accessibility')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Complexity comparison
    axes[1].boxplot([complexities[low_acc_idx], complexities[high_acc_idx]], 
                    labels=['Low\nAccessibility', 'High\nAccessibility'])
    axes[1].set_ylabel('Sequence Complexity')
    axes[1].set_title('Complexity by Accessibility')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # N-content comparison
    axes[2].boxplot([n_contents[low_acc_idx], n_contents[high_acc_idx]], 
                    labels=['Low\nAccessibility', 'High\nAccessibility'])
    axes[2].set_ylabel('N Content')
    axes[2].set_title('N Content by Accessibility')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_quality_by_accessibility.png", dpi=300, bbox_inches='tight')
    print(f"  Saved quality analysis: {output_dir / 'sequence_quality_by_accessibility.png'}")
    plt.close()

def main():
    """Main function."""
    print("=" * 60)
    print("Comprehensive Sequence Analysis")
    print("=" * 60)
    
    # Load data
    sequences, labels, bed_df = load_sequences()
    print(f"\nLoaded {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Perform analysis
    print("\nAnalyzing sequences...")
    analysis_results = analyze_sequences(sequences, labels)
    
    # Print summary
    summary = analysis_results['summary']
    print("\n" + "=" * 60)
    print("Sequence Analysis Summary")
    print("=" * 60)
    print(f"GC Content: {summary['gc_content']['mean']:.3f} ± {summary['gc_content']['std']:.3f}")
    print(f"GC-Label Correlation: {summary['gc_content']['correlation_with_labels']:.3f} (p={summary['gc_content']['correlation_p_value']:.3f})")
    print(f"Sequence Complexity: {summary['complexity']['mean']:.3f} ± {summary['complexity']['std']:.3f}")
    print(f"Complexity-Label Correlation: {summary['complexity']['correlation_with_labels']:.3f} (p={summary['complexity']['correlation_p_value']:.3f})")
    print(f"N Content: {summary['n_content']['mean']:.4f}")
    print("\nNucleotide Frequencies:")
    for nt in ['A', 'C', 'G', 'T']:
        mean_freq = summary['nucleotide_frequencies'][nt]['mean']
        print(f"  {nt}: {mean_freq:.3f}")
    
    # Save summary
    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "sequence_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")
    
    # Create plots
    print("\nGenerating plots...")
    create_comprehensive_plots(analysis_results, labels, output_dir)
    
    print("\n" + "=" * 60)
    print("Sequence analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

