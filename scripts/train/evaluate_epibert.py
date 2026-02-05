#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for EpiBERT Model

Computes:
- ROC-AUC (binary classification via thresholding)
- PR-AUC (precision-recall curve)
- Pearson correlation
- Spearman correlation
- MSE
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from model_multimodal import MultiModalAccessibilityModel
from train_epibert import (
    ChromatinAccessibilityDataset,
    load_data,
    adjust_functional,
)


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}, loss: {loss}")
    return model


def evaluate_model(model, dataloader, device, feature_map, functional_mode, threshold=None):
    """
    Evaluate model and return predictions and labels.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        threshold: Threshold for binary classification (if None, uses median)
    
    Returns:
        predictions: Array of continuous predictions
        labels: Array of true labels
        binary_predictions: Binary predictions (for classification metrics)
        binary_labels: Binary labels (for classification metrics)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            functional = batch.get("functional")
            functional_mask = batch.get("functional_mask")

            if functional is not None:
                functional = functional.to(device)
            if functional_mask is not None:
                functional_mask = functional_mask.to(device)

            functional, functional_mask = adjust_functional(functional, functional_mask, feature_map, functional_mode)

            predictions = model(
                sequences,
                functional=functional,
                functional_mask=functional_mask,
            )

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    
    # Convert to binary for classification metrics
    if threshold is None:
        threshold = np.median(labels)
    
    binary_labels = (labels >= threshold).astype(int)
    binary_predictions = (predictions >= threshold).astype(int)
    
    return predictions, labels, binary_predictions, binary_labels, threshold


def compute_metrics(predictions, labels, binary_predictions, binary_labels):
    """
    Compute all evaluation metrics.
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Regression metrics
    # MSE
    mse = np.mean((predictions - labels) ** 2)
    metrics['MSE'] = float(mse)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(predictions, labels)
    metrics['Pearson_correlation'] = float(pearson_r)
    metrics['Pearson_p_value'] = float(pearson_p)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(predictions, labels)
    metrics['Spearman_correlation'] = float(spearman_r)
    metrics['Spearman_p_value'] = float(spearman_p)
    
    # Classification metrics (using continuous predictions as scores)
    try:
        # ROC-AUC (use continuous predictions as scores)
        roc_auc = roc_auc_score(binary_labels, predictions)
        metrics['ROC_AUC'] = float(roc_auc)
    except ValueError as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        metrics['ROC_AUC'] = None
    
    try:
        # PR-AUC (precision-recall AUC)
        pr_auc = average_precision_score(binary_labels, predictions)
        metrics['PR_AUC'] = float(pr_auc)
    except ValueError as e:
        print(f"Warning: Could not compute PR-AUC: {e}")
        metrics['PR_AUC'] = None
    
    # Accuracy and classification metrics
    accuracy = (binary_predictions == binary_labels).mean()
    metrics['Accuracy'] = float(accuracy)
    
    # Confusion matrix components
    tp = ((binary_predictions == 1) & (binary_labels == 1)).sum()
    tn = ((binary_predictions == 0) & (binary_labels == 0)).sum()
    fp = ((binary_predictions == 1) & (binary_labels == 0)).sum()
    fn = ((binary_predictions == 0) & (binary_labels == 1)).sum()
    
    metrics['True_Positives'] = int(tp)
    metrics['True_Negatives'] = int(tn)
    metrics['False_Positives'] = int(fp)
    metrics['False_Negatives'] = int(fn)
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['Precision'] = float(precision)
    metrics['Recall'] = float(recall)
    metrics['F1_Score'] = float(f1)
    
    # Additional statistics
    metrics['predictions_mean'] = float(np.mean(predictions))
    metrics['predictions_std'] = float(np.std(predictions))
    metrics['labels_mean'] = float(np.mean(labels))
    metrics['labels_std'] = float(np.std(labels))
    metrics['n_samples'] = int(len(predictions))
    
    return metrics


def plot_evaluation_curves(predictions, labels, binary_labels, output_dir):
    """Plot ROC curve and PR curve."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(binary_labels, predictions)
        roc_auc = roc_auc_score(binary_labels, predictions)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
    except ValueError:
        print("Warning: Could not plot ROC curve")
    
    # PR Curve
    try:
        precision, recall, _ = precision_recall_curve(binary_labels, predictions)
        pr_auc = average_precision_score(binary_labels, predictions)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
    except ValueError:
        print("Warning: Could not plot PR curve")
    
    plt.tight_layout()
    plot_path = output_dir / "evaluation_curves.png"
    plt.savefig(plot_path, dpi=300)
    print(f"  Saved evaluation curves: {plot_path}")
    plt.close()
    
    # Scatter plot: predictions vs labels
    plt.figure(figsize=(8, 6))
    plt.scatter(labels, predictions, alpha=0.5, s=20)
    
    # Add correlation to plot
    pearson_r, _ = stats.pearsonr(predictions, labels)
    spearman_r, _ = stats.spearmanr(predictions, labels)
    
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Labels\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}')
    
    # Add diagonal line
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    scatter_path = output_dir / "predictions_vs_labels.png"
    plt.savefig(scatter_path, dpi=300)
    print(f"  Saved scatter plot: {scatter_path}")
    plt.close()


def main(args):
    print("=" * 60)
    print("EpiBERT Comprehensive Evaluation")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    data = load_data(
        args.data_dir,
        use_normalized_func=args.use_normalized_func,
        use_normalized_labels=args.use_normalized_labels,
        use_filtered=getattr(args, 'use_filtered', False),
        split_file=getattr(args, 'split_file', None),
    )

    sequences = data["sequences"]
    labels = data["labels"]
    functional = data["functional"]
    functional_mask = data["functional_mask"]
    access_target = data["access_target"]
    access_valid_mask = data["access_valid_mask"]
    feature_map = data["feature_map"]
    test_indices = data["test_indices"]
    
    # Create test dataset
    test_dataset = ChromatinAccessibilityDataset(
        sequences,
        labels,
        test_indices,
        functional_vectors=functional,
        functional_valid_mask=functional_mask,
        access_targets=access_target,
        access_valid_mask=access_valid_mask,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nTest set size: {len(test_dataset)}")
    
    # Create model
    print(f"\nCreating model...")
    model = MultiModalAccessibilityModel(
        seq_len=sequences.shape[1],
        feature_map=feature_map,
        seq_embed_dim=args.seq_embed_dim,
        fusion_embed_dim=args.fusion_embed_dim,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        dropout=args.dropout,
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model = load_checkpoint(model, checkpoint_path, device)
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    predictions, true_labels, binary_pred, binary_labels, threshold = evaluate_model(
        model,
        test_loader,
        device,
        feature_map,
        args.functional_mode,
        threshold=args.threshold,
    )

    # Denormalize if requested
    if args.use_normalized_labels and args.label_stats:
        stats_path = Path(args.label_stats)
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats_data = json.load(f)
            mean = stats_data.get("mean", 0.0)
            std = stats_data.get("std", 1.0)
            predictions = predictions * std + mean
            true_labels = true_labels * std + mean
    
    print(f"  Threshold for binary classification: {threshold:.4f}")
    print(f"  Positive samples: {binary_labels.sum()}/{len(binary_labels)}")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_metrics(predictions, true_labels, binary_pred, binary_labels)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"Pearson Correlation: {metrics['Pearson_correlation']:.4f} (p={metrics['Pearson_p_value']:.2e})")
    print(f"Spearman Correlation: {metrics['Spearman_correlation']:.4f} (p={metrics['Spearman_p_value']:.2e})")
    print(f"\nClassification Metrics:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    if metrics['ROC_AUC'] is not None:
        print(f"  ROC-AUC: {metrics['ROC_AUC']:.4f}")
    if metrics['PR_AUC'] is not None:
        print(f"  PR-AUC: {metrics['PR_AUC']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-Score: {metrics['F1_Score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['True_Positives']}, TN: {metrics['True_Negatives']}")
    print(f"  FP: {metrics['False_Positives']}, FN: {metrics['False_Negatives']}")
    print(f"\nPredictions - Mean: {metrics['predictions_mean']:.4f}, Std: {metrics['predictions_std']:.4f}")
    print(f"Labels - Mean: {metrics['labels_mean']:.4f}, Std: {metrics['labels_std']:.4f}")
    print("=" * 60)
    
    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics['threshold'] = float(threshold)
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Save predictions and labels for enhanced visualizations
    predictions_path = output_dir / "predictions_and_labels.npz"
    np.savez(predictions_path, predictions=predictions, labels=true_labels)
    print(f"Saved predictions and labels to: {predictions_path}")
    
    # Plot curves
    print(f"\nGenerating plots...")
    plot_evaluation_curves(predictions, true_labels, binary_labels, output_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of EpiBERT model")
    
    # Paths
    parser.add_argument("--checkpoint_path", type=str, 
                       default="models/pretrained/checkpoint_latest.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                       help="Directory to save evaluation results")
    
    # Model hyperparameters (must match training)
    parser.add_argument("--seq_embed_dim", type=int, default=256,
                       help="Sequence encoder embedding dimension")
    parser.add_argument("--fusion_embed_dim", type=int, default=128,
                       help="Cross-modal fusion embedding dimension")
    parser.add_argument("--fusion_layers", type=int, default=2,
                       help="Number of transformer layers in fusion module")
    parser.add_argument("--fusion_heads", type=int, default=8,
                       help="Number of attention heads in fusion module")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")
    parser.add_argument("--functional_mode", choices=["full", "atac", "none"], default="full",
                       help="Functional modality selection (default: full)")
    parser.add_argument("--use_normalized_func", action="store_true",
                       help="Use normalized functional vectors")
    parser.add_argument("--use_normalized_labels", action="store_true",
                       help="Use normalized labels")
    parser.add_argument("--use_filtered", action="store_true",
                       help="Use filtered high-quality sequences")
    parser.add_argument("--split_file", type=str, default=None,
                       help="Custom split file (default: auto-detect based on use_filtered)")
    parser.add_argument("--label_stats", type=str, default="data/processed/labels/accessibility_scores.norm.stats.json",
                       help="Normalization stats JSON for labels (used when denormalizing)")
    
    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold for binary classification (default: median of labels)")
    
    args = parser.parse_args()
    main(args)

