#!/usr/bin/env python3
"""
Training script for EpiBERT model.

Trains the transformer model to predict chromatin accessibility from DNA sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from model_multimodal import MultiModalAccessibilityModel


class CombinedLoss(nn.Module):
    """
    Combined loss function: MSE + (1 - Pearson correlation)
    This encourages the model to minimize prediction error AND maximize correlation.
    """
    def __init__(self, mse_weight=0.5, correlation_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, labels):
        # MSE component
        mse = self.mse_loss(predictions, labels)
        
        # Pearson correlation component (maximize correlation = minimize 1 - correlation)
        # Normalize predictions and labels
        pred_mean = predictions.mean()
        label_mean = labels.mean()
        pred_centered = predictions - pred_mean
        label_centered = labels - label_mean
        
        # Compute correlation
        numerator = (pred_centered * label_centered).sum()
        pred_std = (pred_centered ** 2).sum()
        label_std = (label_centered ** 2).sum()
        denominator = torch.sqrt(pred_std * label_std + 1e-8)  # Add epsilon for numerical stability
        
        # Compute correlation
        correlation = numerator / denominator
        
        # Correlation loss: 1 - correlation (we want to maximize correlation)
        # Clip correlation to valid range [-1, 1] for stability
        correlation = torch.clamp(correlation, -1.0, 1.0)
        correlation_loss = 1.0 - correlation
        
        # Combined loss
        total_loss = self.mse_weight * mse + self.correlation_weight * correlation_loss
        
        return total_loss


class ChromatinAccessibilityDataset(Dataset):
    """Dataset for chromatin accessibility prediction."""
    
    def __init__(
        self,
        sequences,
        labels,
        indices=None,
        functional_vectors=None,
        functional_valid_mask=None,
        access_targets=None,
        access_valid_mask=None,
    ):
        """
        Args:
            sequences: numpy array of shape (N, seq_len, 4) - one-hot encoded sequences
            labels: numpy array of shape (N,) - accessibility scores
            indices: optional list of indices to subset the data
            functional_vectors: optional numpy array of shape (N, n_features)
            functional_valid_mask: optional numpy array of shape (N, n_features)
            access_targets: optional numpy array of shape (N, n_bins)
            access_valid_mask: optional numpy array of shape (N, n_bins)
        """
        if indices is not None:
            self.sequences = sequences[indices]
            self.labels = labels[indices]
            self.functional_vectors = (
                functional_vectors[indices] if functional_vectors is not None else None
            )
            self.functional_valid_mask = (
                functional_valid_mask[indices] if functional_valid_mask is not None else None
            )
            self.access_targets = (
                access_targets[indices] if access_targets is not None else None
            )
            self.access_valid_mask = (
                access_valid_mask[indices] if access_valid_mask is not None else None
            )
        else:
            self.sequences = sequences
            self.labels = labels
            self.functional_vectors = functional_vectors
            self.functional_valid_mask = functional_valid_mask
            self.access_targets = access_targets
            self.access_valid_mask = access_valid_mask
        
        # Convert to float32
        self.sequences = self.sequences.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        
        if self.functional_vectors is not None:
            self.functional_vectors = self.functional_vectors.astype(np.float32)
        if self.functional_valid_mask is not None:
            self.functional_valid_mask = self.functional_valid_mask.astype(np.float32)
        if self.access_targets is not None:
            self.access_targets = self.access_targets.astype(np.float32)
        if self.access_valid_mask is not None:
            self.access_valid_mask = self.access_valid_mask.astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            'sequence': torch.from_numpy(self.sequences[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        
        if self.functional_vectors is not None:
            sample['functional'] = torch.from_numpy(self.functional_vectors[idx])
        if self.functional_valid_mask is not None:
            sample['functional_mask'] = torch.from_numpy(self.functional_valid_mask[idx])
        if self.access_targets is not None:
            sample['access_target'] = torch.from_numpy(self.access_targets[idx])
        if self.access_valid_mask is not None:
            sample['access_valid_mask'] = torch.from_numpy(self.access_valid_mask[idx])
        
        return sample


def adjust_functional(functional, functional_mask, feature_map, mode):
    """
    Adjust functional inputs based on ablation mode.

    Args:
        functional: tensor or None
        functional_mask: tensor or None
        feature_map: dict with bigwigs/n_bins
        mode: "full", "atac", or "none"
    """
    if functional is None:
        return None, None

    if mode == "none":
        return None, None

    if feature_map is None:
        return functional, functional_mask

    if mode not in {"full", "atac"}:
        raise ValueError(f"Unsupported functional_mode: {mode}")

    if mode == "full":
        return functional, functional_mask

    # ATAC-only: zero out other modalities but keep tensor shape for model compatibility
    n_bins = feature_map.get("n_bins", functional.size(-1))
    bigwigs = feature_map.get("bigwigs", [])

    functional = functional.clone()
    if functional_mask is not None:
        functional_mask = functional_mask.clone()

    for idx in range(len(bigwigs)):
        start = idx * n_bins
        end = start + n_bins
        if idx != 0:
            functional[:, start:end] = 0.0
            if functional_mask is not None:
                functional_mask[:, start:end] = 0

    return functional, functional_mask


def load_data(
    data_dir="data/processed",
    use_normalized_func: bool = False,
    use_normalized_labels: bool = False,
    use_filtered: bool = False,
    split_file: str = None,
    sequences_npz: str | None = None,
    labels_npy: str | None = None,
    splits_npz: str | None = None,
):
    """Load sequences, labels, functional vectors, and splits."""
    data_dir = Path(data_dir)
    
    print("Loading data...")
    using_custom_paths = bool(sequences_npz or labels_npy or splits_npz)
    if using_custom_paths:
        print("  Using custom dataset paths")
    if use_filtered:
        print("  Using FILTERED data (high quality sequences only)")
    
    # Load sequences
    if sequences_npz:
        sequences_path = Path(sequences_npz)
    else:
        if use_filtered:
            sequences_path = data_dir / "sequences" / "encoded_sequences.filtered.npz"
            if not sequences_path.exists():
                print(f"  [WARNING] Filtered sequences not found: {sequences_path}")
                print(f"  Falling back to original sequences")
                sequences_path = data_dir / "sequences" / "encoded_sequences.npz"
        else:
            sequences_path = data_dir / "sequences" / "encoded_sequences.npz"
    
    sequences_data = np.load(sequences_path, allow_pickle=True)
    sequences = sequences_data['X']  # (N, seq_len, 4)
    print(f"  Sequences: {sequences.shape}")
    
    # Load labels
    if labels_npy:
        labels_path = Path(labels_npy)
    else:
        if use_filtered:
            labels_base = "accessibility_scores.filtered"
        else:
            labels_base = "accessibility_scores"
        
        labels_path = (
            data_dir / "labels" / (f"{labels_base}.norm.npy" if use_normalized_labels else f"{labels_base}.npy")
        )
        if not labels_path.exists() and use_filtered:
            print(f"  [WARNING] Filtered labels not found, using original")
            labels_path = data_dir / "labels" / ("accessibility_scores.norm.npy" if use_normalized_labels else "accessibility_scores.npy")
    
    labels = np.load(labels_path)
    print(f"  Labels: {labels.shape}")
    
    # Load functional vectors and masks if available
    func_dir = data_dir / "funcvecs"
    func_vectors = None
    func_valid_mask = None
    access_target = None
    access_valid_mask = None
    feature_map = None
    
    # IMPORTANT: if we're using a custom dataset (e.g. k562_1kb), the existing funcvecs
    # in data/processed/funcvecs are for the old 455-peak dataset and will not align.
    # So we skip loading functional vectors unless the user explicitly wires in matching ones.
    if func_dir.exists() and (not using_custom_paths):
        # Try filtered first if requested
        if use_filtered:
            # Try normalized filtered first (most common)
            func_vectors_path = func_dir / "func_vectors.norm.filtered.npy"
            if func_vectors_path.exists():
                func_vectors = np.load(func_vectors_path)
                print(f"  Functional vectors: {func_vectors.shape}")
            else:
                # Try non-normalized filtered
                func_vectors_path = func_dir / "func_vectors.filtered.npy"
                if func_vectors_path.exists():
                    func_vectors = np.load(func_vectors_path)
                    print(f"  Functional vectors: {func_vectors.shape}")
                else:
                    # Fallback: filter original vectors
                    func_vectors_path = func_dir / ("func_vectors.norm.npy" if use_normalized_func else "func_vectors.npy")
                    if func_vectors_path.exists():
                        original_vecs = np.load(func_vectors_path)
                        filter_indices_path = data_dir / "sequences" / "filtered_indices.npy"
                        if filter_indices_path.exists():
                            filter_indices = np.load(filter_indices_path)
                            func_vectors = original_vecs[filter_indices]
                            print(f"  Functional vectors: {func_vectors.shape} (filtered from original)")
                        else:
                            func_vectors = original_vecs
                            print(f"  Functional vectors (original, not filtered): {func_vectors.shape}")
        else:
            # Use original (non-filtered)
            func_base = "func_vectors.norm" if use_normalized_func else "func_vectors"
            func_vectors_path = func_dir / f"{func_base}.npy"
            if func_vectors_path.exists():
                func_vectors = np.load(func_vectors_path)
                print(f"  Functional vectors: {func_vectors.shape}")
        
        valid_mask_path = func_dir / "valid_mask.npy"
        if valid_mask_path.exists():
            func_valid_mask = np.load(valid_mask_path)
            print(f"  Functional valid mask: {func_valid_mask.shape}")
        access_target_path = func_dir / "access_target.npy"
        if access_target_path.exists():
            access_target = np.load(access_target_path)
            print(f"  Access target: {access_target.shape}")
        access_valid_mask_path = func_dir / "access_valid_mask.npy"
        if access_valid_mask_path.exists():
            access_valid_mask = np.load(access_valid_mask_path)
            print(f"  Access valid mask: {access_valid_mask.shape}")
        feature_map_path = func_dir / "feature_map.json"
        if feature_map_path.exists():
            with open(feature_map_path, "r") as f:
                feature_map = json.load(f)
            print(f"  Feature map tracks: {len(feature_map.get('bigwigs', []))}")
    
    # Load splits
    if splits_npz:
        splits_path = Path(splits_npz)
    elif split_file:
        splits_path = data_dir / "labels" / split_file
    elif use_filtered:
        splits_path = data_dir / "labels" / "splits_filtered_90_5_5.npz"
        if not splits_path.exists():
            print(f"  [WARNING] Filtered splits not found, using original")
            splits_path = data_dir / "labels" / "splits.npz"
    else:
        splits_path = data_dir / "labels" / "splits.npz"
    
    splits = np.load(splits_path, allow_pickle=True)
    train_indices = splits['train']
    val_indices = splits['val']
    test_indices = splits['test']
    
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return {
        "sequences": sequences,
        "labels": labels,
        "functional": func_vectors,
        "functional_mask": func_valid_mask,
        "access_target": access_target,
        "access_valid_mask": access_valid_mask,
        "feature_map": feature_map,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def train_epoch(model, dataloader, criterion, optimizer, device, feature_map, functional_mode):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        functional = batch.get('functional')
        functional_mask = batch.get('functional_mask')

        if functional is not None:
            functional = functional.to(device)
        if functional_mask is not None:
            functional_mask = functional_mask.to(device)
        
        functional, functional_mask = adjust_functional(functional, functional_mask, feature_map, functional_mode)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(
            sequences,
            functional=functional,
            functional_mask=functional_mask,
        )
        
        # Use combined loss (MSE + Correlation)
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, criterion, device, feature_map, functional_mode):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            functional = batch.get('functional')
            if functional is not None:
                functional = functional.to(device)
            functional_mask = batch.get('functional_mask')
            if functional_mask is not None:
                functional_mask = functional_mask.to(device)
            
            functional, functional_mask = adjust_functional(functional, functional_mask, feature_map, functional_mode)

            predictions = model(
                sequences,
                functional=functional,
                functional_mask=functional_mask,
            )
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / n_batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Calculate correlation
    correlation = np.corrcoef(all_predictions, all_labels)[0, 1]
    
    return avg_loss, correlation


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Also save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def plot_training_curves(train_losses, val_losses, val_correlations=None, output_dir=None):
    """Plot training curves with loss and correlation."""
    output_dir = Path(output_dir) if output_dir else Path("results/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Correlation curve
    if val_correlations:
        axes[1].plot(val_correlations, label='Val Correlation', marker='^', color='green', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero correlation')
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Pearson Correlation', fontweight='bold')
        axes[1].set_title('Validation Correlation Over Time', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved training curves: {plot_path}")
    plt.close()


def main(args):
    print("=" * 60)
    print("EpiBERT Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  Using CPU (GPU not available)")
    
    # Load data
    data = load_data(
        args.data_dir,
        use_normalized_func=args.use_normalized_func,
        use_normalized_labels=args.use_normalized_labels,
        use_filtered=args.use_filtered,
        split_file=args.split_file,
        sequences_npz=args.sequences_npz,
        labels_npy=args.labels_npy,
        splits_npz=args.splits_npz,
    )
    sequences = data["sequences"]
    labels = data["labels"]
    functional = data["functional"]
    functional_mask = data["functional_mask"]
    access_target = data["access_target"]
    access_valid_mask = data["access_valid_mask"]
    feature_map = data["feature_map"]
    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    test_indices = data["test_indices"]
    
    # Create datasets
    train_dataset = ChromatinAccessibilityDataset(
        sequences,
        labels,
        train_indices,
        functional_vectors=functional,
        functional_valid_mask=functional_mask,
        access_targets=access_target,
        access_valid_mask=access_valid_mask,
    )
    val_dataset = ChromatinAccessibilityDataset(
        sequences,
        labels,
        val_indices,
        functional_vectors=functional,
        functional_valid_mask=functional_mask,
        access_targets=access_target,
        access_valid_mask=access_valid_mask,
    )
    test_dataset = ChromatinAccessibilityDataset(
        sequences,
        labels,
        test_indices,
        functional_vectors=functional,
        functional_valid_mask=functional_mask,
        access_targets=access_target,
        access_valid_mask=access_valid_mask,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print(f"\nCreating model...")
    model = MultiModalAccessibilityModel(
        seq_len=sequences.shape[1],
        feature_map=data["feature_map"],
        seq_embed_dim=args.seq_embed_dim,
        fusion_embed_dim=args.fusion_embed_dim,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        dropout=args.dropout,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function - Use combined MSE + Correlation loss to prevent collapse
    print(f"\nLoss function: Combined MSE + Correlation Loss")
    print(f"  MSE weight: {args.mse_weight:.2f}")
    print(f"  Correlation weight: {args.correlation_weight:.2f}")
    criterion = CombinedLoss(mse_weight=args.mse_weight, correlation_weight=args.correlation_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    def warmup_lambda(epoch):
        if epoch <= 3:
            return (epoch + 1) / 4  # Warmup for first 3 epochs
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Reduce LR on plateau after warmup
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    val_correlations = []
    best_val_loss = float('inf')
    best_val_correlation = -float('inf')
    patience_counter_loss = 0
    patience_counter_corr = 0
    early_stop_patience = args.early_stop_patience
    
    print(f"\nEarly stopping patience: {early_stop_patience} epochs")
    print(f"Early stopping based on: {'Correlation' if args.early_stop_on_correlation else 'Loss'}")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            feature_map,
            args.functional_mode,
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_correlation = validate(
            model,
            val_loader,
            criterion,
            device,
            feature_map,
            args.functional_mode,
        )
        val_losses.append(val_loss)
        val_correlations.append(val_correlation)
        
        # Learning rate scheduling - use correlation if available, else loss
        if epoch <= 3:
            warmup_scheduler.step()  # Warmup phase
        else:
            # Use negative correlation for plateau scheduler (we want to maximize correlation)
            if args.lr_schedule_on_correlation:
                plateau_scheduler.step(-val_correlation)  # Negative because we want to maximize
            else:
                plateau_scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val Correlation: {val_correlation:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best checkpoint based on correlation (more important than loss)
        improved = False
        if args.early_stop_on_correlation:
            # Early stop and save based on correlation
            if val_correlation > best_val_correlation:
                best_val_correlation = val_correlation
                best_val_loss = val_loss
                patience_counter_corr = 0
                improved = True
                save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
                print(f"  New best model! (val_correlation: {val_correlation:.4f})")
            else:
                patience_counter_corr += 1
                print(f"  No correlation improvement ({patience_counter_corr}/{early_stop_patience})")
            
            # Also track loss improvement for monitoring
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        else:
            # Early stop and save based on loss (original behavior)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_correlation = val_correlation
                patience_counter_loss = 0
                improved = True
                save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
                print(f"  New best model! (val_loss: {val_loss:.6f})")
            else:
                patience_counter_loss += 1
                print(f"  No improvement ({patience_counter_loss}/{early_stop_patience})")
        
        # Early stopping
        if args.early_stop_on_correlation:
            if patience_counter_corr >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (correlation not improving)")
                print(f"Best validation correlation: {best_val_correlation:.4f}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
        else:
            if patience_counter_loss >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (loss not improving)")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"Best validation correlation: {best_val_correlation:.4f}")
                break
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
    
    # Final evaluation on test set
    print(f"\nEvaluating on test set...")
    test_loss, test_correlation = validate(
        model,
        test_loader,
        criterion,
        device,
        feature_map,
        args.functional_mode,
    )
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Correlation: {test_correlation:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_correlations, args.output_dir)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_correlations': val_correlations,
        'best_val_loss': float(best_val_loss),
        'best_val_correlation': float(best_val_correlation),
        'best_epoch': int(np.argmin(val_losses) + 1),
        'final_test_loss': float(test_loss),
        'final_test_correlation': float(test_correlation),
        'total_epochs_trained': len(train_losses)
    }
    
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history: {history_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EpiBERT model")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--checkpoint_dir", type=str, default="models/pretrained",
                       help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, default="results/training",
                       help="Directory to save training outputs")
    
    # Model hyperparameters
    parser.add_argument("--seq_embed_dim", type=int, default=256,
                       help="Sequence encoder embedding dimension (default: 256)")
    parser.add_argument("--fusion_embed_dim", type=int, default=128,
                       help="Cross-modal fusion embedding dimension (default: 128)")
    parser.add_argument("--fusion_layers", type=int, default=2,
                       help="Number of transformer layers in fusion module (default: 2)")
    parser.add_argument("--fusion_heads", type=int, default=8,
                       help="Number of attention heads in fusion module (default: 8)")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate used across modules (default: 0.3 for small dataset)")
    parser.add_argument("--use_normalized_func", action="store_true",
                        help="Use normalized functional vectors (func_vectors.norm.npy)")
    parser.add_argument("--use_normalized_labels", action="store_true",
                       help="Use normalized accessibility labels (accessibility_scores.norm.npy)")
    parser.add_argument("--use_filtered", action="store_true",
                       help="Use filtered high-quality sequences (recommended)")
    parser.add_argument("--split_file", type=str, default=None,
                       help="Custom split file (default: auto-detect based on use_filtered)")
    parser.add_argument("--sequences_npz", type=str, default=None,
                       help="Override sequences NPZ path (e.g. data/processed/sequences/encoded_sequences.k562_1kb.npz)")
    parser.add_argument("--labels_npy", type=str, default=None,
                       help="Override labels NPY path (e.g. data/processed/labels/accessibility_scores.k562_1kb.npy)")
    parser.add_argument("--splits_npz", type=str, default=None,
                       help="Override splits NPZ path (e.g. data/processed/labels/splits_k562_1kb_90_5_5.npz)")
    parser.add_argument("--functional_mode", choices=["full", "atac", "none"], default="full",
                       help="Select which functional modalities to use (default: full)")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5, lower for small dataset)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay (default: 1e-4, increased to prevent collapse)")
    parser.add_argument("--mse_weight", type=float, default=0.5,
                       help="Weight for MSE component in combined loss (default: 0.5)")
    parser.add_argument("--correlation_weight", type=float, default=0.5,
                       help="Weight for correlation component in combined loss (default: 0.5)")
    parser.add_argument("--early_stop_on_correlation", action="store_true",
                       help="Early stop based on correlation improvement instead of loss")
    parser.add_argument("--lr_schedule_on_correlation", action="store_true",
                       help="Schedule learning rate based on correlation instead of loss")
    parser.add_argument("--save_every", type=int, default=5,
                       help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--early_stop_patience", type=int, default=15,
                       help="Early stopping patience (default: 15 epochs)")
    
    args = parser.parse_args()
    main(args)

