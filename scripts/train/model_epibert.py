#!/usr/bin/env python3
"""
EpiBERT Model Architecture

Transformer-based model for predicting chromatin accessibility from DNA sequences.
Implements Tiny EpiBERT configuration for initial testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""
    
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with self-attention."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        # Self-attention with residual connection
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward with residual connection
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class EpiBERT(nn.Module):
    """
    EpiBERT Model: Transformer-based chromatin accessibility predictor.
    
    Architecture:
    1. Sequence embedding (one-hot → dense embeddings)
    2. Positional encoding
    3. Transformer encoder blocks
    4. Output head (regression for accessibility prediction)
    """
    
    def __init__(
        self,
        seq_len=1500,
        n_nucleotides=4,
        embed_dim=256,
        n_layers=6,
        n_heads=8,
        dim_feedforward=1024,
        dropout=0.1,
        output_dim=1
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Sequence embedding: one-hot (4) → dense (embed_dim)
        self.sequence_embedding = nn.Linear(n_nucleotides, embed_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        
        # Transformer encoder blocks
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU instead of ReLU for better gradients
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # Output head: global pooling + MLP for regression
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (batch, embed_dim, 1)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.BatchNorm1d(embed_dim // 4),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim)
            # REMOVED SIGMOID - let model learn the range naturally
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_nucleotides)
               One-hot encoded DNA sequences
        
        Returns:
            output: Accessibility predictions of shape (batch, output_dim)
        """
        # x shape: (batch, seq_len, 4)
        batch_size = x.size(0)
        
        # Embed sequences: (batch, seq_len, 4) → (batch, seq_len, embed_dim)
        x = self.sequence_embedding(x)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, embed_dim)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, embed_dim)
        
        # Global pooling: (batch, seq_len, embed_dim) → (batch, embed_dim)
        # Transpose for pooling: (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.global_pool(x)  # (batch, embed_dim, 1)
        x = x.squeeze(-1)  # (batch, embed_dim) - now 2D for BatchNorm1d
        
        # Output head (BatchNorm1d expects 2D input: batch, features)
        output = self.output_head(x)  # (batch, output_dim)
        
        return output.squeeze(-1)  # (batch,)
    
    def _initialize_weights(self):
        """Initialize model weights with better strategy."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Special initialization for output layer (last linear layer)
                    if 'output_head' in name and module.out_features == 1:
                        nn.init.constant_(module.bias, 0.5)  # Start near label mean
                    else:
                        nn.init.constant_(module.bias, 0)


def create_tiny_epibert(seq_len=1500, **kwargs):
    """
    Create Tiny EpiBERT model for initial testing.
    
    Args:
        seq_len: Sequence length (default: 1500)
        **kwargs: Additional arguments to override defaults
    
    Returns:
        EpiBERT model with small configuration
    """
    defaults = {
        'embed_dim': 256,
        'n_layers': 6,
        'n_heads': 8,
        'dim_feedforward': 1024,
        'dropout': 0.1
    }
    defaults.update(kwargs)
    
    return EpiBERT(seq_len=seq_len, **defaults)


if __name__ == "__main__":
    # Test model with dummy data
    print("Testing EpiBERT model...")
    
    # Create Tiny EpiBERT
    model = create_tiny_epibert(seq_len=1500)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1500, 4)  # Random one-hot like input
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("\nSUCCESS: Model test passed!")

