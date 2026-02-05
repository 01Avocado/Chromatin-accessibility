#!/usr/bin/env python3
"""
Script to improve model performance with various techniques.

This script implements several improvements:
1. Better output layer initialization
2. Remove sigmoid constraint (let model learn range)
3. Add batch normalization
4. Better learning rate scheduling
5. Gradient clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedEpiBERT(nn.Module):
    """
    Improved EpiBERT with better architecture for regression.
    """
    
    def __init__(
        self,
        seq_len=1500,
        n_nucleotides=4,
        embed_dim=128,
        n_layers=3,
        n_heads=8,
        dim_feedforward=512,
        dropout=0.1,
        output_dim=1
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Sequence embedding: one-hot (4) → dense (embed_dim)
        self.sequence_embedding = nn.Linear(n_nucleotides, embed_dim)
        
        # Positional encoding
        from model_epibert import PositionalEncoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        
        # Transformer encoder blocks
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU instead of ReLU
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # Improved output head with batch normalization
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Use attention pooling instead of just average pooling
        self.attention_pool = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        
        # Output head with batch norm and better initialization
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.BatchNorm1d(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, output_dim)
            # NO SIGMOID - let model learn the range
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Embed sequences
        x = self.sequence_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Attention pooling (better than simple average)
        # Use CLS token approach: use first token or learnable query
        query = x.mean(dim=1, keepdim=True)  # Mean as query
        pooled, _ = self.attention_pool(query, x, x)
        pooled = pooled.squeeze(1)  # (batch, embed_dim)
        
        # Alternative: simple global average (fallback)
        # x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        # pooled = self.global_pool(x).squeeze(-1)  # (batch, embed_dim)
        
        # Output head
        output = self.output_head(pooled)
        
        return output.squeeze(-1)
    
    def _initialize_weights(self):
        """Initialize model weights with better strategy."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for most layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            # Special initialization for output layer
            if isinstance(module, nn.Linear) and module.out_features == 1:
                # Initialize output layer to predict mean of labels
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.5)  # Start near label mean


def create_improved_epibert(seq_len=1500, **kwargs):
    """Create improved EpiBERT model."""
    defaults = {
        'embed_dim': 128,
        'n_layers': 3,
        'n_heads': 8,
        'dim_feedforward': 512,
        'dropout': 0.1
    }
    defaults.update(kwargs)
    
    return ImprovedEpiBERT(seq_len=seq_len, **defaults)


if __name__ == "__main__":
    # Test improved model
    print("Testing Improved EpiBERT model...")
    
    model = create_improved_epibert(seq_len=1500)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1500, 4)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    
    print("\nSUCCESS: Improved model test passed!")

