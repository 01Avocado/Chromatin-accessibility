#!/usr/bin/env python3
"""
Multi-Modal Transformer model for chromatin accessibility prediction.

Architecture (mirrors project design diagram):
  - Sequence encoder: transformer over one-hot DNA sequence
  - Histone encoder: 1D CNN over H3K27ac/H3K27me3/H3K4me3 bins
  - ATAC encoder: linear projection of ATAC aggregated bins
  - TF encoder: MLP over TF binding profile bins (e.g., CTCF)
  - Cell embedding: learned embedding (single cell type -> learned vector)
  - Cross-modal fusion: transformer encoder over modality embeddings
  - Regression head: predicts continuous accessibility score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_track_index_map(feature_map: Dict) -> Dict[str, Tuple[int, int]]:
    """
    Build mapping from track filename to (start_idx, end_idx).
    Assumes each track contributes `n_bins` consecutive features.
    """
    if feature_map is None:
        return {}

    n_bins = feature_map.get("n_bins", 16)
    track_map = {}
    for i, track in enumerate(feature_map.get("bigwigs", [])):
        start = i * n_bins
        end = start + n_bins
        track_map[track] = (start, end)
    return track_map


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(4, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: (batch, seq_len, 4)
        Returns:
            encoded representation (batch, embed_dim)
        """
        x = self.input_proj(sequence)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.dropout(x)


class HistoneEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, n_bins: int = 16, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.in_channels = in_channels
        self.n_bins = n_bins

    def forward(self, histone: torch.Tensor) -> torch.Tensor:
        # histone: (batch, in_channels * n_bins)
        batch = histone.shape[0]
        x = histone.view(batch, self.in_channels, self.n_bins)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.out(x)


class ATACEncoder(nn.Module):
    def __init__(self, n_bins: int = 16, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bins, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, atac: torch.Tensor) -> torch.Tensor:
        return self.net(atac)


class TFEncoder(nn.Module):
    def __init__(self, n_bins: int = 16, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bins, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, tf: torch.Tensor) -> torch.Tensor:
        return self.net(tf)


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, modalities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modalities: (batch, num_modalities, embed_dim)
        Returns:
            fused vector (batch, embed_dim)
        """
        x = self.encoder(modalities)
        return x.mean(dim=1)


@dataclass
class ModalityConfig:
    atac_track: Optional[str]
    histone_tracks: List[str]
    tf_tracks: List[str]
    n_bins: int


def infer_modality_config(feature_map: Optional[Dict]) -> ModalityConfig:
    if not feature_map:
        return ModalityConfig(atac_track=None, histone_tracks=[], tf_tracks=[], n_bins=16)

    tracks = feature_map.get("bigwigs", [])
    n_bins = feature_map.get("n_bins", 16)

    atac = None
    histones: List[str] = []
    tf_tracks: List[str] = []

    for track in tracks:
        track_lower = track.lower()
        if "atac" in track_lower:
            atac = track
        elif "h3k" in track_lower:
            histones.append(track)
        else:
            tf_tracks.append(track)

    return ModalityConfig(atac_track=atac, histone_tracks=histones, tf_tracks=tf_tracks, n_bins=n_bins)


class ModalityGate(nn.Module):
    def __init__(self, embed_dim: int, init_value: float = 0.0):
        super().__init__()
        self.logit = nn.Parameter(torch.full((1,), init_value))
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.logit)
        return gate * self.proj(x)


class MultiModalAccessibilityModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_map: Optional[Dict] = None,
        seq_embed_dim: int = 256,
        fusion_embed_dim: int = 128,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_cfg = infer_modality_config(feature_map)
        self.track_index_map = _build_track_index_map(feature_map)

        self.sequence_encoder = SequenceEncoder(
            seq_len=seq_len,
            embed_dim=seq_embed_dim,
            n_heads=8,
            n_layers=4,
            dim_feedforward=1024,
            dropout=dropout,
        )

        self.sequence_projection = nn.Sequential(
            nn.Linear(seq_embed_dim, fusion_embed_dim),
            nn.LayerNorm(fusion_embed_dim),
        )

        self.histone_encoder = HistoneEncoder(
            in_channels=len(self.modality_cfg.histone_tracks) or 1,
            n_bins=self.modality_cfg.n_bins,
            embed_dim=fusion_embed_dim,
            dropout=dropout,
        )

        self.histone_gate = ModalityGate(fusion_embed_dim, init_value=0.5)

        self.atac_encoder = ATACEncoder(
            n_bins=self.modality_cfg.n_bins,
            embed_dim=fusion_embed_dim,
            dropout=dropout,
        )

        self.atac_gate = ModalityGate(fusion_embed_dim, init_value=0.5)

        self.tf_encoder = TFEncoder(
            n_bins=self.modality_cfg.n_bins * max(len(self.modality_cfg.tf_tracks), 1),
            embed_dim=fusion_embed_dim,
            dropout=dropout,
        )

        self.tf_gate = ModalityGate(fusion_embed_dim, init_value=0.5)

        self.cell_embedding = nn.Parameter(torch.randn(1, fusion_embed_dim))

        self.fusion = CrossModalFusion(
            embed_dim=fusion_embed_dim,
            n_heads=fusion_heads,
            n_layers=fusion_layers,
            dropout=dropout,
        )

        self.output_head = nn.Sequential(
            nn.Linear(fusion_embed_dim, fusion_embed_dim // 2),
            nn.LayerNorm(fusion_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_embed_dim // 2, 1),
        )

    def _slice_track(self, functional: torch.Tensor, track: str) -> Optional[torch.Tensor]:
        if track not in self.track_index_map:
            return None
        start, end = self.track_index_map[track]
        return functional[:, start:end]

    def forward(
        self,
        sequence: torch.Tensor,
        functional: Optional[torch.Tensor] = None,
        functional_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence: (batch, seq_len, 4)
            functional: (batch, n_features) or None
        """
        batch_size = sequence.size(0)

        # Sequence modality
        seq_repr = self.sequence_encoder(sequence)
        seq_repr = self.sequence_projection(seq_repr)

        modality_embeddings = [seq_repr.unsqueeze(1)]

        if functional is not None:
            # ATAC
            atac_track = None
            if self.modality_cfg.atac_track:
                atac_track = self._slice_track(functional, self.modality_cfg.atac_track)
            if atac_track is None and functional.size(1) >= self.modality_cfg.n_bins:
                atac_track = functional[:, : self.modality_cfg.n_bins]
            atac_repr = self.atac_gate(self.atac_encoder(atac_track))
            modality_embeddings.append(atac_repr.unsqueeze(1))

            # Histones
            histone_tracks = self.modality_cfg.histone_tracks
            if histone_tracks:
                histone_slices = [self._slice_track(functional, t) for t in histone_tracks]
                histone_tensor = torch.stack(histone_slices, dim=1)  # (batch, n_histone, n_bins)
                histone_tensor = histone_tensor.view(batch_size, -1)
            else:
                histone_tensor = atac_track  # fallback to ATAC-only
            histone_repr = self.histone_gate(self.histone_encoder(histone_tensor))
            modality_embeddings.append(histone_repr.unsqueeze(1))

            # TF tracks
            tf_tracks = self.modality_cfg.tf_tracks
            if tf_tracks:
                tf_slices = [self._slice_track(functional, t) for t in tf_tracks]
                tf_tensor = torch.stack(tf_slices, dim=1).view(batch_size, -1)
            else:
                tf_tensor = atac_track
            tf_repr = self.tf_gate(self.tf_encoder(tf_tensor))
            modality_embeddings.append(tf_repr.unsqueeze(1))

        # Cell embedding (single learned vector)
        cell_embed = self.cell_embedding.expand(batch_size, -1).unsqueeze(1)
        modality_embeddings.append(cell_embed)

        fused_input = torch.cat(modality_embeddings, dim=1)  # (batch, num_modalities, embed_dim)
        fused = self.fusion(fused_input)
        prediction = self.output_head(fused).squeeze(-1)
        return prediction

